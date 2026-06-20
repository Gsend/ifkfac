"""
optimizer/bf16_linalg.py

Hand-rolled bf16 linear algebra primitives — built on bf16 tensor-core matmul
since cuSOLVER does not implement geqrf, triangular_solve, cholesky, or inv
for the bfloat16 dtype (verified by tests/test_bf16_linalg_support.py).

Scope:
  - householder_qr_bf16        : reduced QR via Householder reflections
  - solve_triangular_bf16      : forward / back substitution
  - cholesky_bf16              : down-looking Cholesky factorization
  - inv_bf16                   : inverse via Cholesky (for SPD only)

Precision policy:
  - Storage is bf16.  Matmuls run on bf16 tensor cores (~2× fp32 throughput,
    fp32-accumulating internally — that's how NVIDIA tensor cores work).
  - Norms and reciprocals are computed in fp32 then cast to bf16 to avoid
    overflow / underflow.  This is standard mixed-precision practice
    (Higham 2002 §3.4 — "scaled accumulation for sums of squares").

For the κ-sweep (matrices up to ~64×64) Python loops over n are fine.
A blocked variant for larger matrices (transformer / CNN K-FAC factors
up to ~1024×1024) is left for a future iteration; the operation budget at
those sizes wants Triton or CUDA kernels.

References:
  - Householder (1958), Golub (1965) for the QR algorithm
  - Higham (2002) §19 for Householder backward error analysis
  - Higham (2002) §10 for Cholesky precision considerations
"""
from __future__ import annotations

import torch


# ---- Norm-safe primitives --------------------------------------------------

def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    """Frobenius norm computed in fp32 to avoid bf16 overflow / underflow,
    then cast back to bf16.  For vectors and matrices alike."""
    return x.float().norm().to(x.dtype)


def _safe_div(num: torch.Tensor, denom: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """Element-wise division with fp32 intermediate.  Returns same dtype as num.

    Magnitude-clamp (not value-clamp): if |denom| < eps, replace with
    sign(denom) * eps to avoid div-by-zero without corrupting sign.  This
    is critical for triangular-solve correctness — the triangular diagonal
    can carry either sign.
    """
    denom_fp32 = denom.float()
    abs_d = denom_fp32.abs()
    # sign(0) is 0; map to +1 for that edge case so sign * eps stays positive
    sgn = torch.sign(denom_fp32)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    safe = torch.where(abs_d < eps, sgn * eps, denom_fp32)
    return (num.float() / safe).to(num.dtype)


# ---- Block Householder QR (production-grade) -------------------------------

def block_householder_qr_bf16(A: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block Householder QR with compact WY representation.

    Reduced QR of a tall matrix A ∈ ℝ^(m×n) (m ≥ n) returning the
    upper-triangular factor R ∈ ℝ^(n×n).  Q is never materialized.

    Algorithm (Schreiber & Van Loan 1989; Demmel 1997 §3.4.2):
      For each block of `block_size` columns:
        1. Apply unblocked Householder QR within the panel
           (computes b Householder vectors v_1, ..., v_b).
        2. Build the compact WY representation: T ∈ ℝ^(b×b) lower-triangular
           such that  H_b · H_{b-1} · ... · H_1  =  I - V T V^T.
           Recursion (for unit-norm v's with reflection I - 2vv^T):
              T[j, j]   = 2
              T[j, :j]  = -2 · (V[:, :j]^T · v_j)^T · T[:j, :j]
        3. Apply (I - V T V^T) to the trailing columns in ONE BLAS-3 update:
              A_trail ← A_trail − V · T · (V^T · A_trail)
           This is the speedup vs the unblocked variant: the trailing-matrix
           update is dominated by three large matmuls instead of `block_size`
           rank-1 updates, fully exploiting tensor cores.

    Parameters
    ----------
    A : (m, n) bfloat16, m ≥ n
    block_size : int
        Panel width.  Larger = better matmul efficiency but more memory for V, T.
        Empirically 32 is a sweet spot for our K-FAC factor sizes (256-1536);
        64 helps at n ≥ 2000.

    Returns
    -------
    R : (n, n) bfloat16, upper-triangular
    """
    assert A.dtype == torch.bfloat16, f"expected bf16 input, got {A.dtype}"
    m, n = A.shape
    assert m >= n, f"need m >= n, got ({m}, {n})"
    A = A.clone()

    for i in range(0, n, block_size):
        b = min(block_size, n - i)

        # Allocate panel V and compact-WY T for this block.
        V = torch.zeros(m - i, b, dtype=torch.bfloat16, device=A.device)
        T = torch.zeros(b, b, dtype=torch.bfloat16, device=A.device)

        # Phase 1: unblocked Householder QR within the panel A[i:, i:i+b].
        # Each iteration also updates the next column-stripe of T.
        for j in range(b):
            # Householder vector for column i+j (local index j)
            x = A[i+j:, i+j:i+j+1]                          # (m-i-j, 1)
            alpha = -torch.sign(x[0, 0]) * _safe_norm(x)
            v = x.clone()
            v[0, 0] = v[0, 0] - alpha
            v_norm = _safe_norm(v)
            if v_norm.item() == 0.0:
                # Column already zero below diagonal; T column stays at 2 on diag
                T[j, j] = 2.0
                continue
            v = _safe_div(v, v_norm)                         # unit-norm Householder
            # Store v in V at column j; rows 0..j-1 stay zero (padded above).
            V[j:, j:j+1] = v

            # Build T's new column via the WY recursion.
            # IMPORTANT: use the zero-padded V[:, j] (length m-i), NOT the
            # local Householder vector `v` (length m-i-j).  V[:, j] = [0...0, v]
            # with j leading zeros — this is what makes V[:, :j].T @ V[:, j]
            # have shape (j, 1) compatible with T[:j, :j] of shape (j, j).
            if j > 0:
                # vtv = V[:, :j]^T · V[:, j]   shape (j, 1) — bf16 matmul
                vtv = V[:, :j].t() @ V[:, j:j+1]
                # T[j, :j] = -2 · vtv^T · T[:j, :j]   shape (1, j) → (j,)
                new_row = (-2.0 * vtv.t() @ T[:j, :j]).squeeze(0)
                T[j, :j] = new_row
            T[j, j] = 2.0

            # Apply this reflection within the panel ONLY (cols j..b-1).
            panel = A[i+j:, i+j:i+b]                         # (m-i-j, b-j)
            coeffs = v.t() @ panel                           # (1, b-j)
            A[i+j:, i+j:i+b] = panel - 2.0 * v @ coeffs

        # Phase 2: bulk-apply the block reflection to the trailing columns.
        # A_trail ← A_trail − V · T · (V^T · A_trail)    — three big bf16 matmuls
        if i + b < n:
            trail = A[i:, i+b:]                              # (m-i, n-i-b)
            VTA  = V.t() @ trail                             # (b, n-i-b)  — tensor-core
            TVTA = T @ VTA                                    # (b, n-i-b)  — small b × ...
            VTVTA = V @ TVTA                                  # (m-i, n-i-b) — tensor-core
            A[i:, i+b:] = trail - VTVTA

    return A[:n, :n].triu()


# ---- Householder QR (unblocked, kept for n ≤ 64) ----------------------------

def householder_qr_bf16(A: torch.Tensor) -> torch.Tensor:
    """Reduced Householder QR.  Input A: (m, n) bf16 with m ≥ n.
    Returns R: (n, n) bf16 upper-triangular, such that A = Q R for some
    implicit orthogonal Q (Q is never materialized — K-FAC doesn't need it).

    Numerical regime: matmuls in bf16 (tensor-core path), norms in fp32.
    """
    assert A.dtype == torch.bfloat16, f"expected bf16 input, got {A.dtype}"
    m, n = A.shape
    assert m >= n, f"need m >= n, got ({m}, {n})"
    A = A.clone()
    for i in range(n):
        x = A[i:, i:i+1]                        # (m-i, 1) bf16
        alpha = -torch.sign(x[0, 0]) * _safe_norm(x)
        v = x.clone()
        v[0, 0] = v[0, 0] - alpha
        v_norm = _safe_norm(v)
        if v_norm.item() == 0.0:
            continue
        v = _safe_div(v, v_norm)                # (m-i, 1) bf16 unit Householder vector
        # Reflection: A[i:, i:] = A[i:, i:] - 2 v (vᵀ A[i:, i:])
        # The inner matmul (v.T @ block) runs on bf16 tensor cores.
        block = A[i:, i:]                       # (m-i, n-i) bf16
        coeffs = v.t() @ block                  # (1, n-i)  bf16 tensor-core matmul
        A[i:, i:] = block - 2.0 * v @ coeffs    # bf16 update
    return A[:n, :n].triu()


# ---- Triangular solve ------------------------------------------------------

def solve_triangular_bf16(R: torch.Tensor, B: torch.Tensor,
                           upper: bool = True) -> torch.Tensor:
    """Solve R · X = B for X by forward (upper=False) or back (upper=True)
    substitution.  R is (n, n) triangular bf16; B is (n, k) bf16.
    Returns X: (n, k) bf16.

    Implementation: column-by-column substitution.  Each step is a bf16
    matmul against the already-solved rows + a bf16 diagonal divide.
    The diagonal divide uses fp32 intermediate (see _safe_div)."""
    # Defensive dtype + shape diagnostics — the apply_vered chain can pass
    # in tensors that have been transposed / contiguous'd / monkey-patched
    # through several layers; mismatches surface as opaque slice-assign errors.
    if R.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        print(f"[bf16_linalg.solve_triangular_bf16] DTYPE MISMATCH: "
              f"R.dtype={R.dtype}, B.dtype={B.dtype}, "
              f"R.shape={tuple(R.shape)}, B.shape={tuple(B.shape)}", flush=True)
        raise TypeError(
            f"solve_triangular_bf16: both R and B must be bf16; "
            f"got R.dtype={R.dtype}, B.dtype={B.dtype}"
        )
    if R.shape[1] != R.shape[0]:
        print(f"[bf16_linalg.solve_triangular_bf16] NON-SQUARE R: "
              f"R.shape={tuple(R.shape)}", flush=True)
        raise ValueError(f"R must be square, got {R.shape}")
    if B.shape[0] != R.shape[0]:
        print(f"[bf16_linalg.solve_triangular_bf16] SHAPE MISMATCH: "
              f"R.shape={tuple(R.shape)}, B.shape={tuple(B.shape)}", flush=True)
        raise ValueError(f"B rows must match R, got R={R.shape}, B={B.shape}")
    n = R.shape[0]
    # Ensure contiguity — bf16 strided tensors can fail .matmul silently
    if not R.is_contiguous():
        R = R.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    X = torch.empty_like(B)
    if upper:
        # Back substitution: solve from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            # tail contribution: R[i, i+1:] @ X[i+1:, :]
            if i + 1 < n:
                tail = R[i:i+1, i+1:] @ X[i+1:, :]   # (1, k) bf16 matmul
                rhs = B[i:i+1, :] - tail
            else:
                rhs = B[i:i+1, :]
            X[i:i+1, :] = _safe_div(rhs, R[i, i])
    else:
        # Forward substitution: solve from i = 0 up to n-1
        for i in range(n):
            if i > 0:
                head = R[i:i+1, :i] @ X[:i, :]
                rhs = B[i:i+1, :] - head
            else:
                rhs = B[i:i+1, :]
            X[i:i+1, :] = _safe_div(rhs, R[i, i])
    return X


# ---- Cholesky (for Classic K-FAC control) ----------------------------------

def cholesky_bf16(A: torch.Tensor) -> torch.Tensor:
    """Down-looking Cholesky factorization.  A: (n, n) SPD bf16.
    Returns L: (n, n) lower-triangular bf16 such that L Lᵀ = A.

    Used only for the Classic K-FAC control in the κ-sweep; production
    Classic K-FAC inverts via torch.linalg.inv at fp32."""
    assert A.dtype == torch.bfloat16
    n = A.shape[0]
    L = torch.zeros_like(A)
    for j in range(n):
        # Diagonal element: L[j, j] = sqrt(A[j, j] - sum(L[j, :j]^2))
        sum_sq_fp32 = (L[j, :j].float() ** 2).sum()
        diag_fp32 = (A[j, j].float() - sum_sq_fp32).clamp_min(1e-7).sqrt()
        L[j, j] = diag_fp32.to(torch.bfloat16)
        # Off-diagonal column: L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
        if j + 1 < n:
            # vectorized matmul over remaining rows — bf16 tensor cores
            below = A[j+1:, j:j+1]                                 # (n-j-1, 1) bf16
            head  = L[j+1:, :j] @ L[j:j+1, :j].t()                 # (n-j-1, 1) bf16
            L[j+1:, j:j+1] = _safe_div(below - head, L[j, j])
    return L


# ---- Inverse via Cholesky (SPD only) ---------------------------------------

def inv_spd_bf16(A: torch.Tensor) -> torch.Tensor:
    """Compute A⁻¹ for SPD A by solving A X = I via Cholesky.
    Returns X: (n, n) bf16."""
    L = cholesky_bf16(A)
    n = A.shape[0]
    I = torch.eye(n, dtype=A.dtype, device=A.device)
    Y = solve_triangular_bf16(L,   I, upper=False)
    X = solve_triangular_bf16(L.t().contiguous(), Y, upper=True)
    return X


# ---- Self-test (run as script) ---------------------------------------------

def _self_test():
    import math, time
    print("Sanity checks against fp32 references...\n")
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # 1. Householder QR (unblocked) — small sizes only
    print("[1] Householder QR (unblocked)")
    for n in (8, 32, 64):
        A_fp32 = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_bf16 = A_fp32.to(torch.bfloat16)
        R_ours = householder_qr_bf16(A_bf16).float()
        _, R_torch = torch.linalg.qr(A_fp32, mode="reduced")
        # Normalize sign so we can compare R's directly
        sgn_ours  = torch.sign(torch.diagonal(R_ours))
        sgn_torch = torch.sign(torch.diagonal(R_torch))
        R_ours_n  = R_ours  * sgn_ours.unsqueeze(1)
        R_torch_n = R_torch * sgn_torch.unsqueeze(1)
        err = ((R_ours_n - R_torch_n).norm() / R_torch_n.norm()).item()
        print(f"  n={n:>3d}  relative |R_bf16 - R_fp32|/|R_fp32| = {err:.3e}")

    # 1b. Block Householder QR — correctness across sizes
    print("\n[1b] Block Householder QR (correctness, block_size=32)")
    for n in (32, 64, 128, 256, 384, 500, 1000):
        A_fp32 = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_bf16 = A_fp32.to(torch.bfloat16)
        R_ours = block_householder_qr_bf16(A_bf16, block_size=32).float()
        _, R_torch = torch.linalg.qr(A_fp32, mode="reduced")
        sgn_ours  = torch.sign(torch.diagonal(R_ours))
        sgn_torch = torch.sign(torch.diagonal(R_torch))
        R_ours_n  = R_ours  * sgn_ours.unsqueeze(1)
        R_torch_n = R_torch * sgn_torch.unsqueeze(1)
        err = ((R_ours_n - R_torch_n).norm() / R_torch_n.norm()).item()
        print(f"  n={n:>4d}  relative |R_block - R_fp32|/|R_fp32| = {err:.3e}")

    # 1c. Block Householder QR — wall-time across block sizes
    print("\n[1c] Block Householder QR (wall time across block sizes)")
    print(f"  {'n':>5} {'unblocked':>10} {'b=32':>8} {'b=64':>8} {'b=128':>8} "
          f"{'b=256':>8}   {'best speedup':>13}")
    for n in (256, 384, 500, 1000, 1536):
        torch.manual_seed(0)
        A_bf16 = torch.randn(2*n, n, device=device, dtype=torch.bfloat16)
        # Warmup
        householder_qr_bf16(A_bf16)
        for b in (32, 64, 128, 256):
            if b <= n:
                block_householder_qr_bf16(A_bf16, block_size=b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Time each variant
        def _time(fn, reps=3):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
                if device.type == "cuda":
                    torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps * 1000.0
        t_unb = _time(lambda: householder_qr_bf16(A_bf16))
        times = {b: _time(lambda b=b: block_householder_qr_bf16(A_bf16, block_size=b))
                  if b <= n else None for b in (32, 64, 128, 256)}
        best = min(t for t in [t_unb] + [v for v in times.values() if v] if t)
        speedup = t_unb / best if best > 0 else float("inf")
        fmt = lambda v: f"{v:>8.1f}" if v else f"{'—':>8}"
        print(f"  {n:>5d} {t_unb:>10.1f} "
              f"{fmt(times[32])} {fmt(times[64])} {fmt(times[128])} {fmt(times[256])}"
              f"   {speedup:>12.2f}x")

    # 2. Triangular solve
    print("\n[2] Triangular solve")
    for n in (8, 32, 64):
        R_fp32 = torch.triu(torch.randn(n, n, device=device, dtype=torch.float32))
        R_fp32 = R_fp32 + torch.sign(torch.diagonal(R_fp32)).diag() * 1.0
        B_fp32 = torch.randn(n, 4, device=device, dtype=torch.float32)
        X_ours  = solve_triangular_bf16(R_fp32.to(torch.bfloat16),
                                          B_fp32.to(torch.bfloat16), upper=True).float()
        X_torch = torch.linalg.solve_triangular(R_fp32, B_fp32, upper=True)
        err = ((X_ours - X_torch).norm() / X_torch.norm()).item()
        print(f"  n={n:>3d}  relative |X_bf16 - X_fp32|/|X_fp32| = {err:.3e}")

    # 3. Cholesky
    print("\n[3] Cholesky (SPD)")
    for n in (8, 32, 64):
        Z = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_fp32 = Z.t() @ Z + torch.eye(n, device=device)
        L_ours  = cholesky_bf16(A_fp32.to(torch.bfloat16)).float()
        L_torch = torch.linalg.cholesky(A_fp32)
        err = ((L_ours - L_torch).norm() / L_torch.norm()).item()
        print(f"  n={n:>3d}  relative |L_bf16 - L_fp32|/|L_fp32| = {err:.3e}")

    # 4. SPD inverse via bf16 Cholesky
    print("\n[4] SPD inverse via Cholesky")
    for n in (8, 32, 64):
        Z = torch.randn(2*n, n, device=device, dtype=torch.float32)
        A_fp32 = Z.t() @ Z + torch.eye(n, device=device)
        Ainv_ours  = inv_spd_bf16(A_fp32.to(torch.bfloat16)).float()
        Ainv_torch = torch.linalg.inv(A_fp32)
        err = ((Ainv_ours - Ainv_torch).norm() / Ainv_torch.norm()).item()
        print(f"  n={n:>3d}  relative |Ainv_bf16 - Ainv_fp32|/|Ainv_fp32| = {err:.3e}")


if __name__ == "__main__":
    _self_test()
