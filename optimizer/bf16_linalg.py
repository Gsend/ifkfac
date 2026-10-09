"""
optimizer/bf16_linalg.py

Pure-bf16 dense linear algebra built from bf16 tensor operations.  PyTorch
implements no bf16 QR, Cholesky, triangular solve or inverse (torch.linalg
raises "not implemented for 'BFloat16'" on CPU and CUDA), so these routines
are written from bf16 matmuls and element-wise bf16 operations.

Precision contract
------------------
Every tensor these functions create, read or return is bfloat16: there is
no .float() / .to(float32) anywhere.  bf16 has the same exponent range as
fp32, so norms and divisions need no fp32 detour to avoid overflow.  Inside
a single kernel, bf16 matmuls and reductions accumulate in fp32 and round the
result to bf16; that is how every bf16 kernel works on current hardware and
it never produces an fp32 tensor.  benchmark/bf16_guard.py checks this
contract at run time, and every entry point checks on every call that its
inputs are bf16 (optimizer/dtype_check.py; DtypeCheckError otherwise).

Routines (all accept a leading batch dimension unless noted)
------------------------------------------------------------
qr_r_bf16(A)                      R factor of A = QR, positive diagonal (Householder,
                                  compact-WY blocked); Q is never formed
TriSolver(T, lower=False)         blocked triangular solves T X = B and Tᵀ X = B,
                                  using inverted diagonal blocks + bf16 GEMMs
cholesky_bf16_checked(A)          blocked Cholesky, returns (L, ok); ok is False if a
                                  pivot is not positive or not finite (no clamping)
inv_spd_bf16_checked(A)           A⁻¹ through Cholesky, returns (X, ok)
Legacy names kept for tests/test_kappa_scaling.py: householder_qr_bf16,
block_householder_qr_bf16, solve_triangular_bf16, cholesky_bf16, inv_spd_bf16.

References: Householder (1958); Schreiber & Van Loan (1989) compact WY;
Higham (2002) §8, §10, §19.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from optimizer.dtype_check import expect

BF16 = torch.bfloat16


def _need_bf16(site: str, *ts: torch.Tensor):
    """Inputs must be bf16 (raises optimizer.dtype_check.DtypeCheckError, a TypeError)."""
    expect(f"bf16_linalg.{site}", *ts)


def _batched(A: torch.Tensor):
    return (A, False) if A.dim() == 3 else (A.unsqueeze(0), True)


# ---- QR -------------------------------------------------------------------------

def qr_r_bf16(A: torch.Tensor, block_size: int = 32, overwrite: bool = False) -> torch.Tensor:
    """R factor of the reduced QR of A (…, m, n), m ≥ n, with a positive diagonal.

    Blocked Householder with the compact-WY trailing update
    A_trail ← A_trail − V Tᵀ (Vᵀ A_trail), so the bulk of the work is bf16 GEMM.
    No host synchronisation: degenerate (zero) columns are handled with
    torch.where instead of a Python branch.  overwrite=True factors A in
    place (no working copy) - for callers that own a scratch stack.
    """
    _need_bf16("qr_r_bf16.input", A)
    A, squeeze = _batched(A)
    Bn, m, n = A.shape
    if m < n:
        raise ValueError(f"qr_r_bf16 needs m >= n, got m={m}, n={n}")
    if not overwrite:
        A = A.clone()
    dev = A.device
    one = torch.ones((), dtype=BF16, device=dev)
    for i in range(0, n, block_size):
        b = min(block_size, n - i)
        V = torch.zeros(Bn, m - i, b, dtype=BF16, device=dev)
        T = torch.zeros(Bn, b, b, dtype=BF16, device=dev)
        for j in range(b):
            c = i + j
            x = A[:, c:, c]                                         # (Bn, m-c)
            normx = torch.linalg.vector_norm(x, dim=1, keepdim=True)  # (Bn, 1)
            sgn = torch.where(x[:, :1] >= 0, one, -one)
            v = x.clone()
            v[:, :1] = x[:, :1] + sgn * normx                       # x0 - alpha, alpha = -sgn*|x|
            vn = torch.linalg.vector_norm(v, dim=1, keepdim=True)
            v = torch.where(vn > 0, v / vn, torch.zeros_like(v))    # unit Householder vector
            V[:, j:, j] = v
            if j > 0:
                vtv = torch.bmm(V[:, :, :j].transpose(1, 2), V[:, :, j:j + 1])      # (Bn, j, 1)
                T[:, j, :j] = (-2.0 * torch.bmm(vtv.transpose(1, 2), T[:, :j, :j])).squeeze(1)
            T[:, j, j] = 2.0
            panel = A[:, c:, c:i + b]                                # (Bn, m-c, b-j), view
            coeffs = torch.bmm(v.unsqueeze(1), panel)                # (Bn, 1, b-j)
            panel.baddbmm_(v.unsqueeze(2), coeffs, alpha=-2.0)       # in place
        if i + b < n:
            trail = A[:, i:, i + b:]                                 # view
            VtA = torch.bmm(V.transpose(1, 2), trail)                # (Bn, b, n-i-b)
            trail.baddbmm_(V, torch.bmm(T, VtA), alpha=-1.0)         # in place, no full-size temporary
    R = torch.triu(A[:, :n, :n])
    d = torch.diagonal(R, dim1=1, dim2=2)
    s = torch.where(d < 0, -one, one)
    R = R * s.unsqueeze(2)
    return R.squeeze(0) if squeeze else R


# ---- triangular solves -----------------------------------------------------------

def _inv_upper_blocks(U: torch.Tensor) -> torch.Tensor:
    """Inverses of a batch of small upper-triangular blocks U (K, b, b), bf16."""
    K, b, _ = U.shape
    X = torch.zeros_like(U)
    eye = torch.eye(b, dtype=BF16, device=U.device)
    diag = torch.diagonal(U, dim1=1, dim2=2)                          # (K, b)
    for i in range(b - 1, -1, -1):
        rhs = eye[i].expand(K, b)
        if i + 1 < b:
            rhs = rhs - torch.bmm(U[:, i:i + 1, i + 1:], X[:, i + 1:, :]).squeeze(1)
        X[:, i, :] = rhs / diag[:, i:i + 1]
    return X


class TriSolver:
    """Blocked solves with a fixed triangular matrix T (…, n, n), bf16.

    The inverses of the nb×nb diagonal blocks are computed once, when the
    solver is built (for K-FAC: at each factor refresh); each solve is then
    n/nb block steps of two bf16 GEMMs.  T may carry a leading batch
    dimension (several layers of the same shape solved together).
    """

    def __init__(self, T: torch.Tensor, lower: bool = False, nb: int = 64):
        _need_bf16("TriSolver.factor", T)
        U, self.squeeze = _batched(T)
        self.U = (U.transpose(1, 2) if lower else U).contiguous()     # stored upper
        self.lower = lower
        Bn, n, _ = self.U.shape
        self.n = n
        self.blocks: List[Tuple[int, int]] = [(s, min(s + nb, n)) for s in range(0, n, nb)]
        self.Dinv: List[torch.Tensor] = []
        full = [(s, e) for s, e in self.blocks if e - s == nb]
        if full:
            D = torch.stack([self.U[:, s:e, s:e] for s, e in full], dim=1)   # (Bn, K, nb, nb)
            Di = _inv_upper_blocks(D.reshape(-1, nb, nb)).reshape(Bn, len(full), nb, nb)
            self.Dinv = [Di[:, k] for k in range(len(full))]
        s, e = self.blocks[-1]
        if e - s != nb:
            self.Dinv.append(_inv_upper_blocks(self.U[:, s:e, s:e].contiguous()))

    def _back(self, B):            # U X = B
        X = torch.empty_like(B)
        for k in range(len(self.blocks) - 1, -1, -1):
            s, e = self.blocks[k]
            rhs = B[:, s:e]
            if e < self.n:
                rhs = rhs - torch.bmm(self.U[:, s:e, e:], X[:, e:])
            X[:, s:e] = torch.bmm(self.Dinv[k], rhs)
        return X

    def _fwd(self, B):             # Uᵀ X = B
        X = torch.empty_like(B)
        for k, (s, e) in enumerate(self.blocks):
            rhs = B[:, s:e]
            if s > 0:
                rhs = rhs - torch.bmm(self.U[:, :s, s:e].transpose(1, 2), X[:, :s])
            X[:, s:e] = torch.bmm(self.Dinv[k].transpose(1, 2), rhs)
        return X

    def _run(self, B, transpose):
        _need_bf16("TriSolver.rhs", B)
        B3 = B if B.dim() == 3 else B.unsqueeze(0)
        use_upper = (not self.lower) != transpose        # T X = B with T upper, or Tᵀ with T lower
        X = self._back(B3) if use_upper else self._fwd(B3)
        return X if B.dim() == 3 else X.squeeze(0)

    def solve(self, B: torch.Tensor) -> torch.Tensor:
        """X with T X = B."""
        return self._run(B, transpose=False)

    def solve_t(self, B: torch.Tensor) -> torch.Tensor:
        """X with Tᵀ X = B."""
        return self._run(B, transpose=True)


# ---- Cholesky and SPD inverse ------------------------------------------------------

def cholesky_bf16_checked(A: torch.Tensor, nb: int = 64) -> Tuple[torch.Tensor, bool]:
    """Lower Cholesky factor of the symmetric matrix A (n, n), bf16, blocked.

    Only the lower triangle of A is read.  No pivot is clamped: a pivot that
    is not positive (or not finite) makes ok False, and the returned L then
    contains NaN.  One host synchronisation, at the end.
    """
    _need_bf16("cholesky.input", A)
    n = A.shape[0]
    W = A.clone()
    L = torch.zeros_like(A)
    piv = torch.empty(n, dtype=BF16, device=A.device)
    for s in range(0, n, nb):
        e = min(s + nb, n)
        for J in range(s, e):
            d = W[J, J]
            if J > s:
                r = L[J, s:J]
                d = d - (r * r).sum()
            piv[J] = d
            ljj = torch.sqrt(d)
            L[J, J] = ljj
            if J + 1 < e:
                col = W[J + 1:e, J]
                if J > s:
                    col = col - L[J + 1:e, s:J] @ L[J, s:J]
                L[J + 1:e, J] = col / ljj
        if e < n:
            L11inv_T = TriSolver(L[s:e, s:e], lower=True, nb=e - s).Dinv[0]   # (L11ᵀ)⁻¹ as upper
            L[e:, s:e] = W[e:, s:e] @ L11inv_T[0]                 # W21 L11⁻ᵀ
            P = L[e:, s:e]
            W[e:, e:] = W[e:, e:] - P @ P.t()
    ok = bool((torch.isfinite(piv) & (piv > 0)).all()) and bool(torch.isfinite(L).all())
    return L, ok


def inv_spd_bf16_checked(A: torch.Tensor, nb: int = 64) -> Tuple[Optional[torch.Tensor], bool]:
    """A⁻¹ = L⁻ᵀ L⁻¹ for symmetric positive-definite A (n, n), bf16.
    Returns (None, False) if the Cholesky factorization fails."""
    L, ok = cholesky_bf16_checked(A, nb=nb)
    if not ok:
        return None, False
    S = TriSolver(L, lower=True, nb=nb)
    eye = torch.eye(A.shape[0], dtype=BF16, device=A.device)
    return S.solve_t(S.solve(eye)), True


# ---- legacy names (tests/test_kappa_scaling.py) --------------------------------------

def block_householder_qr_bf16(A: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    return qr_r_bf16(A, block_size=block_size)


def householder_qr_bf16(A: torch.Tensor) -> torch.Tensor:
    return qr_r_bf16(A, block_size=32)


def solve_triangular_bf16(R: torch.Tensor, B: torch.Tensor, upper: bool = True) -> torch.Tensor:
    return TriSolver(R, lower=not upper).solve(B)


def cholesky_bf16(A: torch.Tensor) -> torch.Tensor:
    """Legacy: lower Cholesky factor without clamping (NaN if A is not PD)."""
    return cholesky_bf16_checked(A)[0]


def inv_spd_bf16(A: torch.Tensor) -> torch.Tensor:
    """Legacy: SPD inverse; all-NaN if the Cholesky factorization fails."""
    X, ok = inv_spd_bf16_checked(A)
    if not ok:
        return torch.full_like(A, float("nan"))
    return X
