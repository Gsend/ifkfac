"""
Backend abstraction for Gram matrix operations.

Tries to import the compiled olssm Rust module first.
Falls back to numpy-based implementations that are API-compatible.
This allows development and benchmarking without a Rust compiler,
while seamlessly using the Rust backend when available.
"""

import numpy as np
from scipy import linalg as sp_linalg

# ---------------------------------------------------------------------------
# Try importing the compiled Rust module
# ---------------------------------------------------------------------------
try:
    import olssm as _rust_backend

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# ---------------------------------------------------------------------------
# Numpy fallback implementations
# ---------------------------------------------------------------------------

def _np_lu_solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve gram @ X = rhs via LU factorisation (scipy)."""
    lu, piv = sp_linalg.lu_factor(gram)
    if rhs.ndim == 1:
        return sp_linalg.lu_solve((lu, piv), rhs)
    # Handle matrix RHS column-by-column for stability
    result = np.empty_like(rhs)
    for j in range(rhs.shape[1]):
        result[:, j] = sp_linalg.lu_solve((lu, piv), rhs[:, j])
    return result

def _np_lu_inverse_gram(gram: np.ndarray) -> np.ndarray:
    """Compute gram⁻¹ via LU factorisation (scipy)."""
    lu, piv = sp_linalg.lu_factor(gram)
    identity = np.eye(gram.shape[0])
    return sp_linalg.lu_solve((lu, piv), identity)

# ---------------------------------------------------------------------------
# Public API — dispatches to Rust or numpy
# ---------------------------------------------------------------------------

def lu_solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve gram @ X = rhs via LU factorisation.

    Uses the compiled olssm Rust backend if available,
    otherwise falls back to scipy.linalg.lu_factor/lu_solve.

    Parameters
    ----------
    gram : np.ndarray, shape (p, p)
        Symmetric positive-(semi)definite Gram matrix.
    rhs : np.ndarray, shape (p, k) or (p,)
        Right-hand side matrix or vector.

    Returns
    -------
    np.ndarray, shape (p, k) or (p,)
        Solution X such that gram @ X ≈ rhs.
    """
    if _HAS_RUST and rhs.ndim == 2:
        return _rust_backend.lu_solve_gram(
            np.ascontiguousarray(gram, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
        )
    if _HAS_RUST and rhs.ndim == 1:
        return _rust_backend.lu_solve_gram_vec(
            np.ascontiguousarray(gram, dtype=np.float64),
            np.ascontiguousarray(rhs, dtype=np.float64),
        )
    return _np_lu_solve_gram(gram.astype(np.float64), rhs.astype(np.float64))

def lu_inverse_gram(gram: np.ndarray) -> np.ndarray:
    """Compute gram⁻¹ via LU factorisation.

    Uses the compiled olssm Rust backend if available,
    otherwise falls back to scipy.linalg.lu_factor/lu_solve.

    Parameters
    ----------
    gram : np.ndarray, shape (p, p)
        Square matrix.

    Returns
    -------
    np.ndarray, shape (p, p)
        Inverse matrix gram⁻¹.
    """
    if _HAS_RUST:
        return _rust_backend.lu_inverse_gram(
            np.ascontiguousarray(gram, dtype=np.float64),
        )
    return _np_lu_inverse_gram(gram.astype(np.float64))

def lu_damped_inverse_f32(gram: np.ndarray, damping: float) -> np.ndarray:
    """Compute ``(gram + damping·I)⁻¹`` on f32 data — fast path for K-FAC.

    Uses the compiled olssm Rust backend if available (zero dtype cast,
    damping applied inside Rust, only 2 copies vs 6+).
    Falls back to scipy on float64 when Rust is unavailable.

    Parameters
    ----------
    gram : np.ndarray, shape (n, n), float32 or float64
        Symmetric positive-(semi)definite Gram matrix.
    damping : float
        Scalar λ added to the diagonal before inversion.

    Returns
    -------
    np.ndarray, shape (n, n), same dtype as input (float32 if Rust path taken)
    """
    if _HAS_RUST:
        g32 = np.ascontiguousarray(gram, dtype=np.float32)
        return _rust_backend.lu_damped_inverse_f32(g32, float(damping))
    # numpy/scipy fallback — use float64 for stability
    g64 = gram.astype(np.float64)
    n = g64.shape[0]
    g64 += damping * np.eye(n)
    return _np_lu_inverse_gram(g64).astype(gram.dtype)

def eigh_f32(gram: np.ndarray, damping: float):
    """Symmetric eigendecomposition of a Gram matrix — fast K-FAC path.

    Uses faer's SIMD-accelerated EVD on f32.  Returns the eigenvector matrix
    **Q** and damped inverse eigenvalues **1/(λᵢ + δ)**, both as float32.

    Damping is applied in eigenvalue space so Q can be cached across multiple
    damping values without re-decomposing.

    Parameters
    ----------
    gram : np.ndarray, shape (n, n), float32 or float64
        Symmetric positive-(semi)definite Gram matrix.
    damping : float
        Scalar δ — returns ``1 / max(λᵢ + δ, 1e-8)``.

    Returns
    -------
    q : np.ndarray, shape (n, n), float32 — eigenvector matrix (columns = eigenvectors)
    inv_lambda : np.ndarray, shape (n,), float32 — damped inverse eigenvalues
    """
    if _HAS_RUST:
        g32 = np.ascontiguousarray(gram, dtype=np.float32)
        return _rust_backend.eigh_f32(g32, float(damping))
    # numpy fallback
    g64 = gram.astype(np.float64)
    eigenvalues, q = np.linalg.eigh(g64)
    inv_lam = (1.0 / np.maximum(eigenvalues + damping, 1e-8)).astype(np.float32)
    return q.astype(np.float32), inv_lam

def apply_kfac_eigen_f32(
    q_g: np.ndarray, inv_lam_g: np.ndarray,
    grad: np.ndarray,
    q_a: np.ndarray, inv_lam_a: np.ndarray,
) -> np.ndarray:
    """Apply K-FAC eigen-basis preconditioner: ΔW = Q_G d_G Q_Gᵀ grad Q_A d_A Q_Aᵀ.

    All 4 matrix products and element-wise scaling are fused in a single Rust
    call, eliminating Python dispatch overhead for the per-step hot path.

    Parameters
    ----------
    q_g, inv_lam_g : eigenvectors (d_out×d_out) and damped inverse eigenvalues (d_out,) of G
    grad           : weight gradient (d_out × d_in)
    q_a, inv_lam_a : eigenvectors (d_in×d_in) and damped inverse eigenvalues (d_in,) of A

    Returns
    -------
    np.ndarray, shape (d_out, d_in), float32 — preconditioned gradient
    """
    if _HAS_RUST:
        return _rust_backend.apply_kfac_eigen_f32(
            np.ascontiguousarray(q_g,      dtype=np.float32),
            np.ascontiguousarray(inv_lam_g, dtype=np.float32),
            np.ascontiguousarray(grad,     dtype=np.float32),
            np.ascontiguousarray(q_a,      dtype=np.float32),
            np.ascontiguousarray(inv_lam_a, dtype=np.float32),
        )
    # numpy fallback: explicit 4-matmul apply
    tmp = q_g.T @ grad.astype(np.float32) @ q_a
    tmp = tmp * np.outer(inv_lam_g, inv_lam_a)
    return (q_g @ tmp @ q_a.T)

def eigh_topk_f32(gram: np.ndarray, k: int, damping: float):
    """Low-rank symmetric EVD: top-k eigenvectors and damped inverse eigenvalues.

    Like ``eigh_f32`` but returns only the k largest eigenvectors, giving an n×k
    matrix Q_k.  The apply step then costs O((k_g+k_a)·d_out·d_in) instead of
    O(4·n·d_out·d_in) — the genuine speedup over full-rank eigen.

    Parameters
    ----------
    gram : np.ndarray, shape (n, n), float32 or float64
        Symmetric positive-(semi)definite Gram matrix.
    k : int
        Number of top eigenvectors to keep.
    damping : float
        Scalar δ — returns ``1 / max(λᵢ + δ, 1e-8)``.

    Returns
    -------
    q_k : np.ndarray, shape (n, k), float32 — top-k eigenvectors as columns
    inv_lambda_k : np.ndarray, shape (k,), float32 — damped inverse eigenvalues
    """
    if _HAS_RUST:
        g32 = np.ascontiguousarray(gram, dtype=np.float32)
        return _rust_backend.eigh_topk_f32(g32, int(k), float(damping))
    # numpy fallback — eigh returns ascending order, take last k
    g64 = gram.astype(np.float64)
    eigenvalues, q = np.linalg.eigh(g64)
    top_k_idx = slice(-k, None)
    inv_lam_k = (1.0 / np.maximum(eigenvalues[top_k_idx] + damping, 1e-8)).astype(np.float32)
    q_k = q[:, top_k_idx].astype(np.float32)
    return q_k, inv_lam_k

def apply_kfac_lowrank_f32(
    q_g_k: np.ndarray, inv_lam_g_k: np.ndarray,
    grad: np.ndarray,
    q_a_k: np.ndarray, inv_lam_a_k: np.ndarray,
) -> np.ndarray:
    """Apply low-rank K-FAC preconditioner: ΔW ≈ Q_G_k d_G_k Q_G_kᵀ grad Q_A_k d_A_k Q_A_kᵀ.

    Uses rank-k approximations of A and G, reducing apply cost from
    O(4n·d_out·d_in) to O((k_g+k_a)·d_out·d_in).

    Parameters
    ----------
    q_g_k, inv_lam_g_k : eigenvectors (d_out×k_g) and inverse eigenvalues (k_g,) of G
    grad               : weight gradient (d_out × d_in)
    q_a_k, inv_lam_a_k : eigenvectors (d_in×k_a) and inverse eigenvalues (k_a,) of A

    Returns
    -------
    np.ndarray, shape (d_out, d_in), float32 — preconditioned gradient
    """
    if _HAS_RUST:
        return _rust_backend.apply_kfac_lowrank_f32(
            np.ascontiguousarray(q_g_k,      dtype=np.float32),
            np.ascontiguousarray(inv_lam_g_k, dtype=np.float32),
            np.ascontiguousarray(grad,        dtype=np.float32),
            np.ascontiguousarray(q_a_k,       dtype=np.float32),
            np.ascontiguousarray(inv_lam_a_k, dtype=np.float32),
        )
    # numpy fallback
    tmp = q_g_k.T @ grad.astype(np.float32) @ q_a_k        # (k_g × k_a)
    tmp = tmp * np.outer(inv_lam_g_k, inv_lam_a_k)
    return (q_g_k @ tmp @ q_a_k.T)                         # (d_out × d_in)

def randomized_eigh_f32(gram: np.ndarray, k: int, n_iter: int = 1, damping: float = 0.0):
    """Randomized symmetric EVD — approximate top-k eigenvectors in O(k·n²).

    Uses the Halko-Martinsson-Tropp randomized range-finder algorithm:
      1. Random Gaussian projection Ω (n×k)
      2. Power iteration: Y = A^(2·n_iter+1) · Ω  (sharpens subspace alignment)
      3. Modified Gram-Schmidt: Y → Q  (n×k orthonormal)
      4. Small sketch: B = QᵀAQ  (k×k)
      5. Exact EVD of B

    Cost vs exact EVD: O(k·n²·n_iter) vs O(n³) — 24× cheaper for k=32, n=784.
    Accuracy: bounded by σ_{k+1} (first discarded eigenvalue). For K-FAC Gram
    matrices with rapidly decaying spectra, n_iter=1 gives <1% relative error.

    Parameters
    ----------
    gram   : np.ndarray, shape (n, n), float32 or float64
    k      : number of top eigenvectors to approximate
    n_iter : power-iteration passes (0 = pure random projection; 1–2 recommended)
    damping: scalar δ — returns ``1 / max(λᵢ + δ, 1e-8)``

    Returns
    -------
    q_k         : np.ndarray, shape (n, k), float32
    inv_lambda_k: np.ndarray, shape (k,),   float32
    """
    if _HAS_RUST:
        g32 = np.ascontiguousarray(gram, dtype=np.float32)
        return _rust_backend.randomized_eigh_f32(g32, int(k), int(n_iter), float(damping))
    # numpy fallback — exact truncated EVD (no randomization in fallback)
    g64 = gram.astype(np.float64)
    eigenvalues, q = np.linalg.eigh(g64)
    inv_lam_k = (1.0 / np.maximum(eigenvalues[-k:] + damping, 1e-8)).astype(np.float32)
    q_k = q[:, -k:].astype(np.float32)
    return q_k, inv_lam_k

def get_backend_name() -> str:
    """Return the name of the active backend."""
    return "olssm (Rust)" if _HAS_RUST else "numpy/scipy (fallback)"


def get_backend_info() -> dict:
    """Return a dict describing the active backend.

    Useful for verifying at runtime which backend is loaded and for
    logging/debugging.  Example output::

        {'rust_available': True, 'backend': 'olssm (Rust)', 'version': '0.3.0'}

    Returns
    -------
    dict with keys:
        ``rust_available`` bool   — True if the compiled Rust module loaded
        ``backend``        str    — human-readable backend name
        ``version``        str    — Rust module version, or ``'n/a'``
    """
    version = "n/a"
    if _HAS_RUST:
        version = getattr(_rust_backend, "__version__", "n/a")
    return {
        "rust_available": _HAS_RUST,
        "backend": get_backend_name(),
        "version": version,
    }
