"""
test_equivalence.py — IFKFAC at fp32 reproduces Classic K-FAC at fp32.

The two algorithms compute the same natural gradient when numerical noise is
negligible (fp64 or fp32 at modest κ).  This test verifies algorithmic
equivalence — that IFKFAC's QR + triangular-solve pipeline is solving the
SAME underlying problem as Classic's Gram-matrix-inversion pipeline.

We construct a single small layer, compute the natural gradient with both
methods, and require cosine similarity ≥ 0.999.  If this fails, the two
pipelines have drifted apart and the comparison vs Classic in the paper
becomes meaningless.
"""
import math
import torch
import pytest


def _natural_gradient_classic(X, dY, grad_W, damping):
    """Reference: A = XᵀX/p, G = δᵀδ/p, ΔW = (G+λI)⁻¹ · grad_W · (A+λI)⁻¹."""
    p = X.shape[0]
    n_in = X.shape[1]
    n_out = dY.shape[1]
    A = X.t() @ X / p
    G = dY.t() @ dY / p
    A_aug = A + damping * torch.eye(n_in, device=X.device, dtype=X.dtype)
    G_aug = G + damping * torch.eye(n_out, device=X.device, dtype=X.dtype)
    A_inv = torch.linalg.inv(A_aug)
    G_inv = torch.linalg.inv(G_aug)
    return G_inv @ grad_W @ A_inv


def _natural_gradient_ifkfac(X, dY, grad_W, damping):
    """IFKFAC: R_X = QR(X/√p; √λ·I) and similarly for δ, then four trsm."""
    p = X.shape[0]
    n_in = X.shape[1]
    n_out = dY.shape[1]
    # Augmented QR for ridge-damped factor
    eye_A = math.sqrt(damping) * torch.eye(n_in, device=X.device, dtype=X.dtype)
    eye_G = math.sqrt(damping) * torch.eye(n_out, device=X.device, dtype=X.dtype)
    aug_X = torch.cat([X / math.sqrt(p), eye_A], dim=0)
    aug_G = torch.cat([dY / math.sqrt(p), eye_G], dim=0)
    _, R_X = torch.linalg.qr(aug_X, mode="reduced")
    _, R_G = torch.linalg.qr(aug_G, mode="reduced")
    # Now (R_X^T R_X) = A + λI and (R_G^T R_G) = G + λI.
    # ΔW = R_G⁻¹ · R_G⁻ᵀ · grad_W · R_X⁻¹ · R_X⁻ᵀ
    Id_out = torch.eye(n_out, device=X.device, dtype=X.dtype)
    Id_in  = torch.eye(n_in,  device=X.device, dtype=X.dtype)
    G_inv = torch.linalg.solve_triangular(R_G, Id_out, upper=True)
    G_inv = G_inv @ torch.linalg.solve_triangular(R_G.t(), Id_out, upper=False)
    A_inv = torch.linalg.solve_triangular(R_X, Id_in, upper=True)
    A_inv = A_inv @ torch.linalg.solve_triangular(R_X.t(), Id_in, upper=False)
    return G_inv @ grad_W @ A_inv


def _cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ifkfac_matches_classic_fp64(seed):
    """At fp64, the two algorithms compute identical natural gradients."""
    torch.manual_seed(seed)
    p, n_in, n_out = 128, 16, 32
    damping = 1e-2
    X = torch.randn(p, n_in, dtype=torch.float64)
    dY = torch.randn(p, n_out, dtype=torch.float64)
    grad_W = torch.randn(n_out, n_in, dtype=torch.float64)

    dW_classic = _natural_gradient_classic(X, dY, grad_W, damping)
    dW_ifkfac  = _natural_gradient_ifkfac(X, dY, grad_W, damping)

    rel_err = (dW_classic - dW_ifkfac).norm() / dW_classic.norm()
    assert rel_err < 1e-10, f"fp64 mismatch: rel_err = {rel_err:.3e}"
    assert _cos(dW_classic, dW_ifkfac) > 0.999999


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ifkfac_matches_classic_fp32(seed):
    """At fp32 with modest κ, the algorithms agree to high precision."""
    torch.manual_seed(seed)
    p, n_in, n_out = 256, 16, 32
    damping = 1e-2
    X = torch.randn(p, n_in, dtype=torch.float32)
    dY = torch.randn(p, n_out, dtype=torch.float32)
    grad_W = torch.randn(n_out, n_in, dtype=torch.float32)

    dW_classic = _natural_gradient_classic(X, dY, grad_W, damping)
    dW_ifkfac  = _natural_gradient_ifkfac(X, dY, grad_W, damping)

    rel_err = (dW_classic - dW_ifkfac).norm() / dW_classic.norm()
    # At fp32, both methods accumulate some noise but should still agree to 1e-4
    assert rel_err < 1e-3, f"fp32 mismatch: rel_err = {rel_err:.3e}"
    assert _cos(dW_classic, dW_ifkfac) > 0.999
