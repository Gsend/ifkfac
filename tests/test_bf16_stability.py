"""
test_bf16_stability.py — the κ²/κ contrast on synthetic data.

This is the headline empirical claim of IFKFAC, demonstrable in a fast unit
test.  Construct X with controlled condition number κ via SVD, compute the
natural gradient via Classic and IFKFAC paths under both fp32 and bf16, and
verify:

  - At fp32 with κ ≤ 100: both methods agree to high precision (κ² · ε_fp32
    is still small).
  - At bf16 with κ = 100: Classic's error is O(κ²·ε_bf16) ≈ 40 (saturated);
    IFKFAC's is O(κ·ε_bf16) ≈ 0.4 (preconditioner direction preserved).

The test asserts that the Classic-to-IFKFAC bf16 error RATIO is at least 5×
— enough margin to be unambiguous, conservative enough to be robust across
hardware/seed.
"""
import math
import torch
import pytest


def _make_X_with_kappa(p: int, n: int, kappa: float, seed: int = 0):
    """Build X ∈ ℝ^(p×n) with prescribed κ(X) = kappa via SVD."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(p, n, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    sigmas = torch.logspace(0, math.log10(kappa), n, dtype=torch.float64)
    return U @ torch.diag(sigmas) @ V.T


def _classic_natural_gradient(X, dY, grad_W, damping, precision):
    """Form A = XᵀX, invert.  Quantize to bf16 if precision == 'bf16'."""
    p = X.shape[0]
    A = (X.t() @ X) / p
    G = (dY.t() @ dY) / p
    if precision == "bf16":
        A = A.to(torch.bfloat16).to(torch.float32)
        G = G.to(torch.bfloat16).to(torch.float32)
    A_aug = A + damping * torch.eye(A.shape[0], dtype=A.dtype)
    G_aug = G + damping * torch.eye(G.shape[0], dtype=G.dtype)
    A_inv = torch.linalg.inv(A_aug)
    G_inv = torch.linalg.inv(G_aug)
    # The factor pipeline above runs at the test precision; the final product is
    # promoted so the fp64 comparison measures error from the factorisation, not
    # from this matmul.  grad_W has to be promoted with the inverses — callers
    # pass it at the same precision as X, so leaving it out is a dtype mismatch.
    return (G_inv.to(torch.float64)
            @ grad_W.to(torch.float64)
            @ A_inv.to(torch.float64))


def _ifkfac_natural_gradient(X, dY, grad_W, damping, precision):
    """QR + triangular solves.  Quantize R factors to bf16 if precision='bf16'."""
    p = X.shape[0]
    n_in, n_out = X.shape[1], dY.shape[1]
    X_use = X.to(torch.bfloat16).to(torch.float32) if precision == "bf16" else X.to(torch.float32)
    dY_use = dY.to(torch.bfloat16).to(torch.float32) if precision == "bf16" else dY.to(torch.float32)
    sqrt_p = math.sqrt(p)
    eye_A = math.sqrt(damping) * torch.eye(n_in, dtype=X_use.dtype)
    eye_G = math.sqrt(damping) * torch.eye(n_out, dtype=X_use.dtype)
    _, R_X = torch.linalg.qr(torch.cat([X_use / sqrt_p, eye_A], dim=0), mode="reduced")
    _, R_G = torch.linalg.qr(torch.cat([dY_use / sqrt_p, eye_G], dim=0), mode="reduced")
    if precision == "bf16":
        R_X = R_X.to(torch.bfloat16).to(torch.float32)
        R_G = R_G.to(torch.bfloat16).to(torch.float32)
    Id_in  = torch.eye(n_in,  dtype=R_X.dtype)
    Id_out = torch.eye(n_out, dtype=R_X.dtype)
    A_inv = torch.linalg.solve_triangular(R_X, Id_in, upper=True)
    A_inv = A_inv @ torch.linalg.solve_triangular(R_X.t(), Id_in, upper=False)
    G_inv = torch.linalg.solve_triangular(R_G, Id_out, upper=True)
    G_inv = G_inv @ torch.linalg.solve_triangular(R_G.t(), Id_out, upper=False)
    # Promoted for the same reason as in _classic_natural_gradient.
    return (G_inv.to(torch.float64)
            @ grad_W.to(torch.float64)
            @ A_inv.to(torch.float64))


def _rel_err(approx, exact):
    return float((approx - exact).norm() / (exact.norm() + 1e-30))


@pytest.mark.parametrize("kappa", [10.0, 100.0])
def test_classic_and_ifkfac_agree_at_fp32_with_moderate_kappa(kappa):
    """Sanity: at fp32 + κ ≤ 100, both methods are equally accurate."""
    torch.manual_seed(0)
    p, n_in, n_out = 256, 16, 16
    damping = 1e-3
    X = _make_X_with_kappa(p, n_in, kappa)
    dY = torch.randn(p, n_out, dtype=torch.float64)
    grad_W = torch.randn(n_out, n_in, dtype=torch.float64)
    # fp64 reference
    dW_ref = _classic_natural_gradient(X, dY, grad_W, damping, precision="fp32")
    dW_classic = _classic_natural_gradient(X.float(), dY.float(), grad_W.float(), damping, "fp32")
    dW_ifkfac  = _ifkfac_natural_gradient(X.float(), dY.float(), grad_W.float(), damping, "fp32")
    err_classic = _rel_err(dW_classic, dW_ref)
    err_ifkfac  = _rel_err(dW_ifkfac,  dW_ref)
    # Both should be within an order of magnitude
    assert err_classic < 0.1, f"Classic fp32 should not have saturated: err = {err_classic:.3e}"
    assert err_ifkfac  < 0.1, f"IFKFAC fp32 should be accurate: err = {err_ifkfac:.3e}"


def test_ifkfac_beats_classic_at_bf16_high_kappa():
    """At κ=100 and bf16, IFKFAC's error must be substantially smaller than Classic's.

    Predicted from theory: Classic ≈ κ²·ε_bf16 = 40 (saturated); IFKFAC ≈ κ·ε_bf16 = 0.4.
    Conservatively require Classic/IFKFAC error ratio ≥ 5 to allow for seed variance.
    """
    torch.manual_seed(0)
    p, n_in, n_out = 1024, 16, 16
    kappa = 100.0
    damping = 1e-3
    X = _make_X_with_kappa(p, n_in, kappa)
    dY = torch.randn(p, n_out, dtype=torch.float64)
    grad_W = torch.randn(n_out, n_in, dtype=torch.float64)
    # fp64 reference (algorithmically Classic; at fp64 the two algos agree)
    dW_ref = _classic_natural_gradient(X, dY, grad_W, damping, "fp32")
    dW_classic_bf16 = _classic_natural_gradient(X.float(), dY.float(), grad_W.float(), damping, "bf16")
    dW_ifkfac_bf16  = _ifkfac_natural_gradient(X.float(), dY.float(), grad_W.float(), damping, "bf16")
    err_classic = _rel_err(dW_classic_bf16, dW_ref)
    err_ifkfac  = _rel_err(dW_ifkfac_bf16,  dW_ref)
    print(f"\n  κ=100, bf16:  Classic err = {err_classic:.3e}   IFKFAC err = {err_ifkfac:.3e}")
    print(f"  ratio Classic/IFKFAC = {err_classic / err_ifkfac:.1f}× (predicted ≈ κ = 100×)")
    assert err_classic > err_ifkfac * 5, (
        f"Expected κ²/κ contrast: Classic err {err_classic:.3e} should exceed "
        f"IFKFAC err {err_ifkfac:.3e} by at least 5×"
    )
