"""
test_basic.py — smoke test: IFKFAC reduces MSE loss on a small MLP.

A standard convergence sanity check. The model should reduce its loss by at
least 10× over 60 steps; if it doesn't, the optimizer is broken (no need to
isolate a specific failure mode — this is the canary test).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from ifkfac import IFKFAC


def _build_small_mlp(in_dim=16, hidden=32, out_dim=8, seed=42):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def test_ifkfac_reduces_loss():
    """IFKFAC should reduce MSE on a small MLP within 60 steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_small_mlp().to(device)

    g = torch.Generator(device='cpu').manual_seed(0)
    X = torch.randn(128, 16, generator=g).to(device)
    Y = torch.randn(128, 8, generator=g).to(device)

    optimizer = IFKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=10)
    initial_loss = F.mse_loss(model(X), Y).item()

    for _ in range(60):
        loss = F.mse_loss(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = F.mse_loss(model(X), Y).item()
    assert final_loss < initial_loss / 10, (
        f"IFKFAC failed to reduce loss: {initial_loss:.4f} → {final_loss:.4f}"
    )


def test_ifkfac_handles_bias():
    """Layers with bias should also converge — bias is updated alongside W."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 4, bias=True),
    ).to(device)

    g = torch.Generator(device='cpu').manual_seed(1)
    X = torch.randn(64, 8, generator=g).to(device)
    Y = torch.randn(64, 4, generator=g).to(device)

    opt = IFKFAC(model, lr=1e-2, damping=1e-2, factor_update_freq=5)
    L0 = F.mse_loss(model(X), Y).item()
    for _ in range(40):
        loss = F.mse_loss(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    L1 = F.mse_loss(model(X), Y).item()
    assert L1 < L0 / 5
