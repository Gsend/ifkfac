"""
test_low_precision_qr.py — the bf16 path that no existing test covered.

`tests/test_bf16_stability.py` only *simulates* bf16: it round-trips values
through bfloat16 and back to float32, then calls every linalg op in fp32.  That
is the right test for the κ-vs-κ² numerical claim, but it means no test ever
handed a genuine bfloat16 tensor to `torch.linalg.qr` — and neither LAPACK nor
cuSOLVER implements `geqrf` for bf16/fp16, so every QR site in `triangular.py`
raised "not implemented for 'BFloat16'" the moment a model was trained under
AMP.  The failure was a hard crash, not silent drift, which is why it survived
the whole suite.

These tests feed real low-precision tensors through each QR site and through
the full hook/optimizer stack.  Every test is parametrised over fp32 and bf16
so the two precisions are held to the same assertions, plus explicit
cross-precision agreement checks.

Correctness reference throughout: RᵀR must reproduce the Gram matrix XᵀX of
the *same input values*.  Because `_qr` promotes bf16 → fp32 losslessly, a bf16
input is expected to match its fp32-viewed Gram matrix to fp32 precision — the
bf16 rounding happens when the caller builds X, not inside the factorisation.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ifkfac import IFKFAC
from ifkfac.activation_hooks import RawActivationHooks
from ifkfac.triangular import (
    batched_streaming_tsqr_update,
    finalize_R,
    streaming_tsqr_update,
    tsqr,
)

DTYPES = [torch.float32, torch.bfloat16]
IDS = ["fp32", "bf16"]


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative Frobenius error, both operands viewed in fp64."""
    a, e = actual.double(), expected.double()
    return (a - e).norm().item() / max(e.norm().item(), 1e-30)


def _gram(*chunks: torch.Tensor) -> torch.Tensor:
    """Σ XᵢᵀXᵢ over the chunks, accumulated in fp64 from their exact values."""
    total = None
    for c in chunks:
        c64 = c.double()
        g = c64.t() @ c64
        total = g if total is None else total + g
    return total


def _chunk(p: int, n: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(p, n, generator=g).to(dtype)


# ---------------------------------------------------------------------------
# QR primitives — one test per site that raised on bf16
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_streaming_tsqr_update_first_chunk(dtype):
    """The exact call that crashed: a bf16 activation chunk on the first pass."""
    X = _chunk(128, 8, dtype, seed=0)

    R = streaming_tsqr_update(None, X)

    assert R.shape == (8, 8)
    assert torch.isfinite(R).all()
    assert _rel_err(R.t() @ R, _gram(X)) < 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_streaming_tsqr_update_accumulates_chunks(dtype):
    """Folding several chunks must give RᵀR = Σ XᵢᵀXᵢ, not just the last one."""
    chunks = [_chunk(64, 8, dtype, seed=s) for s in range(5)]

    R = None
    for c in chunks:
        R = streaming_tsqr_update(R, c)

    assert _rel_err(R.t() @ R, _gram(*chunks)) < 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_tsqr_tiled_and_single_tile(dtype):
    """Both tsqr branches — the single-tile shortcut and the tree merge."""
    X = _chunk(1024, 8, dtype, seed=1)

    R_tree = tsqr(X, tile_size=128)      # 8 tiles -> 3 merge rounds
    R_single = tsqr(X, tile_size=4096)   # single-tile shortcut

    expected = _gram(X)
    assert _rel_err(R_tree.t() @ R_tree, expected) < 1e-4
    assert _rel_err(R_single.t() @ R_single, expected) < 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_batched_streaming_tsqr_update(dtype):
    """The batched-flush path uses its own QR calls — bf16 broke there too."""
    B, p, n = 3, 64, 8
    chunks = torch.stack([_chunk(p, n, dtype, seed=10 + i) for i in range(B)])
    running = torch.zeros(B, n, n, dtype=dtype)

    R = batched_streaming_tsqr_update(running, chunks)

    assert R.shape == (B, n, n)
    for i in range(B):
        assert _rel_err(R[i].t() @ R[i], _gram(chunks[i])) < 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_finalize_R_applies_damping(dtype):
    """finalize_R runs on every get_factors() call — reached in every mode.

    Also pins the damping precision: √λ is built in fp32 even for a bf16
    accumulator, because √1e-2 = 0.1 is not representable in bf16 and would
    perturb λ by ~0.4% before the QR ran.
    """
    X = _chunk(128, 8, dtype, seed=2)
    damping = 1e-2

    R = finalize_R(streaming_tsqr_update(None, X), damping)

    expected = _gram(X) + damping * torch.eye(8, dtype=torch.float64)
    assert _rel_err(R.t() @ R, expected) < 1e-4


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_accumulator_is_promoted_to_fp32(dtype):
    """Regression guard on the design decision behind `_qr`.

    The R accumulator is folded once per forward pass, so rounding it back to
    bf16 after every merge compounds over hundreds of chunks and erodes the
    κ(X) conditioning advantage the module exists to provide.  `_qr` therefore
    returns fp32 regardless of input dtype; bf16 *storage* is applied once at
    the end of the pipeline by `use_true_bf16`.
    """
    R = streaming_tsqr_update(None, _chunk(64, 8, dtype, seed=3))
    assert R.dtype == torch.float32

    R = streaming_tsqr_update(R, _chunk(64, 8, dtype, seed=4))
    assert R.dtype == torch.float32
    assert finalize_R(R, 1e-2).dtype == torch.float32


def test_bf16_and_fp32_factors_agree():
    """Cross-precision check: identical values, so identical factors.

    Casting the *same* fp32 chunk to bf16 and back is the only precision loss;
    the factorisation itself must not add any.  A regression that silently
    downcast the accumulator would show up here as error at bf16 epsilon
    (~8e-3) rather than fp32 epsilon.
    """
    X32 = _chunk(256, 8, torch.float32, seed=5)
    Xbf = X32.to(torch.bfloat16)

    R32 = finalize_R(streaming_tsqr_update(None, X32.to(torch.float32)), 1e-2)
    Rbf = finalize_R(streaming_tsqr_update(None, Xbf), 1e-2)

    # Rbf reproduces the Gram matrix of the bf16 values to fp32 precision...
    assert _rel_err(Rbf.t() @ Rbf,
                    _gram(Xbf) + 1e-2 * torch.eye(8, dtype=torch.float64)) < 1e-4
    # ...and tracks the fp32 factor to within the input quantisation itself.
    assert _rel_err(Rbf, R32) < 5e-2


# ---------------------------------------------------------------------------
# Hook-level — the scenario that actually crashed training
# ---------------------------------------------------------------------------

def _conv_net(dtype: torch.dtype) -> nn.Module:
    """Small UNet-shaped stack: strided conv, plain conv, linear head.

    The head is pooled to 8 features on purpose.  A wide head (8·8·8 = 512
    inputs against 8 rows per pass) never reaches p ≥ n within the test's step
    budget, and batched mode turns that into an IFKFACRankError rather than the
    per-layer skip the streaming path uses — which would mask the dtype
    behaviour these tests are actually about.
    """
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
@pytest.mark.parametrize("mode", ["streaming", "batched", "deferred"])
def test_hooks_get_factors(dtype, mode):
    """End-to-end hooks in every accumulation mode.

    Before the fix this raised "not implemented for 'BFloat16'" inside the
    forward hook on the first conv layer for bf16 — and, for the modes that
    defer their QR, later inside finalize_R.
    """
    model = _conv_net(dtype)
    hooks = RawActivationHooks(
        model,
        damping=1e-2,
        batched=(mode == "batched"),
        deferred=(mode == "deferred"),
    )
    hooks.enable()

    x = torch.randn(8, 3, 16, 16).to(dtype)
    for _ in range(4):
        loss = model(x).pow(2).mean()
        model.zero_grad()
        loss.backward()

    if mode == "batched":
        hooks.flush()
    factors = hooks.get_factors()

    assert factors, "no layer produced factors"
    for module, (R_X, R_G) in factors.items():
        assert torch.isfinite(R_X).all(), f"non-finite R_X for {module}"
        assert torch.isfinite(R_G).all(), f"non-finite R_G for {module}"
        assert R_X.shape[0] == R_X.shape[1]
        assert R_G.shape[0] == R_G.shape[1]
        # Damping guarantees a strictly positive diagonal.
        assert (R_X.diag() > 0).all()
        assert (R_G.diag() > 0).all()


# ---------------------------------------------------------------------------
# Optimizer-level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_ifkfac_step_on_conv_model(dtype):
    """A full step(): hooks → get_factors → apply_vered → weight update.

    Covers the dtype seam this fix introduced as well as the crash it removed:
    the R accumulators are fp32 while a wholesale-cast model produces bf16
    gradients, and `solve_triangular` rejects mismatched operands.
    """
    model = _conv_net(dtype)
    opt = IFKFAC(model, lr=1e-3, damping=1e-2, factor_update_freq=2)

    x = torch.randn(8, 3, 16, 16).to(dtype)
    y = torch.randn(8, 4).to(dtype)

    losses = []
    for _ in range(12):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(math.isfinite(v) for v in losses), f"loss went non-finite: {losses}"
    assert any(p.grad is not None for p in model.parameters())
    for p in model.parameters():
        assert torch.isfinite(p).all(), "parameters diverged"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} → {losses[-1]}"


def test_ifkfac_step_under_autocast():
    """The real AMP shape: bf16 activations, fp32 master weights.

    This is what a UNet trained with `torch.autocast` produces — the hooks see
    bf16 chunks even though every parameter is fp32.  Note that autocast puts
    `linalg_qr` on its fp32 cast list, so the *forward* hook may be promoted for
    free while the backward hook (which runs outside the autocast region) is
    not; the factors must come out fp32 either way.
    """
    if not hasattr(torch, "autocast"):
        pytest.skip("torch.autocast unavailable")

    model = _conv_net(torch.float32)
    opt = IFKFAC(model, lr=1e-3, damping=1e-2, factor_update_freq=2)
    x = torch.randn(8, 3, 16, 16)
    y = torch.randn(8, 4)

    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            probe = model(x)
    except (RuntimeError, ValueError) as exc:      # no CPU bf16 autocast support
        pytest.skip(f"cpu bf16 autocast unavailable: {exc}")
    assert probe.dtype == torch.bfloat16, "autocast did not engage"

    for _ in range(6):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = F.mse_loss(model(x), y.to(torch.bfloat16))
        opt.zero_grad()
        loss.float().backward()
        opt.step()

    for p in model.parameters():
        assert p.dtype == torch.float32, "master weights should stay fp32"
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_ifkfac_ema_blend(dtype):
    """gamma > 0 re-QRs the blended factors — its own bf16-unsafe QR site.

    With use_true_bf16 the cached factors are stored as bf16, so the blend on
    the *second* factor update is handed bf16 operands.  It only fires once
    factors already exist, which is why a short run never reached it.
    """
    model = _conv_net(torch.float32)
    opt = IFKFAC(
        model, lr=1e-3, damping=1e-2, factor_update_freq=2,
        gamma=0.9, use_true_bf16=(dtype == torch.bfloat16),
    )

    x = torch.randn(8, 3, 16, 16)
    y = torch.randn(8, 4)
    for _ in range(8):          # >= 3 factor updates, so the blend runs twice
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert opt._factors, "no factors cached"
    for R_X, R_G in opt._factors.values():
        assert torch.isfinite(R_X).all()
        assert torch.isfinite(R_G).all()
        if dtype == torch.bfloat16:
            assert R_X.dtype == torch.bfloat16, "use_true_bf16 storage not applied"
    for p in model.parameters():
        assert torch.isfinite(p).all()
