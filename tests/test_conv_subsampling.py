"""
test_conv_subsampling.py — conv row subsampling happens *before* materialisation.

`F.unfold` expands (B, C_in, H, W) into (B, C_in·kH·kW, L), and the permute +
reshape that follows copies it again.  For a UNet encoder that is multiple GB
per layer per step (64 imgs × 576 cols × 65536 locations in bf16 ≈ 4.8 GB), so
capping rows *after* the unfold — which is what the hooks used to do — still
paid the full allocation on every forward pass.  The same applies to the
backward hook's permute+reshape of (B, C_out, H, W) gradients.

The fix subsamples spatially first: the unfold stride is raised to a multiple of
the conv's own stride, and gradients are strided along H/W.  Both yield an
*exact subset* of the rows the unsubsampled path would have produced, drawn from
every image in the batch.

These tests pin the three properties that make that substitution safe:
  1. the rows are a genuine subset of the true rows (not shifted or dilated),
  2. every image in the batch still contributes rows,
  3. the large intermediate is never allocated.
"""
import pytest
import torch
import torch.nn as nn

from ifkfac.activation_hooks import RawActivationHooks as H


# ---------------------------------------------------------------------------
# Output-geometry arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kernel, stride, padding, dilation, hw",
    [
        (3, 1, 1, 1, (16, 16)),
        (3, 2, 1, 1, (16, 16)),     # strided downsample — L is H·W/4, not H·W
        (5, 2, 2, 1, (32, 24)),     # non-square input
        (3, 1, 2, 2, (16, 16)),     # dilated
        (1, 1, 0, 1, (7, 7)),       # 1x1 projection
        (4, 4, 0, 1, (16, 16)),     # patchify-style
    ],
)
def test_conv_out_hw_matches_pytorch(kernel, stride, padding, dilation, hw):
    """The row-count estimate must use the conv's *output* size.

    An earlier attempt at this fix budgeted rows using the input H·W.  Because
    `_unfold_conv_input` passes `stride=module.stride`, patches per image is
    H_out·W_out — off by stride² — which silently cut the sample budget by 4×
    on every stride-2 layer.  Checking against the module's real output shape
    is the only way to keep that honest.
    """
    conv = nn.Conv2d(3, 4, kernel_size=kernel, stride=stride,
                     padding=padding, dilation=dilation)
    x = torch.randn(2, 3, *hw)

    assert H._conv_out_hw(conv, hw) == tuple(conv(x).shape[-2:])


def test_conv_out_hw_declines_string_padding():
    """'same'/'valid' padding doesn't fit the arithmetic — say so, don't guess."""
    conv = nn.Conv2d(3, 4, kernel_size=3, padding="same")
    assert H._conv_out_hw(conv, (16, 16)) is None


@pytest.mark.parametrize(
    "total, cap, expected",
    [
        (100, 512, 1),      # already under the cap
        (512, 512, 1),      # exactly at it
        (2048, 512, 2),     # 4x over -> stride 2 (rows scale as 1/s²)
        (4608, 512, 3),
        (16_777_216, 512, 182),
        (1000, 0, 1),       # cap disabled
    ],
)
def test_spatial_subsample_factor(total, cap, expected):
    assert H._spatial_subsample_factor(total, cap) == expected


@pytest.mark.parametrize("cap", [64, 256, 512])
def test_subsample_factor_actually_reaches_the_cap(cap):
    for total in (cap + 1, cap * 3, cap * 97, cap * 5000):
        s = H._spatial_subsample_factor(total, cap)
        assert total / (s * s) <= cap


# ---------------------------------------------------------------------------
# Forward path — unfold
# ---------------------------------------------------------------------------

def _row_set(t: torch.Tensor) -> set:
    return {tuple(r.tolist()) for r in t}


@pytest.mark.parametrize("stride", [1, 2])
def test_unfold_rows_are_a_subset_of_the_true_patches(stride):
    """Raising the unfold stride must select real patches, not new ones.

    Locations at multiples of s·stride are a subset of locations at multiples of
    stride, so every subsampled row has to appear verbatim in the full unfold.
    Slicing the input spatially instead (x[:, :, ::s, ::s]) would *not* satisfy
    this — it changes each patch's neighbourhood into a dilated one.
    """
    conv = nn.Conv2d(2, 4, kernel_size=3, stride=stride, padding=1)
    x = torch.randn(3, 2, 12, 12)

    full = H._unfold_conv_input(x, conv, 0)
    sub = H._unfold_conv_input(x, conv, 16)

    assert sub.shape[1] == full.shape[1]
    assert sub.shape[0] < full.shape[0], "subsampling did not engage"
    assert _row_set(sub) <= _row_set(full)


def test_unfold_keeps_every_image_in_the_batch():
    """Each image must still contribute rows.

    Dropping whole images off the front of the batch (x[:n_img]) hits the same
    row budget but estimates the factor from one image's worth of spatially
    correlated patches, which collapses the effective rank — the widest layers
    then trip the partial-rank skip in get_factors() and are never
    preconditioned.
    """
    conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
    B = 6
    # Image i is constant-valued i+1, so every patch from it is all (i+1)s.
    x = torch.arange(1, B + 1, dtype=torch.float32).view(B, 1, 1, 1)
    x = x.expand(B, 1, 10, 10).contiguous()

    rows = H._unfold_conv_input(x, conv, 12)

    assert rows.shape[0] <= 12 * 4, "row cap wildly exceeded"
    contributors = {int(r[0].item()) for r in rows}
    assert contributors == set(range(1, B + 1)), (
        f"only images {sorted(contributors)} contributed, expected all {B}"
    )


def test_unfold_respects_disabled_cap():
    """max_rows=0 must reproduce the original full-resolution behaviour."""
    conv = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
    x = torch.randn(4, 2, 8, 8)

    rows = H._unfold_conv_input(x, conv, 0)

    assert rows.shape == (4 * 8 * 8, 2 * 3 * 3)


def test_unfold_never_materialises_the_full_intermediate(monkeypatch):
    """The whole point of the fix: the big tensor is not allocated at all.

    Guards against a regression that reinstates the cap-after-unfold ordering,
    which would still pass every row-count assertion above while allocating the
    multi-GB intermediate that made a step take 23 seconds.
    """
    import ifkfac.activation_hooks as ah

    seen = []
    real_unfold = ah.F.unfold

    def spy(*args, **kwargs):
        out = real_unfold(*args, **kwargs)
        seen.append(out.numel())
        return out

    monkeypatch.setattr(ah.F, "unfold", spy)

    conv = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
    x = torch.randn(16, 4, 64, 64)       # full unfold would be 16·36·4096 = 2.36M
    cap = 512

    rows = H._unfold_conv_input(x, conv, cap)

    assert len(seen) == 1
    # Allow generous slack for the ceil() in the stride factor, but nothing
    # close to the unsubsampled 2.36M elements.
    assert seen[0] <= cap * conv.in_channels * 9 * 4, (
        f"unfold materialised {seen[0]} elements — subsampling ran too late"
    )
    assert rows.shape[0] <= cap * 4


# ---------------------------------------------------------------------------
# Backward path — gradient reshape
# ---------------------------------------------------------------------------

def test_grad_reshape_rows_are_a_subset():
    """Each row is one spatial location's C_out vector, so striding selects rows."""
    delta = torch.randn(3, 5, 12, 12)

    full = H._reshape_conv_grad(delta, 0)
    sub = H._reshape_conv_grad(delta, 16)

    assert full.shape == (3 * 12 * 12, 5)
    assert sub.shape[1] == 5
    assert sub.shape[0] < full.shape[0], "subsampling did not engage"
    assert _row_set(sub) <= _row_set(full)


def test_grad_reshape_keeps_every_image():
    B = 6
    delta = torch.arange(1, B + 1, dtype=torch.float32).view(B, 1, 1, 1)
    delta = delta.expand(B, 3, 10, 10).contiguous()

    rows = H._reshape_conv_grad(delta, 12)

    contributors = {int(r[0].item()) for r in rows}
    assert contributors == set(range(1, B + 1))


def test_grad_reshape_was_previously_uncapped():
    """The backward hook had no conv row cap at all before this change.

    Only the forward hook capped conv rows, so R_G accumulated B·H·W rows per
    step against R_X's 512 — the merge QR for the gradient factor ran on chunks
    orders of magnitude larger than the input factor's.  The cap is now applied
    on both sides.
    """
    delta = torch.randn(8, 4, 32, 32)      # 8192 rows unsubsampled
    assert H._reshape_conv_grad(delta, 0).shape[0] == 8192
    assert H._reshape_conv_grad(delta, 512).shape[0] <= 512 * 4


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_conv_rows", [64, 256])
def test_hooks_cap_rows_on_both_factors(max_conv_rows):
    """Row accounting must reflect what actually went into the QR.

    get_factors() normalises by 1/sqrt(n_rows), so a bookkeeping drift between
    the rows counted and the rows factored would rescale the preconditioner.
    """
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
    )
    hooks = H(model, damping=1e-2, max_conv_rows=max_conv_rows)
    hooks.enable()

    x = torch.randn(8, 3, 32, 32)
    steps = 3
    for _ in range(steps):
        model(x).pow(2).mean().backward()

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for conv in convs:
        # 4x slack covers the ceil() in the stride factor; the point is that the
        # count is bounded per step, not that it hits the cap exactly.
        assert hooks._n_rows_X[conv] <= max_conv_rows * 4 * steps
        assert hooks._n_rows_G[conv] <= max_conv_rows * 4 * steps
        assert hooks._n_rows_X[conv] > 0
        assert hooks._n_rows_G[conv] > 0
