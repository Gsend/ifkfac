"""
benchmark/pure_bf16.py  --  building blocks for the pure-bf16 experiments.

"Pure bf16" = the model's weights, activations, gradients, the optimizer
state and every stage of the K-FAC pipeline are bf16 tensors; the dtype is
set once, on the model, and flows from stage to stage (hooks see bf16
activations and gradients, the factors are bf16, the solves are bf16, the
update is bf16).  benchmark/bf16_guard.py verifies it: no op may produce a
floating tensor that is not bf16.

Pieces:
  to_pure_bf16(model)     swap LayerNorm / BatchNorm2d for explicit-bf16 versions,
                          cast the model to bf16, cast floating inputs to bf16
  BF16LayerNorm           LayerNorm from bf16 ops (the fused CUDA kernel returns
                          fp32 mean / rstd tensors)
  BF16BatchNorm2d         BatchNorm2d from bf16 ops, bf16 running statistics
  cross_entropy_bf16      log_softmax + gather in bf16 (replaces F.cross_entropy
                          for bf16 logits; other dtypes go to the original)
  bce_with_logits_bf16    per-pixel BCE from logits (softplus form), bf16
  bf16 AdamW              optimizer/bf16_adamw.py

Self-test:  python -m benchmark.pure_bf16 --selftest
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BF16 = torch.bfloat16
_ORIG_CE = F.cross_entropy


# ---- layers --------------------------------------------------------------------

class BF16LayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = tuple(ln.normalized_shape)
        self.eps = ln.eps
        self.weight = nn.Parameter(ln.weight.detach().clone()) if ln.weight is not None else None
        self.bias = nn.Parameter(ln.bias.detach().clone()) if ln.bias is not None else None

    def forward(self, x):
        dims = tuple(range(-len(self.normalized_shape), 0))
        mu = x.mean(dim=dims, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=dims, keepdim=True)
        y = xc * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class BF16BatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.eps, self.momentum, self.affine = bn.eps, bn.momentum, bn.affine
        self.track_running_stats = bn.track_running_stats
        self.weight = nn.Parameter(bn.weight.detach().clone()) if bn.affine else None
        self.bias = nn.Parameter(bn.bias.detach().clone()) if bn.affine else None
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())

    def forward(self, x):
        if self.training or not self.track_running_stats:
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            xc = x - mu
            var = (xc * xc).mean(dim=(0, 2, 3), keepdim=True)
            if self.training and self.track_running_stats:
                n = x.numel() // x.shape[1]
                with torch.no_grad():
                    m = self.momentum
                    self.running_mean.mul_(1 - m).add_(mu.flatten(), alpha=m)
                    self.running_var.mul_(1 - m).add_(var.flatten(), alpha=m * n / max(n - 1, 1))
        else:
            mu = self.running_mean.view(1, -1, 1, 1)
            xc = x - mu
            var = self.running_var.view(1, -1, 1, 1)
        y = xc * torch.rsqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return y


def _swap_norms(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, BF16LayerNorm(child))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, BF16BatchNorm2d(child))
        else:
            _swap_norms(child)


def _cast_inputs(_module, args):
    return tuple(a.to(BF16) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                 for a in args)


def to_pure_bf16(model: nn.Module) -> nn.Module:
    """In place: explicit-bf16 norms, bf16 parameters and buffers, bf16 inputs."""
    _swap_norms(model)
    model.to(BF16)
    model.register_forward_pre_hook(_cast_inputs)
    model._pure_bf16 = True
    return model


# ---- losses --------------------------------------------------------------------

def cross_entropy_bf16(input, target, weight=None, size_average=None, ignore_index=-100,
                       reduce=None, reduction="mean", label_smoothing=0.0):
    if input.dtype != BF16:
        return _ORIG_CE(input, target, weight=weight, size_average=size_average,
                        ignore_index=ignore_index, reduce=reduce, reduction=reduction,
                        label_smoothing=label_smoothing)
    if weight is not None or label_smoothing != 0.0 or size_average is not None or reduce is not None:
        raise NotImplementedError("cross_entropy_bf16: only the plain form is supported")
    logp = torch.log_softmax(input, dim=-1)
    valid = target != ignore_index
    tgt = torch.where(valid, target, torch.zeros_like(target))
    nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    n_valid = int(valid.sum())                    # Python int: no float tensor
    return nll.sum() / max(n_valid, 1)


def bce_with_logits_bf16(logits, target, reduction="mean"):
    """-[t log sigmoid(z) + (1-t) log(1 - sigmoid(z))] = softplus(z) - t z,
    per pixel; 'mean' = sum over pixels, mean over the batch (the AE recipe)."""
    per = F.softplus(logits) - target * logits
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    return per.sum(dim=1).mean()


def install_bf16_losses():
    """Route F.cross_entropy through cross_entropy_bf16 (bf16 logits only)."""
    F.cross_entropy = cross_entropy_bf16
    torch.nn.functional.cross_entropy = cross_entropy_bf16


class NoAutocast:
    """Replacement for torch.autocast in pure-bf16 runs: the model is already
    bf16, and autocast would run its fp32-listed ops (softmax, layer_norm,
    losses, ...) in fp32."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---- self-test -------------------------------------------------------------------

def _selftest():
    from benchmark.bf16_guard import Bf16Guard
    from optimizer.bf16_adamw import BF16AdamW
    torch.manual_seed(0)
    # BF16AdamW == torch AdamW in fp32
    p1 = nn.Parameter(torch.randn(5, 3)); p2 = nn.Parameter(p1.detach().clone())
    o1 = torch.optim.AdamW([p1], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    o2 = BF16AdamW([p2], lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    for _ in range(5):
        g = torch.randn(5, 3); p1.grad = g.clone(); p2.grad = g.clone(); o1.step(); o2.step()
    print("BF16AdamW vs torch.optim.AdamW (fp32) max|diff| =", (p1 - p2).abs().max().item())
    # norms and losses match the fp32 modules
    x = torch.randn(4, 6, 8)
    ln = nn.LayerNorm(8); b = BF16LayerNorm(ln)
    print("BF16LayerNorm vs LayerNorm (fp32) max|diff| =", (ln(x) - b(x)).abs().max().item())
    bn = nn.BatchNorm2d(3); bb = BF16BatchNorm2d(bn); xi = torch.randn(4, 3, 5, 5)
    print("BF16BatchNorm2d vs BatchNorm2d (fp32) max|diff| =", (bn(xi) - bb(xi)).abs().max().item(),
          " running_var diff", (bn.running_var - bb.running_var).abs().max().item())
    z = torch.randn(7, 11); t = torch.randint(0, 11, (7,)); t[2] = -100
    ref = _ORIG_CE(z, t, ignore_index=-100)
    got = cross_entropy_bf16(z.to(BF16), t, ignore_index=-100)
    print(f"cross_entropy_bf16 vs F.cross_entropy: {got.item():.4f} vs {ref.item():.4f}")
    zl = torch.randn(3, 9); tl = (torch.rand(3, 9) > 0.5).float()
    ref = F.binary_cross_entropy_with_logits(zl, tl, reduction="sum") / 3
    print(f"bce_with_logits_bf16 vs torch: {bce_with_logits_bf16(zl.to(BF16), tl.to(BF16)).item():.4f} "
          f"vs {ref.item():.4f}")
    # the pieces themselves are clean under the guard
    m = to_pure_bf16(nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(4 * 3 * 3, 5), nn.LayerNorm(5)))
    opt = BF16AdamW(m.parameters(), lr=1e-3)
    xb = torch.randn(2, 3, 5, 5).to(BF16)          # the loaders hand over bf16 batches
    with Bf16Guard("layers+losses+adamw") as g:
        out = m(xb)
        loss = cross_entropy_bf16(out, torch.tensor([1, 3]))
        loss.backward(); opt.step()
    g.report()
    assert g.clean


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
