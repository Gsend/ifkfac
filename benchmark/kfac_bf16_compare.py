"""
benchmark/kfac_bf16_compare.py

Compare Classic / IFKFAC / IFKFAC+WGSO at each variant's optimal params,
with the K-FAC factor accumulation and application done in bf16.

Tests the kappa-scaling hypothesis at the precision regime where it
should bite: bf16 ε ≈ 4e-3, so kappa^4·ε terms blow up much more than
kappa^1·ε. If kappa-scaling matters anywhere, it should be visible here.

Cells (all at matched-screen champion mom=0.7, lr=2e-3, gamma=0.9, freq=20):
    Classic       damping=1e-4 (baseline 922 in fp32)
    IFKFAC         damping=1e-4 (baseline 921 in fp32)
    IFKFAC+WGSO    damping=1e-6 (best WGSO point, ~898 in fp32)

Model weights, gradients, and forward/backward stay in fp32. Only the
K-FAC factor pipeline (TSQR accumulation, inverse/triangular solve,
natgrad direction) operates in bf16.

Output: benchmark/results/per_step_bf16_{label}_s1000.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, make_optimizers, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info
import optimizer.raw_activation_hooks as rah
import optimizer.sgso as sgso
import optimizer.classic_kfac as ck
from optimizer.dtype_check import expect_bf16_values


# ---- bf16 monkey-patch for IFKFAC streaming TSQR + apply_ifkfac ------------

_orig_tsqr = sgso.streaming_tsqr_update
_orig_finalize = sgso.finalize_R
_orig_apply_ifkfac = sgso.apply_ifkfac
_orig_apply_ifkfac_bias = sgso.apply_ifkfac_bias


# CUDA geqrf doesn't support bf16, so we quantize to bf16 (precision loss)
# then upcast to fp32 for the actual QR kernel. The bf16 round-trips at every
# storage point simulate the precision regime; the fp32 inside the kernel is
# unavoidable until hardware supports it natively.

def _bf16q(t):
    """Quantize to bf16 then upcast to fp32 (emulates bf16 storage)."""
    if t is None:
        return None
    return t.to(torch.bfloat16).to(torch.float32)


def _tsqr_bf16(running, x):
    out = _orig_tsqr(_bf16q(running), _bf16q(x))
    return _bf16q(out)


def _finalize_bf16(running_R, damping):
    return _bf16q(_orig_finalize(_bf16q(running_R), damping))


def _apply_ifkfac_bf16(grad_W, R_X, R_G):
    out = _orig_apply_ifkfac(_bf16q(grad_W), _bf16q(R_X), _bf16q(R_G))
    return out.to(grad_W.dtype)


def _apply_ifkfac_bias_bf16(grad_b, R_G):
    out = _orig_apply_ifkfac_bias(_bf16q(grad_b), _bf16q(R_G))
    return out.to(grad_b.dtype)


# ---- WGSO (row-weighting) ------------------------------------------------
# Applied per-step via RawActivationHooks.chunk_transform_X.  This stays
# correct under streaming / batched / deferred modes because it weights
# each step's chunk independently, before it's stored.  (The old approach
# of monkey-patching streaming_tsqr_update broke under deferred mode —
# the residual drain at refresh bypassed the patch, mixing weighted and
# unweighted data and degrading ppl by ~300.)

WGSO_EPS_FRAC = 1.0
_WGSO_HOOK_REFS: list = []  # keep refs so the toggle can detach them


def _wgso_weight_rows(M):
    M32 = M.to(torch.float32)
    sq = (M32 * M32).sum(dim=1)
    med = sq.median()
    w = 1.0 / (sq + WGSO_EPS_FRAC * med + 1e-30)
    return M32 * w.sqrt().unsqueeze(1)


def _install_wgso_on_hooks(hooks):
    """Set WGSO row-weighting as the per-step chunk transform on a hooks instance."""
    hooks.chunk_transform_X = _wgso_weight_rows
    hooks.chunk_transform_G = _wgso_weight_rows


# ---- Phase 1: bf16 via chunk-level quantization --------------------------
# Previous design: patched streaming_tsqr_update / finalize_R / apply_ifkfac
# to .to(bf16).to(fp32) round-trip their inputs and outputs.  This re-quantized
# at every QR call (3 round-trips × ~96 calls/step) and broke deferred mode
# because the wrapper round-tripped 5x-larger cat'd chunks at flush time.
#
# Phase 1: each chunk gets bf16-quantized exactly once when it enters the hook
# buffer (via chunk_transform_X / chunk_transform_G).  R factors built from
# those chunks then naturally carry bf16 precision through the whole pipeline,
# in fp32 storage so cuSOLVER QR still works.  Two consequences:
#   1. ~5x fewer .to() round-trips per step → ~2x wall-time speedup at bf16
#   2. Deferred mode now works at bf16 (was anti-helpful under the wrappers)
# apply_ifkfac still bf16-quantizes the incoming grad_W (which arrives fresh
# from fp32 backward), but does NOT re-quantize R_X / R_G since they already
# carry bf16 precision from the chunk transform.

def _bf16_chunk(x):
    """Chunk-level bf16 quantization: precision loss, fp32 storage."""
    return x.to(torch.bfloat16).to(torch.float32)


def _wgso_then_bf16(x):
    """Compose: WGSO row-weighting first (algorithmic), then bf16 quantization
    (precision regime).  Quantizing after WGSO captures the precision of the
    weighted data, which is what we want — WGSO is what the algorithm does;
    bf16 is the precision it runs in."""
    return _bf16_chunk(_wgso_weight_rows(x))


def _apply_ifkfac_grad_only_bf16(grad_W, R_X, R_G):
    """Lightweight wrapper: quantize only grad_W (fresh from fp32 backward);
    R_X, R_G already carry bf16 precision via chunk-level quantization."""
    out = _orig_apply_ifkfac(_bf16q(grad_W), R_X, R_G)
    return out.to(grad_W.dtype)


def _apply_ifkfac_bias_grad_only_bf16(grad_b, R_G):
    out = _orig_apply_ifkfac_bias(_bf16q(grad_b), R_G)
    return out.to(grad_b.dtype)


def _install_bf16_on_hooks(hooks):
    """Install bf16 chunk-level quantization on a hooks instance."""
    hooks.chunk_transform_X = _bf16_chunk
    hooks.chunk_transform_G = _bf16_chunk


def _install_bf16_wgso_on_hooks(hooks):
    """Compose WGSO + bf16 quantization in one chunk transform per direction."""
    hooks.chunk_transform_X = _wgso_then_bf16
    hooks.chunk_transform_G = _wgso_then_bf16


_INIT_PATCH_REFS: list = []


def enable_bf16(wgso=False):
    # Only patch apply_ifkfac (grad_W still needs quantization since it bypasses
    # the hook chunk transforms).  streaming_tsqr_update and finalize_R are
    # NOT patched — R factors carry bf16 precision from the chunk transforms.
    sgso.apply_ifkfac = _apply_ifkfac_grad_only_bf16
    sgso.apply_ifkfac_bias = _apply_ifkfac_bias_grad_only_bf16

    # Patch IFKFAC.__init__ to install the chunk transforms on the hooks.
    from optimizer import ifkfac_kfac as _vk
    orig_init = _vk.IFKFAC.__init__
    install = _install_bf16_wgso_on_hooks if wgso else _install_bf16_on_hooks
    def init_with_bf16(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        install(self.hooks)
    _vk.IFKFAC.__init__ = init_with_bf16
    _INIT_PATCH_REFS.append(("ifkfac_init", orig_init))


def disable_bf16():
    sgso.apply_ifkfac = _orig_apply_ifkfac
    sgso.apply_ifkfac_bias = _orig_apply_ifkfac_bias
    # Restore IFKFAC.__init__
    while _INIT_PATCH_REFS:
        tag, orig = _INIT_PATCH_REFS.pop()
        if tag == "ifkfac_init":
            from optimizer import ifkfac_kfac as _vk
            _vk.IFKFAC.__init__ = orig
    # Legacy WGSO refs (in case old code paths added entries)
    while _WGSO_HOOK_REFS:
        tag, orig = _WGSO_HOOK_REFS.pop()
        if tag == "ifkfac_init":
            from optimizer import ifkfac_kfac as _vk
            _vk.IFKFAC.__init__ = orig


# ---- Classic bf16 patch: store its cached A_inv/G_inv in bf16 -------------
# Classic computes the damped inverses (A+lambda*I)^-1, (G+lambda*I)^-1 in fp32
# at every refresh; this patch rounds them to bf16 right after they are
# computed, so every step applies bf16-stored inverses (fp32 matmuls, fp32
# gradient).  Only the cached inverses are rounded; the Gram EMA stays fp32.
#
# FIX (2026-09-27).  The previous version wrapped ClassicKFAC.step(): it
# rounded a copy of the inverses, called step(), then restored the pre-step
# dict.  On a refresh step that restore replaced the freshly computed
# inverses with the (initially empty) old dict, so Classic never kept any
# inverses: it applied unrounded fp32 inverses on the refresh step only and
# fell back to plain SGD (no preconditioner, no momentum) on the other
# factor_update_freq - 1 steps.  All "Classic bf16" training results made
# with that version measure this artefact, not bf16 Classic K-FAC.

_orig_classic_step = ck.ClassicKFAC.step            # kept for reference; not patched
_orig_update_inverses = ck.ClassicKFAC._update_inverses


def _update_inverses_bf16(self):
    _orig_update_inverses(self)
    for m in list(self._inverses.keys()):
        Ai, Gi = self._inverses[m]
        Ai, Gi = _bf16q(Ai), _bf16q(Gi)
        expect_bf16_values("classic.storage.inverse_rounded", Ai, Gi)
        self._inverses[m] = (Ai, Gi)


def _check_applied_inverses(self, module, A_inv, G_inv):
    """ClassicKFAC.inverse_check while the patch is on: the inverses a step
    applies must hold bf16 values.  Each new pair of inverse tensors is
    checked once (they are replaced, never modified, at a refresh)."""
    seen = self.__dict__.setdefault("_bf16_checked_inverses", {})
    last = seen.get(module)
    if last is not None and last[0] is A_inv and last[1] is G_inv:
        return
    expect_bf16_values("classic.storage.inverse_applied", A_inv, G_inv)
    seen[module] = (A_inv, G_inv)


def enable_classic_bf16():
    ck.ClassicKFAC._update_inverses = _update_inverses_bf16
    ck.ClassicKFAC.inverse_check = _check_applied_inverses


def disable_classic_bf16():
    ck.ClassicKFAC._update_inverses = _orig_update_inverses
    ck.ClassicKFAC.step = _orig_classic_step
    ck.ClassicKFAC.inverse_check = None


# ---- Config + cells -------------------------------------------------------

CFG_BASE = dict(gamma=0.9, momentum=0.7, kfac_lr=2e-3, grad_clip=300.0,
                max_steps=1000, warmup_steps=200, seed=42, factor_update_freq=20)

# (label, variant, damping, wgso)
CELLS = [
    ("classic",    "ClassicKFAC", 1e-4, False),
    ("ifkfac",      "IFKFAC",   1e-4, False),
    ("ifkfac_wgso", "IFKFAC",   1e-6, True),
]


def out_path(label):
    return OUT / f"per_step_bf16_{label}_s1000.json"


def run_one(label, variant, damping, wgso, tlf, vl, vocab, pad, device, hw):
    p = out_path(label)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    if variant == "IFKFAC":
        enable_bf16(wgso=wgso)
    elif variant == "ClassicKFAC":
        enable_classic_bf16()

    try:
        torch.manual_seed(CFG_BASE["seed"])
        model = SmallGPT(vocab_size=vocab).to(device)
        kfac, emb, _ = make_optimizers(
            variant, model, CFG_BASE["kfac_lr"], damping, CFG_BASE["momentum"],
            grad_clip=CFG_BASE["grad_clip"], gamma=CFG_BASE["gamma"],
            factor_update_freq=CFG_BASE["factor_update_freq"])
        w = CFG_BASE["warmup_steps"]
        sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
            torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG_BASE["max_steps"]),
        ], milestones=[w])
        esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG_BASE["max_steps"])

        it = iter(tlf())
        recs, prev = [], None
        print(f"\n=== bf16  {label}  variant={variant}  damp={damping}  wgso={wgso} ===")
        t0 = time.perf_counter()
        for step in range(1, CFG_BASE["max_steps"]+1):
            try: x, y = next(it)
            except StopIteration: it = iter(tlf()); x, y = next(it)
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), ignore_index=pad)
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step}  NaN, aborting"); break
            model.zero_grad(); loss.backward()
            kfac.step(); emb.step(); sched.step(); esched.step()
            dl = (lv - prev) if prev is not None else 0.0; prev = lv
            recs.append({"step": step, "loss": lv, "delta_loss": dl,
                         "refreshed": (step-1) % CFG_BASE["factor_update_freq"] == 0,
                         "steps_since_refresh": (step-1) % CFG_BASE["factor_update_freq"]})

        fp = evaluate_ppl(model, vl, device, pad)
        wall = time.perf_counter() - t0
        out = {"label": label, "variant": variant, "wgso": wgso,
               "damping": damping, "precision": "bf16_kfac_only",
               "config": CFG_BASE, "hw": hw, "wall_s": wall,
               "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    finally:
        disable_bf16()
        disable_classic_bf16()

    return out


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1

    results = []
    for label, variant, damping, wgso in CELLS:
        results.append(run_one(label, variant, damping, wgso, tlf, vl, vocab, pad, device, hw))

    print("\n=== bf16 K-FAC comparison ===")
    print(f"  {'cell':>14}  {'final_ppl':>10}  {'fp32 baseline':>14}")
    fp32 = {"classic": 922, "ifkfac": 921, "ifkfac_wgso": 898}
    for r in results:
        label = r.get("label")
        fp = r.get("final_ppl")
        baseline = fp32.get(label, 0)
        diff = fp - baseline if fp else None
        diff_s = f"{diff:+.0f}" if diff is not None else "--"
        print(f"  {label:>14}  {(f'{fp:.0f}' if fp else 'DIV'):>10}  {baseline:>11} ({diff_s})")


if __name__ == "__main__":
    main()
