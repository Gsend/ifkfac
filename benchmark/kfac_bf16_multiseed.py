"""
benchmark/kfac_bf16_multiseed.py

Multi-seed bf16 K-FAC comparison across two architectures.

Cells (each repeated across 5 seeds × 2 archs = 30 runs):
    classic      ClassicKFAC, damp=1e-4
    ifkfac        IFKFAC + batched_qr, damp=1e-4
    ifkfac_wgso   IFKFAC + batched_qr + WGSO, damp=1e-6

Architectures:
    small   SmallGPT defaults     (4 blocks, d=256, FFN=1024)  — match prior
    medium  SmallGPT scaled up    (6 blocks, d=384, FFN=1536)  — kappa-scale check

IFKFAC variants use deferred_qr=True (2x speedup vs streaming TSQR), bringing
IFKFAC per-step wall time within ~1.77x of Classic — close enough that the
"just run Classic longer in fp32" reviewer-1 objection doesn't survive.
This is the multi-seed statistical evidence needed for the paper claim
"Classic K-FAC collapses in bf16; IFKFAC degrades gracefully."

Champion hyperparams (mom=0.7, lr=2e-3, gamma=0.9, freq=20) from prior screen.

Output: benchmark/results/per_step_bf16_{arch}_{label}_seed{seed}_s1000.json
Total wall time on RTX 3080: ~3-4 hours (resumable; skips existing).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, make_optimizers, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info
from benchmark.kfac_bf16_compare import (
    enable_bf16, disable_bf16,
    enable_classic_bf16, disable_classic_bf16,
)


# ---- Cells ----------------------------------------------------------------

CFG_BASE = dict(gamma=0.9, momentum=0.7, kfac_lr=2e-3, grad_clip=300.0,
                max_steps=1000, warmup_steps=200, factor_update_freq=20)

# (label, variant, damping, wgso)
CELLS = [
    ("classic",    "ClassicKFAC", 1e-4, False),
    ("ifkfac",      "IFKFAC",   1e-4, False),
    ("ifkfac_wgso", "IFKFAC",   1e-6, True),
    ("singd",      "SINGD",       1e-4, False),    # Lin et al. ICML 2024 inverse-free KFAC
]

# ---- Architectures --------------------------------------------------------

# Each entry: (label, kwargs for SmallGPT constructor).
# Default SmallGPT has d=256, 4 blocks, d_ff=1024.  'medium' scales up to
# stress test whether the bf16 gap holds with more layers / wider activations.
ARCHS = [
    ("small",  dict()),                                                  # defaults
    ("medium", dict(d_model=384, n_heads=6, n_layers=6, d_ff=1536)),     # ~22M params
]

SEEDS = [42, 43, 44, 45, 46]


# ---- Run ------------------------------------------------------------------

def out_path(arch, label, seed):
    return OUT / f"per_step_bf16_{arch}_{label}_seed{seed}_s1000.json"


def run_one(arch_label, arch_kwargs, label, variant, damping, wgso,
            seed, tlf, vl, vocab, pad, device, hw):
    p = out_path(arch_label, label, seed)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    # Engage bf16 patches for this cell (SINGD manages its own precision).
    if variant == "IFKFAC":
        enable_bf16(wgso=wgso)
    elif variant == "ClassicKFAC":
        enable_classic_bf16()

    try:
        torch.manual_seed(seed)
        model = SmallGPT(vocab_size=vocab, **arch_kwargs).to(device)

        if variant == "SINGD":
            kfac, emb = _build_singd_optimizers(model, damping)
        else:
            # IFKFAC/Classic: existing dual-optimizer build path.
            kfac, emb, _ = _build_optimizers(
                variant, model, CFG_BASE["kfac_lr"], damping, CFG_BASE["momentum"],
                grad_clip=CFG_BASE["grad_clip"], gamma=CFG_BASE["gamma"],
                factor_update_freq=CFG_BASE["factor_update_freq"],
            )

        w = CFG_BASE["warmup_steps"]
        sched = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
            torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=CFG_BASE["max_steps"]),
        ], milestones=[w])
        esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG_BASE["max_steps"])

        # SINGD runs the model itself in bf16 (autocast); other variants
        # stay fp32 with the K-FAC factor pipeline patched to bf16.
        use_amp = (variant == "SINGD")

        it = iter(tlf())
        recs, prev = [], None
        print(f"\n=== bf16  arch={arch_label}  {label}  seed={seed}  damp={damping} ===")
        t0 = time.perf_counter()
        for step in range(1, CFG_BASE["max_steps"] + 1):
            try: x, y = next(it)
            except StopIteration: it = iter(tlf()); x, y = next(it)
            x, y = x.to(device), y.to(device)
            model.train()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            y.reshape(-1), ignore_index=pad)
            else:
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
        out = {"arch": arch_label, "arch_kwargs": arch_kwargs,
               "label": label, "variant": variant, "wgso": wgso,
               "damping": damping, "seed": seed,
               "precision": "bf16_kfac_only" if variant != "SINGD" else "bf16_autocast",
               "config": CFG_BASE, "hw": hw,
               "wall_s": wall, "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    finally:
        # Release the optimizer's hook handles + cached K-FAC factors before
        # leaving the run.  Without this, the next run starts with the prior
        # run's ~1 GB of hooks + cached factors still alive in GPU memory,
        # eventually tripping the allocator's slow path ("stuck" runs).
        try:
            kfac.cleanup()
        except Exception:
            pass
        try:
            del kfac, emb, model
        except Exception:
            pass
        disable_bf16()
        disable_classic_bf16()
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def _build_singd_optimizers(model, damping):
    """Build (SINGD, AdamW) pair matching our K-FAC dual-optimizer setup.

    SINGD preconditions all parameters that belong to nn.Linear / nn.Conv2d
    layers below KFAC_MAX_DIM (excluding the LM head, matching our K-FAC variants
    so the comparison is apples-to-apples).  AdamW handles the rest
    (embeddings, LayerNorm, LM head).
    """
    from singd.optim.optimizer import SINGD
    from torch import nn
    from benchmark.stability_benchmark import KFAC_MAX_DIM

    # Identify which parameters belong to K-FAC-eligible Linear/Conv2d layers.
    kfac_param_ids = set()
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        out_dim = module.out_features if isinstance(module, nn.Linear) else module.out_channels
        if KFAC_MAX_DIM > 0 and out_dim > KFAC_MAX_DIM:
            continue
        for p in module.parameters():
            kfac_param_ids.add(id(p))

    kfac_params = [p for p in model.parameters() if id(p) in kfac_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kfac_param_ids]

    # SINGD-Dense (full Kronecker, == INGD).  Hyperparameters picked from a
    # 27-cell tuning sweep (see benchmark/singd_tuning_sweep.py + JSONs at
    # benchmark/results/singd_tune2_*.json).
    #
    # Winning config (seed=42, small arch, bf16):
    #   lr_cov=1e-1, alpha1=0.5, kfac_like=False  →  ppl=953
    # SINGD's defaults (lr_cov=1e-2) gave 1782 ppl — 10x off; their
    # preconditioner stepsize default is calibrated for a different setup.
    #
    # damping matters less than expected because lr_cov dominates the
    # preconditioner update magnitude.  We keep damping=1e-3 (SINGD's default)
    # for the multiseed sweep so SINGD isn't disadvantaged by our K-FAC's
    # 1e-4 setting.
    singd_opt = SINGD(
        model,
        params=kfac_params,
        lr=CFG_BASE["kfac_lr"],
        damping=1e-3,                 # SINGD default; damping had near-zero effect
        momentum=CFG_BASE["momentum"],
        T=CFG_BASE["factor_update_freq"],
        structures=("dense", "dense"),
        loss_average="batch+sequence",
        lr_cov=1e-1,                  # ← tuned: their default 1e-2 was 10x too small
        alpha1=0.5,                   # ← SINGD default; sweep confirmed it's best
        warn_unsupported=False,
    )
    emb_opt = torch.optim.AdamW(other_params, lr=1e-3, weight_decay=0.0)
    return singd_opt, emb_opt


def _build_optimizers(variant, model, kfac_lr, damping, momentum,
                       grad_clip, gamma, factor_update_freq):
    """Wrapper around make_optimizers.

    Phase 1 change: deferred_qr is now safe under bf16 because chunk-level
    quantization replaced the per-call .to(bf16) wrappers.  The 5-chunk cat'd
    flush no longer triggers per-call round-trips, and deferred's launch-
    reduction benefit is recovered.  Expected: streaming bf16 was 14 min →
    deferred bf16 ~7 min.
    """
    if variant == "IFKFAC":
        from optimizer import ifkfac_kfac as _vk
        orig_init = _vk.IFKFAC.__init__
        def patched_init(self, *args, **kwargs):
            kwargs.setdefault("deferred_qr", True)
            return orig_init(self, *args, **kwargs)
        _vk.IFKFAC.__init__ = patched_init
        try:
            return make_optimizers(
                variant, model, kfac_lr, damping, momentum,
                grad_clip=grad_clip, gamma=gamma,
                factor_update_freq=factor_update_freq,
            )
        finally:
            _vk.IFKFAC.__init__ = orig_init
    return make_optimizers(
        variant, model, kfac_lr, damping, momentum,
        grad_clip=grad_clip, gamma=gamma,
        factor_update_freq=factor_update_freq,
    )


def summarize(results_by_arch):
    """Print per-arch summary table: mean ± std of final ppl across seeds."""
    import statistics
    for arch_label, results in results_by_arch.items():
        print(f"\n=== {arch_label} ===")
        print(f"  {'cell':>14}  {'mean ppl':>10}  {'std':>8}  {'seeds':>8}")
        by_cell = {}
        for r in results:
            by_cell.setdefault(r["label"], []).append(r.get("final_ppl"))
        for label, ppls in by_cell.items():
            ppls = [p for p in ppls if p is not None and math.isfinite(p)]
            if not ppls:
                print(f"  {label:>14}  {'DIV':>10}")
                continue
            mean = statistics.mean(ppls)
            std  = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
            print(f"  {label:>14}  {mean:>10.1f}  {std:>8.1f}  {len(ppls):>8}")


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1

    total = len(ARCHS) * len(CELLS) * len(SEEDS)
    done = 0
    results_by_arch = {}
    for arch_label, arch_kwargs in ARCHS:
        results_by_arch[arch_label] = []
        for label, variant, damping, wgso in CELLS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}]")
                r = run_one(arch_label, arch_kwargs, label, variant, damping, wgso,
                             seed, tlf, vl, vocab, pad, device, hw)
                results_by_arch[arch_label].append(r)

    summarize(results_by_arch)


if __name__ == "__main__":
    main()
