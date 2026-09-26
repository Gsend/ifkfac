"""
benchmark/kfac_wd_tuning_sweep.py

K-FAC weight_decay sweep to match the AdamW tuning fairness.

AdamW with lr=2e-3, wd=0.1, beta2=0.95 reached 672 ppl on transformer fp32.
Our K-FAC champion (wd=0) reached 804.  Closing the gap requires testing
whether K-FAC also benefits from non-zero weight_decay — an axis we never
explored in the original 2D (momentum × lr) screen.

Sweep:
    variant ∈ {ClassicKFAC, IFKFAC, SINGD}
    weight_decay ∈ {0.0, 0.01, 0.05, 0.1}
    arch = transformer-medium, precision = fp32, seed = 42, 1000 steps

= 12 cells × ~7 min = ~85 min.

Output: benchmark/results/kfac_wd_{variant}_wd{wd}.json
        plus printed summary.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = ["ClassicKFAC", "IFKFAC", "SINGD"]
WDS = [0.0, 0.01, 0.05, 0.1]
SEED = 42
MAX_STEPS = 1000
WARMUP = 200
KFAC_LR = 2e-3
KFAC_MOMENTUM = 0.7
KFAC_GAMMA = 0.9
KFAC_FREQ = 20
GRAD_CLIP = 300.0
DAMPING = {"ClassicKFAC": 1e-4, "IFKFAC": 1e-4, "SINGD": 1e-3}


def out_path(variant, wd):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"kfac_wd_{variant.lower()}_wd{wd:g}.json"


def build_kfac(variant, model, wd):
    """Build a K-FAC variant with the specified weight_decay set on the optimizer."""
    KFAC_MAX_DIM = 4096
    from optimizer import ifkfac_kfac as _vk
    from optimizer import classic_kfac as _ck

    # Identify K-FAC-eligible params
    kfac_pids = set()
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            out_dim = (mod.out_features if isinstance(mod, nn.Linear)
                        else mod.out_channels)
            if out_dim <= KFAC_MAX_DIM:
                for p in mod.parameters():
                    kfac_pids.add(id(p))
    kfac_params = [p for p in model.parameters() if id(p) in kfac_pids]
    other = [p for p in model.parameters() if id(p) not in kfac_pids]

    if variant == "IFKFAC":
        kfac = _vk.IFKFAC(
            model, lr=KFAC_LR, damping=DAMPING["IFKFAC"],
            factor_update_freq=KFAC_FREQ, weight_decay=wd,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=True,
        )
    elif variant == "ClassicKFAC":
        kfac = _ck.ClassicKFAC(
            model, lr=KFAC_LR, damping=DAMPING["ClassicKFAC"],
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=wd, momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_gram_dim=KFAC_MAX_DIM,
        )
    elif variant == "SINGD":
        from singd.optim.optimizer import SINGD
        kfac = SINGD(
            model, params=kfac_params,
            lr=KFAC_LR, damping=1e-3, momentum=KFAC_MOMENTUM,
            T=KFAC_FREQ, structures=("dense", "dense"),
            loss_average="batch+sequence",
            lr_cov=1e-1, alpha1=0.5, weight_decay=wd,
            warn_unsupported=False,
        )
    else:
        raise ValueError(variant)

    emb = torch.optim.AdamW(other, lr=1e-3, weight_decay=wd, betas=(0.9, 0.95))
    return kfac, emb


def run_one(variant, wd, ctx, device, hw):
    p = out_path(variant, wd)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    try:
        torch.manual_seed(SEED)
        from benchmark.gpu_benchmark import SmallGPT
        model = SmallGPT(vocab_size=ctx["vocab"],
                         d_model=384, n_heads=6, n_layers=6, d_ff=1536).to(device)
        kfac, emb = build_kfac(variant, model, wd)

        sched_k = torch.optim.lr_scheduler.SequentialLR(kfac, schedulers=[
            torch.optim.lr_scheduler.LinearLR(kfac, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
            torch.optim.lr_scheduler.ConstantLR(kfac, factor=1.0, total_iters=MAX_STEPS),
        ], milestones=[WARMUP])
        sched_e = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=MAX_STEPS)

        it = iter(ctx["tlf"]())
        recs = []
        print(f"\n=== {variant}  wd={wd} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            try: batch = next(it)
            except StopIteration: it = iter(ctx["tlf"]()); batch = next(it)
            x, y = (t.to(device, non_blocking=True) for t in batch)
            model.train()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), ignore_index=ctx["pad"])
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting"); break
            model.zero_grad(set_to_none=True); loss.backward()
            kfac.step(); emb.step(); sched_k.step(); sched_e.step()
            recs.append({"step": step, "loss": lv})

        wall = time.perf_counter() - t0
        from benchmark.stability_benchmark import evaluate_ppl
        try:
            fp = evaluate_ppl(model, ctx["vl"], device, ctx["pad"])
        except Exception:
            fp = None
        out = {"variant": variant, "weight_decay": wd, "seed": SEED,
               "wall_s": wall, "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = f"ppl={fp:.0f}" if fp else "ppl=DIV"
        print(f"  -> {msg}  wall={wall/60:.1f}m")
    finally:
        try:
            if hasattr(kfac, 'cleanup'): kfac.cleanup()
        except Exception: pass
        try: del kfac, emb, model
        except Exception: pass
        import gc as _gc; _gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    from benchmark.stability_benchmark import build_data
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1
    ctx = {"tlf": tlf, "vl": vl, "vocab": vocab, "pad": pad}

    total = len(VARIANTS) * len(WDS)
    done = 0
    results = {}
    for variant in VARIANTS:
        for wd in WDS:
            done += 1
            print(f"\n[{done}/{total}]")
            r = run_one(variant, wd, ctx, device, hw)
            results[(variant, wd)] = r.get("final_ppl")

    print("\n=== K-FAC weight_decay sweep summary (transformer-medium, fp32, seed 42) ===")
    print(f"  AdamW best (tuned): 446 ppl  ← target to beat")
    print(f"")
    print(f"  {'variant':>15}  {'wd=0':>7}  {'wd=0.01':>8}  {'wd=0.05':>8}  {'wd=0.1':>7}")
    for variant in VARIANTS:
        cells = [results.get((variant, wd)) for wd in WDS]
        row = "  ".join(
            f"{v:>7.0f}" if v is not None and math.isfinite(v) else f"{'DIV':>7}"
            for v in cells
        )
        print(f"  {variant:>15}  {row}")

    print("\n  ← lower is better; bold the cell that beats AdamW 672")
    print("\n  If any K-FAC variant beats 672 at some wd, K-FAC > AdamW under matched tuning.")
    print("  If none beats 672, paper pivots: K-FAC vs AdamW story now framed around bf16 stability,")
    print("  not raw fp32 perplexity.")


if __name__ == "__main__":
    main()
