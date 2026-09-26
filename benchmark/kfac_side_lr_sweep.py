"""
benchmark/kfac_side_lr_sweep.py

Does K-FAC lose to AdamW on SmallGPT because of the "side" AdamW?

K-FAC preconditions only the nn.Linear layers with out_dim <= 4096.  Everything
else (the token embedding, which is TIED to the 50257-way output head, the
positional embedding and the LayerNorms) is trained by a side AdamW.  In the
§5.7 matched-tuning numbers (K-FAC 625/626 ppl vs AdamW 446 ppl, SmallGPT-medium,
fp32, seed 42) that side AdamW ran at lr=1e-3 with no warmup, while the tuned
AdamW baseline trains the same parameters at lr=3e-3 with a 200-step warmup.
The embedding/head is the largest matrix in the model, so this may account for
much of the gap.  This script tests that directly.

K-FAC settings are those of kfac_wd_tuning_sweep.py at its best weight decay
(wd=0.1): lr 2e-3, momentum 0.7, gamma 0.9, T=20, damping 1e-4, clip 300,
1000 steps, warmup 200, SmallGPT-medium (d=384, 6 layers), fp32.

Side-AdamW configurations (betas 0.9/0.95, weight decay 0.1 throughout):
    ref     lr 1e-3, no warmup    -> reproduces the §5.7 cells (625 / 626 ppl)
    wu1e-3  lr 1e-3, warmup 200
    wu3e-3  lr 3e-3, warmup 200   -> exactly what tuned AdamW gives these params
    wu6e-3  lr 6e-3, warmup 200

Phases
    python -m benchmark.kfac_side_lr_sweep              # seed 42 grid: 2 variants x 4 configs (~1.6 h)
    python -m benchmark.kfac_side_lr_sweep --confirm    # best config per variant + AdamW, seeds 43 44 (~1 h)
    python -m benchmark.kfac_side_lr_sweep --summary    # print the table from saved JSONs
Resumable: finished cells (JSON on disk) are skipped.

Outputs (benchmark/results/):
    kfac_sidelr_{variant}_{cfg}_seed{seed}.json
    adamw_ref_transformer_fp32_seed{seed}.json     (AdamW lr 3e-3, wd 0.1, beta2 0.95)
Seed 42 AdamW reuses adamw_tune_transformer_fp32_lr3e-03_wd0.1.json (446 ppl).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = ROOT / "benchmark" / "results"
VARIANTS = ["IFKFAC", "ClassicKFAC"]
SIDE_CFGS = {
    "ref":    {"lr": 1e-3, "warmup": False},
    "wu1e-3": {"lr": 1e-3, "warmup": True},
    "wu3e-3": {"lr": 3e-3, "warmup": True},
    "wu6e-3": {"lr": 6e-3, "warmup": True},
}
SIDE_WD = 0.1
SIDE_BETAS = (0.9, 0.95)
KFAC_WD = 0.1
KFAC_LR = 2e-3
KFAC_MOMENTUM = 0.7
KFAC_GAMMA = 0.9
KFAC_FREQ = 20
KFAC_DAMPING = 1e-4
KFAC_MAX_DIM = 4096
GRAD_CLIP = 300.0
MAX_STEPS = 1000
WARMUP = 200
ADAMW_LR, ADAMW_WD = 3e-3, 0.1
CONFIRM_SEEDS = [43, 44]


def kfac_path(variant, cfg, seed):
    return OUT / f"kfac_sidelr_{variant.lower()}_{cfg}_seed{seed}.json"


def adamw_path(seed):
    if seed == 42:
        return OUT / "adamw_tune_transformer_fp32_lr3e-03_wd0.1.json"
    return OUT / f"adamw_ref_transformer_fp32_seed{seed}.json"


def build_model(ctx, device):
    from benchmark.gpu_benchmark import SmallGPT
    return SmallGPT(vocab_size=ctx["vocab"], d_model=384, n_heads=6,
                    n_layers=6, d_ff=1536).to(device)


def split_params(model):
    """(kfac_params, side_params, report) using the same rule as the §5.7 sweeps."""
    kfac_ids = set()
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            out_dim = mod.out_features if isinstance(mod, nn.Linear) else mod.out_channels
            if out_dim <= KFAC_MAX_DIM:
                kfac_ids.update(id(p) for p in mod.parameters())
    seen, kfac, side = set(), [], []
    for p in model.parameters():                 # tied weights appear once here
        if id(p) in seen:
            continue
        seen.add(id(p))
        (kfac if id(p) in kfac_ids else side).append(p)
    names = {}
    for n, p in model.named_parameters():
        names.setdefault(id(p), n)
    n_k = sum(p.numel() for p in kfac)
    n_s = sum(p.numel() for p in side)
    biggest = sorted(((p.numel(), names[id(p)]) for p in side), reverse=True)[:3]
    report = {"kfac_params": n_k, "side_params": n_s,
              "side_fraction": n_s / (n_k + n_s),
              "largest_side_tensors": [(n, int(c)) for c, n in biggest]}
    return kfac, side, report


def build_kfac(variant, model, kfac_params):
    if variant == "IFKFAC":
        from optimizer.ifkfac_kfac import IFKFAC
        return IFKFAC(model, lr=KFAC_LR, damping=KFAC_DAMPING,
                         factor_update_freq=KFAC_FREQ, weight_decay=KFAC_WD,
                         momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
                         gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM, deferred_qr=True)
    if variant == "ClassicKFAC":
        from optimizer.classic_kfac import ClassicKFAC
        return ClassicKFAC(model, lr=KFAC_LR, damping=KFAC_DAMPING,
                           factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
                           weight_decay=KFAC_WD, momentum=KFAC_MOMENTUM,
                           grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA, max_gram_dim=KFAC_MAX_DIM)
    raise ValueError(variant)


def warmup_sched(opt, on):
    if not on:
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=MAX_STEPS)
    return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
        torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=MAX_STEPS),
    ], milestones=[WARMUP])


def train(model, opts, scheds, ctx, device, steps):
    it = iter(ctx["tlf"]())
    recs = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(ctx["tlf"]()); batch = next(it)
        x, y = (t.to(device, non_blocking=True) for t in batch)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=ctx["pad"])
        lv = float(loss.item())
        if not math.isfinite(lv):
            print(f"  step={step} non-finite loss, aborting", flush=True)
            break
        model.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        recs.append({"step": step, "loss": lv})
        if step % 200 == 0:
            print(f"  step {step}: loss={lv:.3f}", flush=True)
    wall = time.perf_counter() - t0
    from benchmark.stability_benchmark import evaluate_ppl
    try:
        ppl = float(evaluate_ppl(model, ctx["vl"], device, ctx["pad"]))
    except Exception as e:
        print(f"  eval failed: {e}", flush=True)
        ppl = None
    return recs, wall, ppl


def run_kfac(variant, cfg, seed, ctx, device, hw, steps):
    p = kfac_path(variant, cfg, seed)
    if p.exists():
        print(f"[skip] {p.name}", flush=True)
        return json.loads(p.read_text())
    sc = SIDE_CFGS[cfg]
    torch.manual_seed(seed)
    model = build_model(ctx, device)
    kfac_params, side_params, report = split_params(model)
    kfac = build_kfac(variant, model, kfac_params)
    side = torch.optim.AdamW(side_params, lr=sc["lr"], weight_decay=SIDE_WD, betas=SIDE_BETAS)
    s_k, s_s = warmup_sched(kfac, True), warmup_sched(side, sc["warmup"])
    print(f"\n=== {variant}  side={cfg} (lr {sc['lr']:g}, warmup {sc['warmup']})  seed {seed} ===",
          flush=True)
    try:
        recs, wall, ppl = train(model, [kfac, side], [s_k, s_s], ctx, device, steps)
    finally:
        try:
            kfac.cleanup()
        except Exception:
            pass
    out = {"variant": variant, "side_cfg": cfg, "side_lr": sc["lr"], "side_warmup": sc["warmup"],
           "side_wd": SIDE_WD, "side_betas": SIDE_BETAS, "kfac_wd": KFAC_WD, "seed": seed,
           "steps": steps, "wall_s": wall, "final_ppl": ppl, "param_split": report,
           "hw": hw, "per_step": recs}
    if steps == MAX_STEPS:
        p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={ppl:.0f}  wall={wall/60:.1f} min" if ppl else "  -> diverged", flush=True)
    del kfac, side, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_adamw(seed, ctx, device, hw, steps):
    p = adamw_path(seed)
    if p.exists():
        print(f"[skip] {p.name}", flush=True)
        return json.loads(p.read_text())
    torch.manual_seed(seed)
    model = build_model(ctx, device)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WD, betas=(0.9, 0.95))
    print(f"\n=== AdamW lr {ADAMW_LR:g} wd {ADAMW_WD:g}  seed {seed} ===", flush=True)
    recs, wall, ppl = train(model, [opt], [warmup_sched(opt, True)], ctx, device, steps)
    out = {"optimizer": "adamw", "lr": ADAMW_LR, "weight_decay": ADAMW_WD, "seed": seed,
           "steps": steps, "wall_s": wall, "final_ppl": ppl, "hw": hw, "per_step": recs}
    if steps == MAX_STEPS:
        p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={ppl:.0f}  wall={wall/60:.1f} min" if ppl else "  -> diverged", flush=True)
    del opt, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def load_ppl(p):
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("final_ppl")
    return float(v) if v is not None and math.isfinite(float(v)) else None


def best_cfg(variant):
    cells = {c: load_ppl(kfac_path(variant, c, 42)) for c in SIDE_CFGS}
    cells = {c: v for c, v in cells.items() if v is not None}
    return min(cells, key=cells.get) if cells else None


def summary():
    print("\n=== SmallGPT-medium fp32, 1000 steps: final validation ppl ===", flush=True)
    a42 = load_ppl(adamw_path(42))
    print(f"  AdamW (lr 3e-3, wd 0.1), seed 42: {a42:.0f}" if a42 else "  AdamW seed 42: missing")
    print(f"  {'variant':<12} " + "  ".join(f"{c:>8}" for c in SIDE_CFGS), flush=True)
    for v in VARIANTS:
        row = [load_ppl(kfac_path(v, c, 42)) for c in SIDE_CFGS]
        print(f"  {v:<12} " + "  ".join(f"{x:>8.0f}" if x else f"{'-':>8}" for x in row))
    print("\n  seeds 42-44 at each variant's best side config (and AdamW):")
    for v in VARIANTS:
        c = best_cfg(v)
        if c is None:
            continue
        xs = [load_ppl(kfac_path(v, c, s)) for s in [42] + CONFIRM_SEEDS]
        xs = [x for x in xs if x is not None]
        if xs:
            sd = f" ± {stdev(xs):.0f}" if len(xs) > 1 else ""
            print(f"  {v:<12} side={c:<7} n={len(xs)}  {mean(xs):.0f}{sd}")
    xs = [load_ppl(adamw_path(s)) for s in [42] + CONFIRM_SEEDS]
    xs = [x for x in xs if x is not None]
    if xs:
        sd = f" ± {stdev(xs):.0f}" if len(xs) > 1 else ""
        print(f"  {'AdamW':<12} {'':<12} n={len(xs)}  {mean(xs):.0f}{sd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="run each variant's best seed-42 side config and AdamW on seeds 43 44")
    ap.add_argument("--summary", action="store_true", help="only print the table")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--steps", type=int, default=MAX_STEPS,
                    help="for a quick test only; results are saved only at the full 1000 steps")
    args = ap.parse_args()
    if args.summary:
        summary(); return

    from benchmark.gpu_benchmark import get_device, get_hardware_info
    from benchmark.stability_benchmark import build_data
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}", flush=True)
    tlf, vl, vocab = build_data(device)
    ctx = {"tlf": tlf, "vl": vl, "vocab": vocab, "pad": vocab - 1}

    _, _, report = split_params(build_model(ctx, "cpu"))
    print(f"param split: K-FAC {report['kfac_params']/1e6:.1f}M, side AdamW "
          f"{report['side_params']/1e6:.1f}M ({100*report['side_fraction']:.0f}%); largest side "
          f"tensors: {report['largest_side_tensors']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.confirm:
        for v in args.variants:
            for c in SIDE_CFGS:
                run_kfac(v, c, 42, ctx, device, hw, args.steps)
    else:
        for v in args.variants:
            c = best_cfg(v)
            if c is None:
                print(f"  no seed-42 results for {v}; run the grid first", flush=True)
                continue
            for s in CONFIRM_SEEDS:
                run_kfac(v, c, s, ctx, device, hw, args.steps)
        for s in CONFIRM_SEEDS:
            run_adamw(s, ctx, device, hw, args.steps)
    summary()


if __name__ == "__main__":
    main()
