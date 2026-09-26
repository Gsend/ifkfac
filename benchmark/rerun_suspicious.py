"""
benchmark/rerun_suspicious.py

Rerun every benchmark cell whose stored JSON does not match the number
quoted in the TMLR paper (see audit against results/*.json).

Two suspicious clusters:

  A. §5.4 fp32 sanity, SmallGPT-small, champion HP (single seed=42).
     Article: Classic 922, IFKFAC 921, SINGD-Dense ~920
     Log:     Classic 932.7 (per_step_ClassicKFAC_champion_s1000.json),
              IFKFAC  no file on disk,
              SINGD   no file on disk.
     Action:  Regenerate all three fp32 cells with the champion HP so
              §5.4 has a coherent, up-to-date single-seed baseline.

  B. §5.9 Laplace CIFAR-10 ResNet-18, IFKFAC rows (3 seeds x fp32/bf16).
     Article's IFKFAC numbers are ~1.5% acc / ~0.04 NLL BETTER than the
     JSON files on disk (laplace_cifar10_ifkfac_{fp32,bf16}_seed{42..44}).
     Classic rows match to 4 decimals.
     Action:  Delete the 6 ifkfac JSON files and let laplace_cifar10.py
              re-run only those cells (its writer skips existing files,
              so Classic cells are preserved and not repeated).

Recipe details are read verbatim from the sibling scripts:
  §5.4: benchmark/kfac_bf16_compare.py (CFG_BASE, damping=1e-4)
  §5.9: benchmark/laplace_cifar10.py

Outputs (rewritten in place):
  per_step_fp32_small_classic_champion_s1000.json
  per_step_fp32_small_ifkfac_champion_s1000.json
  per_step_fp32_small_singd_champion_s1000.json
  laplace_cifar10_ifkfac_{fp32,bf16}_seed{42,43,44}.json

Run:
    python -m benchmark.rerun_suspicious                 # both parts
    python -m benchmark.rerun_suspicious --part fp32     # §5.4 only
    python -m benchmark.rerun_suspicious --part laplace  # §5.9 only
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import (
    build_data,
    make_optimizers,
    evaluate_ppl,
    KFAC_MAX_DIM,
    OUT,
)
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info


# =============================================================================
# Part A: §5.4 fp32 sanity — SmallGPT-small, champion HP, single seed
# =============================================================================
CFG_FP32 = dict(
    gamma=0.9,
    momentum=0.7,
    kfac_lr=2e-3,
    grad_clip=300.0,
    max_steps=1000,
    warmup_steps=200,
    seed=42,
    factor_update_freq=20,
    damping=1e-4,
)


def fp32_out_path(label: str) -> Path:
    return OUT / f"per_step_fp32_small_{label}_champion_s1000.json"


def _run_kfac_fp32_cell(variant, label, tlf, vl, vocab, pad, device, hw):
    """variant is 'ClassicKFAC' or 'IFKFAC'."""
    p = fp32_out_path(label)
    if p.exists():
        print(f"[skip fp32/{label}] {p.name}")
        return json.loads(p.read_text())

    torch.manual_seed(CFG_FP32["seed"])
    model = SmallGPT(vocab_size=vocab).to(device)
    kfac, emb, _ = make_optimizers(
        variant, model,
        CFG_FP32["kfac_lr"], CFG_FP32["damping"], CFG_FP32["momentum"],
        grad_clip=CFG_FP32["grad_clip"], gamma=CFG_FP32["gamma"],
        factor_update_freq=CFG_FP32["factor_update_freq"],
    )
    w = CFG_FP32["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(
        kfac,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                kfac, start_factor=0.1, end_factor=1.0, total_iters=w
            ),
            torch.optim.lr_scheduler.ConstantLR(
                kfac, factor=1.0, total_iters=CFG_FP32["max_steps"]
            ),
        ],
        milestones=[w],
    )
    esched = torch.optim.lr_scheduler.ConstantLR(
        emb, factor=1.0, total_iters=CFG_FP32["max_steps"]
    )

    it = iter(tlf())
    recs, prev = [], None
    print(f"\n=== [A/fp32] {label}  (champion HP, seed={CFG_FP32['seed']}) ===")
    t0 = time.perf_counter()
    for step in range(1, CFG_FP32["max_steps"] + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad,
        )
        lv = float(loss.item())
        if not math.isfinite(lv):
            print(f"  step={step} NaN, aborting"); break
        model.zero_grad(); loss.backward()
        kfac.step(); emb.step(); sched.step(); esched.step()
        dl = (lv - prev) if prev is not None else 0.0
        prev = lv
        recs.append({
            "step": step, "loss": lv, "delta_loss": dl,
            "refreshed": (step - 1) % CFG_FP32["factor_update_freq"] == 0,
            "steps_since_refresh": (step - 1) % CFG_FP32["factor_update_freq"],
        })

    fp = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0
    out = {
        "label": label, "variant": variant,
        "precision": "fp32", "config": CFG_FP32, "hw": hw,
        "wall_s": wall, "final_ppl": fp, "per_step": recs,
    }
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={fp:.1f}  wall={wall/60:.1f}m")

    try:
        kfac.cleanup()
    except Exception:
        pass
    del kfac, emb, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _run_singd_fp32_cell(tlf, vl, vocab, pad, device, hw):
    from singd.optim.optimizer import SINGD
    from torch import nn

    p = fp32_out_path("singd")
    if p.exists():
        print(f"[skip fp32/singd] {p.name}")
        return json.loads(p.read_text())

    torch.manual_seed(CFG_FP32["seed"])
    model = SmallGPT(vocab_size=vocab).to(device)

    kfac_param_ids = set()
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        out_dim = (
            module.out_features
            if isinstance(module, nn.Linear)
            else module.out_channels
        )
        if KFAC_MAX_DIM > 0 and out_dim > KFAC_MAX_DIM:
            continue
        for p_ in module.parameters():
            kfac_param_ids.add(id(p_))

    kfac_params = [pp for pp in model.parameters() if id(pp) in kfac_param_ids]
    other_params = [pp for pp in model.parameters() if id(pp) not in kfac_param_ids]

    # SINGD hyperparameters: tune2 winner for bf16 (lr_cov=0.1); at fp32
    # this cell just checks that the fp32 baseline is ~920. Keep the same
    # winner HP for consistency with §5.5/§5.7.
    singd = SINGD(
        model,
        params=kfac_params,
        lr=CFG_FP32["kfac_lr"],
        damping=1e-3,
        momentum=CFG_FP32["momentum"],
        T=CFG_FP32["factor_update_freq"],
        structures=("dense", "dense"),
        loss_average="batch+sequence",
        lr_cov=1e-1,
        alpha1=0.5,
        kfac_like=False,
        warn_unsupported=False,
    )
    print(f"  [singd] hooked {len(singd.module_names)} modules")
    emb = torch.optim.AdamW(other_params, lr=1e-3, weight_decay=0.0)

    w = CFG_FP32["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(
        singd,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                singd, start_factor=0.1, end_factor=1.0, total_iters=w
            ),
            torch.optim.lr_scheduler.ConstantLR(
                singd, factor=1.0, total_iters=CFG_FP32["max_steps"]
            ),
        ],
        milestones=[w],
    )
    esched = torch.optim.lr_scheduler.ConstantLR(
        emb, factor=1.0, total_iters=CFG_FP32["max_steps"]
    )

    it = iter(tlf())
    recs, prev = [], None
    print(f"\n=== [A/fp32] singd  (champion HP, seed={CFG_FP32['seed']}) ===")
    t0 = time.perf_counter()
    for step in range(1, CFG_FP32["max_steps"] + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad,
        )
        lv = float(loss.item())
        if not math.isfinite(lv):
            print(f"  step={step} NaN, aborting"); break
        model.zero_grad(); loss.backward()
        singd.step(); emb.step(); sched.step(); esched.step()
        dl = (lv - prev) if prev is not None else 0.0
        prev = lv
        recs.append({
            "step": step, "loss": lv, "delta_loss": dl,
            "refreshed": (step - 1) % CFG_FP32["factor_update_freq"] == 0,
            "steps_since_refresh": (step - 1) % CFG_FP32["factor_update_freq"],
        })

    fp = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0
    out = {
        "label": "singd", "variant": "SINGD",
        "precision": "fp32", "config": CFG_FP32, "hw": hw,
        "wall_s": wall, "final_ppl": fp, "per_step": recs,
    }
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={fp:.1f}  wall={wall/60:.1f}m")
    try:
        singd.cleanup()
    except Exception:
        pass
    del singd, emb, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def rerun_fp32_sanity():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    print("\n### Part A: §5.4 fp32 sanity ###")
    results = []
    results.append(
        _run_kfac_fp32_cell("ClassicKFAC", "classic", tlf, vl, vocab, pad, device, hw)
    )
    results.append(
        _run_kfac_fp32_cell("IFKFAC", "ifkfac", tlf, vl, vocab, pad, device, hw)
    )
    results.append(
        _run_singd_fp32_cell(tlf, vl, vocab, pad, device, hw)
    )

    print("\n=== §5.4 fp32 sanity summary (article: Classic 922, IFKFAC 921, SINGD ~920) ===")
    for r in results:
        fp = r.get("final_ppl")
        print(f"  {r['label']:>8}: ppl={fp:.1f}  wall={r['wall_s']/60:.1f}m")
    return results


# =============================================================================
# Part B: §5.9 Laplace CIFAR-10 IFKFAC (ifkfac) — 3 seeds x fp32/bf16
# =============================================================================
def rerun_laplace_ifkfac():
    """Force re-run of the 6 suspicious ifkfac laplace cells.

    laplace_cifar10.py's run_cell() skips existing files, so we delete the
    ifkfac JSONs first. classic files are left untouched — those numbers
    match the paper.
    """
    print("\n### Part B: §5.9 Laplace ifkfac re-run ###")
    to_delete = []
    for prec in ("fp32", "bf16"):
        for seed in (42, 43, 44):
            p = OUT / f"laplace_cifar10_ifkfac_{prec}_seed{seed}.json"
            if p.exists():
                to_delete.append(p)
    for p in to_delete:
        print(f"  deleting {p.name}")
        p.unlink()

    # Import late to avoid pulling laplace deps for a fp32-only run.
    from benchmark import laplace_cifar10 as lap

    # Restrict CELLS to just ifkfac, keep SEEDS from the module.
    original_cells = list(lap.CELLS)
    lap.CELLS = [("ifkfac", "fp32"), ("ifkfac", "bf16")]
    try:
        lap.main()
    finally:
        lap.CELLS = original_cells

    # Print a quick recompute of the summary.
    print("\n=== §5.9 Laplace ifkfac summary (article: fp32 84.39/0.4687, bf16 84.00/0.4762) ===")
    from statistics import mean, stdev
    for prec in ("fp32", "bf16"):
        accs, nlls = [], []
        for seed in (42, 43, 44):
            p = OUT / f"laplace_cifar10_ifkfac_{prec}_seed{seed}.json"
            if p.exists():
                d = json.loads(p.read_text())
                m = d.get("metrics", {})
                if isinstance(m.get("accuracy"), (int, float)):
                    accs.append(m["accuracy"])
                if isinstance(m.get("nll"), (int, float)):
                    nlls.append(m["nll"])
        if accs:
            am = 100 * mean(accs); asd = 100 * stdev(accs) if len(accs) > 1 else 0
            nm = mean(nlls); nsd = stdev(nlls) if len(nlls) > 1 else 0
            print(f"  ifkfac/{prec}: acc%={am:.2f} ± {asd:.2f}   NLL={nm:.4f} ± {nsd:.4f}")


# =============================================================================
# Driver
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part", choices=("all", "fp32", "laplace"), default="all",
        help="Which suspicious block to rerun (default: all)",
    )
    args = parser.parse_args()

    if args.part in ("all", "fp32"):
        rerun_fp32_sanity()
    if args.part in ("all", "laplace"):
        rerun_laplace_ifkfac()


if __name__ == "__main__":
    main()
