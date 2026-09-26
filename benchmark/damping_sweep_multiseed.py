"""
benchmark/damping_sweep_multiseed.py

Multi-seed bf16 damping-sensitivity sweep for §5.5 (three seeds, three
optimizers, seven damping values = 63 cells).  Supersedes the earlier
single-seed scripts by folding all three methods into one grid.

Grid:
    methods  ∈ {classic, ifkfac, singd}
    seeds    ∈ {42, 43, 44}                          (paper §5.7 convention)
    lambda   ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}   (Classic 7-point grid)

Recipe (identical to benchmark/kfac_bf16_classic_damping.py):
    mom=0.7, lr=2e-3, gamma=0.9, freq=20, max_steps=1000, warmup=200,
    SmallGPT-small, bf16 K-FAC-only.
    SINGD hyperparameters: tune2 winner (lr_cov=1e-1, alpha1=0.5, kfac_like=False).

Output naming (new, consistent across methods):
    per_step_bf16_{method}_damp_d{d}_seed{seed}_s1000.json

Backward compat with the earlier single-seed runs (seed=42):
    - Classic old: per_step_bf16_classic_d{d}_s1000.json
    - IFKFAC old:   per_step_bf16_damp_ifkfac_d{d}_s1000.json
    - SINGD old:   per_step_bf16_singd_damp_d{d}_s1000.json
    If an old-named file exists and the new-named file does not, the old
    file's content is copied to the new name (no re-run required).

Run:
    python -m benchmark.damping_sweep_multiseed
Each cell caches its own JSON, so the script is safely resumable.
"""
from __future__ import annotations
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
from benchmark.kfac_bf16_compare import (
    enable_bf16, disable_bf16,
    enable_classic_bf16, disable_classic_bf16,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CFG = dict(
    gamma=0.9,
    momentum=0.7,
    kfac_lr=2e-3,
    grad_clip=300.0,
    max_steps=1000,
    warmup_steps=200,
    factor_update_freq=20,
)
SEEDS    = [42, 43, 44]
DAMPINGS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# SINGD hyperparameters (tune2 winner).
SINGD_HP = dict(lr_cov=1e-1, alpha1=0.5, kfac_like=False)


# -----------------------------------------------------------------------------
# Filename helpers + backward-compat migration
# -----------------------------------------------------------------------------
def new_path(method: str, damping: float, seed: int) -> Path:
    return OUT / f"per_step_bf16_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json"


def old_path_seed42(method: str, damping: float) -> Path:
    """Existing single-seed (seed=42) filenames from the earlier scripts."""
    if method == "classic":
        return OUT / f"per_step_bf16_classic_d{damping:.0e}_s1000.json"
    if method == "ifkfac":
        return OUT / f"per_step_bf16_damp_ifkfac_d{damping:.0e}_s1000.json"
    if method == "singd":
        return OUT / f"per_step_bf16_singd_damp_d{damping:.0e}_s1000.json"
    raise ValueError(method)


def migrate_seed42():
    """Copy any seed-42 old-named results to the new naming scheme so the
    multi-seed loop can find them and skip the recompute."""
    migrated = 0
    for method in ("classic", "ifkfac", "singd"):
        for d in DAMPINGS:
            old = old_path_seed42(method, d)
            new = new_path(method, d, 42)
            if old.exists() and not new.exists():
                new.write_text(old.read_text())
                migrated += 1
                print(f"  [migrate] {old.name} -> {new.name}")
    print(f"  migrated {migrated} seed-42 files to new naming.")


# -----------------------------------------------------------------------------
# Common training loop for K-FAC-family optimizers (Classic, IFKFAC)
# -----------------------------------------------------------------------------
def _train_kfac(variant, damping, seed, model, tlf, vl, vocab, pad, device):
    torch.manual_seed(seed)
    kfac, emb, _ = make_optimizers(
        variant, model,
        CFG["kfac_lr"], damping, CFG["momentum"],
        grad_clip=CFG["grad_clip"], gamma=CFG["gamma"],
        factor_update_freq=CFG["factor_update_freq"],
    )
    w = CFG["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(
        kfac,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                kfac, start_factor=0.1, end_factor=1.0, total_iters=w
            ),
            torch.optim.lr_scheduler.ConstantLR(
                kfac, factor=1.0, total_iters=CFG["max_steps"]
            ),
        ],
        milestones=[w],
    )
    esched = torch.optim.lr_scheduler.ConstantLR(
        emb, factor=1.0, total_iters=CFG["max_steps"]
    )
    return _run_loop(kfac, emb, sched, esched, model, tlf, vl, pad, device, autocast=False)


def _build_singd(model, damping, seed):
    from singd.optim.optimizer import SINGD
    from torch import nn

    torch.manual_seed(seed)
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
        for p in module.parameters():
            kfac_param_ids.add(id(p))
    kfac_params = [p for p in model.parameters() if id(p) in kfac_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kfac_param_ids]

    singd = SINGD(
        model,
        params=kfac_params,
        lr=CFG["kfac_lr"],
        damping=damping,
        momentum=CFG["momentum"],
        T=CFG["factor_update_freq"],
        structures=("dense", "dense"),
        loss_average="batch+sequence",
        lr_cov=SINGD_HP["lr_cov"],
        alpha1=SINGD_HP["alpha1"],
        kfac_like=SINGD_HP["kfac_like"],
        warn_unsupported=False,
    )
    print(f"  [singd] hooked {len(singd.module_names)} modules")
    emb = torch.optim.AdamW(other_params, lr=1e-3, weight_decay=0.0)
    return singd, emb


def _train_singd(damping, seed, model, tlf, vl, vocab, pad, device):
    singd, emb = _build_singd(model, damping, seed)
    w = CFG["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(
        singd,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                singd, start_factor=0.1, end_factor=1.0, total_iters=w
            ),
            torch.optim.lr_scheduler.ConstantLR(
                singd, factor=1.0, total_iters=CFG["max_steps"]
            ),
        ],
        milestones=[w],
    )
    esched = torch.optim.lr_scheduler.ConstantLR(
        emb, factor=1.0, total_iters=CFG["max_steps"]
    )
    return _run_loop(singd, emb, sched, esched, model, tlf, vl, pad, device, autocast=True)


def _run_loop(kfac, emb, sched, esched, model, tlf, vl, pad, device, autocast):
    it = iter(tlf())
    recs, prev = [], None
    t0 = time.perf_counter()
    for step in range(1, CFG["max_steps"] + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        if autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=pad,
                )
        else:
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
            "refreshed": (step - 1) % CFG["factor_update_freq"] == 0,
            "steps_since_refresh": (step - 1) % CFG["factor_update_freq"],
        })

    ppl = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0

    try:
        kfac.cleanup()
    except Exception:
        pass
    return ppl, wall, recs


# -----------------------------------------------------------------------------
# Per-method dispatch: enter precision mode, build model, train, tear down.
# -----------------------------------------------------------------------------
def run_cell(method, damping, seed, tlf, vl, vocab, pad, device, hw):
    p = new_path(method, damping, seed)
    if p.exists():
        return

    print(f"\n=== [{method}] seed={seed}  damp={damping:.0e} ===")
    model = None
    try:
        if method == "classic":
            enable_classic_bf16()
            torch.manual_seed(seed)
            model = SmallGPT(vocab_size=vocab).to(device)
            ppl, wall, recs = _train_kfac(
                "ClassicKFAC", damping, seed, model, tlf, vl, vocab, pad, device
            )
        elif method == "ifkfac":
            enable_bf16(wgso=False)
            torch.manual_seed(seed)
            model = SmallGPT(vocab_size=vocab).to(device)
            ppl, wall, recs = _train_kfac(
                "IFKFAC", damping, seed, model, tlf, vl, vocab, pad, device
            )
        elif method == "singd":
            torch.manual_seed(seed)
            model = SmallGPT(vocab_size=vocab).to(device)
            ppl, wall, recs = _train_singd(
                damping, seed, model, tlf, vl, vocab, pad, device
            )
        else:
            raise ValueError(method)

        out = {
            "method": method, "damping": damping, "seed": seed,
            "precision": "bf16_kfac_only", "config": CFG,
            "singd_hp": SINGD_HP if method == "singd" else None,
            "hw": hw, "wall_s": wall,
            "final_ppl": ppl, "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={ppl:.0f}  wall={wall/60:.1f}m")
    finally:
        if method == "classic":
            disable_classic_bf16()
        elif method == "ifkfac":
            disable_bf16()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Driver + summary
# -----------------------------------------------------------------------------
def print_summary():
    from statistics import mean, stdev

    print("\n=== bf16 damping-sensitivity sweep — multi-seed summary ===")
    print(f"  {'':<10}" + "  ".join(f"{d:>10.0e}" for d in DAMPINGS))
    for method in ("classic", "ifkfac", "singd"):
        row = []
        for d in DAMPINGS:
            vals = []
            for seed in SEEDS:
                p = new_path(method, d, seed)
                if p.exists():
                    fp = json.loads(p.read_text()).get("final_ppl")
                    if fp is not None and math.isfinite(fp):
                        vals.append(fp)
            if len(vals) == len(SEEDS):
                m = mean(vals); s = stdev(vals) if len(vals) > 1 else 0
                row.append(f"{m:>4.0f}±{s:>3.0f}")
            elif vals:
                m = mean(vals)
                row.append(f"({m:>4.0f}) {len(vals)}/{len(SEEDS)}")
            else:
                row.append("       --")
        print(f"  {method:<10}" + "  ".join(f"{c:>10}" for c in row))


def main():
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    print("\n### Migration: reuse existing seed-42 files ###")
    migrate_seed42()

    tlf, vl, vocab = build_data(device)
    pad = vocab - 1

    total = len(("classic", "ifkfac", "singd")) * len(SEEDS) * len(DAMPINGS)
    done = 0
    # Order the loops so seed 42 (mostly cached) prints first, then 43, 44.
    for seed in SEEDS:
        for method in ("classic", "ifkfac", "singd"):
            for d in DAMPINGS:
                done += 1
                print(f"\n[{done}/{total}]")
                run_cell(method, d, seed, tlf, vl, vocab, pad, device, hw)

    print_summary()


if __name__ == "__main__":
    main()
