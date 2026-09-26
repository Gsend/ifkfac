"""
benchmark/autoencoder_mnist_adamw_screen.py

AdamW hyperparameter screen on the MNIST autoencoder.  Required for matched
tuning fairness — K-FAC was tuned through screens 1+2+3 (~100 cells); AdamW
should get a similar effort before we can honestly compare.

Grid (fp32, seed 42, 2000 steps):
    lr     ∈ {1e-4, 3e-4, 1e-3, 3e-3}
    wd     ∈ {0, 0.01, 0.1}
    beta2  ∈ {0.999, 0.99}
= 24 cells × ~1.5 min ≈ 36 min.

Output: benchmark/results/ae_mnist_adamw_screen_lr{lr}_wd{wd}_b2{b2}.json
Plus printed summary tables.
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from benchmark.autoencoder_mnist import (
    DeepAutoencoder, get_mnist, compute_loss, evaluate_recon,
)


SEED      = 42
MAX_STEPS = 2000
WARMUP    = 100

LRS    = [1e-4, 3e-4, 1e-3, 3e-3]
WDS    = [0.0, 0.01, 0.1]
BETA2S = [0.999, 0.99]


def out_path(lr, wd, b2):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"ae_mnist_adamw_screen_lr{lr:g}_wd{wd:g}_b2{b2:g}.json"


def run_one(lr, wd, b2, ctx, device, hw):
    p = out_path(lr, wd, b2)
    if p.exists():
        return json.loads(p.read_text())

    opt = model = None
    try:
        torch.manual_seed(SEED)
        model = DeepAutoencoder().to(device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr, weight_decay=wd, betas=(0.9, b2),
        )

        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
                torch.optim.lr_scheduler.ConstantLR(
                    opt, factor=1.0, total_iters=MAX_STEPS),
            ], milestones=[WARMUP],
        )

        train_iter = iter(ctx["tlf"]())
        recs = []
        print(f"\n=== adamw_screen lr={lr:g} wd={wd:g} b2={b2:g} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]())
                batch = next(train_iter)

            model.train()
            loss = compute_loss(model, batch, None)
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting")
                break

            model.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            recs.append({"step": step, "loss": lv})

        wall = time.perf_counter() - t0
        final_recon = None
        try:
            final_recon = evaluate_recon(model, ctx["Xte"])
        except Exception as e:
            print(f"  eval failed: {e}")

        out = {
            "benchmark": "ae_mnist_adamw_screen",
            "lr": lr, "wd": wd, "beta2": b2,
            "seed": SEED, "wall_s": wall, "hw": hw,
            "final_recon_bce": final_recon,
            "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = f"recon_bce={final_recon:.2f}  " if final_recon is not None else "DIV  "
        print(f"  -> {msg}wall={wall/60:.1f}m")
    finally:
        try:
            del opt, model
        except Exception:
            pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    print("[setup] loading MNIST...")
    tlf, Xte = get_mnist(device)
    ctx = {"tlf": tlf, "Xte": Xte}

    cells = [(lr, wd, b2) for lr in LRS for wd in WDS for b2 in BETA2S]
    total = len(cells)
    print(f"\n{total} AdamW cells")

    for i, (lr, wd, b2) in enumerate(cells, 1):
        print(f"\n[{i}/{total}]")
        run_one(lr, wd, b2, ctx, device, hw)

    # Summary by beta2
    for b2 in BETA2S:
        print(f"\n=== AdamW screen, beta2 = {b2:g} (final test BCE @ 2000 steps) ===")
        header = f"  {'lr':>7}  " + "  ".join(f"wd={wd:g}".rjust(10) for wd in WDS)
        print(header)
        for lr in LRS:
            row = []
            for wd in WDS:
                p = out_path(lr, wd, b2)
                v = None
                if p.exists():
                    d = json.loads(p.read_text())
                    v = d.get("final_recon_bce")
                row.append(
                    f"{v:>10.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>10}"
                )
            print(f"  {lr:>7g}  " + "  ".join(row))

    # Global winner
    best = None
    for lr in LRS:
        for wd in WDS:
            for b2 in BETA2S:
                p = out_path(lr, wd, b2)
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                v = d.get("final_recon_bce")
                if v is None or not math.isfinite(v):
                    continue
                if best is None or v < best[3]:
                    best = (lr, wd, b2, v)
    if best:
        print(f"\nWinning AdamW config @ 2000 steps:  "
              f"lr={best[0]:g}  wd={best[1]:g}  b2={best[2]:g}  →  BCE = {best[3]:.2f}")
    print("\nReference points:")
    print("  AdamW main sweep (untuned, 10k steps): 136.28 BCE")
    print("  Classic+IFKFAC joint winner (2k steps):  35.1 BCE  (lr=1e-3 dmp=3e-2)")
    print("  M&G 2015 (gold):                        ~58 BCE")


if __name__ == "__main__":
    main()
