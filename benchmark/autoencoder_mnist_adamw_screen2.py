"""
benchmark/autoencoder_mnist_adamw_screen2.py

Push the AdamW lr direction one more time.  Screen 1 found lr=3e-3 to be
the best of {1e-4, 3e-4, 1e-3, 3e-3}, with monotonic improvement up the
ladder.  This script tests lr ∈ {1e-2, 3e-2} to verify we're past AdamW's
optimum and not artificially capping it.

Grid (fp32, seed 42, 2000 steps):
    lr  ∈ {1e-2, 3e-2}
    wd  ∈ {0, 0.01, 0.1}
    b2  = 0.999 (default — screen 1 showed b2 doesn't matter)
= 6 cells × ~1.5 min ≈ 9 min.
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

from benchmark.autoencoder_mnist import (
    DeepAutoencoder, get_mnist, compute_loss, evaluate_recon,
)


SEED      = 42
MAX_STEPS = 2000
WARMUP    = 100

LRS = [1e-2, 3e-2]
WDS = [0.0, 0.01, 0.1]
B2  = 0.999


def out_path(lr, wd):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"ae_mnist_adamw_screen2_lr{lr:g}_wd{wd:g}.json"


def run_one(lr, wd, ctx, device, hw):
    p = out_path(lr, wd)
    if p.exists():
        return json.loads(p.read_text())

    opt = model = None
    try:
        torch.manual_seed(SEED)
        model = DeepAutoencoder().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                 betas=(0.9, B2))

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
        print(f"\n=== adamw_screen2 lr={lr:g} wd={wd:g} ===")
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
            "benchmark": "ae_mnist_adamw_screen2",
            "lr": lr, "wd": wd, "beta2": B2, "seed": SEED,
            "wall_s": wall, "hw": hw,
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

    cells = [(lr, wd) for lr in LRS for wd in WDS]
    for i, (lr, wd) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}]")
        run_one(lr, wd, ctx, device, hw)

    print(f"\n=== AdamW screen2, beta2 = {B2:g} (final test BCE @ 2000 steps) ===")
    print(f"  {'lr':>7}  " + "  ".join(f"wd={wd:g}".rjust(10) for wd in WDS))
    for lr in LRS:
        row = []
        for wd in WDS:
            p = out_path(lr, wd)
            v = None
            if p.exists():
                d = json.loads(p.read_text())
                v = d.get("final_recon_bce")
            row.append(
                f"{v:>10.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>10}"
            )
        print(f"  {lr:>7g}  " + "  ".join(row))

    print("\nReference points:")
    print("  Screen 1 best:                 lr=3e-3 wd=0.1 b2=0.999 → 150.32 BCE")
    print("  Tuned K-FAC (joint Cl+Vd):     lr=1e-3 dmp=3e-2        →  35.1 BCE")
    print("  M&G 2015 gold:                                          ~58 BCE")


if __name__ == "__main__":
    main()
