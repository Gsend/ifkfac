"""
benchmark/autoencoder_mnist_adamw_bf16_screen.py

bf16-specific AdamW lr screen on the MNIST autoencoder.  The fp32 winner
(lr=1e-2, wd=0.01) overflows bf16's numerical range within ~300 steps,
producing partially-trained junk numbers in the main sweep.  This screen
finds the largest lr that survives 2000 steps at bf16.

Grid (bf16, seed 42, 2000 steps):
    lr  ∈ {3e-4, 1e-3, 3e-3, 1e-2}
    wd  = 0.01 (fixed — fp32 screen showed wd barely matters)
    b2  = 0.999 (fixed)
= 4 cells × ~1.5 min ≈ 6 min.

Output: benchmark/results/ae_mnist_adamw_bf16_screen_lr{lr}.json
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
WD        = 0.01
B2        = 0.999

LRS = [3e-4, 1e-3, 3e-3, 1e-2]


def out_path(lr):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"ae_mnist_adamw_bf16_screen_lr{lr:g}.json"


def run_one(lr, ctx, device, hw):
    p = out_path(lr)
    if p.exists():
        return json.loads(p.read_text())

    opt = model = None
    aborted_step = None
    try:
        torch.manual_seed(SEED)
        model = DeepAutoencoder().to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                 lr=lr, weight_decay=WD, betas=(0.9, B2))

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
        print(f"\n=== adamw_bf16_screen lr={lr:g} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]())
                batch = next(train_iter)

            model.train()
            # bf16 autocast — same as main sweep's AdamW bf16 path
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compute_loss(model, batch, None)
            lv = float(loss.item())
            if not math.isfinite(lv):
                aborted_step = step
                print(f"  step={step} NaN/Inf, aborting")
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
            "benchmark": "ae_mnist_adamw_bf16_screen",
            "lr": lr, "wd": WD, "beta2": B2,
            "seed": SEED, "wall_s": wall, "hw": hw,
            "final_recon_bce": final_recon,
            "aborted_step": aborted_step,
            "completed_steps": len(recs),
            "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        steps_msg = f" (stopped at step {aborted_step})" if aborted_step else ""
        msg = (f"recon_bce={final_recon:.2f}{steps_msg}"
               if final_recon is not None else f"DIV{steps_msg}")
        print(f"  -> {msg}  wall={wall/60:.1f}m")
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

    for i, lr in enumerate(LRS, 1):
        print(f"\n[{i}/{len(LRS)}]")
        run_one(lr, ctx, device, hw)

    print(f"\n=== AdamW bf16 lr screen (wd={WD:g}, b2={B2:g}, "
          f"{MAX_STEPS} steps, seed {SEED}) ===")
    print(f"  {'lr':>7}  {'BCE':>9}  {'completed':>9}  {'wall_min':>9}")
    for lr in LRS:
        p = out_path(lr)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        v = d.get("final_recon_bce")
        n = d.get("completed_steps", 0)
        w = d.get("wall_s", 0) / 60.0
        bce_str = f"{v:>9.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>9}"
        print(f"  {lr:>7g}  {bce_str}  {n:>9d}  {w:>9.2f}")

    print("\nReference (from main sweep):")
    print("  AdamW fp32 lr=1e-2 :  86.0 BCE  (10k steps, mean 3 seeds)")
    print("  AdamW bf16 lr=1e-2 : ~175  BCE  (aborted at step 264-331 — overflow)")
    print("  IFKFAC  bf16        :  30.8 BCE  (stable, target to compare)")


if __name__ == "__main__":
    main()
