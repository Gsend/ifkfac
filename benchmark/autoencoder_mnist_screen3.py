"""
benchmark/autoencoder_mnist_screen3.py

Mini-screen to close two gaps from round-2:

1. Classic K-FAC was never tested at lr=1e-3 (round-1 grid started at 3e-3).
   IFKFAC's optimum is at lr=1e-3 → 35.2 BCE.  Test Classic in the same
   neighborhood to determine whether the optimum is shared (expected, since
   Classic and IFKFAC share Kronecker geometry) or IFKFAC-specific.

2. IFKFAC's round-2 winning column (lr=1e-3) showed monotonic improvement with
   higher damping (81.9 → 41.5 → 35.2 at damping 1e-3 → 1e-2 → 3e-2).  Push
   one more cell at lr=1e-3 damping=1e-1 to check if we've hit the bottom.
   Also push lr down to 3e-4 at damping=3e-2 to check the lr direction.

Cells (5):
  classic  lr=1e-3 damping=1e-2
  classic  lr=1e-3 damping=3e-2
  classic  lr=1e-3 damping=1e-1
  ifkfac    lr=1e-3 damping=1e-1
  ifkfac    lr=3e-4 damping=3e-2

Budget: ~12 minutes.
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
    KFAC_MAX_DIM, KFAC_MOMENTUM, KFAC_GAMMA, KFAC_FREQ, GRAD_CLIP,
)

SEED      = 42
MAX_STEPS = 2000
WARMUP    = 100


CELLS = [
    ("classic", {"lr": 1e-3, "damping": 1e-2}),
    ("classic", {"lr": 1e-3, "damping": 3e-2}),
    ("classic", {"lr": 1e-3, "damping": 1e-1}),
    ("ifkfac",   {"lr": 1e-3, "damping": 1e-1}),
    ("ifkfac",   {"lr": 3e-4, "damping": 3e-2}),
]


def out_path(method, **hp):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    key = "_".join(f"{k}{v:g}" for k, v in sorted(hp.items()))
    return OUT / f"ae_mnist_screen3_{method}_{key}.json"


def build_classic(model, lr, damping):
    from optimizer.classic_kfac import ClassicKFAC
    return ClassicKFAC(
        model, lr=lr, damping=damping,
        factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
        weight_decay=0.0, momentum=KFAC_MOMENTUM,
        grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
        max_gram_dim=KFAC_MAX_DIM,
    )


def build_ifkfac(model, lr, damping):
    from optimizer.ifkfac_kfac import IFKFAC
    return IFKFAC(
        model, lr=lr, damping=damping,
        factor_update_freq=KFAC_FREQ, weight_decay=0.0,
        momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
        gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
        deferred_qr=True,
    )


def run_one(method, hp, ctx, device, hw):
    p = out_path(method, **hp)
    if p.exists():
        return json.loads(p.read_text())

    opt = model = None
    try:
        torch.manual_seed(SEED)
        model = DeepAutoencoder().to(device)
        if method == "classic":
            opt = build_classic(model, hp["lr"], hp["damping"])
        elif method == "ifkfac":
            opt = build_ifkfac(model, hp["lr"], hp["damping"])

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
        hp_str = " ".join(f"{k}={v:g}" for k, v in hp.items())
        print(f"\n=== screen3/{method} {hp_str} ===")
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
            "benchmark": "ae_mnist_screen3",
            "method": method, "hp": hp, "seed": SEED,
            "wall_s": wall, "hw": hw,
            "final_recon_bce": final_recon,
            "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = f"recon_bce={final_recon:.2f}  " if final_recon is not None else "DIV  "
        print(f"  -> {msg}wall={wall/60:.1f}m")
    finally:
        try:
            if opt is not None and hasattr(opt, "cleanup"):
                opt.cleanup()
        except Exception:
            pass
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

    for i, (method, hp) in enumerate(CELLS, 1):
        print(f"\n[{i}/{len(CELLS)}]")
        run_one(method, hp, ctx, device, hw)

    print("\n=== Mini-screen results (final test BCE @ 2000 steps, seed 42) ===")
    print(f"  {'method':>8}  {'lr':>7}  {'damping':>8}  {'BCE':>8}")
    for method, hp in CELLS:
        p = out_path(method, **hp)
        v = None
        if p.exists():
            d = json.loads(p.read_text())
            v = d.get("final_recon_bce")
        cell = f"{v:>8.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>8}"
        print(f"  {method:>8}  {hp['lr']:>7g}  {hp['damping']:>8g}  {cell}")

    print("\nReference points:")
    print("  AdamW main sweep : 136.28 BCE   (10k steps, mean 3 seeds)")
    print("  Classic screen1  :  91.1  BCE   (lr=3e-3 dmp=1e-2)")
    print("  IFKFAC screen2    :  35.2  BCE   (lr=1e-3 dmp=3e-2)  ← previous best")
    print("  M&G 2015 (gold)  :  ~58  BCE")


if __name__ == "__main__":
    main()
