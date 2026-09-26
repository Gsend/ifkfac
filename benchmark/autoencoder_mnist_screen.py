"""
benchmark/autoencoder_mnist_screen.py

Hyperparameter screen for Classic K-FAC and SINGD on the Hinton-Salakhutdinov
MNIST autoencoder.  The first 30-cell sweep showed Classic and SINGD both
underperforming our reference numbers from Martens & Grosse 2015; this script
finds where they actually train.

Screen design (fp32, single seed, 2000 steps — 5x faster than the main sweep):
    Classic:  lr ∈ {3e-3, 1e-2, 3e-2, 1e-1}
              damping ∈ {1e-4, 1e-3, 1e-2}
              mom=0.9 fixed, freq=10 fixed   = 12 cells
    SINGD:    lr ∈ {3e-3, 1e-2, 3e-2}
              damping ∈ {1e-3, 1e-2}
              lr_cov ∈ {1e-2, 1e-1}
              alpha1=0.5 fixed               = 12 cells

Output: benchmark/results/ae_mnist_screen_{method}_lr{lr}_dmp{dmp}[_lrcov{lrcov}].json
Plus a printed summary.

Use the winner of each method's screen to update autoencoder_mnist.py defaults.
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
import torch.nn.functional as F


# ---- Imports from the main AE benchmark ------------------------------------

from benchmark.autoencoder_mnist import (
    DeepAutoencoder, get_mnist, compute_loss, evaluate_recon,
    KFAC_MAX_DIM, KFAC_MOMENTUM, KFAC_GAMMA, KFAC_FREQ, GRAD_CLIP,
    BATCH_SIZE,
)


# ---- Screen config ----------------------------------------------------------

SEED       = 42
MAX_STEPS  = 2000           # 5x faster than the main 10k sweep
WARMUP     = 100

CLASSIC_GRID = {
    "lr":      [3e-3, 1e-2, 3e-2, 1e-1],
    "damping": [1e-4, 1e-3, 1e-2],
}

SINGD_GRID = {
    "lr":      [3e-3, 1e-2, 3e-2],
    "damping": [1e-3, 1e-2],
    "lr_cov":  [1e-2, 1e-1],
}


def out_path(method, **hp):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    key = "_".join(f"{k}{v:g}" for k, v in sorted(hp.items()))
    return OUT / f"ae_mnist_screen_{method}_{key}.json"


# ---- Optimizer builders -----------------------------------------------------

def build_classic(model, lr, damping):
    from optimizer.classic_kfac import ClassicKFAC
    return ClassicKFAC(
        model, lr=lr, damping=damping,
        factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
        weight_decay=0.0, momentum=KFAC_MOMENTUM,
        grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
        max_gram_dim=KFAC_MAX_DIM,
    )


def build_singd(model, lr, damping, lr_cov):
    from singd.optim.optimizer import SINGD
    kfac_params = [
        p for mod in model.modules() if isinstance(mod, nn.Linear)
        for p in mod.parameters()
    ]
    return SINGD(
        model, params=kfac_params,
        lr=lr, damping=damping, momentum=KFAC_MOMENTUM,
        T=KFAC_FREQ, structures=("dense", "dense"),
        loss_average="batch",
        lr_cov=lr_cov, alpha1=0.5,
        warn_unsupported=False,
    )


# ---- Single-cell executor ---------------------------------------------------

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
        elif method == "singd":
            opt = build_singd(model, hp["lr"], hp["damping"], hp["lr_cov"])
        else:
            raise ValueError(method)

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
        print(f"\n=== screen/{method} {hp_str} ===")
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
            "benchmark": "ae_mnist_screen",
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

    return out


# ---- Main -------------------------------------------------------------------

def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    print("[setup] loading MNIST...")
    tlf, Xte = get_mnist(device)
    ctx = {"tlf": tlf, "Xte": Xte}

    classic_cells = [
        {"lr": lr, "damping": dmp}
        for lr in CLASSIC_GRID["lr"]
        for dmp in CLASSIC_GRID["damping"]
    ]
    singd_cells = [
        {"lr": lr, "damping": dmp, "lr_cov": lrc}
        for lr in SINGD_GRID["lr"]
        for dmp in SINGD_GRID["damping"]
        for lrc in SINGD_GRID["lr_cov"]
    ]
    cells = [("classic", hp) for hp in classic_cells] \
          + [("singd",   hp) for hp in singd_cells]

    total = len(cells)
    print(f"\n{total} cells total "
          f"(classic: {len(classic_cells)}, singd: {len(singd_cells)})")
    for i, (method, hp) in enumerate(cells, 1):
        print(f"\n[{i}/{total}]")
        run_one(method, hp, ctx, device, hw)

    # Summary tables
    print("\n=== Classic K-FAC screen (final test BCE @ 2000 steps, seed 42) ===")
    print(f"  {'lr':>7}  {'dmp=1e-4':>10}  {'dmp=1e-3':>10}  {'dmp=1e-2':>10}")
    for lr in CLASSIC_GRID["lr"]:
        cells_row = []
        for dmp in CLASSIC_GRID["damping"]:
            p = out_path("classic", lr=lr, damping=dmp)
            v = None
            if p.exists():
                d = json.loads(p.read_text())
                v = d.get("final_recon_bce")
            cells_row.append(
                f"{v:>10.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>10}"
            )
        print(f"  {lr:>7g}  " + "  ".join(cells_row))

    print("\n=== SINGD screen (final test BCE @ 2000 steps, seed 42) ===")
    for lrc in SINGD_GRID["lr_cov"]:
        print(f"  lr_cov = {lrc:g}")
        print(f"    {'lr':>7}  {'dmp=1e-3':>10}  {'dmp=1e-2':>10}")
        for lr in SINGD_GRID["lr"]:
            cells_row = []
            for dmp in SINGD_GRID["damping"]:
                p = out_path("singd", lr=lr, damping=dmp, lr_cov=lrc)
                v = None
                if p.exists():
                    d = json.loads(p.read_text())
                    v = d.get("final_recon_bce")
                cells_row.append(
                    f"{v:>10.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>10}"
                )
            print(f"    {lr:>7g}  " + "  ".join(cells_row))

    print("\nReference values:")
    print("  AdamW 30-cell sweep result (fp32, seed mean):  136.28 BCE")
    print("  IFKFAC 30-cell sweep result:              252.62 BCE")
    print("  Martens & Grosse 2015 (M&G adaptive damping):  ~58 BCE  (gold)")


if __name__ == "__main__":
    main()
