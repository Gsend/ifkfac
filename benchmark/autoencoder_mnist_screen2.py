"""
benchmark/autoencoder_mnist_screen2.py

Second-round hyperparameter screen for the MNIST autoencoder.  Round 1
(autoencoder_mnist_screen.py) found Classic K-FAC's winner at lr=3e-3,
damping=1e-2 → 91 BCE.  This script does three follow-ups:

1. IFKFAC screen around Classic's winning region.
2. IFKFAC + WGSO screen — Round 1's main sweep had WGSO diverging at 3491 BCE
   with damping=1e-4.  WGSO row-equilibrates so we expect a slightly smaller
   damping than plain IFKFAC, but 1e-4 was clearly too small.  Sweep
   (lr × damping) to find a stable point.
3. SINGD richer screen — Round 1 found SINGD stuck at the per-pixel marginal
   floor (206 BCE = predicting the marginal of binarized MNIST).  This is
   consistent with SINGD's preconditioner not actually engaging on the
   sigmoid AE.  Try alpha1 (Riemannian momentum), T (refresh frequency),
   and lr_cov to find a config where it actually preconditions.

Output: benchmark/results/ae_mnist_screen2_{method}_{key}.json
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
    KFAC_MAX_DIM, KFAC_MOMENTUM, KFAC_GAMMA, KFAC_FREQ, GRAD_CLIP,
)


# ---- Screen config ----------------------------------------------------------

SEED       = 42
MAX_STEPS  = 2000
WARMUP     = 100

# IFKFAC: explore around Classic's winning region
IFKFAC_GRID = {
    "lr":      [1e-3, 3e-3, 1e-2],
    "damping": [1e-3, 1e-2, 3e-2],
}                                  # 9 cells

# WGSO: explore where the row-equilibration tolerates ridge
WGSO_GRID = {
    "lr":      [1e-3, 3e-3, 1e-2],
    "damping": [1e-4, 1e-3, 1e-2],
}                                  # 9 cells

# SINGD: fix lr=3e-3 damping=1e-2 (Classic's winner) and tune SINGD-specific knobs
SINGD_GRID = {
    "alpha1":  [0.1, 0.5, 0.9],    # Riemannian momentum on the inverse Cholesky factor
    "lr_cov":  [1e-3, 1e-2],       # preconditioner-update stepsize
    "T":       [1, 10],            # factor refresh frequency
}                                  # 12 cells
SINGD_LR_FIXED      = 3e-3
SINGD_DAMPING_FIXED = 1e-2


def out_path(method, **hp):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    key = "_".join(f"{k}{v:g}" for k, v in sorted(hp.items()))
    return OUT / f"ae_mnist_screen2_{method}_{key}.json"


# ---- Optimizer builders -----------------------------------------------------

def build_ifkfac(model, lr, damping):
    from optimizer.ifkfac_kfac import IFKFAC
    return IFKFAC(
        model, lr=lr, damping=damping,
        factor_update_freq=KFAC_FREQ, weight_decay=0.0,
        momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
        gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
        deferred_qr=True,
    )


def build_wgso(model, lr, damping):
    from optimizer.ifkfac_kfac import IFKFAC
    from benchmark.kfac_bf16_compare import _wgso_weight_rows
    kfac = IFKFAC(
        model, lr=lr, damping=damping,
        factor_update_freq=KFAC_FREQ, weight_decay=0.0,
        momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
        gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
        deferred_qr=True,
    )
    kfac.hooks.chunk_transform_X = _wgso_weight_rows
    kfac.hooks.chunk_transform_G = _wgso_weight_rows
    return kfac


def build_singd(model, alpha1, lr_cov, T):
    from singd.optim.optimizer import SINGD
    kfac_params = [
        p for mod in model.modules() if isinstance(mod, nn.Linear)
        for p in mod.parameters()
    ]
    return SINGD(
        model, params=kfac_params,
        lr=SINGD_LR_FIXED, damping=SINGD_DAMPING_FIXED, momentum=KFAC_MOMENTUM,
        T=T, structures=("dense", "dense"),
        loss_average="batch",
        lr_cov=lr_cov, alpha1=alpha1,
        warn_unsupported=False,
    )


def build_opt(method, model, hp):
    if method == "ifkfac":
        return build_ifkfac(model, hp["lr"], hp["damping"])
    if method == "wgso":
        return build_wgso(model, hp["lr"], hp["damping"])
    if method == "singd":
        return build_singd(model, hp["alpha1"], hp["lr_cov"], hp["T"])
    raise ValueError(method)


# ---- Single-cell executor ---------------------------------------------------

def run_one(method, hp, ctx, device, hw):
    p = out_path(method, **hp)
    if p.exists():
        return json.loads(p.read_text())

    opt = model = None
    try:
        torch.manual_seed(SEED)
        model = DeepAutoencoder().to(device)
        opt = build_opt(method, model, hp)

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
        print(f"\n=== screen2/{method} {hp_str} ===")
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
            "benchmark": "ae_mnist_screen2",
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


# ---- Summary printers -------------------------------------------------------

def print_2d_table(method, x_name, x_values, y_name, y_values, **fixed):
    print(f"\n=== {method} screen (final test BCE @ {MAX_STEPS} steps, seed {SEED}) ===")
    if fixed:
        print("  fixed: " + ", ".join(f"{k}={v:g}" for k, v in fixed.items()))
    header = f"  {x_name:>7}  " + "  ".join(f"{y_name}={v:g}".rjust(12) for v in y_values)
    print(header)
    for x in x_values:
        cells_row = []
        for y in y_values:
            hp = {x_name: x, y_name: y, **fixed}
            p = out_path(method, **hp)
            v = None
            if p.exists():
                d = json.loads(p.read_text())
                v = d.get("final_recon_bce")
            cells_row.append(
                f"{v:>12.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>12}"
            )
        print(f"  {x:>7g}  " + "  ".join(cells_row))


def print_singd_table():
    print(f"\n=== SINGD screen2 (lr={SINGD_LR_FIXED:g}, damping={SINGD_DAMPING_FIXED:g}, "
          f"{MAX_STEPS} steps, seed {SEED}) ===")
    for T in SINGD_GRID["T"]:
        print(f"  T = {T}")
        header = f"    {'alpha1':>7}  " + "  ".join(
            f"lr_cov={lrc:g}".rjust(14) for lrc in SINGD_GRID["lr_cov"]
        )
        print(header)
        for a in SINGD_GRID["alpha1"]:
            cells_row = []
            for lrc in SINGD_GRID["lr_cov"]:
                p = out_path("singd", alpha1=a, lr_cov=lrc, T=T)
                v = None
                if p.exists():
                    d = json.loads(p.read_text())
                    v = d.get("final_recon_bce")
                cells_row.append(
                    f"{v:>14.1f}" if (v is not None and math.isfinite(v)) else f"{'DIV':>14}"
                )
            print(f"    {a:>7g}  " + "  ".join(cells_row))


# ---- Main -------------------------------------------------------------------

def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    print("[setup] loading MNIST...")
    tlf, Xte = get_mnist(device)
    ctx = {"tlf": tlf, "Xte": Xte}

    cells = []
    for lr in IFKFAC_GRID["lr"]:
        for dmp in IFKFAC_GRID["damping"]:
            cells.append(("ifkfac", {"lr": lr, "damping": dmp}))
    for lr in WGSO_GRID["lr"]:
        for dmp in WGSO_GRID["damping"]:
            cells.append(("wgso", {"lr": lr, "damping": dmp}))
    for a in SINGD_GRID["alpha1"]:
        for lrc in SINGD_GRID["lr_cov"]:
            for T in SINGD_GRID["T"]:
                cells.append(("singd", {"alpha1": a, "lr_cov": lrc, "T": T}))

    total = len(cells)
    n_ifkfac = len(IFKFAC_GRID["lr"]) * len(IFKFAC_GRID["damping"])
    n_wgso  = len(WGSO_GRID["lr"])  * len(WGSO_GRID["damping"])
    n_singd = len(SINGD_GRID["alpha1"]) * len(SINGD_GRID["lr_cov"]) * len(SINGD_GRID["T"])
    print(f"\n{total} cells (ifkfac {n_ifkfac}, wgso {n_wgso}, singd {n_singd})")

    for i, (method, hp) in enumerate(cells, 1):
        print(f"\n[{i}/{total}]")
        run_one(method, hp, ctx, device, hw)

    print_2d_table("ifkfac", "lr", IFKFAC_GRID["lr"],
                   "damping", IFKFAC_GRID["damping"])
    print_2d_table("wgso", "lr", WGSO_GRID["lr"],
                   "damping", WGSO_GRID["damping"])
    print_singd_table()

    print("\nReference (from 30-cell main sweep + round-1 screen):")
    print("  AdamW   : 136.28 BCE   (10000 steps, mean across 3 seeds)")
    print("  Classic : 91.1 BCE     (2000 steps, screen winner: lr=3e-3 dmp=1e-2)")
    print("  IFKFAC   : 252.62 BCE   (10000 steps, mean — under-tuned)")
    print("  M&G '15 : ~58 BCE      (gold reference)")


if __name__ == "__main__":
    main()
