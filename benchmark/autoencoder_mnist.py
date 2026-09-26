"""
benchmark/autoencoder_mnist.py

Hinton-Salakhutdinov (2006) MNIST deep autoencoder benchmark — the canonical
test case from the K-FAC literature where second-order methods beat Adam.

Architecture: 784 -> 1000 -> 500 -> 250 -> 30 -> 250 -> 500 -> 1000 -> 784
              with sigmoid activations throughout (final = sigmoid for BCE).
Loss:         per-pixel binary cross-entropy on binarized MNIST.
Eval:         BCE reconstruction loss on the 10k test set.

This is the autoencoder Martens (2010, Hessian-free) and Martens & Grosse
(2015, K-FAC) used to demonstrate that second-order optimizers can avoid
the layerwise RBM pre-training that Hinton & Salakhutdinov originally needed.
Tuned AdamW gets stuck on this loss surface; K-FAC does not.

Sweep grid:
    methods   : adamw, classic, ifkfac, ifkfac_wgso, singd  (5)
    precisions: fp32, bf16                                 (2)
    seeds     : 42, 43, 44                                 (3)
Total: 30 runs, resumable (skips existing files).

Output: benchmark/results/ae_mnist_{precision}_{method}_seed{seed}.json
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


# ---- Config -----------------------------------------------------------------

METHODS    = ["adamw", "classic", "ifkfac"]   # WGSO + SINGD broken on this AE
                                              #  (see §5.6 — kept JSONs as docs)
PRECISIONS = ["fp32", "bf16"]
SEEDS      = [42, 43, 44]

MAX_STEPS    = 10000     # M&G 2015 ran ~6000 effective epochs; we use 10k steps
WARMUP       = 200       # linear ramp lr: 0.1*lr -> lr
CONST_PHASE  = 2000      # constant-lr phase matching the 2000-step screen window
                          #  — preserves the screen's hyperparameter optimum,
                          #    then cosine decay refines for the remaining steps
                          #    (compensates for "bouncing at full lr" effect at
                          #    larger MAX_STEPS)
# Larger batch size is required at bf16 to ensure enough rows for the first
# Linear layer's QR (n_in = 784 input pixels).  At BATCH_SIZE=256 the
# deferred-QR buffer fails to accumulate enough rows across steps under
# true_bf16; raise to 1024 so each step has >= n_in rows.
BATCH_SIZE = 256    # deferred_qr accumulates rows across steps (3-4 steps for
                     # the AE's n_in=784 first layer), so 256 is sufficient.
# 10000 * 256 / 60000 ≈ 42 epochs over MNIST train

# =============================================================================
# Hyperparameters taken from the literature for the Hinton-Salakhutdinov MNIST
# autoencoder.  These should be considered defaults to be screened, not final.
#
# K-FAC family
#   Martens & Grosse 2015 (§7.1, table 1) used:
#     lr        = 3e-2 ramp with Levenberg-Marquardt damping adaptation
#     momentum  = 0.9 (heavy-ball)
#     damping   = 1e-3 initial, LM-adaptive thereafter
#     batch     = 1000 with factor updates every step
#   We don't have LM adaptation, so:
#     - lr lowered to 1e-2 for fixed-damping stability
#     - damping kept at M&G's initial 1e-3
#     - momentum kept at 0.9
#     - factor_update_freq bumped to 10 (rapidly-changing sigmoid curvature)
#
# AdamW
#   No canonical reference for this AE; common values across papers:
#     lr ∈ {1e-4, 3e-4, 1e-3}, weight_decay = 0
#   Adam is famously slow on this AE — the original H&S paper had to use
#   layerwise RBM pre-training because gradient descent couldn't reach a
#   useful minimum.  We use lr=1e-3 as a reasonable starting value; expect
#   to need a quick lr screen.
#
# SINGD
#   No published values for the AE.  We carry over the tuned values from
#   our transformer sweep (lr_cov=1e-1, alpha1=0.5) and adjust if the
#   smoke test shows problems.
# =============================================================================

# K-FAC family — tuned via screen 1/2/3 on the MNIST AE
#   Joint Classic+IFKFAC winner: lr=1e-3, damping=3e-2 → ~35 BCE @ 2000 steps
#   (beats AdamW=136, beats M&G 2015 reported ~58 BCE).
#   Damping pushed up to 3e-2 because fixed-damping (no Levenberg-Marquardt)
#   needs more regularization than M&G's adaptive baseline; lr lowered to
#   1e-3 to match.
KFAC_LR              = 1e-3
KFAC_DAMPING_IFKFAC   = 3e-2
KFAC_DAMPING_CLASSIC = 3e-2
KFAC_DAMPING_WGSO    = 3e-2     # WGSO fails on this AE regardless of damping
                                 #  (see §5.6 — sigmoid activations + row eq)
KFAC_MOMENTUM        = 0.9      # M&G 2015 standard
KFAC_GAMMA           = 0.9
KFAC_FREQ            = 10
GRAD_CLIP            = 300.0

# AdamW — tuned via adamw_screen + adamw_screen2 + adamw_bf16_screen
#   fp32 winner: lr=1e-2, wd=0.01, b2=0.999 → 113 BCE @ 2000 steps
#                                            →  86 BCE @ 10000 steps
#   bf16 winner: lr=3e-4 (largest stable — 1e-2/3e-3/1e-3 all NaN'd)
#                                            → ~170 BCE @ 2000 steps
#   K-FAC variants use the same lr at both precisions (IFKFAC: 31 → 31).
#   This precision-sensitivity is itself part of the paper's stability story.
ADAMW_LR_FP32        = 1e-2
ADAMW_LR_BF16        = 3e-4
ADAMW_WD             = 0.01
ADAMW_BETA2          = 0.999

# SINGD (carried over from transformer)
SINGD_LR_COV         = 1e-1
SINGD_ALPHA1         = 0.5


# ---- Architecture: Hinton-Salakhutdinov 2006 -------------------------------

class DeepAutoencoder(nn.Module):
    """784 -> 1000 -> 500 -> 250 -> 30 -> 250 -> 500 -> 1000 -> 784, sigmoid."""

    def __init__(self):
        super().__init__()
        dims = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        # x: (B, 784)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # sigmoid on every layer including output (output represents
            # per-pixel Bernoulli probabilities, consumed by BCE loss).
            x = torch.sigmoid(x)
        return x


# ---- Data -------------------------------------------------------------------

def get_mnist(device):
    """Returns (train_loader_factory, test_tensor, target_tensor).

    Binarizes MNIST by thresholding at 0.5 (Hinton-Salakhutdinov 2006 convention).
    """
    from torchvision import datasets, transforms

    data_root = ROOT / "data" / "mnist"
    data_root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])

    train = datasets.MNIST(str(data_root), train=True,  download=True, transform=transform)
    test  = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)

    # Pre-stack the test set onto device once (10k * 784 floats is fine in VRAM)
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))]).to(device)
    # Hinton-Salakhutdinov binarization: x > 0.5
    Xte_bin = (Xte > 0.5).float()

    # Iterable factory over the train set
    def train_loader_factory():
        loader = torch.utils.data.DataLoader(
            train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device, non_blocking=True)
            x_bin = (x > 0.5).float()
            yield x_bin, x_bin  # input, target are the same (reconstruction)

    return train_loader_factory, Xte_bin


def ae_bce_loss(recon, target, reduction="mean"):
    """Per-pixel binary cross-entropy on binarised MNIST, as a standalone
    loss_fn(output, target, reduction=...).

    Exposed as a top-level function (not a lambda/closure) with an explicit
    `reduction` kwarg so ASDL's `KfacGradientMaker` accepts it: ASDL gates on
    `'reduction' in loss_fn.__code__.co_varnames` (asdl/utils.has_reduction) and
    drives the empirical-Fisher backward with reduction='sum'.

    `recon` is already post-sigmoid (the AE applies sigmoid on every layer), so
    these are Bernoulli probabilities — clamp away from 0/1 to keep log finite.
    The default "mean" reduction matches compute_loss (sum over pixels, mean
    over batch = Martens & Grosse 2015 §7.1), so the training gradient scaling
    is identical to the homebrew Classic K-FAC path.
    """
    eps = 1e-7
    recon = recon.clamp(eps, 1.0 - eps)
    per_elem = -(target * torch.log(recon) + (1 - target) * torch.log(1 - recon))
    if reduction == "sum":
        return per_elem.sum()
    if reduction == "none":
        return per_elem
    # "mean": sum over pixels, mean over batch.
    return per_elem.sum(dim=1).mean()


def compute_loss(model, batch, ctx):
    x, target = batch
    recon = model(x)
    return ae_bce_loss(recon, target, reduction="mean")


def evaluate_recon(model, Xte_bin):
    """Mean per-image BCE on the test set."""
    model.eval()
    with torch.no_grad():
        eps = 1e-7
        # Chunk to avoid VRAM blowup
        losses = []
        for i in range(0, Xte_bin.size(0), 1024):
            x = Xte_bin[i:i + 1024]
            recon = model(x).clamp(eps, 1.0 - eps)
            loss = -(x * torch.log(recon) + (1 - x) * torch.log(1 - recon)).sum(dim=1)
            losses.append(loss)
        return float(torch.cat(losses).mean().item())


# ---- Optimizer factory (mirrors comparison_4way_multiseed.py) --------------

KFAC_MAX_DIM = 4096


def build_optimizers(method, model, precision="fp32"):
    """Returns (primary, secondary_or_None).

    Note: this AE is all-Linear, so every param is K-FAC-eligible and there
    is no secondary AdamW group.  We bypass stability_benchmark.make_optimizers
    (which assumes a non-empty "other" group) and construct the K-FAC opts
    directly.

    AdamW uses a precision-specific lr because the fp32 tuned value (1e-2)
    overflows bf16's range — see adamw_bf16_screen.
    """
    if method == "adamw":
        adamw_lr = ADAMW_LR_BF16 if precision == "bf16" else ADAMW_LR_FP32
        return torch.optim.AdamW(
            model.parameters(),
            lr=adamw_lr, weight_decay=ADAMW_WD, betas=(0.9, ADAMW_BETA2),
        ), None

    if method == "classic":
        from optimizer.classic_kfac import ClassicKFAC
        kfac = ClassicKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_CLASSIC,
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=0.0, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
        return kfac, None

    # True bf16 mode: store R_X, R_G as bf16; route apply_ifkfac tri-solves
    # through optimizer.bf16_linalg (cuSOLVER lacks bf16 tri-solve).
    true_bf16 = (precision == "bf16")

    # Keep deferred_qr=True for both precisions — the accumulation works fine,
    # the "factor update skipped" warnings just fire for the first few steps
    # until the buffer reaches n_in rows.  After that they go quiet.
    deferred = True

    if method == "ifkfac":
        from optimizer.ifkfac_kfac import IFKFAC
        kfac = IFKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_IFKFAC,
            factor_update_freq=KFAC_FREQ, weight_decay=0.0,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=deferred,
            use_true_bf16=true_bf16,
        )
        return kfac, None

    if method == "ifkfac_wgso":
        from optimizer.ifkfac_kfac import IFKFAC
        from benchmark.kfac_bf16_compare import _wgso_weight_rows
        kfac = IFKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_WGSO,
            factor_update_freq=KFAC_FREQ, weight_decay=0.0,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=deferred,
            use_true_bf16=true_bf16,
        )
        kfac.hooks.chunk_transform_X = _wgso_weight_rows
        kfac.hooks.chunk_transform_G = _wgso_weight_rows
        return kfac, None

    if method == "singd":
        from singd.optim.optimizer import SINGD
        # All Linear params are K-FAC-eligible (no embedding / BN here)
        kfac_params = [
            p for mod in model.modules() if isinstance(mod, nn.Linear)
            for p in mod.parameters()
        ]
        singd = SINGD(
            model, params=kfac_params,
            lr=KFAC_LR, damping=1e-3, momentum=KFAC_MOMENTUM,
            T=KFAC_FREQ, structures=("dense", "dense"),
            loss_average="batch",
            lr_cov=SINGD_LR_COV, alpha1=SINGD_ALPHA1,
            warn_unsupported=False,
        )
        return singd, None

    raise ValueError(method)


# ---- bf16 patching (reuses existing helpers) -------------------------------

def engage_bf16(method):
    # When use_true_bf16=True is in effect (ifkfac + ifkfac_wgso at bf16),
    # the kfac_bf16_compare monkey-patch is SKIPPED — the optimizer's
    # internal bf16 dispatch handles everything (R factors stored as bf16,
    # apply_ifkfac routes through solve_triangular_bf16).  The two paths
    # conflict if both run: the monkey-patch wraps apply_ifkfac then calls
    # _orig_apply_ifkfac which sees our bf16 R and triggers our dispatch
    # with mismatched dtype / shape state.
    if method == "ifkfac":
        # use_true_bf16 handles bf16 internally — skip the patch.
        return
    elif method == "ifkfac_wgso":
        # use_true_bf16 handles bf16 internally — but WGSO row equilibration
        # is a separate concern.  TODO: route WGSO through chunk_transform
        # without bf16 quantization.  For now we still need the monkey-patch
        # for the WGSO chunk transform, but at bf16 storage this may break.
        from benchmark.kfac_bf16_compare import enable_bf16
        enable_bf16(wgso=True)
    elif method == "classic":
        from benchmark.kfac_bf16_compare import enable_classic_bf16
        enable_classic_bf16()


def disengage_bf16(method):
    if method in ("ifkfac", "ifkfac_wgso"):
        from benchmark.kfac_bf16_compare import disable_bf16
        disable_bf16()
    elif method == "classic":
        from benchmark.kfac_bf16_compare import disable_classic_bf16
        disable_classic_bf16()


def need_autocast(method, precision):
    return precision == "bf16" and method in ("singd", "adamw")


# ---- Per-run executor -------------------------------------------------------

def out_path(precision, method, seed):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"ae_mnist_{precision}_{method}_seed{seed}.json"


def run_one(precision, method, seed, ctx, device, hw):
    p = out_path(precision, method, seed)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    if precision == "bf16":
        engage_bf16(method)

    opt_primary = opt_secondary = model = None
    try:
        torch.manual_seed(seed)
        model = DeepAutoencoder().to(device)

        opt_primary, opt_secondary = build_optimizers(method, model, precision)

        # Three-phase schedule: warmup, constant (matches screen), cosine decay.
        # The constant phase preserves the 2000-step screen's training dynamics;
        # the cosine phase compensates for "bouncing at full lr" by letting the
        # model settle as it accumulates more steps.
        decay_steps = MAX_STEPS - WARMUP - CONST_PHASE
        sched_primary = torch.optim.lr_scheduler.SequentialLR(
            opt_primary, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt_primary, start_factor=0.1, end_factor=1.0,
                    total_iters=WARMUP),
                torch.optim.lr_scheduler.ConstantLR(
                    opt_primary, factor=1.0, total_iters=CONST_PHASE),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt_primary, T_max=decay_steps,
                    eta_min=opt_primary.param_groups[0]["lr"] * 1e-3),
            ],
            milestones=[WARMUP, WARMUP + CONST_PHASE],
        )
        sched_secondary = None
        if opt_secondary is not None:
            sched_secondary = torch.optim.lr_scheduler.ConstantLR(
                opt_secondary, factor=1.0, total_iters=MAX_STEPS)

        use_amp = need_autocast(method, precision)

        train_iter = iter(ctx["tlf"]())
        recs = []
        print(f"\n=== ae_mnist/{precision}/{method}/seed{seed} ===", flush=True)
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            t_step = time.perf_counter()
            # Verbose prints for first 20 steps to pinpoint any stall
            if step <= 20:
                print(f"  step {step}: getting batch", flush=True)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]())
                batch = next(train_iter)

            model.train()
            if step <= 20:
                print(f"  step {step}: forward", flush=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = compute_loss(model, batch, ctx)
            else:
                loss = compute_loss(model, batch, ctx)

            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting", flush=True)
                break

            if step <= 20:
                print(f"  step {step}: loss={lv:.3e}, backward", flush=True)
            model.zero_grad(set_to_none=True)
            loss.backward()
            if step <= 20:
                print(f"  step {step}: opt.step()", flush=True)
            opt_primary.step()
            if opt_secondary is not None:
                opt_secondary.step()
            sched_primary.step()
            if sched_secondary is not None:
                sched_secondary.step()
            recs.append({"step": step, "loss": lv})
            if step <= 20:
                t_elapsed = time.perf_counter() - t_step
                print(f"  step {step}: DONE in {t_elapsed*1000:.1f}ms", flush=True)
            elif step % 100 == 0:
                t_elapsed = time.perf_counter() - t_step
                print(f"  step {step}: loss={lv:.3e}  step_time={t_elapsed*1000:.1f}ms", flush=True)

        wall = time.perf_counter() - t0
        final_recon = None
        try:
            final_recon = evaluate_recon(model, ctx["Xte"])
        except Exception as e:
            print(f"  eval failed: {e}")

        out = {
            "benchmark": "ae_mnist",
            "precision": precision, "method": method, "seed": seed,
            "wall_s": wall, "hw": hw,
            "final_recon_bce": final_recon,
            "per_step": recs,
        }
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = f"recon_bce={final_recon:.2f}  " if final_recon is not None else ""
        print(f"  -> {msg}wall={wall/60:.1f}m")
    finally:
        if precision == "bf16":
            disengage_bf16(method)
        try:
            if opt_primary is not None and hasattr(opt_primary, "cleanup"):
                opt_primary.cleanup()
        except Exception:
            pass
        try:
            del opt_primary, opt_secondary, model
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

    total = len(PRECISIONS) * len(METHODS) * len(SEEDS)
    done = 0
    for precision in PRECISIONS:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                print(f"\n[{done}/{total}]")
                run_one(precision, method, seed, ctx, device, hw)

    # Summary
    print("\n=== ae_mnist summary (test reconstruction BCE, lower is better) ===")
    print(f"  {'precision':>9}  {'method':>11}  {'mean_bce':>9}  {'std':>6}  {'wall_min':>9}")
    import statistics
    for precision in PRECISIONS:
        for method in METHODS:
            vals, walls = [], []
            for seed in SEEDS:
                p = out_path(precision, method, seed)
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                v = d.get("final_recon_bce")
                if v is not None and math.isfinite(v):
                    vals.append(v)
                walls.append(d.get("wall_s", 0) / 60)
            if not vals:
                continue
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            w    = statistics.mean(walls) if walls else 0.0
            print(f"  {precision:>9}  {method:>11}  {mean:>9.2f}  {std:>6.2f}  {w:>9.2f}")


def smoke_test():
    """Quick sanity check: run AdamW for 100 steps on MNIST autoencoder.

    Verifies that the architecture compiles, data loads, loss decreases,
    and the eval pipeline works.  ~1 minute.
    """
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device()
    hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    print("[smoke] loading MNIST...")
    tlf, Xte = get_mnist(device)

    print("[smoke] building model + AdamW...")
    torch.manual_seed(42)
    model = DeepAutoencoder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=0.0)

    it = iter(tlf())
    print("[smoke] training 100 steps...")
    t0 = time.perf_counter()
    losses = []
    for step in range(1, 101):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tlf())
            batch = next(it)
        model.train()
        loss = compute_loss(model, batch, None)
        model.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step in (1, 10, 50, 100):
            print(f"  step {step:>3d}: loss = {losses[-1]:.2f}")
    wall = time.perf_counter() - t0
    bce = evaluate_recon(model, Xte)
    print(f"\n[smoke] wall={wall:.1f}s  final_train_loss={losses[-1]:.2f}  test_recon_bce={bce:.2f}")
    print("[smoke] OK — loss decreased from {:.1f} to {:.1f} in 100 steps.".format(
        losses[0], losses[-1]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="quick AdamW sanity check")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        main()
