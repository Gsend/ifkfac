"""
benchmark/comparison_4way_multiseed.py

Loss + wall-time comparison for the paper:
    methods   : AdamW, Classic K-FAC, IFKFAC, SINGD-Dense
    archs     : transformer (SmallGPT-medium, 22M params) + cnn (ResNet-34 CIFAR-10, 21M)
    precisions: fp32, bf16
    seeds     : 3 (42, 43, 44)

Total cells: 4 × 2 × 2 × 3 = 48 runs.  Resumable (skips existing files).
Champion hyperparams from prior screen (mom=0.7, lr=2e-3 for K-FAC; AdamW lr=2e-3
with warmup; weight_decay=0 for both).

Output: benchmark/results/per_step_4way_{arch}_{precision}_{method}_seed{seed}.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Configuration --------------------------------------------------------

ARCHS = ["transformer", "cnn"]
PRECISIONS = ["fp32", "bf16"]
METHODS = ["adamw", "classic", "ifkfac", "ifkfac_wgso", "singd"]
SEEDS = [42, 43, 44]

# Standard sweep config
MAX_STEPS = 1000
WARMUP = 200
KFAC_LR = 2e-3
KFAC_DAMPING_IFKFAC = 1e-4
KFAC_DAMPING_CLASSIC = 1e-4
KFAC_MOMENTUM = 0.7
KFAC_GAMMA = 0.9
KFAC_FREQ = 20
GRAD_CLIP = 300.0

# AdamW config
ADAMW_LR = 2e-3

# SINGD tuned config
SINGD_LR_COV = 1e-1
SINGD_ALPHA1 = 0.5


# ---- Per-arch model + data setup -----------------------------------------

def make_transformer(vocab: int):
    from benchmark.gpu_benchmark import SmallGPT
    return SmallGPT(vocab_size=vocab, d_model=384, n_heads=6, n_layers=6, d_ff=1536)


def make_cnn():
    from benchmark.models_cnn import ResNet34_CIFAR
    return ResNet34_CIFAR(num_classes=10)


def get_data(arch: str, device):
    if arch == "transformer":
        from benchmark.stability_benchmark import build_data
        tlf, vl, vocab = build_data(device)
        return {"tlf": tlf, "vl": vl, "vocab": vocab, "pad": vocab - 1}
    elif arch == "cnn":
        from benchmark.models_cnn import cifar10_loaders
        tlf, vl, num_classes = cifar10_loaders(batch_size=128)
        return {"tlf": tlf, "vl": vl, "num_classes": num_classes}
    raise ValueError(arch)


# ---- Per-arch loss computation -------------------------------------------

def compute_loss_transformer(model, batch, ctx):
    x, y = batch
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=ctx["pad"],
    )
    return loss


def compute_loss_cnn(model, batch, ctx):
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits, y)


# ---- Optimizer factories per method ---------------------------------------

def build_optimizers(method: str, model, arch: str, ctx):
    """Returns (primary_opt, secondary_opt_or_None).  AdamW has no secondary."""
    if method == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=0.0), None

    from benchmark.stability_benchmark import make_optimizers, KFAC_MAX_DIM

    if method == "classic":
        kfac, emb, _ = make_optimizers(
            "ClassicKFAC", model, KFAC_LR, KFAC_DAMPING_CLASSIC, KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA, factor_update_freq=KFAC_FREQ,
        )
        return kfac, emb

    if method == "ifkfac":
        from optimizer import ifkfac_kfac as _vk
        orig = _vk.IFKFAC.__init__
        def patched(self, *args, **kw):
            kw.setdefault("deferred_qr", True)
            return orig(self, *args, **kw)
        _vk.IFKFAC.__init__ = patched
        try:
            kfac, emb, _ = make_optimizers(
                "IFKFAC", model, KFAC_LR, KFAC_DAMPING_IFKFAC, KFAC_MOMENTUM,
                grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA, factor_update_freq=KFAC_FREQ,
            )
        finally:
            _vk.IFKFAC.__init__ = orig
        return kfac, emb

    if method == "ifkfac_wgso":
        # IFKFAC with WGSO row equilibration installed on the hooks.
        # WGSO down-weights outlier rows in activations and gradients before
        # they enter the QR pipeline (van der Sluis 1969 optimal diagonal
        # scaling).  Particularly useful for from-scratch training where
        # activation distributions are unsettled.
        # Uses tighter damping (1e-6) because WGSO already regularizes the
        # spectrum, allowing the ridge to be smaller.
        from optimizer import ifkfac_kfac as _vk
        from benchmark.kfac_bf16_compare import _wgso_weight_rows
        orig = _vk.IFKFAC.__init__
        def patched(self, *args, **kw):
            kw.setdefault("deferred_qr", True)
            ret = orig(self, *args, **kw)
            # Install WGSO as a chunk transform on the hooks
            self.hooks.chunk_transform_X = _wgso_weight_rows
            self.hooks.chunk_transform_G = _wgso_weight_rows
            return ret
        _vk.IFKFAC.__init__ = patched
        try:
            kfac, emb, _ = make_optimizers(
                "IFKFAC", model, KFAC_LR, 1e-6, KFAC_MOMENTUM,
                grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA, factor_update_freq=KFAC_FREQ,
            )
        finally:
            _vk.IFKFAC.__init__ = orig
        return kfac, emb

    if method == "singd":
        from singd.optim.optimizer import SINGD
        # Identify K-FAC-eligible params (matching our other methods)
        kfac_pids = set()
        for mod in model.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                out_dim = (mod.out_features if isinstance(mod, nn.Linear)
                            else mod.out_channels)
                if out_dim <= KFAC_MAX_DIM:
                    for p in mod.parameters():
                        kfac_pids.add(id(p))
        kfac_params = [p for p in model.parameters() if id(p) in kfac_pids]
        other = [p for p in model.parameters() if id(p) not in kfac_pids]
        singd = SINGD(
            model,
            params=kfac_params,
            lr=KFAC_LR,
            damping=1e-3,
            momentum=KFAC_MOMENTUM,
            T=KFAC_FREQ,
            structures=("dense", "dense"),
            loss_average=("batch+sequence" if arch == "transformer" else "batch"),
            lr_cov=SINGD_LR_COV,
            alpha1=SINGD_ALPHA1,
            warn_unsupported=False,
        )
        emb = torch.optim.AdamW(other, lr=1e-3, weight_decay=0.0)
        return singd, emb

    raise ValueError(method)


# ---- Precision regime helpers --------------------------------------------

def engage_bf16(method: str):
    """Enable bf16 emulation for the K-FAC factor pipeline (Classic/IFKFAC/WGSO)."""
    if method == "ifkfac":
        from benchmark.kfac_bf16_compare import enable_bf16
        enable_bf16(wgso=False)
    elif method == "ifkfac_wgso":
        from benchmark.kfac_bf16_compare import enable_bf16
        enable_bf16(wgso=True)
    elif method == "classic":
        from benchmark.kfac_bf16_compare import enable_classic_bf16
        enable_classic_bf16()
    # singd uses autocast; adamw doesn't care


def disengage_bf16(method: str):
    if method in ("ifkfac", "ifkfac_wgso"):
        from benchmark.kfac_bf16_compare import disable_bf16
        disable_bf16()
    elif method == "classic":
        from benchmark.kfac_bf16_compare import disable_classic_bf16
        disable_classic_bf16()


def need_autocast(method: str, precision: str) -> bool:
    """SINGD uses autocast for bf16; AdamW too. Classic/IFKFAC use the K-FAC patches instead."""
    return precision == "bf16" and method in ("singd", "adamw")


# ---- Per-run executor -----------------------------------------------------

def out_path(arch, precision, method, seed):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"per_step_4way_{arch}_{precision}_{method}_seed{seed}.json"


def run_one(arch, precision, method, seed, ctx, device, hw):
    p = out_path(arch, precision, method, seed)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    if precision == "bf16":
        engage_bf16(method)

    try:
        torch.manual_seed(seed)
        if arch == "transformer":
            model = make_transformer(ctx["vocab"]).to(device)
        else:
            model = make_cnn().to(device)

        opt_primary, opt_secondary = build_optimizers(method, model, arch, ctx)

        # Schedules
        sched_primary = torch.optim.lr_scheduler.SequentialLR(opt_primary, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt_primary, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
            torch.optim.lr_scheduler.ConstantLR(opt_primary, factor=1.0, total_iters=MAX_STEPS),
        ], milestones=[WARMUP])
        sched_secondary = None
        if opt_secondary is not None:
            sched_secondary = torch.optim.lr_scheduler.ConstantLR(opt_secondary, factor=1.0, total_iters=MAX_STEPS)

        use_amp = need_autocast(method, precision)
        loss_fn = compute_loss_transformer if arch == "transformer" else compute_loss_cnn

        train_iter = iter(ctx["tlf"]())
        recs = []
        print(f"\n=== {arch}/{precision}/{method}/seed{seed} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]())
                batch = next(train_iter)
            batch = tuple(t.to(device, non_blocking=True) for t in batch)

            model.train()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = loss_fn(model, batch, ctx)
            else:
                loss = loss_fn(model, batch, ctx)

            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting"); break
            model.zero_grad(set_to_none=True); loss.backward()
            opt_primary.step()
            if opt_secondary is not None:
                opt_secondary.step()
            sched_primary.step()
            if sched_secondary is not None:
                sched_secondary.step()
            recs.append({"step": step, "loss": lv})

        wall = time.perf_counter() - t0
        # eval
        eval_metric = None
        try:
            if arch == "transformer":
                from benchmark.stability_benchmark import evaluate_ppl
                eval_metric = {"final_ppl": evaluate_ppl(model, ctx["vl"], device, ctx["pad"])}
            else:
                from benchmark.models_cnn import evaluate_acc
                eval_metric = {"final_acc": evaluate_acc(model, ctx["vl"], device)}
        except Exception as e:
            print(f"  eval failed: {e}")

        out = {"arch": arch, "precision": precision, "method": method,
               "seed": seed, "wall_s": wall, "hw": hw,
               "per_step": recs, **(eval_metric or {})}
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = ""
        if eval_metric and "final_ppl" in eval_metric:
            msg = f"ppl={eval_metric['final_ppl']:.0f}  "
        elif eval_metric and "final_acc" in eval_metric:
            msg = f"acc={eval_metric['final_acc']*100:.1f}%  "
        print(f"  -> {msg}wall={wall/60:.1f}m")
    finally:
        if precision == "bf16":
            disengage_bf16(method)
        try:
            if hasattr(opt_primary, "cleanup"): opt_primary.cleanup()
        except Exception: pass
        try:
            del opt_primary, opt_secondary, model
        except Exception: pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    # Pre-build per-arch contexts once (saves data-loading time)
    contexts = {}
    for arch in ARCHS:
        print(f"\n[setup] building data for {arch}...")
        contexts[arch] = get_data(arch, device)

    total = len(ARCHS) * len(PRECISIONS) * len(METHODS) * len(SEEDS)
    done = 0
    for arch in ARCHS:
        for precision in PRECISIONS:
            for method in METHODS:
                for seed in SEEDS:
                    done += 1
                    print(f"\n[{done}/{total}]")
                    run_one(arch, precision, method, seed, contexts[arch], device, hw)

    print("\n=== summary ===")
    print(f"  {'arch':>12}  {'prec':>5}  {'method':>8}  {'mean':>8}  {'wall(m)':>8}")
    import statistics
    for arch in ARCHS:
        for precision in PRECISIONS:
            for method in METHODS:
                vals = []
                walls = []
                for seed in SEEDS:
                    p = out_path(arch, precision, method, seed)
                    if not p.exists(): continue
                    d = json.loads(p.read_text())
                    metric_key = "final_ppl" if arch == "transformer" else "final_acc"
                    v = d.get(metric_key)
                    if v is not None: vals.append(v)
                    walls.append(d.get("wall_s", 0) / 60)
                if not vals: continue
                m = statistics.mean(vals)
                w = statistics.mean(walls)
                print(f"  {arch:>12}  {precision:>5}  {method:>8}  {m:>8.2f}  {w:>8.2f}")


if __name__ == "__main__":
    main()
