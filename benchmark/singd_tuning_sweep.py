


"""
benchmark/singd_tuning_sweep.py

Small SINGD hyperparameter sweep at single seed to find a fair operating
point before the multi-seed comparison.  SINGD has different damping
semantics than our K-FAC variants — its `damping` argument scales the
preconditioner momentum update, not a Tikhonov ridge.  Sweep over the
most-impactful parameters and pick the best for the fair comparison cell.

Grid (single seed=42, small arch, bf16):
    lr ∈ {1e-3 (SINGD default), 2e-3 (our default)}
    damping ∈ {1e-4, 1e-3, 1e-2}  (their default is 1e-3)
    normalize_lr_cov ∈ {False, True}  (they recommend True for stability)

= 2 × 3 × 2 = 12 cells.  At ~2 min each on small arch: ~25 min total.

Output: benchmark/results/singd_tune_lr{lr}_d{damp}_nlc{0|1}_s1000.json
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, KFAC_MAX_DIM, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info


CFG_BASE = dict(momentum=0.7, grad_clip=300.0,
                max_steps=1000, warmup_steps=200,
                factor_update_freq=20, seed=42)

# Grid — Pass 2 focuses on the previously-missed axes lr_cov and alpha1.
# Damping had no effect in Pass 1; fix it at 1e-3 (SINGD default).
# normalize_lr_cov=True was strictly worse; fix False.
# lr=2e-3 was best; keep it.
LRS = [2e-3]
DAMPINGS = [1e-3]
NORMALIZE_FLAGS = [False]
LR_COVS = [1e-3, 1e-2, 1e-1]               # SINGD default is 1e-2
ALPHA1S = [0.0, 0.5, 0.9]                   # SINGD default is 0.5; 0=no Riem.mom
KFAC_LIKE_FLAGS = [False, True]             # True = IKFAC-like simplified update


def out_path(lr, damp, lr_cov, alpha1, kfac_like):
    kl = 1 if kfac_like else 0
    return OUT / f"singd_tune2_lr{lr:.0e}_d{damp:.0e}_c{lr_cov:.0e}_a{alpha1}_kl{kl}_s1000.json"


def build_optimizers(model, lr, damping, lr_cov, alpha1, kfac_like):
    from singd.optim.optimizer import SINGD
    from torch import nn

    kfac_param_ids = set()
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        out_dim = module.out_features if isinstance(module, nn.Linear) else module.out_channels
        if KFAC_MAX_DIM > 0 and out_dim > KFAC_MAX_DIM:
            continue
        for p in module.parameters():
            kfac_param_ids.add(id(p))

    kfac_params = [p for p in model.parameters() if id(p) in kfac_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in kfac_param_ids]

    singd = SINGD(
        model,
        params=kfac_params,
        lr=lr,
        damping=damping,
        momentum=CFG_BASE["momentum"],
        T=CFG_BASE["factor_update_freq"],
        structures=("dense", "dense"),
        loss_average="batch+sequence",
        lr_cov=lr_cov,
        alpha1=alpha1,
        kfac_like=kfac_like,
        warn_unsupported=False,
    )
    # Sanity check: how many modules did SINGD actually hook?
    print(f"  [singd] hooked {len(singd.module_names)} modules")
    emb = torch.optim.AdamW(other_params, lr=1e-3, weight_decay=0.0)
    return singd, emb


def run_one(lr, damping, lr_cov, alpha1, kfac_like, tlf, vl, vocab, pad, device, hw):
    p = out_path(lr, damping, lr_cov, alpha1, kfac_like)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    torch.manual_seed(CFG_BASE["seed"])
    model = SmallGPT(vocab_size=vocab).to(device)
    singd, emb = build_optimizers(model, lr, damping, lr_cov, alpha1, kfac_like)

    w = CFG_BASE["warmup_steps"]
    sched = torch.optim.lr_scheduler.SequentialLR(singd, schedulers=[
        torch.optim.lr_scheduler.LinearLR(singd, start_factor=0.1, end_factor=1.0, total_iters=w),
        torch.optim.lr_scheduler.ConstantLR(singd, factor=1.0, total_iters=CFG_BASE["max_steps"]),
    ], milestones=[w])
    esched = torch.optim.lr_scheduler.ConstantLR(emb, factor=1.0, total_iters=CFG_BASE["max_steps"])

    it = iter(tlf())
    print(f"\n=== singd tuning  lr={lr:.0e}  damp={damping:.0e}  lr_cov={lr_cov:.0e}  alpha1={alpha1}  kfac_like={kfac_like} ===")
    t0 = time.perf_counter()
    final_loss = None
    for step in range(1, CFG_BASE["max_steps"] + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    y.reshape(-1), ignore_index=pad)
        lv = float(loss.item())
        if not math.isfinite(lv):
            print(f"  step={step}  NaN, aborting"); break
        final_loss = lv
        model.zero_grad(); loss.backward()
        singd.step(); emb.step(); sched.step(); esched.step()

    fp = evaluate_ppl(model, vl, device, pad)
    wall = time.perf_counter() - t0
    out = {"variant": "SINGD", "lr": lr, "damping": damping,
           "lr_cov": lr_cov, "alpha1": alpha1, "kfac_like": kfac_like,
           "config": CFG_BASE, "hw": hw,
           "wall_s": wall, "final_ppl": fp, "final_loss": final_loss}
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")

    import gc as _gc
    try: singd.cleanup()
    except Exception: pass
    del singd, emb, model
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1

    results = []
    total = len(LRS) * len(DAMPINGS) * len(LR_COVS) * len(ALPHA1S) * len(KFAC_LIKE_FLAGS)
    i = 0
    for lr in LRS:
        for d in DAMPINGS:
            for lc in LR_COVS:
                for a1 in ALPHA1S:
                    for kl in KFAC_LIKE_FLAGS:
                        if kl and a1 != 0.0:
                            # SINGD ignores alpha1 when kfac_like=True (it forces alpha1=0)
                            continue
                        i += 1
                        print(f"\n[{i}/{total}]")
                        results.append(run_one(lr, d, lc, a1, kl, tlf, vl, vocab, pad, device, hw))

    print(f"\n=== SINGD tuning Pass 2 summary (small arch, bf16, seed=42) ===")
    print(f"  {'lr':>5}  {'damp':>5}  {'lr_cov':>6}  {'alpha1':>6}  {'kfac_like':>9}  {'final_ppl':>9}  {'wall_m':>6}")
    for r in sorted(results, key=lambda r: r.get("final_ppl") or 1e9):
        fp = r.get("final_ppl")
        fp_s = f"{fp:.0f}" if fp is not None and math.isfinite(fp) else "DIV"
        print(f"  {r['lr']:>5.0e}  {r['damping']:>5.0e}  {r['lr_cov']:>6.0e}  {r['alpha1']:>6}  {str(r['kfac_like']):>9}  {fp_s:>9}  {r['wall_s']/60:>6.1f}")


if __name__ == "__main__":
    main()
