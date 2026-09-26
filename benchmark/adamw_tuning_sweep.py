"""
benchmark/adamw_tuning_sweep.py

AdamW hyperparameter sweep matched to the K-FAC / SINGD tuning we did:
    lr            ∈ {6e-4, 1e-3, 2e-3, 3e-3}
    weight_decay  ∈ {0.0, 0.01, 0.1}
    archs         : transformer (SmallGPT-medium) + cnn (ResNet-34/CIFAR-10)
    precisions    : fp32, bf16

= 4 × 3 × 2 × 2 = 48 cells × 1 seed × ~3 min ≈ 2.5 hours.

Then user re-runs the comparison_4way_multiseed.py at 3 seeds for the
winning AdamW config per cell (which gets coded back into
comparison_4way_multiseed.py's ADAMW_LR constant).

Output: benchmark/results/adamw_tune_{arch}_{precision}_lr{lr}_wd{wd}.json
        plus a printed summary table.
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


LRS = [6e-4, 1e-3, 2e-3, 3e-3]
WDS = [0.0, 0.01, 0.1]
ARCHS = ["transformer", "cnn"]
PRECISIONS = ["fp32", "bf16"]
SEED = 42
MAX_STEPS = 1000
WARMUP = 200


def out_path(arch, precision, lr, wd):
    OUT = ROOT / "benchmark" / "results"
    OUT.mkdir(exist_ok=True)
    return OUT / f"adamw_tune_{arch}_{precision}_lr{lr:.0e}_wd{wd:g}.json"


def build_data(arch, device):
    if arch == "transformer":
        from benchmark.stability_benchmark import build_data as _bd
        tlf, vl, vocab = _bd(device)
        return {"tlf": tlf, "vl": vl, "vocab": vocab, "pad": vocab - 1}
    from benchmark.models_cnn import cifar10_loaders
    tlf, vl, num_classes = cifar10_loaders(batch_size=128)
    return {"tlf": tlf, "vl": vl, "num_classes": num_classes}


def build_model(arch, ctx, device):
    if arch == "transformer":
        from benchmark.gpu_benchmark import SmallGPT
        return SmallGPT(vocab_size=ctx["vocab"],
                        d_model=384, n_heads=6, n_layers=6, d_ff=1536).to(device)
    from benchmark.models_cnn import ResNet34_CIFAR
    return ResNet34_CIFAR(num_classes=10).to(device)


def loss_fn(arch, model, batch, ctx):
    x, y = batch
    if arch == "transformer":
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1), ignore_index=ctx["pad"])
    return F.cross_entropy(model(x), y)


def run_one(arch, precision, lr, wd, ctx, device, hw):
    p = out_path(arch, precision, lr, wd)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    try:
        torch.manual_seed(SEED)
        model = build_model(arch, ctx, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                 betas=(0.9, 0.95 if arch == "transformer" else 0.999))
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP),
            torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=MAX_STEPS),
        ], milestones=[WARMUP])
        use_amp = (precision == "bf16")

        train_iter = iter(ctx["tlf"]())
        recs = []
        print(f"\n=== {arch}/{precision}/adamw  lr={lr:.0e} wd={wd:g} ===")
        t0 = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            try: batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]()); batch = next(train_iter)
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            model.train()
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = loss_fn(arch, model, batch, ctx)
            else:
                loss = loss_fn(arch, model, batch, ctx)
            lv = float(loss.item())
            if not math.isfinite(lv):
                print(f"  step={step} NaN, aborting"); break
            opt.zero_grad(set_to_none=True); loss.backward()
            opt.step(); sched.step()
            recs.append({"step": step, "loss": lv})

        wall = time.perf_counter() - t0
        eval_metric = {}
        try:
            if arch == "transformer":
                from benchmark.stability_benchmark import evaluate_ppl
                eval_metric["final_ppl"] = evaluate_ppl(model, ctx["vl"], device, ctx["pad"])
            else:
                from benchmark.models_cnn import evaluate_acc
                eval_metric["final_acc"] = evaluate_acc(model, ctx["vl"], device)
        except Exception as e:
            print(f"  eval failed: {e}")

        out = {"arch": arch, "precision": precision, "lr": lr, "weight_decay": wd,
               "seed": SEED, "wall_s": wall, "hw": hw, "per_step": recs, **eval_metric}
        p.write_text(json.dumps(out, indent=2, default=str))
        msg = (f"ppl={eval_metric['final_ppl']:.0f}" if "final_ppl" in eval_metric
               else f"acc={eval_metric['final_acc']*100:.1f}%" if "final_acc" in eval_metric
               else "")
        print(f"  -> {msg}  wall={wall/60:.1f}m")
    finally:
        try: del opt, model
        except Exception: pass
        import gc as _gc; _gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return out


def main():
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")

    contexts = {}
    for arch in ARCHS:
        print(f"\n[setup] building data for {arch}...")
        contexts[arch] = build_data(arch, device)

    total = len(ARCHS) * len(PRECISIONS) * len(LRS) * len(WDS)
    done = 0
    results = {}
    for arch in ARCHS:
        for precision in PRECISIONS:
            best = None
            for lr in LRS:
                for wd in WDS:
                    done += 1
                    print(f"\n[{done}/{total}]")
                    r = run_one(arch, precision, lr, wd, contexts[arch], device, hw)
                    key = "final_ppl" if arch == "transformer" else "final_acc"
                    score = r.get(key)
                    if score is None or not math.isfinite(score):
                        continue
                    # Lower ppl is better; higher acc is better
                    better = (score < best[1]) if arch == "transformer" and best else \
                              (score > best[1]) if arch == "cnn" and best else \
                              (best is None)
                    if better:
                        best = ((lr, wd), score)
            results[(arch, precision)] = best

    print("\n=== best AdamW config per (arch, precision) ===")
    print(f"  {'arch':>12}  {'prec':>5}  {'lr':>6}  {'wd':>5}  {'metric':>10}")
    for (arch, precision), best in results.items():
        if best is None: continue
        (lr, wd), score = best
        m = f"{score:.0f}" if arch == "transformer" else f"{score*100:.1f}%"
        print(f"  {arch:>12}  {precision:>5}  {lr:>6.0e}  {wd:>5g}  {m:>10}")

    print("\nNext step: edit benchmark/comparison_4way_multiseed.py ADAMW_LR and")
    print("run kfac_bf16_multiseed/adamw rows with the winning configs at 3 seeds.")


if __name__ == "__main__":
    main()
