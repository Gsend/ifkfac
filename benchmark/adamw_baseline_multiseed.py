"""
benchmark/adamw_baseline_multiseed.py

AdamW baseline matching the K-FAC bf16 multiseed setup exactly:
  - SmallGPT, both small (4-block) and medium (6-block) archs
  - WikiText-2, batch 64, seq 128
  - 1000 steps, warmup 200, bf16 via torch.autocast
  - lr 2e-3 (matching K-FAC champion); weight_decay 0.0

Output: benchmark/results/per_step_bf16_{arch}_adamw_seed{seed}_s1000.json

Provides AdamW comparator for the loss-curve plot in the paper.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, evaluate_ppl, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info


CFG_BASE = dict(lr=2e-3, weight_decay=0.0,
                max_steps=1000, warmup_steps=200)

ARCHS = [
    ("small",  dict()),
    ("medium", dict(d_model=384, n_heads=6, n_layers=6, d_ff=1536)),
]
SEEDS = [42, 43, 44, 45, 46]


def out_path(arch, seed):
    return OUT / f"per_step_bf16_{arch}_adamw_seed{seed}_s1000.json"


def run_one(arch_label, arch_kwargs, seed, tlf, vl, vocab, pad, device, hw):
    p = out_path(arch_label, seed)
    if p.exists():
        print(f"[skip] {p.name}")
        return json.loads(p.read_text())

    try:
        torch.manual_seed(seed)
        model = SmallGPT(vocab_size=vocab, **arch_kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG_BASE["lr"],
                                weight_decay=CFG_BASE["weight_decay"])
        w = CFG_BASE["warmup_steps"]
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=CFG_BASE["max_steps"]),
        ], milestones=[w])

        it = iter(tlf())
        recs, prev = [], None
        print(f"\n=== bf16  arch={arch_label}  adamw  seed={seed} ===")
        t0 = time.perf_counter()
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
            opt.zero_grad(set_to_none=True); loss.backward()
            opt.step(); sched.step()
            dl = (lv - prev) if prev is not None else 0.0; prev = lv
            recs.append({"step": step, "loss": lv, "delta_loss": dl})

        fp = evaluate_ppl(model, vl, device, pad)
        wall = time.perf_counter() - t0
        out = {"arch": arch_label, "arch_kwargs": arch_kwargs,
               "label": "adamw", "variant": "AdamW", "seed": seed,
               "precision": "bf16_autocast", "config": CFG_BASE, "hw": hw,
               "wall_s": wall, "final_ppl": fp, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={fp:.0f}  wall={wall/60:.1f}m")
    finally:
        try: del opt, model
        except Exception: pass
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, vl, vocab = build_data(device); pad = vocab - 1
    total = len(ARCHS) * len(SEEDS)
    done = 0
    for arch_label, arch_kwargs in ARCHS:
        for seed in SEEDS:
            done += 1
            print(f"\n[{done}/{total}]")
            run_one(arch_label, arch_kwargs, seed, tlf, vl, vocab, pad, device, hw)


if __name__ == "__main__":
    main()
