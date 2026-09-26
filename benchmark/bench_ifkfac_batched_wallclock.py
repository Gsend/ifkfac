"""
benchmark/bench_ifkfac_batched_wallclock.py

Quick A/B wall-time comparison: IFKFAC baseline vs IFKFAC with batched_qr=True
on the real SmallGPT (12 transformer blocks, ~72 Linear layers under K-FAC).

This is where the batching payoff actually shows: tiny TinyGPT bucket sizes
(~2 layers/bucket) saw 1.22x; SmallGPT with 12 same-shape attention layers
per bucket should see 2-3x.

Reports per-step time over 100 steps for both modes.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark.stability_benchmark import build_data, make_optimizers
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info
from optimizer import ifkfac_kfac as _vk


N_STEPS = 100
WARMUP = 10
SEED = 42
KFAC_LR = 2e-3
DAMPING = 1e-4
MOM = 0.7
GAMMA = 0.9
FREQ = 20


def run(mode: str, tlf, vocab, pad, device, label):
    # mode: "baseline" | "batched" | "deferred"
    orig_init = _vk.IFKFAC.__init__
    def patched(self, *args, **kwargs):
        if mode == "batched":
            kwargs.setdefault("batched_qr", True)
        elif mode == "deferred":
            kwargs.setdefault("deferred_qr", True)
        return orig_init(self, *args, **kwargs)
    _vk.IFKFAC.__init__ = patched
    try:
        torch.manual_seed(SEED)
        model = SmallGPT(vocab_size=vocab).to(device)
        # make_optimizers applies KFAC_MAX_DIM=4096 → excludes LM head
        # (vocab×vocab QR would be ~50k×50k and take minutes per step).
        opt, emb, _ = make_optimizers(
            "IFKFAC", model, KFAC_LR, DAMPING, MOM,
            grad_clip=300.0, gamma=GAMMA, factor_update_freq=FREQ,
        )
    finally:
        _vk.IFKFAC.__init__ = orig_init

    it = iter(tlf())
    per_step = []
    print(f"\n=== {label}  mode={mode} ===", flush=True)
    for step in range(1, N_STEPS + WARMUP + 1):
        try: x, y = next(it)
        except StopIteration: it = iter(tlf()); x, y = next(it)
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1), ignore_index=pad)
        model.zero_grad()
        loss.backward()
        opt.step(); emb.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step > WARMUP:
            per_step.append(dt)
        if step <= 3 or step % 20 == 0:
            print(f"  step={step:>3}  dt={dt*1000:.1f}ms", flush=True)
    import numpy as np
    arr = np.array(per_step)
    print(f"  n={len(arr)} steps  mean={arr.mean()*1000:.1f}ms  "
          f"p50={np.median(arr)*1000:.1f}ms  p99={np.percentile(arr, 99)*1000:.1f}ms",
          flush=True)
    return arr


def main():
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}")
    tlf, _, vocab = build_data(device); pad = vocab - 1

    times_base = run("baseline", tlf, vocab, pad, device, "baseline (streaming TSQR / step)")
    times_def  = run("deferred", tlf, vocab, pad, device, "deferred (one big QR / refresh)")

    speedup = times_base.mean() / times_def.mean()
    print(f"\nspeedup deferred vs baseline: {speedup:.2f}x  "
          f"({times_base.mean()*1000:.1f}ms -> {times_def.mean()*1000:.1f}ms)", flush=True)


if __name__ == "__main__":
    main()
