"""
benchmark/rerun_ifkfac_true_bf16.py

Rerun the bf16 IFKFAC cells of §5.2, §5.7 and §5.5 with the
applied factor stored in bf16, so that both K-FAC methods are treated alike.

Why
---
In the original bf16 runs (benchmark/kfac_bf16_compare.py regime):
    Classic K-FAC  Grams formed and inverted in fp32; the cached inverses
                   (A+λI)^-1, (G+λI)^-1 are rounded to bf16 before use.
    IFKFAC         activation / gradient chunks rounded to bf16 before the
                   streaming QR; the triangular factor R stays in fp32.
So Classic's applied factor was stored in bf16 and IFKFAC's was not.

Here IFKFAC runs with use_true_bf16=True (the §5.8 autoencoder regime):
R is formed in fp32 from fp32 data, like Classic's inverse, then STORED in
bf16 between refreshes (including the moving-average blend) and upcast to
fp32 for the four triangular solves (cuSOLVER has no bf16 triangular solve).
Classic and SINGD cells are not rerun; everything else (recipe, seeds,
damping, schedules) comes unchanged from the original scripts, which are
imported and called with three patches: IFKFAC is built with
use_true_bf16=True, the input-chunk rounding is switched off, and results go
to new filenames (nothing is overwritten).

Run
    python -m benchmark.rerun_ifkfac_true_bf16 --smoke     # 30-step check that R is stored in bf16 (~1 min)
    python -m benchmark.rerun_ifkfac_true_bf16             # §5.2 (10 runs) + §5.7 (6 runs), ~6 h
    python -m benchmark.rerun_ifkfac_true_bf16 --damping   # also the §5.5 IFKFAC bf16 curve (21 runs, ~4.5 h)
    python -m benchmark.rerun_ifkfac_true_bf16 --summary   # tables: new vs original vs Classic
Resumable: finished runs are skipped.

Outputs (benchmark/results/)
    §5.2  per_step_bf16tb_{small|medium}_ifkfac_seed{42..46}_s1000.json
    §5.7  per_step_4way_{transformer|cnn}_bf16tb_ifkfac_seed{42..44}.json
    §5.5  per_step_bf16tb_ifkfac_damp_d{d}_seed{42..44}_s1000.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import benchmark.kfac_bf16_compare as KC
import benchmark.kfac_bf16_multiseed as MS
import benchmark.comparison_4way_multiseed as C4
import benchmark.damping_sweep_multiseed as DS
from benchmark.stability_benchmark import OUT
import optimizer.ifkfac_kfac as VK

TAG = {"precision": "bf16_true_storage", "r_storage": "bf16",
       "input_rounding": False,
       "note": "IFKFAC with use_true_bf16=True: R formed in fp32, stored in bf16, fp32 triangular solves"}


# ---- patches -----------------------------------------------------------------
_ORIG_INIT = VK.IFKFAC.__init__


def _init_true_bf16(self, *args, **kwargs):
    kwargs["use_true_bf16"] = True
    _ORIG_INIT(self, *args, **kwargs)


def _noop(*args, **kwargs):
    return None


def install_patches():
    VK.IFKFAC.__init__ = _init_true_bf16
    # switch off the chunk-rounding regime for IFKFAC wherever the scripts engage it
    KC.enable_bf16 = _noop
    MS.enable_bf16 = _noop
    DS.enable_bf16 = _noop
    # new result filenames
    MS.out_path = lambda arch, label, seed: OUT / f"per_step_bf16tb_{arch}_{label}_seed{seed}_s1000.json"
    C4.out_path = lambda arch, precision, method, seed: (
        ROOT / "benchmark" / "results" / f"per_step_4way_{arch}_bf16tb_{method}_seed{seed}.json")
    DS.new_path = lambda method, damping, seed: (
        OUT / f"per_step_bf16tb_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json")


def tag_file(p: Path):
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("precision") != TAG["precision"]:
            d.update(TAG)
            p.write_text(json.dumps(d, indent=2, default=str))


# ---- runs --------------------------------------------------------------------
def run_52(device, hw):
    from benchmark.stability_benchmark import build_data
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    for arch_label, arch_kwargs in MS.ARCHS:
        for seed in MS.SEEDS:
            MS.run_one(arch_label, arch_kwargs, "ifkfac", "IFKFAC", 1e-4, False,
                       seed, tlf, vl, vocab, pad, device, hw)
            tag_file(MS.out_path(arch_label, "ifkfac", seed))


def run_57(device, hw):
    for arch in C4.ARCHS:
        ctx = C4.get_data(arch, device)
        for seed in C4.SEEDS:
            C4.run_one(arch, "bf16", "ifkfac", seed, ctx, device, hw)
            tag_file(C4.out_path(arch, "bf16", "ifkfac", seed))


def run_55(device, hw):
    from benchmark.stability_benchmark import build_data
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    for seed in DS.SEEDS:
        for d in DS.DAMPINGS:
            DS.run_cell("ifkfac", d, seed, tlf, vl, vocab, pad, device, hw)
            tag_file(DS.new_path("ifkfac", d, seed))


def smoke(device):
    """30 steps of SmallGPT-small: checks that R is stored in bf16 after a refresh."""
    from benchmark.stability_benchmark import build_data, make_optimizers
    from benchmark.gpu_benchmark import SmallGPT
    import torch.nn.functional as F
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    torch.manual_seed(42)
    model = SmallGPT(vocab_size=vocab).to(device)
    kfac, emb, _ = make_optimizers("IFKFAC", model, 2e-3, 1e-4, 0.7, grad_clip=300.0,
                                   gamma=0.9, factor_update_freq=20)
    assert kfac.use_true_bf16, "use_true_bf16 patch not applied"
    assert getattr(kfac.hooks, "chunk_transform_X", None) is None, "input rounding still active"
    it = iter(tlf())
    for step in range(1, 31):
        x, y = (t.to(device) for t in next(it))
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
        model.zero_grad(); loss.backward()
        kfac.step(); emb.step()
    dtypes = {R.dtype for pair in kfac._factors.values() for R in pair}
    print(f"smoke: {len(kfac._factors)} layers with factors, stored dtypes {dtypes}, "
          f"loss at step 30 = {loss.item():.3f}", flush=True)
    assert kfac._factors and dtypes == {torch.bfloat16}, "R factors are not stored in bf16"
    try:
        kfac.cleanup()
    except Exception:
        pass
    print("smoke OK", flush=True)


# ---- summary -----------------------------------------------------------------
def _stats(paths, key):
    vals = []
    for p in paths:
        if p.exists():
            v = json.loads(p.read_text()).get(key)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
    if not vals:
        return "--"
    s = f"{mean(vals):.4g}"
    if len(vals) > 1:
        s += f" ± {stdev(vals):.2g}"
    return s + f" (n={len(vals)})"


def summary():
    R = OUT
    print("\n=== §5.2 SmallGPT bf16, final ppl (5 seeds) ===")
    for arch, _ in MS.ARCHS:
        new = [R / f"per_step_bf16tb_{arch}_ifkfac_seed{s}_s1000.json" for s in MS.SEEDS]
        old = [R / f"per_step_bf16_{arch}_ifkfac_seed{s}_s1000.json" for s in MS.SEEDS]
        cla = [R / f"per_step_bf16_{arch}_classic_seed{s}_s1000.json" for s in MS.SEEDS]
        print(f"  {arch:<7} IFKFAC R-in-bf16: {_stats(new, 'final_ppl'):<24} "
              f"IFKFAC original: {_stats(old, 'final_ppl'):<24} Classic: {_stats(cla, 'final_ppl')}")
    print("\n=== §5.7 bf16 (3 seeds): transformer ppl, CNN accuracy ===")
    for arch, key in (("transformer", "final_ppl"), ("cnn", "final_acc")):
        new = [R / f"per_step_4way_{arch}_bf16tb_ifkfac_seed{s}.json" for s in C4.SEEDS]
        old = [R / f"per_step_4way_{arch}_bf16_ifkfac_seed{s}.json" for s in C4.SEEDS]
        cla = [R / f"per_step_4way_{arch}_bf16_classic_seed{s}.json" for s in C4.SEEDS]
        print(f"  {arch:<11} IFKFAC R-in-bf16: {_stats(new, key):<24} "
              f"IFKFAC original: {_stats(old, key):<24} Classic: {_stats(cla, key)}")
    rows = [(d, [R / f"per_step_bf16tb_ifkfac_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS],
             [R / f"per_step_bf16_ifkfac_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS])
            for d in DS.DAMPINGS]
    if any(p.exists() for _, new, _ in rows for p in new):
        print("\n=== §5.5 IFKFAC bf16 damping curve, final ppl (3 seeds) ===")
        for d, new, old in rows:
            print(f"  λ={d:.0e}  R-in-bf16: {_stats(new, 'final_ppl'):<24} original: {_stats(old, 'final_ppl')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--damping", action="store_true", help="also rerun the §5.5 IFKFAC bf16 damping curve")
    ap.add_argument("--only", choices=["5.2", "5.7", "5.5"], help="run just one section")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); return
    install_patches()
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}", flush=True)
    if args.smoke:
        smoke(device); return
    sections = [args.only] if args.only else ["5.2", "5.7"] + (["5.5"] if args.damping else [])
    for sec in sections:
        {"5.2": run_52, "5.7": run_57, "5.5": run_55}[sec](device, hw)
    summary()


if __name__ == "__main__":
    main()
