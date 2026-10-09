"""
benchmark/rerun_classic_bf16_fixed.py

Rerun every Classic K-FAC bf16 training cell of the paper with the corrected
bf16 patch (benchmark/kfac_bf16_compare.py, fix of 2026-09-27).

Why
---
The old patch wrapped ClassicKFAC.step() and restored a pre-step copy of the
cached inverses afterwards.  On every refresh step that restore discarded the
freshly computed inverses, so Classic never kept any: it applied unrounded
fp32 inverses on the refresh step only and ran plain SGD (no preconditioner,
no momentum) on the other factor_update_freq - 1 steps.  The fixed patch
rounds the inverses to bf16 right after they are computed and leaves step()
alone, so every step applies bf16-stored inverses (fp32 arithmetic, fp32
gradient) - the regime the paper describes, and the same storage-only regime
as IFKFAC's use_true_bf16 rerun (benchmark/rerun_ifkfac_true_bf16.py).

Everything else (recipe, seeds, damping, schedules) comes unchanged from the
original scripts, which are imported and called; only the result filenames
change (nothing is overwritten).

Run
    python -m benchmark.rerun_classic_bf16_fixed --smoke          # 30-step check (~1 min)
    python -m benchmark.rerun_classic_bf16_fixed                  # §5.2 + §5.7 + §5.8 (19 runs, ~2.5 h)
    python -m benchmark.rerun_classic_bf16_fixed --only 5.5       # §5.5 Classic bf16 damping curve (21 runs, ~1.5 h)
    python -m benchmark.rerun_classic_bf16_fixed --summary
Resumable: finished runs are skipped.

Outputs (benchmark/results/)
    §5.2  per_step_bf16fix_{small|medium}_classic_seed{42..46}_s1000.json
    §5.7  per_step_4way_{transformer|cnn}_bf16fix_classic_seed{42..44}.json
    §5.8  ae_mnist_bf16fix_classic_seed{42..44}.json
    §5.5  per_step_bf16fix_classic_damp_d{d}_seed{42..44}_s1000.json
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
import benchmark.autoencoder_mnist as AE
from benchmark.stability_benchmark import OUT
import optimizer.classic_kfac as CK
import optimizer.dtype_check as DC
from benchmark import bf16_checks as BC

RES = ROOT / "benchmark" / "results"
TAG = {"precision": "bf16_inverse_storage", "classic_bf16_patch": "fixed-2026-09-27",
       "note": "Classic K-FAC: Gram EMA and inverses computed in fp32, cached inverses stored "
               "rounded to bf16 and applied at every step (fp32 matmuls, fp32 gradient)"}


# ---- checks and redirection ----------------------------------------------------
def check_fixed_patch():
    """Refuse to run with the old (buggy) step-wrapping patch."""
    assert hasattr(KC, "_update_inverses_bf16"), \
        "benchmark/kfac_bf16_compare.py still has the old Classic bf16 patch"
    step0, inv0 = CK.ClassicKFAC.step, CK.ClassicKFAC._update_inverses
    KC.enable_classic_bf16()
    ok = (CK.ClassicKFAC.step is step0) and (CK.ClassicKFAC._update_inverses is KC._update_inverses_bf16)
    KC.disable_classic_bf16()
    assert ok, "enable_classic_bf16() does not install the fixed patch"
    # the harnesses must call the fixed functions (module-level imports in MS / DS)
    for mod in (MS, DS):
        assert mod.enable_classic_bf16 is KC.enable_classic_bf16, f"{mod.__name__} holds a stale patch"


def install_paths():
    MS.out_path = lambda arch, label, seed: OUT / f"per_step_bf16fix_{arch}_{label}_seed{seed}_s1000.json"
    C4.out_path = lambda arch, precision, method, seed: (
        RES / f"per_step_4way_{arch}_bf16fix_{method}_seed{seed}.json")
    DS.new_path = lambda method, damping, seed: (
        OUT / f"per_step_bf16fix_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json")
    AE.out_path = lambda precision, method, seed: RES / f"ae_mnist_bf16fix_{method}_seed{seed}.json"


def tag_file(p: Path):
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("classic_bf16_patch") != TAG["classic_bf16_patch"]:
            d.update(TAG)
            p.write_text(json.dumps(d, indent=2, default=str))


def _cell(fn, path: Path, *args):
    """One run with dtype checks: the inverses must be rounded to bf16 at every
    refresh and every step must apply bf16-valued inverses
    (benchmark/kfac_bf16_compare.py); the counts go into the result file."""
    already = path.exists()
    DC.reset()
    fn(*args)
    if path.exists() and not already:
        BC.record(path, json.loads(path.read_text()), methods=["classic"], storage=True)
    tag_file(path)


# ---- runs ------------------------------------------------------------------------
def run_52(device, hw):
    from benchmark.stability_benchmark import build_data
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    for arch_label, arch_kwargs in MS.ARCHS:
        for seed in MS.SEEDS:
            _cell(MS.run_one, MS.out_path(arch_label, "classic", seed),
                  arch_label, arch_kwargs, "classic", "ClassicKFAC", 1e-4, False,
                  seed, tlf, vl, vocab, pad, device, hw)


def run_57(device, hw):
    for arch in C4.ARCHS:
        ctx = C4.get_data(arch, device)
        for seed in C4.SEEDS:
            _cell(C4.run_one, C4.out_path(arch, "bf16", "classic", seed),
                  arch, "bf16", "classic", seed, ctx, device, hw)


def run_58(device, hw):
    tlf, Xte = AE.get_mnist(device)
    ctx = {"tlf": tlf, "Xte": Xte}
    for seed in AE.SEEDS:
        _cell(AE.run_one, AE.out_path("bf16", "classic", seed), "bf16", "classic", seed, ctx, device, hw)


def run_55(device, hw):
    from benchmark.stability_benchmark import build_data
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    for seed in DS.SEEDS:
        for d in DS.DAMPINGS:
            _cell(DS.run_cell, DS.new_path("classic", d, seed), "classic", d, seed, tlf, vl, vocab, pad, device, hw)


def smoke(device):
    """30 steps of SmallGPT-small: inverses must be cached, bf16-valued and kept every step."""
    from benchmark.stability_benchmark import build_data, make_optimizers
    from benchmark.gpu_benchmark import SmallGPT
    import torch.nn.functional as F
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    KC.enable_classic_bf16()
    DC.reset()
    try:
        torch.manual_seed(42)
        model = SmallGPT(vocab_size=vocab).to(device)
        kfac, emb, _ = make_optimizers("ClassicKFAC", model, 2e-3, 1e-4, 0.7, grad_clip=300.0,
                                       gamma=0.9, factor_update_freq=20)
        it = iter(tlf())
        n_with_inverses = 0
        for step in range(1, 31):
            x, y = (t.to(device) for t in next(it))
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad)
            model.zero_grad(); loss.backward()
            kfac.step(); emb.step()
            invs = list(kfac._inverses.values())
            assert invs, f"no cached inverses after step {step}"
            assert all(torch.equal(A, KC._bf16q(A)) and torch.equal(G, KC._bf16q(G)) for A, G in invs), \
                f"cached inverses are not bf16-valued after step {step}"
            n_with_inverses += 1
        print(f"smoke: {len(kfac._inverses)} layers with cached bf16 inverses on all "
              f"{n_with_inverses}/30 steps, {len(kfac._momentum_buffers)} momentum buffers, "
              f"loss at step 30 = {loss.item():.3f}", flush=True)
        print("smoke " + BC.line(BC.run_summary(methods=["classic"], storage=True)), flush=True)
        try:
            kfac.cleanup()
        except Exception:
            pass
    finally:
        KC.disable_classic_bf16()
    print("smoke OK", flush=True)


# ---- summary -------------------------------------------------------------------
def _stats(paths, key, scale=1.0):
    vals = []
    for p in paths:
        if p.exists():
            v = json.loads(p.read_text()).get(key)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v) * scale)
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
        f = lambda tag, lab: [R / f"per_step_{tag}_{arch}_{lab}_seed{s}_s1000.json" for s in MS.SEEDS]
        print(f"  {arch:<7} Classic fixed: {_stats(f('bf16fix', 'classic'), 'final_ppl'):<22} "
              f"Classic old patch: {_stats(f('bf16', 'classic'), 'final_ppl'):<22} "
              f"IFKFAC R-in-bf16: {_stats(f('bf16tb', 'ifkfac'), 'final_ppl')}")
    print("\n=== §5.7 (3 seeds): transformer ppl, CNN accuracy % ===")
    for arch, key, sc in (("transformer", "final_ppl", 1.0), ("cnn", "final_acc", 100.0)):
        f = lambda tag, m: [RES / f"per_step_4way_{arch}_{tag}_{m}_seed{s}.json" for s in C4.SEEDS]
        print(f"  {arch:<11} Classic fixed: {_stats(f('bf16fix', 'classic'), key, sc):<22} "
              f"old patch: {_stats(f('bf16', 'classic'), key, sc):<22} "
              f"Classic fp32: {_stats(f('fp32', 'classic'), key, sc):<22} "
              f"IFKFAC R-in-bf16: {_stats(f('bf16tb', 'ifkfac'), key, sc)}")
    print("\n=== §5.8 MNIST autoencoder, test BCE (3 seeds) ===")
    f = lambda tag, m: [RES / f"ae_mnist_{tag}_{m}_seed{s}.json" for s in AE.SEEDS]
    print(f"  Classic fixed: {_stats(f('bf16fix', 'classic'), 'final_recon_bce')}   "
          f"old patch: {_stats(f('bf16', 'classic'), 'final_recon_bce')}   "
          f"Classic fp32: {_stats(f('fp32', 'classic'), 'final_recon_bce')}   "
          f"IFKFAC bf16: {_stats(f('bf16', 'ifkfac'), 'final_recon_bce')}")
    rows = [(d, [R / f"per_step_bf16fix_classic_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS],
             [R / f"per_step_bf16_classic_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS])
            for d in DS.DAMPINGS]
    if any(p.exists() for _, new, _ in rows for p in new):
        print("\n=== §5.5 Classic bf16 damping curve, final ppl (3 seeds) ===")
        for d, new, old in rows:
            print(f"  λ={d:.0e}  fixed: {_stats(new, 'final_ppl'):<22} old patch: {_stats(old, 'final_ppl')}")
    BC.report(list(OUT.glob("*bf16fix_*classic*.json")) + list(RES.glob("*bf16fix_*classic*.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=["5.2", "5.7", "5.8", "5.5"], help="run just one section")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); return
    check_fixed_patch()
    install_paths()
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}", flush=True)
    if args.smoke:
        smoke(device); return
    sections = [args.only] if args.only else ["5.2", "5.7", "5.8"]
    for sec in sections:
        {"5.2": run_52, "5.7": run_57, "5.8": run_58, "5.5": run_55}[sec](device, hw)
    summary()


if __name__ == "__main__":
    main()
