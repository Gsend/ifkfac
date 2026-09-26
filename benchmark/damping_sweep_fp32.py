"""
benchmark/damping_sweep_fp32.py

fp32 control for the §5.5 damping-sensitivity figure (Figure 2).

Same recipe, seeds and damping grid as benchmark/damping_sweep_multiseed.py
(the bf16 sweep): SmallGPT-small, mom 0.7, lr 2e-3, gamma 0.9, T=20,
clip 300, 1000 steps, warmup 200, seeds {42, 43, 44},
lambda in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}.
The only difference: the K-FAC factor pipeline is NOT switched to bf16
(no enable_bf16 / enable_classic_bf16), so everything runs in fp32.
SINGD is excluded, as in Figure 2 (its `damping` is not a Tikhonov ridge).

Question it answers: do Classic K-FAC and IFKFAC have the same damping
curve at fp32?  If so, the gap in the bf16 curves is caused by precision,
not by a damping preference of either method.

Run:
    python -m benchmark.damping_sweep_fp32            # 2 methods x 7 lambdas x 3 seeds = 42 runs, ~6 h (RTX 3080 Laptop)
    python -m benchmark.damping_sweep_fp32 --low      # also lambda 1e-5, 3e-5 (use if the fp32 minimum sits at 1e-4)
    python -m benchmark.damping_sweep_fp32 --summary  # table only, fp32 next to bf16
    python benchmark/plot_damping_sweep.py            # redraw Figure 2 with the fp32 curves
Resumable: finished runs (JSON on disk) are skipped.

Output: benchmark/results/per_step_fp32_{classic|ifkfac}_damp_d{d}_seed{seed}_s1000.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmark.damping_sweep_multiseed import CFG, SEEDS, DAMPINGS, _train_kfac, new_path as bf16_path
from benchmark.stability_benchmark import build_data, OUT
from benchmark.gpu_benchmark import SmallGPT, get_device, get_hardware_info

METHODS = {"classic": "ClassicKFAC", "ifkfac": "IFKFAC"}
LOW_DAMPINGS = [1e-5, 3e-5]


def fp32_path(method: str, damping: float, seed: int) -> Path:
    return OUT / f"per_step_fp32_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json"


def run_cell(method, damping, seed, tlf, vl, vocab, pad, device, hw):
    p = fp32_path(method, damping, seed)
    if p.exists():
        return
    print(f"\n=== [fp32 {method}] seed={seed}  damp={damping:.0e} ===", flush=True)
    model = None
    try:
        torch.manual_seed(seed)
        model = SmallGPT(vocab_size=vocab).to(device)
        ppl, wall, recs = _train_kfac(METHODS[method], damping, seed, model,
                                      tlf, vl, vocab, pad, device)
        out = {"method": method, "damping": damping, "seed": seed,
               "precision": "fp32", "config": CFG, "hw": hw, "wall_s": wall,
               "final_ppl": ppl, "per_step": recs}
        p.write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> ppl={ppl:.0f}  wall={wall/60:.1f}m", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cell(path_fn, method, d, seeds):
    vals = []
    for s in seeds:
        p = path_fn(method, d, s)
        if p.exists():
            v = json.loads(p.read_text()).get("final_ppl")
            if v is not None and math.isfinite(v):
                vals.append(v)
    if not vals:
        return "      --"
    if len(vals) == len(seeds):
        return f"{mean(vals):>5.0f}±{stdev(vals) if len(vals) > 1 else 0:>3.0f}"
    return f"({mean(vals):>4.0f}) {len(vals)}/{len(seeds)}"


def print_summary(dampings):
    print("\n=== damping sensitivity, SmallGPT-small, final ppl (mean ± std over seeds) ===")
    print(f"  {'':<14}" + "  ".join(f"{d:>10.0e}" for d in dampings))
    for method in METHODS:
        for prec, fn in (("fp32", fp32_path), ("bf16", bf16_path)):
            row = [_cell(fn, method, d, SEEDS) for d in dampings]
            print(f"  {method + ' ' + prec:<14}" + "  ".join(f"{c:>10}" for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", action="store_true", help="also run lambda 1e-5 and 3e-5")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--summary", action="store_true", help="print the table only")
    args = ap.parse_args()
    dampings = (LOW_DAMPINGS if args.low else []) + DAMPINGS
    if args.summary:
        print_summary(dampings); return

    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}", flush=True)
    tlf, vl, vocab = build_data(device)
    pad = vocab - 1
    cells = [(s, m, d) for s in args.seeds for m in args.methods for d in dampings]
    for i, (s, m, d) in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}]", flush=True)
        run_cell(m, d, s, tlf, vl, vocab, pad, device, hw)
    print_summary(dampings)


if __name__ == "__main__":
    main()
