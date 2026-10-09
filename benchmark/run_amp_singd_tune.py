"""
benchmark/run_amp_singd_tune.py  --  the SINGD tuning screen in mixed precision.

Runs benchmark/singd_tuning_sweep.py (12 cells: lr_cov in {1e-3, 1e-2, 1e-1}
x alpha1 in {0, 0.5, 0.9} x kfac_like; SmallGPT-small, seed 42, 1000 steps)
in the regime of benchmark/run_amp_bf16.py: fp32 master weights, bf16 autocast
forward/backward, SINGD preconditioner_dtype=(bf16, bf16).  SINGD's default
operating point (lr_cov=1e-2, alpha1=0.5, kfac_like=False) is one of the cells.

    python -m benchmark.run_amp_singd_tune          # ~35 min on an RTX 3080 Laptop
Resumable.  Outputs: benchmark/results/singd_tune2_bf16amp_*.json
Each run is dtype-checked as in run_amp_bf16.py (SINGD built with a (bf16, bf16)
preconditioner; H terms, Kronecker factors, momenta and the factors used for
the natural gradient checked bf16 on every call); the per-site counts are in
each result file under "dtype_checks".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark.run_amp_bf16 as RA
import benchmark.singd_tuning_sweep as T
from benchmark import bf16_checks as BC
import optimizer.dtype_check as DC


def main():
    RA.install()
    gpt = T.SmallGPT
    T.SmallGPT = lambda *a, **k: RA.to_amp(gpt(*a, **k))
    T.out_path = lambda lr, damp, lr_cov, alpha1, kfac_like: T.OUT / (
        f"singd_tune2_bf16amp_lr{lr:.0e}_d{damp:.0e}_c{lr_cov:.0e}_a{alpha1}"
        f"_kl{1 if kfac_like else 0}_s1000.json")
    orig = T.run_one

    def run_one(lr, damping, lr_cov, alpha1, kfac_like, *rest):
        p = T.out_path(lr, damping, lr_cov, alpha1, kfac_like)
        new = not p.exists()
        RA._OPTS.clear()
        DC.reset()
        out = orig(lr, damping, lr_cov, alpha1, kfac_like, *rest)
        if new and p.exists():
            d = json.loads(p.read_text())
            d.update(RA.TAG)
            d["optimizer_stats"] = RA._opt_stats()
            out = BC.record(p, d, RA._OPTS, methods=["singd"])
        return out
    T.run_one = run_one
    T.main()


if __name__ == "__main__":
    main()
