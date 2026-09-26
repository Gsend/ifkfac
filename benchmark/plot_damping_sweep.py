"""
Damping-sensitivity plot for §5.5 (Figure 2), read from the result JSONs.

Classic K-FAC vs IFKFAC only: same Tikhonov-ridge semantics.
SINGD is excluded because its `damping` argument is a preconditioner-
update scale, not a Tikhonov ridge.

bf16 curves (solid):  per_step_bf16_{classic|ifkfac}_damp_d{d}_seed{s}_s1000.json
                      (benchmark/damping_sweep_multiseed.py)
fp32 curves (dashed): per_step_fp32_{classic|ifkfac}_damp_d{d}_seed{s}_s1000.json
                      (benchmark/damping_sweep_fp32.py), drawn when present.
Without fp32 sweep results, the single-seed fp32 Classic champion run
(per_step_fp32_small_classic_champion_s1000.json, lambda = 1e-4) is drawn as
a dotted reference line instead.

Each point is the mean over the seeds found (shaded band: ± std).

Run:  python benchmark/plot_damping_sweep.py [--out PATH]
Default output: tmlr_submission/figures/damping_sweep.png when that folder
exists, else benchmark/results/damping_sweep.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"
SEEDS = [42, 43, 44]
LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
METHODS = {
    "classic": {"label": "Classic K-FAC ($\\kappa^2$)", "color": "#D55E00", "marker": "s"},
    "ifkfac":   {"label": "IFKFAC ($\\kappa^1$)",        "color": "#0072B2", "marker": "o"},
}


def load_series(prec, method):
    """-> (lambdas, mean, std, n_seeds) over lambdas with at least one seed."""
    lam, mu, sd, n = [], [], [], []
    for d in LAMBDAS:
        vals = []
        for s in SEEDS:
            p = RES / f"per_step_{prec}_{method}_damp_d{d:.0e}_seed{s}_s1000.json"
            if p.exists():
                v = json.loads(p.read_text()).get("final_ppl")
                if v is not None and math.isfinite(v):
                    vals.append(v)
        if vals:
            lam.append(d); mu.append(np.mean(vals))
            sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0); n.append(len(vals))
    return np.array(lam), np.array(mu), np.array(sd), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ((ROOT / "tmlr_submission" / "figures" / "damping_sweep.png")
                       if (ROOT / "tmlr_submission" / "figures").is_dir()
                       else RES / "damping_sweep.png")

    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=300)
    all_y, all_x, have_fp32 = [], [], False
    for method, st in METHODS.items():
        for prec, ls, fill in (("bf16", "-", st["color"]), ("fp32", "--", "white")):
            lam, mu, sd, n = load_series(prec, method)
            if len(lam) == 0:
                continue
            have_fp32 |= prec == "fp32"
            ax.fill_between(lam, mu - sd, mu + sd, color=st["color"],
                            alpha=0.18 if prec == "bf16" else 0.10, linewidth=0, zorder=2)
            partial = "" if all(k == len(SEEDS) for k in n) else f", {min(n)}-{max(n)} seeds"
            ax.plot(lam, mu, color=st["color"], linestyle=ls, linewidth=2.0 if prec == "bf16" else 1.6,
                    marker=st["marker"], markersize=6 if prec == "bf16" else 5,
                    markerfacecolor=fill, markeredgecolor=st["color"], markeredgewidth=1.0,
                    label=f"{st['label']}, {prec}{partial}", zorder=3)
            all_y += list(mu - sd) + list(mu + sd); all_x += list(lam)
            if prec == "bf16":
                i = int(np.argmin(mu))
                dx, dy = (1.5, -160) if method == "classic" else (0.15, 140)
                ax.annotate(f"min: {mu[i]:.0f}", xy=(lam[i], mu[i]),
                            xytext=(lam[i] * dx, mu[i] + dy), fontsize=7,
                            color=st["color"], ha="left", va="center",
                            arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.6))

    if not have_fp32:
        ref = RES / "per_step_fp32_small_classic_champion_s1000.json"
        if ref.exists():
            v = float(json.loads(ref.read_text())["final_ppl"])
            ax.axhline(v, color="#666666", linestyle=":", linewidth=1.0, zorder=1)
            ax.text(max(all_x) * 1.4, v, f"fp32 Classic\n(λ=1e-4, seed 42: {v:.0f})",
                    fontsize=7, color="#666666", va="center", ha="left")
            all_y.append(v)

    ax.set_xscale("log")
    ax.set_xlabel(r"Tikhonov damping $\lambda$")
    ax.set_ylabel("Final perplexity (WikiText-2)")
    ax.set_ylim(max(0, min(all_y) - 150), max(all_y) + 250)
    ax.set_xlim(min(all_x) / 1.6, max(all_x) * 4)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.4, color="#DDDDDD")
    ax.grid(True, which="minor", axis="x", linestyle="-", linewidth=0.3, color="#EEEEEE")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
    ax.tick_params(colors="#444444", labelsize=8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"$10^{{{int(round(np.log10(x)))}}}$"
        if abs(x - 10 ** round(np.log10(x))) < 1e-12 else f"{x:g}"))
    ax.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.5, labelspacing=0.4)
    plt.tight_layout(pad=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
