"""
benchmark/plot_4way_comparison.py

Two figures from the 4-way comparison sweep:
  - Figure A (2×2 grid): training-loss curves for transformer + cnn, fp32 + bf16
  - Figure B (2×2 grid): wall-time bar charts (mean per method, with seed range)

Output: benchmark/results/comparison_loss.png
        benchmark/results/comparison_wall.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = Path(__file__).resolve().parent / "results"

METHODS = [
    ("classic",    "Classic K-FAC", "#cc4444"),
    ("ifkfac",      "IFKFAC",        "#2266aa"),
    ("singd",      "SINGD-Dense",   "#44aa66"),
]
ARCHS = [("transformer", "SmallGPT-medium (22M)"),
         ("cnn",         "ResNet-34 CIFAR-10 (21M)")]
PRECISIONS = ["fp32", "bf16"]
SEEDS = [42, 43, 44]
# Per-step losses are logged every 10 training steps (101 records over 1000
# steps), so a rolling window of 5 records ≈ 50 steps — enough to denoise
# without flattening the curve.
SMOOTH = 5


def load_run(arch, precision, method, seed):
    p = RES / f"per_step_4way_{arch}_{precision}_{method}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def smooth(x, w):
    if len(x) < w or w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / float(w)


# ---- Figure A: loss curves -----------------------------------------------

def plot_loss():
    fig, axes = plt.subplots(len(ARCHS), len(PRECISIONS),
                              figsize=(11.5, 7), sharex=True)
    for ai, (arch, arch_title) in enumerate(ARCHS):
        for pi, precision in enumerate(PRECISIONS):
            ax = axes[ai, pi]
            for method, label, color in METHODS:
                curves, step_arrs = [], []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    losses = np.array([r["loss"] for r in d["per_step"]])
                    stepvals = np.array([r["step"] for r in d["per_step"]],
                                        dtype=float)
                    # Require a near-complete run (full runs log 101 records).
                    # This avoids a badly-aborted seed truncating the shared
                    # min_len and collapsing the other seeds' curves.
                    if len(losses) >= 50:
                        curves.append(losses)
                        step_arrs.append(stepvals)
                if not curves:
                    continue
                min_len = min(len(c) for c in curves)
                stacked = np.stack([c[:min_len] for c in curves])
                # x-axis is the ACTUAL training step (losses are logged every 10
                # steps, so the step grid runs 1, 11, …, 1000 — not 1..N).  The
                # seeds share this grid, so take the first.
                xsteps = step_arrs[0][:min_len]
                mean = stacked.mean(axis=0)
                std = stacked.std(axis=0)
                if SMOOTH > 1 and min_len >= SMOOTH:
                    mean = smooth(mean, SMOOTH)
                    std = smooth(std, SMOOTH)
                    xsteps = smooth(xsteps, SMOOTH)
                ax.plot(xsteps, mean, color=color, linewidth=1.8,
                        label=f"{label} (n={len(curves)})")
                ax.fill_between(xsteps, mean - std, mean + std,
                                color=color, alpha=0.15)
            ax.set_title(f"{arch_title} — {precision}", fontsize=10)
            ax.grid(alpha=0.3)
            if ai == len(ARCHS) - 1:
                ax.set_xlabel("training step")
            if pi == 0:
                ax.set_ylabel("loss (cross-entropy)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(0, 1000)
    fig.suptitle(f"Training-loss curves: 4-way comparison "
                 f"(mean ± std over {len(SEEDS)} seeds, "
                 f"{SMOOTH}-step rolling smoothing)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = RES / "comparison_loss.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ---- Figure B: wall-time bars --------------------------------------------

def plot_wall():
    fig, axes = plt.subplots(len(ARCHS), len(PRECISIONS),
                              figsize=(10.5, 6.5), sharey=False)
    for ai, (arch, arch_title) in enumerate(ARCHS):
        for pi, precision in enumerate(PRECISIONS):
            ax = axes[ai, pi]
            labels, means, stds, colors, is_na = [], [], [], [], []
            for method, label, color in METHODS:
                walls = []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    # A run that did not complete (full runs log ~101 records;
                    # a crashed optimizer logs only a handful) has a misleadingly
                    # tiny wall time — it never did the work.  Mark it N/A rather
                    # than plotting a fast-looking bar.
                    if len(d.get("per_step", [])) < 50:
                        continue
                    walls.append(d.get("wall_s", 0) / 60)
                labels.append(label)
                colors.append(color)
                if walls:
                    means.append(np.mean(walls)); stds.append(np.std(walls))
                    is_na.append(False)
                else:
                    # nan height → no bar drawn; we annotate "N/A" instead.
                    means.append(np.nan); stds.append(np.nan)
                    is_na.append(True)
            x = np.arange(len(labels))
            plot_means = [0.0 if na else m for m, na in zip(means, is_na)]
            plot_stds = [0.0 if na else sd for sd, na in zip(stds, is_na)]
            bars = ax.bar(x, plot_means, yerr=plot_stds, color=colors, capsize=4,
                          edgecolor="black", linewidth=0.5)
            for bar, na in zip(bars, is_na):
                if na:
                    bar.set_visible(False)   # don't draw a zero-height stub
            ymax = max([m for m, na in zip(means, is_na) if not na], default=1.0)
            for xi, m, na in zip(x, means, is_na):
                if na:
                    ax.text(xi, ymax * 0.03, "N/A", ha="center", va="bottom",
                            fontsize=8, color="#999999", style="italic")
                else:
                    ax.text(xi, m, f"{m:.1f}m", ha="center", va="bottom",
                            fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
            ax.set_title(f"{arch_title} — {precision}", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            if pi == 0:
                ax.set_ylabel("wall time (min / 1000 steps)")
    fig.suptitle("Wall-time per 1000 training steps  (mean ± std, 3 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = RES / "comparison_wall.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    plot_loss()
    plot_wall()

    # text summary
    print("\n=== final metric (ppl for transformer, acc%% for cnn) and wall time ===")
    print(f"  {'arch':>12}  {'prec':>5}  {'method':>8}  {'metric':>10}  {'wall(m)':>8}")
    for arch, _ in ARCHS:
        for precision in PRECISIONS:
            for method, _, _ in METHODS:
                mvals, walls = [], []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    k = "final_ppl" if arch == "transformer" else "final_acc"
                    v = d.get(k)
                    if v is not None:
                        if k == "final_acc": v *= 100
                        mvals.append(v)
                    walls.append(d.get("wall_s", 0) / 60)
                if not mvals: continue
                mu_m = np.mean(mvals); mu_w = np.mean(walls)
                print(f"  {arch:>12}  {precision:>5}  {method:>8}  {mu_m:>10.2f}  {mu_w:>8.2f}")


if __name__ == "__main__":
    main()
