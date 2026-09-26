"""
benchmark/plot_4way_walltime_pareto.py

Three additional figures for the paper:
  - comparison_loss_vs_walltime.png : loss curves with x-axis = cumulative wall time
  - comparison_time_to_target.png   : bar chart of wall time to reach target ppl
  - comparison_final_ppl.png        : final-perplexity bar chart per method
                                       (call-out so the K-FAC vs AdamW gap is
                                       visually unmistakable)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RES = Path(__file__).resolve().parent / "results"

METHODS = [
    ("adamw",      "AdamW",         "#888888"),
    ("classic",    "Classic K-FAC", "#cc4444"),
    ("ifkfac",      "IFKFAC",   "#2266aa"),
    ("ifkfac_wgso", "IFKFAC + WGSO",  "#1144cc"),
    ("singd",      "SINGD-Dense",   "#44aa66"),
]
ARCHS = [("transformer", "SmallGPT-medium (22M)"),
         ("cnn",         "ResNet-34 CIFAR-10 (21M)")]
PRECISIONS = ["fp32", "bf16"]
SEEDS = [42, 43, 44]
SMOOTH = 20


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


def loss_to_ppl(loss):
    """For transformer cells, perplexity ≈ exp(loss)."""
    return np.exp(np.minimum(loss, 30))   # clamp for safety


# ============================================================================
# Figure A: loss vs cumulative wall time (Pareto-style — speed/quality)
# ============================================================================

def plot_loss_vs_walltime():
    fig, axes = plt.subplots(len(ARCHS), len(PRECISIONS),
                              figsize=(12, 7), sharex=False)
    for ai, (arch, arch_title) in enumerate(ARCHS):
        for pi, precision in enumerate(PRECISIONS):
            ax = axes[ai, pi]
            for method, label, color in METHODS:
                # Pull loss + wall time per seed and average
                step_walls = []
                losses = []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    if not d.get("per_step"): continue
                    L = np.array([r["loss"] for r in d["per_step"]])
                    wall_total = d.get("wall_s", 0)
                    # Approximate per-step wall: linearly interpolate
                    # (we recorded only total wall, so per-step ≈ total/n_steps)
                    n = len(L)
                    if n < 100 or wall_total <= 0: continue
                    step_wall = wall_total / n
                    cum_wall = np.arange(1, n+1) * step_wall / 60.0   # minutes
                    step_walls.append(cum_wall)
                    losses.append(L)
                if not losses: continue
                # Align lengths
                min_len = min(len(L) for L in losses)
                stacked_L = np.stack([L[:min_len] for L in losses])
                stacked_W = np.stack([W[:min_len] for W in step_walls])
                mean_L = stacked_L.mean(axis=0)
                std_L = stacked_L.std(axis=0)
                mean_W = stacked_W.mean(axis=0)
                if SMOOTH > 1 and min_len >= SMOOTH:
                    mean_L = smooth(mean_L, SMOOTH)
                    std_L = smooth(std_L, SMOOTH)
                    mean_W = mean_W[SMOOTH-1:]
                ax.plot(mean_W, mean_L, color=color, linewidth=1.8, label=label)
                ax.fill_between(mean_W, mean_L - std_L, mean_L + std_L,
                                color=color, alpha=0.15)
            ax.set_title(f"{arch_title} — {precision}", fontsize=10)
            ax.grid(alpha=0.3)
            if ai == len(ARCHS) - 1:
                ax.set_xlabel("cumulative wall time (minutes)")
            if pi == 0:
                ax.set_ylabel("training loss")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Training loss vs cumulative wall time  (Pareto frontier; mean ± std, 3 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    out = RES / "comparison_loss_vs_walltime.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ============================================================================
# Figure B: wall time to reach target perplexity (transformer only)
# ============================================================================

def plot_time_to_target_ppl():
    """For transformer, show wall time to first reach target perplexity values."""
    targets = [1500, 1200, 1000, 900]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    arch = "transformer"
    for pi, precision in enumerate(PRECISIONS):
        ax = axes[pi]
        x = np.arange(len(targets))
        width = 0.2
        for mi, (method, label, color) in enumerate(METHODS):
            times = []
            for tgt in targets:
                vals = []
                for s in SEEDS:
                    d = load_run(arch, precision, method, s)
                    if d is None: continue
                    L = [r["loss"] for r in d.get("per_step", [])]
                    P = np.exp(np.minimum(L, 30))
                    wall_total = d.get("wall_s", 0)
                    n = len(L)
                    if n == 0 or wall_total <= 0: continue
                    # find first index where ppl ≤ target
                    idx = np.argmax(P <= tgt) if (P <= tgt).any() else None
                    if idx is None or P[idx] > tgt:
                        vals.append(np.nan)
                    else:
                        vals.append((idx + 1) * (wall_total / n) / 60.0)
                # Average across seeds, ignoring NaN (methods that never hit target)
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    times.append(np.mean(finite))
                else:
                    times.append(np.nan)
            # Plot bars
            xs = x + (mi - (len(METHODS)-1)/2) * width
            for xi, t in zip(xs, times):
                if np.isfinite(t):
                    ax.bar(xi, t, width=width, color=color,
                           edgecolor="black", linewidth=0.5)
                    ax.text(xi, t, f"{t:.1f}", ha="center", va="bottom", fontsize=7)
                else:
                    # Hatch pattern for "never reached"
                    ax.bar(xi, 0, width=width, color=color, alpha=0.2)
                    ax.text(xi, 0.1, "n/r", ha="center", va="bottom", fontsize=7,
                            color="darkred")
        ax.set_xticks(x)
        ax.set_xticklabels([f"ppl ≤ {t}" for t in targets])
        ax.set_title(f"Transformer — {precision}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if pi == 0:
            ax.set_ylabel("wall time to reach target (minutes)")
        # Manual legend
        from matplotlib.patches import Patch
        legend_items = [Patch(facecolor=c, edgecolor="black", label=l)
                        for _, l, c in METHODS]
        ax.legend(handles=legend_items, loc="upper left", fontsize=8)
    fig.suptitle("Wall time to first reach target validation perplexity  "
                 "(transformer, 3 seeds; n/r = never reached)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = RES / "comparison_time_to_target.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ============================================================================
# Figure C: final ppl bar chart (transformer) — the K-FAC vs AdamW headline
# ============================================================================

def plot_final_ppl():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    arch = "transformer"
    for pi, precision in enumerate(PRECISIONS):
        ax = axes[pi]
        labels, means, stds, colors = [], [], [], []
        for method, label, color in METHODS:
            ppls = []
            for s in SEEDS:
                d = load_run(arch, precision, method, s)
                if d is None: continue
                ppl = d.get("final_ppl")
                if ppl is not None and math.isfinite(ppl):
                    ppls.append(ppl)
            if not ppls: continue
            labels.append(label)
            means.append(np.mean(ppls))
            stds.append(np.std(ppls))
            colors.append(color)
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, color=colors,
                      edgecolor="black", linewidth=0.5, capsize=4)
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
        # Annotate the AdamW reference line for visual baseline
        adamw_mean = means[labels.index("AdamW")] if "AdamW" in labels else None
        if adamw_mean is not None:
            ax.axhline(adamw_mean, color="#444444", linestyle="--",
                       linewidth=0.7, alpha=0.5)
            ax.text(len(labels) - 0.4, adamw_mean,
                    "  AdamW baseline", fontsize=7, va="bottom", color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_title(f"Transformer — {precision}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if pi == 0:
            ax.set_ylabel("final validation perplexity (lower is better)")
    fig.suptitle("Final perplexity per method (mean ± std, 3 seeds) — K-FAC vs AdamW",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = RES / "comparison_final_ppl.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    plot_loss_vs_walltime()
    plot_time_to_target_ppl()
    plot_final_ppl()


if __name__ == "__main__":
    main()
