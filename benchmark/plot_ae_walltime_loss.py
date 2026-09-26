"""
benchmark/plot_ae_walltime_loss.py

Plot training BCE vs cumulative wall time for AdamW vs Classic K-FAC vs
IFKFAC on the MNIST autoencoder, with a side-by-side fp32 / bf16
comparison.

Highlights:
  - fp32: K-FAC beats AdamW by ~2.8x (§2.5.2 claim).
  - bf16: Classic K-FAC collapses; IFKFAC stays flat (§5 claim).

Loads from:
  benchmark/results/ae_mnist_{fp32|bf16}_{method}_seed{seed}.json

Output: benchmark/results/ae_walltime_loss.png
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
    ("AdamW (tuned per precision)",       "adamw",   "#888888"),
    ("Classic K-FAC (lr=1e-3, dmp=3e-2)", "classic", "#cc4444"),
    ("IFKFAC (lr=1e-3, dmp=3e-2)",   "ifkfac",   "#2266aa"),
]
SEEDS = [42, 43, 44]


def safe_load(p):
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return json.JSONDecoder().raw_decode(p.read_text())[0]


def smooth(x, w=20):
    if len(x) < w or w <= 1:
        return np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / float(w)


def collect(method, precision):
    """Returns list of (cum_wall_min, smoothed_bce, n_steps, total_wall_min,
    final_test_bce) tuples for each available seed."""
    series = []
    for seed in SEEDS:
        p = RES / f"ae_mnist_{precision}_{method}_seed{seed}.json"
        if not p.exists():
            continue
        d = safe_load(p)
        recs = d.get("per_step", [])
        if not recs:
            continue
        losses = np.array([r["loss"] for r in recs])
        wall_total = d.get("wall_s", 0.0)
        n = len(losses)
        per_step = wall_total / max(n, 1)
        cum_min = np.arange(1, n + 1) * per_step / 60.0
        sm_bce = smooth(losses, w=20)
        sm_wall = cum_min[19:] if len(cum_min) >= 20 else cum_min
        series.append((sm_wall, sm_bce, n, wall_total / 60.0,
                       d.get("final_recon_bce")))
    return series


def panel(ax, precision, title):
    summary = []
    for label, method, color in METHODS:
        series = collect(method, precision)
        if not series:
            print(f"  no seeds yet for {precision}/{method}")
            continue
        n_seeds = len(series)
        finals = [s[4] for s in series if s[4] is not None]
        final_mean = np.mean(finals) if finals else None
        final_std  = np.std(finals, ddof=1)  if len(finals) >= 2 else 0.0
        wall_mean = np.mean([s[3] for s in series])

        # Translucent per-seed traces
        for w, b, _, _, _ in series:
            ax.plot(w, b, color=color, alpha=0.30, linewidth=1.0)

        # Bold mean trace: interpolate seeds onto a common reference grid
        ref_w = max(series, key=lambda s: s[0][-1] if len(s[0]) else 0)[0]
        interp_bces = []
        for w, b, _, _, _ in series:
            if len(w) < 2:
                continue
            interp_bces.append(np.interp(ref_w, w, b))
        if interp_bces:
            mean_bce = np.mean(interp_bces, axis=0)
            label_str = (f"{label}  →  "
                         f"final BCE {final_mean:.1f}"
                         + (f" ± {final_std:.1f}" if final_std > 0 else "")
                         + f" (n={n_seeds})")
            ax.plot(ref_w, mean_bce, color=color, linewidth=2.4,
                    label=label_str)

        summary.append((label, n_seeds, final_mean, final_std, wall_mean))

    # M&G reference line
    ax.axhline(58, color="#444", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.5, 58 * 1.04, "  M&G 2015 reported ≈ 58",
            fontsize=8, va="bottom", color="#444")

    ax.set_xlabel("cumulative wall time (minutes)")
    ax.set_ylabel("training BCE (per image, smoothed, lower is better)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    return summary


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    summary_fp32 = panel(axes[0], "fp32",
                          "fp32: K-FAC beats AdamW")
    summary_bf16 = panel(axes[1], "bf16",
                          "bf16: Classic K-FAC degrades, IFKFAC stable")

    fig.suptitle("MNIST autoencoder (Hinton–Salakhutdinov 784→1000→500→250→30 hourglass)",
                  fontsize=12, y=1.00)
    fig.tight_layout()

    out = RES / "ae_walltime_loss.png"
    fig.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")

    print("\n=== fp32 ===")
    print(f"  {'method':>30}  {'n':>3}  {'BCE':>11}  {'wall(min)':>9}")
    for label, n, m, s, w in summary_fp32:
        mstr = f"{m:.1f} ± {s:.1f}" if m is not None else "—"
        print(f"  {label:>30}  {n:>3d}  {mstr:>11}  {w:>9.2f}")

    print("\n=== bf16 ===")
    print(f"  {'method':>30}  {'n':>3}  {'BCE':>11}  {'wall(min)':>9}")
    for label, n, m, s, w in summary_bf16:
        mstr = f"{m:.1f} ± {s:.1f}" if m is not None else "—"
        print(f"  {label:>30}  {n:>3d}  {mstr:>11}  {w:>9.2f}")


if __name__ == "__main__":
    main()
