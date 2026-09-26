"""
tests/test_kappa_scaling.py

Numerical-analysis verification of the kappa^1 vs kappa^4 stability claim.

Procedure
---------
1. Build synthetic activation matrix X with controlled condition number kappa,
   via SVD with prescribed singular spectrum.  Same for gradient signals δ.
2. Compute the "exact" natural gradient nat* = G^-1 @ grad @ A^-1 in fp64
   using a stable triangular-solve pipeline (this is the reference).
3. For each method (Classic / IFKFAC / IFKFAC+WGSO) at each precision
   (fp32 / bf16-simulated), compute its natgrad and measure relative error
   against nat*.
4. Plot log(rel_err) vs log(kappa) per (method, precision).  Slope of each
   line is the kappa exponent.  Expected:
       Classic:  slope ≈ 4
       IFKFAC:    slope ≈ 1
       WGSO:     slope ≈ 1 (smaller intercept)

Output: benchmark/results/kappa_scaling.png  +  kappa_scaling.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Synthetic data ----------------------------------------------------

def make_data(p: int, n: int, kappa: float, seed: int = 0,
              device="cpu", dtype=torch.float64):
    """Build X of shape (p, n) with cond(X) = kappa, via SVD."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    # Random orthonormal U (p, n) and V (n, n) via QR of random Gaussians
    U, _ = torch.linalg.qr(torch.randn(p, n, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    # Singular values geometrically spaced from 1 to kappa (so sigma_max/sigma_min = kappa)
    sigmas = torch.logspace(0, np.log10(kappa), n, dtype=torch.float64)
    X = U @ torch.diag(sigmas) @ V.T
    return X.to(device=device, dtype=dtype)


def make_grad(n_out: int, n_in: int, seed: int, device="cpu",
              dtype=torch.float64):
    g = torch.Generator(device='cpu').manual_seed(seed + 1000)
    return torch.randn(n_out, n_in, generator=g, dtype=torch.float64).to(
        device=device, dtype=dtype)


# ---- Precision regimes -------------------------------------------------

def quantize(t: torch.Tensor, regime: str) -> torch.Tensor:
    """Convert tensor to the target precision regime.

    fp32:      run the algorithm in fp32 arithmetic.
    bf16:      cuSOLVER lacks bf16 QR, so we simulate by quantizing inputs to
               bf16 precision then upcasting to fp32 for the actual call.  The
               fp32 arithmetic operates on bf16-precision data, exactly matching
               how our multiseed sweep emulates bf16 K-FAC.
    true_bf16: tensors stored as bf16 throughout — calls hand-rolled bf16
               linalg primitives in optimizer/bf16_linalg.py because cuSOLVER
               lacks geqrf/triangular_solve/cholesky/inv for bf16.
    """
    if regime == "fp64":
        return t.to(torch.float64)
    if regime == "fp32":
        return t.to(torch.float32)
    if regime == "bf16":
        return t.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    if regime == "true_bf16":
        return t.to(torch.bfloat16)
    raise ValueError(regime)


# ---- Algorithms --------------------------------------------------------

def classic_natgrad(X, delta, grad_W, damping):
    """Classic K-FAC: form Gram, invert via Cholesky.

    For bf16-stored inputs we dispatch to the hand-rolled inv_spd_bf16
    primitive (cuSOLVER lacks native bf16 inv).
    """
    A = X.T @ X
    G = delta.T @ delta
    n_in, n_out = A.shape[0], G.shape[0]
    eye_A = torch.eye(n_in,  device=A.device, dtype=A.dtype)
    eye_G = torch.eye(n_out, device=G.device, dtype=G.dtype)
    A = A + damping * eye_A
    G = G + damping * eye_G
    if A.dtype == torch.bfloat16:
        from optimizer.bf16_linalg import inv_spd_bf16
        A_inv = inv_spd_bf16(A)
        G_inv = inv_spd_bf16(G)
    else:
        A_inv = torch.linalg.inv(A)
        G_inv = torch.linalg.inv(G)
    return G_inv @ grad_W @ A_inv


def _qr_dispatch(M):
    """Reduced QR; uses hand-rolled bf16 path for bf16 inputs."""
    if M.dtype == torch.bfloat16:
        from optimizer.bf16_linalg import householder_qr_bf16
        return householder_qr_bf16(M)
    _, R = torch.linalg.qr(M, mode="reduced")
    return R


def _solve_tri_dispatch(R, B, upper):
    """Triangular solve R · X = B; uses hand-rolled bf16 path for bf16 inputs."""
    if R.dtype == torch.bfloat16:
        from optimizer.bf16_linalg import solve_triangular_bf16
        return solve_triangular_bf16(R, B, upper=upper)
    return torch.linalg.solve_triangular(R, B, upper=upper)


def _augmented_qr(M, damping):
    """Append sqrt(damping)*I and re-QR — same as finalize_R."""
    n = M.shape[1]
    aug = torch.cat([M, np.sqrt(damping) * torch.eye(n, device=M.device, dtype=M.dtype)],
                    dim=0)
    R = _qr_dispatch(aug)
    diag_signs = torch.sign(torch.diagonal(R))
    diag_signs[diag_signs == 0] = 1.0
    return R * diag_signs.unsqueeze(1)


def ifkfac_natgrad(X, delta, grad_W, damping):
    """IFKFAC: QR factor, then four triangular solves.

    Dispatches to hand-rolled bf16 primitives when inputs are bf16.
    """
    R_X_raw = _qr_dispatch(X)
    R_G_raw = _qr_dispatch(delta)
    R_X = _augmented_qr(R_X_raw, damping)
    R_G = _augmented_qr(R_G_raw, damping)
    # natgrad = R_GᵀR_G⁻¹ · grad_W · R_XᵀR_X⁻¹
    # For bf16 path, R_G.T / R_X.T need to be made contiguous (the bf16
    # solver assumes row-major storage).
    T1 = _solve_tri_dispatch(R_G.t().contiguous(), grad_W, upper=False)
    T2 = _solve_tri_dispatch(R_G,                  T1,     upper=True)
    T3 = _solve_tri_dispatch(R_X.t().contiguous(), T2.t().contiguous(), upper=False)
    T4 = _solve_tri_dispatch(R_X,                  T3,     upper=True)
    return T4.t().contiguous()


def wgso_natgrad(X, delta, grad_W, damping, eps_frac=1.0):
    """IFKFAC + WGSO row weighting.

    Same kappa^1 exponent as IFKFAC; smaller kappa constant because the row
    equilibration reduces cond(W·X) below cond(X) for matrices with outlier
    row norms.
    """
    def wgso(M):
        sq = (M * M).sum(dim=1)
        med = sq.median()
        w = 1.0 / (sq + eps_frac * med + 1e-30)
        return M * w.sqrt().unsqueeze(1)
    return ifkfac_natgrad(wgso(X), wgso(delta), grad_W, damping)


# ---- Driver ------------------------------------------------------------

METHODS = {
    "classic": classic_natgrad,
    "ifkfac":   ifkfac_natgrad,
    # WGSO is intentionally excluded: it computes a DIFFERENT natural
    # gradient (using a row-weighted matrix W·X in place of X), so it
    # cannot be compared against the un-WGSO ground truth in a slope test.
    # WGSO's contribution is to the kappa CONSTANT, not the kappa exponent;
    # it gets evaluated separately in the damping-sensitivity sweep (§5.5).
}

# Apply precision regime at this granularity: quantize the algorithm's INPUTS
# (X, delta, grad_W) to the target precision, then run the algorithm in
# fp64 storage (so cuSOLVER works) - the precision is carried by the data.
# This mirrors what kfac_bf16_compare.py does for bf16.

def run_one(method_name, regime, X, delta, grad_W, damping):
    Xq = quantize(X, regime)
    dq = quantize(delta, regime)
    gq = quantize(grad_W, regime)
    return METHODS[method_name](Xq, dq, gq, damping)


def fit_slope(log_kappa, log_err):
    """Return slope, intercept of log_err = slope * log_kappa + intercept.

    Filters out saturated points (err ≥ 0.5 at the head of the curve, or
    err < machine floor at the tail).  Returns NaN if too few points remain.
    """
    log_kappa = np.asarray(log_kappa); log_err = np.asarray(log_err)
    # Filter out points where err saturated (numerical floor or overflow)
    ok = np.isfinite(log_err) & (log_err < np.log10(0.5)) & (log_err > -16)
    if ok.sum() < 2:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(log_kappa[ok], log_err[ok], 1)
    return slope, intercept


def fit_slope_per_seed_with_ci(stats_dict, method, regime, kappas, seeds):
    """Fit one slope per seed (using the seed's own per-κ trajectory) and
    return (mean_slope, std_slope, 95%_ci_lo, 95%_ci_hi, all_per_seed_slopes).

    More honest than fitting a single slope on per-κ means, because it
    accounts for seed-to-seed variability in the slope estimate itself.
    """
    per_seed_slopes = []
    for s_idx, _ in enumerate(seeds):
        ers = []
        for k in kappas:
            per_seed = stats_dict[(method, regime, k)][2]
            ers.append(per_seed[s_idx])
        log_k = np.log10(kappas)
        log_e = np.log10(np.maximum(ers, 1e-20))
        slope, _ = fit_slope(log_k, log_e)
        if np.isfinite(slope):
            per_seed_slopes.append(slope)
    if not per_seed_slopes:
        return float('nan'), float('nan'), float('nan'), float('nan'), []
    arr = np.asarray(per_seed_slopes)
    return (float(arr.mean()), float(arr.std()),
             float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)),
             per_seed_slopes)


DISPLAY = {"classic": "Classic K-FAC", "ifkfac": "IFKFAC"}


def plot_from_json(json_path, out_png):
    """Draw the 4-panel kappa-scaling figure from the saved JSON (no recompute)."""
    d = json.loads(Path(json_path).read_text())
    regimes = ["fp64", "fp32", "bf16", "true_bf16"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), sharey=True)
    colors = {"classic": "#cc4444", "ifkfac": "#2266aa"}
    markers = {"classic": "o", "ifkfac": "s"}
    for ax, regime in zip(axes, regimes):
        per_method = {}
        for method in METHODS:
            data = d["means"][f"{method}_{regime}"]
            ks = [k for k, _ in data]
            ers = [e for _, e in data]
            stds = [float(np.std(d["per_seed"][f"{method}_{regime}_kappa{k:.0e}"])) for k in ks]
            slope, intercept = fit_slope(np.log10(ks), np.log10(ers))
            per_method[method] = (ks, ers, slope, intercept)
            ss = d.get("slope_summary", {}).get(f"{method}_{regime}")
            name = DISPLAY.get(method, method)
            if ss and ss.get("mean") is not None and np.isfinite(ss["mean"]):
                label = (f"{name}  slope = {ss['mean']:.2f} ± {ss['std']:.2f}"
                         f"  (theory: {ss['theory']:.0f})")
            elif method == "ifkfac" and regime == "fp64":
                label = f"{name}  (fp64 reference, error at numerical floor)"
            else:
                label = f"{name}  slope ≈ {slope:.2f}"
            ax.errorbar(ks, ers, yerr=stds, marker=markers[method], color=colors[method],
                        linewidth=1.7, markersize=8, label=label, zorder=3,
                        capsize=3, capthick=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        kappas_arr = np.array(per_method["ifkfac"][0])
        ks_v, ers_v, _, _ = per_method["ifkfac"]
        kref = np.array([kappas_arr[0], kappas_arr[-1]])
        ax.loglog(kref, ers_v[0] * (kref / ks_v[0]) ** 1.0, color="#2266aa", linestyle=":",
                  linewidth=1, alpha=0.5, zorder=1, label=r"$\kappa^1$ reference")
        ks_c, ers_c, _, _ = per_method["classic"]
        non_sat = [(k, e) for k, e in zip(ks_c, ers_c) if e < 0.5]
        if non_sat:
            k0, e0 = non_sat[0]
            ax.loglog(kref, e0 * (kref / k0) ** 2.0, color="#cc4444", linestyle=":",
                      linewidth=1, alpha=0.5, zorder=1, label=r"$\kappa^2$ reference")
        ax.set_xlabel(r"$\kappa(X)$")
        ax.set_title(f"precision = {regime}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(top=5.0)
    axes[0].set_ylabel("relative error of natural gradient")
    fig.suptitle(r"Numerical stability of K-FAC variants: $\|\hat g - g^*\| / \|g^*\|$ vs $\kappa(X)$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    p, n_in, n_out = 1024, 64, 64
    # Damping must be effectively zero so the kappa amplification is not
    # floored by the ridge.  Pure 0 risks exact singularities; 1e-14 is
    # safely below fp32 machine epsilon so it doesn't suppress amplification
    # while still keeping matrices invertible.
    damping = 1e-14
    # kappa range chosen so IFKFAC's linear regime is visible BEFORE its
    # bf16 saturation at err ≈ 1.0, and Classic's normal-equations
    # quadratic regime lands clearly in-range.  9 points for solid slope fit.
    kappas = [3.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 500.0, 1000.0]
    # Four precision regimes:
    #   fp64       — reference baseline (no quantization noise)
    #   fp32       — algorithmic-only verification
    #   bf16       — fp32-internal simulation (inputs bf16-quantized, math fp32)
    #   true_bf16  — hand-rolled bf16 linalg (storage AND arithmetic both bf16)
    # The contrast between "bf16" and "true_bf16" demonstrates that the κ²
    # algorithmic gap is observable at true bf16 but hidden by the
    # fp32-internal simulation.
    regimes = ["fp64", "fp32", "bf16", "true_bf16"]
    # 10 seeds for tighter error bars on the empirical slope estimates.
    seeds = list(range(10))

    # results[(method, regime)] = list of (kappa, mean_err) — for plotting
    results = {}
    # stats[(method, regime, kappa)] = (mean, std, list_of_per_seed)
    stats = {}

    print(f"\nrunning {len(seeds)} seeds × {len(kappas)} kappas × {len(METHODS)} methods × {len(regimes)} regimes")

    # Build per-seed truth cache (uses IFKFAC fp64 reference per seed)
    truth_cache = {}
    for seed in seeds:
        for kappa in kappas:
            X64 = make_data(p, n_in,  kappa, seed=seed, device=device)
            d64 = make_data(p, n_out, kappa, seed=seed+10000, device=device)
            g64 = make_grad(n_out, n_in, seed=seed, device=device)
            truth_cache[(kappa, seed)] = (X64, d64, g64,
                                           ifkfac_natgrad(X64, d64, g64, damping))

    print("\nmeasuring relative error per method × regime × kappa (averaged over seeds)...")
    for method in METHODS:
        for regime in regimes:
            mean_errs = []
            for kappa in kappas:
                per_seed = []
                for seed in seeds:
                    X64, d64, g64, truth = truth_cache[(kappa, seed)]
                    got = run_one(method, regime, X64, d64, g64, damping)
                    e = ((got - truth).norm() / truth.norm()).item()
                    per_seed.append(e)
                mu = float(np.mean(per_seed))
                sd = float(np.std(per_seed))
                stats[(method, regime, kappa)] = (mu, sd, per_seed)
                mean_errs.append((kappa, mu))
                print(f"  {method:>7} {regime} kappa={kappa:>8.0e}  "
                      f"mean={mu:.3e}  std={sd:.3e}  "
                      f"(min={min(per_seed):.3e}, max={max(per_seed):.3e})")
            results[(method, regime)] = mean_errs

    # Save raw — both means and full per-seed data
    json_out = {
        "means": {f"{m}_{r}": [[k, e] for k, e in v] for (m, r), v in results.items()},
        "per_seed": {
            f"{m}_{r}_kappa{k:.0e}": st[2]
            for (m, r, k), st in stats.items()
        },
        "seeds": list(seeds),
    }
    (out_dir / "kappa_scaling.json").write_text(json.dumps(json_out, indent=2))

    # ---- Per-seed slope summary for the paper ---------------------------
    print("\n" + "=" * 70)
    print("Slope summary  (slope ± std, 95% CI from per-seed fits)")
    print("=" * 70)
    slope_summary = {}
    print(f"  {'method':>9}  {'regime':>5}  "
          f"{'slope':>7}  {'± std':>7}  {'95% CI':>18}  {'theory':>8}")
    theoretical = {"classic": 2.0, "ifkfac": 1.0}
    for method in METHODS:
        for regime in regimes:
            mu, sd, ci_lo, ci_hi, all_s = fit_slope_per_seed_with_ci(
                stats, method, regime, kappas, seeds)
            slope_summary[(method, regime)] = {
                "mean": mu, "std": sd, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "per_seed": all_s, "theory": theoretical[method],
            }
            # Distinguish 'numerical floor' (IFKFAC vs its own fp64 reference)
            # from 'high-kappa saturation' (algorithmic error has hit ≥0.5).
            if np.isfinite(ci_lo):
                ci_str = f"[{ci_lo:>5.2f}, {ci_hi:>5.2f}]"
            elif method == "ifkfac" and regime == "fp64":
                ci_str = "(< numerical floor)"
            else:
                ci_str = "(high-κ saturated)"
            print(f"  {method:>9}  {regime:>5}  "
                  f"{mu:>7.2f}  {sd:>7.3f}  {ci_str:>18}  {theoretical[method]:>8.0f}")
    json_out["slope_summary"] = {
        f"{m}_{r}": v for (m, r), v in slope_summary.items()
    }
    (out_dir / "kappa_scaling.json").write_text(json.dumps(json_out, indent=2))

    out_png = plot_from_json(out_dir / "kappa_scaling.json", out_dir / "kappa_scaling.png")
    print(f"\nsaved {out_png}")
    print(f"saved {out_dir / 'kappa_scaling.json'}")


if __name__ == "__main__":
    if "--replot" in sys.argv:
        out = plot_from_json(ROOT / "benchmark" / "results" / "kappa_scaling.json",
                             ROOT / "benchmark" / "results" / "kappa_scaling.png")
        print(f"saved {out}")
    else:
        main()
