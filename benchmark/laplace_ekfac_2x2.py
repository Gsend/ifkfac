"""
benchmark/laplace_ekfac_2x2.py

§5.9 Laplace CIFAR-10 / ResNet-18: basis x eigenvalue 2x2 with EK-FAC.

Why
---
In exact arithmetic Classic K-FAC (eigh of the Gram) and IFKFAC (SVD of the
QR factor R) produce the SAME eigendecomposition.  Any difference between them
is numerical.  This experiment separates WHERE the numerical difference lives:

                      Kronecker eigenvalues        EK-FAC eigenvalues
                      S = d_G (x) d_A              S = E[(U_G^T dW U_A)^2]
    Classic basis     classic_kfe                  classic_ekfac   (= standard EK-FAC)
    IFKFAC basis      ifkfac_kfe                    ifkfac_ekfac

Both columns use the SAME exact damping in the Kronecker eigenbasis,
    posterior precision per coordinate = N * S + lambda,
so the only difference between columns is the eigenvalues, and the only
difference between rows is the basis.

Also produced, for continuity with the Daxberger form:
    classic, ifkfac  - standard K-FAC Laplace with factored damping
                      (sqrt(N) d_G + sqrt(lambda)) (sqrt(N) d_A + sqrt(lambda)).
    Note: the pre-existing grid-searched ifkfac_* JSONs were compared against
    fixed-lambda classic_* JSONs - a confound.  The default fixed-lambda run
    writes *_lam10.json files for every arm, so nothing is clobbered.  In
    --grid mode, fixed-lambda classic files are backed up as *_fixedlam10.json.

Protocol (EKFAC_LAPLACE_PLAN): FIXED lambda = 10 for every arm by default
(same as the original §5.8 protocol; isolates Fisher quality from lambda
selection).  --grid switches every arm to the same val-set grid search.
Common damping scheme for the 4-arm comparison: exact damping in the
Kronecker eigenbasis (N*S + lambda) - the only scheme EK-FAC admits.  The
factored-damping rows (classic, ifkfac) are kept for continuity with the
Daxberger form.  30-sample MC predictive on test, seeds {42,43,44}, fp32/bf16.
Wall-clock and peak GPU memory are logged per arm (EK-FAC's extra pass is
part of its cost).  Whether torch.linalg.eigh runs natively on bf16 is
recorded; the bf16 regime is bf16 storage + fp32 arithmetic.

Fisher scale (v2, 2026-09-23 - fixes the v1 smoke run)
-----------------------------------------------------
The shared K-FAC hooks average rows and use a MEAN-reduced loss, so their
G = E_rows[d d^T] carries 1/B^2, and for conv layers the row-average drops
the factor L = H_out*W_out that the per-sample Fisher has
(KFC, Grosse & Martens 2016: F ~= L * A (x) G with row-averaged factors).
The per-sample Fisher that the Laplace precision N*F + lambda*I needs is
therefore  F = c_l * (G (x) A)  with  c_l = B^2 * L_l.
v1 omitted c_l (B^2 = 16384, times L up to 1024): N * d_G stayed ~1e-4, far
below sqrt(lambda), so G never entered the factored posterior and every
exact-damping posterior was the prior (10% accuracy).  v2 multiplies d_G and
S_ekfac by c_l.  Batches are drop_last so B is exact.
v1 also left the IFKFAC training ridge (1e-8, R^T R = G + 1e-8 I) in the
Laplace capture; at the v1 scale it swamped G in most layers
(median kappa(G) = 1.0).  v2 captures IFKFAC with ridge 0, like Classic.
v1 IFKFAC also used the training default of 512 random conv patch rows per
batch for A (Classic uses all B*L rows, up to 256x more).  v2 uses all rows
for both, so the two bases are estimated from identical data.
v2 outputs carry the suffix _ps; v1 files are never read.

EK-FAC eigenvalue scale convention (matches the K-FAC hooks exactly)
-------------------------------------------------------------------
Hooks: A = mean over patch rows of a a^T;  G = mean over patch rows of d d^T,
where d is the output grad of the MEAN-reduced CE loss at batch size B.
Under the spatial-independence assumption,
    E_n[(u_G^T (sum_t d_nt a_nt^T) u_A)^2] = L * d_G * d_A,
so we define  S_ekfac = (1/L) * E_n[(U_G^T dW_n U_A)^2]  with the same loss,
labels-sampling and batch size.  S_ekfac == d_G (x) d_A when K-FAC's
assumptions hold exactly.  The per-layer ratio sum(S_ekfac)/sum(S_kron) is
logged as a sanity check (expect O(1)).

Memory: per-sample projected grads are computed layer by layer inside the
backward hook, in chunks of EKFAC_CHUNK samples, squared, accumulated, freed.
Peak extra ~0.3 GB (512 x 4608 layers at chunk 32).  Fits 16 GB VRAM.

Outputs (benchmark/results/):
    laplace_cifar10_{method}_{precision}_seed{seed}_lam10_ps.json   (fixed lambda, default)
    laplace_cifar10_{method}_{precision}_seed{seed}_ps.json         (--grid)
    method in {classic, ifkfac, classic_kfe, ifkfac_kfe, classic_ekfac, ifkfac_ekfac}

Run:
    python -m benchmark.laplace_ekfac_2x2 --smoke --grid   # seed 42, fp32, ~40 min
    python -m benchmark.laplace_ekfac_2x2 --smoke     # seed 42, fp32, ~35 min
    python -m benchmark.laplace_ekfac_2x2             # full, fixed lambda=10, ~3.5-4 h (RTX 3080 Laptop)
    python -m benchmark.laplace_ekfac_2x2 --grid      # every arm grid-searched (secondary table)
Resumable: finished (method, precision, seed) JSONs are skipped.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import laplace_cifar10 as L
from benchmark.metrics_calibration import (
    all_metrics, reliability_diagram_data, negative_log_likelihood,
)
from optimizer.laplace import LaplacePosterior

warnings.filterwarnings("ignore", message=".*Full backward hook.*")

OUT = ROOT / "benchmark" / "results"
SEEDS = [42, 43, 44]
PRECISIONS = ["fp32", "bf16"]
BASES = ["classic", "ifkfac"]
EKFAC_CHUNK = 32          # samples per einsum chunk inside the backward hook
FIXED_LAMBDA = 10.0       # None -> grid search (set by --grid)
EIGH_BF16_NATIVE = "not probed"
WITH_FACTORED = True      # also evaluate the standard factored-damping posteriors
FISHER_TAG = "_ps"        # v2: per-sample Fisher scale (c_l = B^2 * L_l), ridge-0 IFKFAC capture
# Wider than L.LAMBDA_GRID: exact damping leaves prior-dominated KFE coordinates at
# variance 1/lambda, so the selected prior precision can sit above 1e5.
LAMBDA_GRID = [1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


# -----------------------------------------------------------------------------
# Posterior with exact damping in the Kronecker eigenbasis
# -----------------------------------------------------------------------------
class KFEPosterior(LaplacePosterior):
    """Diagonal-in-KFE Laplace posterior.

    Per layer: W = W_map + U_G (Xi * 1/sqrt(N*S + lambda)) U_A^T,
    with S of shape (n_out, n_in).  S = d_G (x) d_A gives Kronecker
    eigenvalues with exact damping; S = EK-FAC second moments gives EK-FAC.
    Reuses LaplacePosterior's predictive_full_loader / restore_map.
    """

    def __init__(self, model, kfe, prior_precision, dataset_size):
        self.model = model
        self.prior_precision = prior_precision
        self.dataset_size = dataset_size
        self._w_map = {mod: mod.weight.data.clone() for mod in kfe}
        self._kfe = {}
        for mod, (U_A, U_G, S) in kfe.items():
            scale = 1.0 / torch.sqrt(float(dataset_size) * S.clamp(min=0.0)
                                     + float(prior_precision))
            self._kfe[mod] = (U_A.float(), U_G.float(), scale.float())
        self._factors = self._kfe          # restore_map iterates these keys

    @torch.no_grad()
    def sample_in_place(self, generator=None):
        for mod, (U_A, U_G, scale) in self._kfe.items():
            mod.weight.data.copy_(self._w_map[mod])
            xi = torch.randn(U_G.shape[0], U_A.shape[0], device=mod.weight.device,
                             dtype=torch.float32, generator=generator)
            delta = U_G @ (xi * scale) @ U_A.t()
            mod.weight.data.add_(delta.reshape(mod.weight.shape).to(mod.weight.dtype))


# -----------------------------------------------------------------------------
# EK-FAC second moments in a given eigenbasis
# -----------------------------------------------------------------------------
class EKFACProjector:
    """Accumulate S = (1/L) * E_n[(U_G^T dW_n U_A)^2] per layer via hooks."""

    def __init__(self, bases):
        self.bases = bases                     # {mod: (U_A, U_G)} fp32 on device
        self.S = {m: torch.zeros(U_G.shape[0], U_A.shape[0], device=U_A.device)
                  for m, (U_A, U_G) in bases.items()}
        self.n = {m: 0 for m in bases}
        self._inp = {}
        self._handles = []
        for m in bases:
            self._handles.append(m.register_forward_hook(self._fwd))
            self._handles.append(m.register_full_backward_hook(self._bwd))

    def _fwd(self, mod, inp, out):
        self._inp[mod] = inp[0].detach()

    @torch.no_grad()
    def _bwd(self, mod, grad_in, grad_out):
        x = self._inp.pop(mod, None)
        if x is None:
            return
        d = grad_out[0].detach()
        U_A, U_G = self.bases[mod]
        B = x.shape[0]
        for s in range(0, B, EKFAC_CHUNK):
            xs, ds = x[s:s + EKFAC_CHUNK].float(), d[s:s + EKFAC_CHUNK].float()
            if isinstance(mod, nn.Conv2d):
                a = F.unfold(xs, kernel_size=mod.kernel_size, dilation=mod.dilation,
                             padding=mod.padding, stride=mod.stride).transpose(1, 2)
                g = ds.flatten(2).transpose(1, 2)            # (b, L, n_out)
            else:
                a = xs.reshape(xs.shape[0], -1, xs.shape[-1])  # (b, L, n_in)
                g = ds.reshape(ds.shape[0], -1, ds.shape[-1])  # (b, L, n_out)
            Lsp = a.shape[1]
            proj = torch.einsum("blo,bli->boi", g @ U_G, a @ U_A)   # (b, n_out, n_in)
            self.S[mod].add_(proj.pow_(2).sum(0), alpha=1.0 / Lsp)
            del a, g, proj
        self.n[mod] += B

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._inp.clear()

    def result(self):
        return {m: self.S[m] / max(self.n[m], 1) for m in self.S}


def compute_ekfac_S(model, bases, train, device, seed):
    """One extra pass: MC-sampled labels, mean-reduced CE, batch size = L.BATCH_SIZE
    (must match the Fisher capture so the 1/B scaling of grads matches G)."""
    torch.manual_seed(seed + 1000); torch.cuda.manual_seed_all(seed + 1000)
    loader = torch.utils.data.DataLoader(train, batch_size=L.BATCH_SIZE, shuffle=True,
                                         num_workers=0, pin_memory=True, drop_last=True)
    model.eval()
    proj = EKFACProjector(bases)
    try:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            with torch.no_grad():
                y = torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1)
            loss = F.cross_entropy(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
    finally:
        proj.remove()
    model.zero_grad(set_to_none=True)
    return proj.result()


@torch.no_grad()
def spatial_sizes(model, mods, loader, device):
    """L_l = number of output positions per sample (H_out*W_out for Conv2d, 1 for Linear)."""
    sizes, handles = {}, []

    def hook(mod, inp, out):
        sizes[mod] = int(out.shape[-2] * out.shape[-1]) if isinstance(mod, nn.Conv2d) else \
            int(out[0].numel() // out.shape[-1])
    for m in mods:
        handles.append(m.register_forward_hook(hook))
    try:
        x, _ = next(iter(loader))
        model.eval()
        model(x[:2].to(device))
    finally:
        for h in handles:
            h.remove()
    return sizes


def per_sample_scale(model, factors, loader, device):
    """c_l = B^2 * L_l: hook factors (mean-reduced loss, row-averaged) -> per-sample Fisher."""
    Lsp = spatial_sizes(model, list(factors), loader, device)
    return {m: float(L.BATCH_SIZE) ** 2 * Lsp[m] for m in factors}, Lsp


def frac_data_dominated(S_by_layer, lam):
    """Fraction of KFE coordinates where the likelihood term N*S exceeds the prior lambda."""
    num = sum(int((float(L.N_TRAIN) * S > lam).sum()) for S in S_by_layer)
    den = sum(S.numel() for S in S_by_layer)
    return num / max(den, 1)


# -----------------------------------------------------------------------------
# Generic grid search + test eval
# -----------------------------------------------------------------------------
def peak_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()


def peak_gb():
    if torch.cuda.is_available():
        torch.cuda.synchronize(); return torch.cuda.max_memory_allocated() / 2**30
    return float("nan")


def grid_and_test(make_post, model, val_loader, test_loader, device, seed):
    trace, best_lam, best_nll = [], None, float("inf")
    t0 = time.perf_counter()
    grid = [] if FIXED_LAMBDA is not None else LAMBDA_GRID
    if FIXED_LAMBDA is not None:
        best_lam, best_nll = FIXED_LAMBDA, float("nan")
        trace = [{"lambda": FIXED_LAMBDA, "val_nll": float("nan"), "fixed": True}]
    for lam in grid:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)   # common random numbers
        post = make_post(lam)
        probs, labels = post.predictive_full_loader(val_loader, device,
                                                    n_samples=L.GRID_SEARCH_SAMPLES)
        nll = negative_log_likelihood(probs, labels)
        trace.append({"lambda": lam, "val_nll": nll})
        mark = "  <- best" if nll < best_nll else ""
        print(f"      lambda={lam:>8.3g}  val_nll={nll:.4f}{mark}", flush=True)
        if nll < best_nll:
            best_nll, best_lam = nll, lam
        del post
    t_grid = time.perf_counter() - t0
    if FIXED_LAMBDA is None and best_lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1]):
        print(f"      WARNING: chosen lambda={best_lam:g} is at the grid edge", flush=True)

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    post = make_post(best_lam)
    t0 = time.perf_counter()
    probs, labels = post.predictive_full_loader(test_loader, device, n_samples=L.LAPLACE_SAMPLES)
    t_test = time.perf_counter() - t0
    post.restore_map()
    metrics = all_metrics(probs, labels)
    rel = {k: v.tolist() if hasattr(v, "tolist") else v
           for k, v in reliability_diagram_data(probs, labels).items()}
    del post
    return best_lam, best_nll, trace, metrics, rel, t_grid, t_test


def tag():
    return ("" if FIXED_LAMBDA is None else f"_lam{FIXED_LAMBDA:g}") + FISHER_TAG


def out_path(method, precision, seed):
    return OUT / f"laplace_cifar10_{method}_{precision}_seed{seed}{tag()}.json"


def is_done(method, precision, seed):
    p = out_path(method, precision, seed)
    if not p.exists():
        return False
    if FIXED_LAMBDA is not None:
        return json.loads(p.read_text()).get("fisher_scale") == "per_sample"
    d = json.loads(p.read_text())
    if d.get("fisher_scale") != "per_sample":
        return False
    return float(d.get("wall_s_grid", 0.0)) > 0.0      # grid mode: fixed-lambda files don't count


def backup_fixed_lambda(method, precision, seed):
    if FIXED_LAMBDA is not None:
        return                                          # suffixed names never clobber
    p = out_path(method, precision, seed)
    if p.exists():
        d = json.loads(p.read_text())
        if float(d.get("wall_s_grid", 0.0)) == 0.0:
            bak = p.with_name(p.stem + "_fixedlam10.json")
            if not bak.exists():
                p.rename(bak)
                print(f"  [backup] {p.name} -> {bak.name}", flush=True)


# -----------------------------------------------------------------------------
# One (seed, precision, basis) block: capture once, evaluate up to 3 posteriors
# -----------------------------------------------------------------------------
def run_block(basis, precision, seed, model, train, val, test, device, force):
    methods = {"kfe": f"{basis}_kfe", "ekfac": f"{basis}_ekfac"}
    if WITH_FACTORED:
        methods = {"std": basis, **methods}
    todo = {k: m for k, m in methods.items() if force or not is_done(m, precision, seed)}
    if not todo:
        print(f"  [skip] {basis}/{precision}/seed{seed}: all 3 posteriors done", flush=True)
        return
    if "std" in todo:
        backup_fixed_lambda(methods["std"], precision, seed)

    print(f"\n=== Block: basis={basis}  precision={precision}  seed={seed}  "
          f"posteriors={list(todo.values())} ===", flush=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    if precision == "bf16":
        from benchmark.kfac_bf16_compare import (
            enable_bf16, enable_classic_bf16, disable_bf16, disable_classic_bf16)
        (enable_bf16(wgso=False) if basis == "ifkfac" else enable_classic_bf16())

    loaders = {
        "train": torch.utils.data.DataLoader(train, batch_size=L.BATCH_SIZE, shuffle=True,
                                             num_workers=0, pin_memory=True, drop_last=True),
        "val": torch.utils.data.DataLoader(val, batch_size=L.BATCH_SIZE, shuffle=False,
                                           num_workers=0, pin_memory=True),
        "test": torch.utils.data.DataLoader(test, batch_size=L.BATCH_SIZE, shuffle=False,
                                            num_workers=0, pin_memory=True),
    }
    opt = None
    try:
        # Phase 2: Fisher capture with the basis's own path (eigh of Gram / SVD of R)
        opt = L.build_kfac(basis, model, precision=precision)
        # v2: IFKFAC capture on the same data as Classic's Gram path:
        #   ridge 0 (Classic adds none), all conv patch rows (the training default
        #   subsamples 512 rows/batch; Classic uses all B*L), streaming (not
        #   deferred) TSQR so the uncapped chunks are not buffered.
        capture_ridge, capture_rows = None, "all"
        hk = getattr(opt, "hooks", None)
        if hasattr(hk, "damping"):
            hk.damping = 0.0
            capture_ridge = 0.0
        if hasattr(hk, "max_conv_rows"):
            hk.max_conv_rows = 0
        if hasattr(hk, "deferred"):
            hk.deferred = False
        peak_reset()
        t0 = time.perf_counter()
        factors = L.capture_fisher(model, opt, loaders["train"], device,
                                   n_passes=L.FISHER_PASSES, precision=precision)
        t_fisher = time.perf_counter() - t0
        mem_fisher = peak_gb()
        try:
            opt.hooks.remove()
        except Exception as e:
            print(f"  hook detach failed: {e}", flush=True)
        print(f"  Fisher capture: {t_fisher/60:.1f} min ({len(factors)} layers)"
              f"   capture ridge: {capture_ridge}  conv rows: {capture_rows}", flush=True)

        # v2: hook factors -> per-sample Fisher scale (see module docstring)
        c_scale, Lsp = per_sample_scale(model, factors, loaders["train"], device)
        factors = {m: (U_A, d_A, U_G, d_G * c_scale[m])
                   for m, (U_A, d_A, U_G, d_G) in factors.items()}
        kron = {m: (U_A, U_G, d_G[:, None] * d_A[None, :])
                for m, (U_A, d_A, U_G, d_G) in factors.items()}
        nS_max = max(float(L.N_TRAIN * k[2].max()) for k in kron.values())
        print(f"  per-sample scale c_l = B^2*L: {min(c_scale.values()):.3g}..{max(c_scale.values()):.3g}"
              f"   d_G(scaled) in [{min(float(f[3].min()) for f in factors.values()):.2e}, "
              f"{max(float(f[3].max()) for f in factors.values()):.2e}]   max N*S_kron = {nS_max:.3g}",
              flush=True)

        ekfac, t_ekfac, ratio_stats, mem_ekfac = None, 0.0, None, float("nan")
        if "ekfac" in todo:
            bases = {m: (U_A.float(), U_G.float()) for m, (U_A, _, U_G, _) in factors.items()}
            peak_reset()
            t0 = time.perf_counter()
            S = compute_ekfac_S(model, bases, train, device, seed)
            t_ekfac = time.perf_counter() - t0
            mem_ekfac = peak_gb()
            S = {m: s * c_scale[m] for m, s in S.items()}          # v2 per-sample scale
            if precision == "bf16":
                S = {m: s.to(torch.bfloat16).to(torch.float32) for m, s in S.items()}
            ratios = [float(S[m].sum() / kron[m][2].clamp(min=0).sum().clamp(min=1e-30))
                      for m in factors]
            ratio_stats = {"min": min(ratios), "median": sorted(ratios)[len(ratios) // 2],
                           "max": max(ratios), "per_layer": ratios}
            print(f"  EK-FAC pass: {t_ekfac/60:.1f} min   sum(S_ekfac)/sum(S_kron) per layer: "
                  f"min={ratio_stats['min']:.3g} med={ratio_stats['median']:.3g} "
                  f"max={ratio_stats['max']:.3g}", flush=True)
            ekfac = {m: (factors[m][0], factors[m][2], S[m]) for m in factors}

        makers = {
            "std":   lambda lam: LaplacePosterior(model, factors, prior_precision=lam,
                                                  dataset_size=L.N_TRAIN),
            "kfe":   lambda lam: KFEPosterior(model, kron, lam, L.N_TRAIN),
            "ekfac": lambda lam: KFEPosterior(model, ekfac, lam, L.N_TRAIN),
        }
        meta = {
            "std":   {"posterior": "factored_damping", "eigenvalues": "kron"},
            "kfe":   {"posterior": "kfe_exact_damping", "eigenvalues": "kron"},
            "ekfac": {"posterior": "kfe_exact_damping", "eigenvalues": "ekfac"},
        }
        for key, method in todo.items():
            print(f"  --- {method}/{precision}/seed{seed}: "
                  f"{'grid search' if FIXED_LAMBDA is None else f'fixed lambda={FIXED_LAMBDA:g}'} ---",
                  flush=True)
            peak_reset()
            lam, vnll, trace, met, rel, t_grid, t_test = grid_and_test(
                makers[key], model, loaders["val"], loaders["test"], device, seed)
            mem_eval = peak_gb()
            S_arm = [v[2] for v in (ekfac if key == "ekfac" else kron).values()]
            fdd = frac_data_dominated(S_arm, lam)
            print(f"  RESULT {method}/{precision}/seed{seed}: lambda*={lam:g}  "
                  f"acc={met['accuracy']*100:.2f}%  NLL={met['nll']:.4f}  "
                  f"ECE={met['ece']*100:.2f}%  Brier={met['brier']:.4f}  "
                  f"frac(N*S>lambda)={fdd:.3g}", flush=True)
            out = {
                "method": method, "basis": basis, "precision": precision, "seed": seed,
                **meta[key],
                "chosen_lambda": lam, "best_val_nll": vnll, "lambda_trace": trace,
                "metrics": met, "reliability": rel,
                "kfac_damping_train": L.KFAC_DAMPING_TRAIN,
                "ekfac_scale_ratio": ratio_stats if key == "ekfac" else None,
                "wall_s_train": 0.0, "wall_s_fisher": t_fisher,
                "wall_s_ekfac_pass": t_ekfac if key == "ekfac" else 0.0,
                "wall_s_grid": t_grid, "wall_s_test": t_test,
                "protocol": "fixed_lambda" if FIXED_LAMBDA is not None else "grid",
                "peak_mem_gb": {"fisher_capture": mem_fisher,
                                "ekfac_pass": mem_ekfac if key == "ekfac" else None,
                                "posterior_eval": mem_eval},
                "eigh_bf16_native": EIGH_BF16_NATIVE,
                "fisher_scale": "per_sample",
                "fisher_scale_c": [c_scale[m] for m in factors],
                "spatial_L": [Lsp[m] for m in factors],
                "laplace_capture_ridge": capture_ridge,
                "laplace_capture_conv_rows": capture_rows,
                "frac_data_dominated": fdd,
                "lambda_grid": None if FIXED_LAMBDA is not None else LAMBDA_GRID,
            }
            out_path(method, precision, seed).write_text(json.dumps(out, indent=2, default=str))
            print(f"  saved {out_path(method, precision, seed).name}", flush=True)
    finally:
        if precision == "bf16":
            (disable_bf16() if basis == "ifkfac" else disable_classic_bf16())
        try:
            if opt is not None:
                opt.cleanup()
        except Exception:
            pass
        del opt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def load_ok(m, prec, s):
    p = out_path(m, prec, s)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if FIXED_LAMBDA is None and float(d.get("wall_s_grid", 0)) == 0:
        return None
    return d


def paired_signs(ref, arm, prec, seeds):
    """Per metric: (#seeds where arm beats ref, #paired seeds)."""
    better = {"accuracy": 1, "nll": -1, "ece": -1, "brier": -1}
    out = {}
    for k, sgn in better.items():
        wins, n = 0, 0
        for s in seeds:
            a, r = load_ok(arm, prec, s), load_ok(ref, prec, s)
            if a is None or r is None:
                continue
            n += 1
            wins += int(sgn * (a["metrics"][k] - r["metrics"][k]) > 0)
        out[k] = (wins, n)
    return out


def summarize(seeds):
    rows = ["classic", "ifkfac", "classic_kfe", "ifkfac_kfe", "classic_ekfac", "ifkfac_ekfac"]
    print("\n=== §5.9 Laplace 2x2 summary (mean ± std over seeds) ===", flush=True)
    print(f"  {'method':<14} {'prec':<5} {'n':>2}  {'acc%':>13}  {'NLL':>15}  "
          f"{'ECE%':>11}  {'Brier':>15}  lambda*", flush=True)
    for prec in PRECISIONS:
        for m in rows:
            ds = [d for d in (load_ok(m, prec, s) for s in seeds) if d is not None]
            if not ds:
                continue
            def ms(key, k=1.0):
                v = [d["metrics"][key] * k for d in ds]
                return f"{mean(v):.4f} ± {stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
            lams = sorted({d["chosen_lambda"] for d in ds})
            print(f"  {m:<14} {prec:<5} {len(ds):>2}  {ms('accuracy', 100):>13}  "
                  f"{ms('nll'):>15}  {ms('ece', 100):>11}  {ms('brier'):>15}  {lams}",
                  flush=True)
    print("\n  Paired vs classic_kfe (arm 1 under the common exact-damping scheme): "
          "wins/n per metric", flush=True)
    for prec in PRECISIONS:
        for m in ["ifkfac_kfe", "classic_ekfac", "ifkfac_ekfac"]:
            ps = paired_signs("classic_kfe", m, prec, seeds)
            if all(n == 0 for _, n in ps.values()):
                continue
            print(f"    {m:<14} {prec:<5} " + "  ".join(f"{k}={w}/{n}" for k, (w, n) in ps.items()),
                  flush=True)
    print("\n  Per-arm cost (mean over seeds): fisher capture / EK-FAC pass / eval wall (min), "
          "peak GB", flush=True)
    for prec in PRECISIONS:
        for m in rows:
            ds = [d for d in (load_ok(m, prec, s) for s in seeds) if d is not None]
            if not ds:
                continue
            w = lambda k: mean(d.get(k, 0.0) for d in ds) / 60
            pk = max((v for d in ds for v in (d.get("peak_mem_gb") or {}).values()
                      if isinstance(v, (int, float)) and not math.isnan(v)), default=float("nan"))
            print(f"    {m:<14} {prec:<5} {w('wall_s_fisher'):5.1f} / {w('wall_s_ekfac_pass'):5.1f}"
                  f" / {w('wall_s_grid') + w('wall_s_test'):5.1f}   peak {pk:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="seed 42, fp32 only")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--precisions", nargs="+", default=PRECISIONS, choices=PRECISIONS)
    ap.add_argument("--bases", nargs="+", default=BASES, choices=BASES)
    ap.add_argument("--force", action="store_true", help="recompute even if JSON exists")
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0,
                    help="fixed prior precision for every arm (default 10, the §5.8 protocol)")
    ap.add_argument("--grid", action="store_true", help="grid-search lambda for every arm instead")
    ap.add_argument("--no-factored", action="store_true",
                    help="skip the factored-damping (Daxberger-form) classic/ifkfac rows")
    args = ap.parse_args()
    global FIXED_LAMBDA, WITH_FACTORED, EIGH_BF16_NATIVE
    FIXED_LAMBDA = None if args.grid else args.lam
    WITH_FACTORED = not args.no_factored
    seeds = [42] if args.smoke else args.seeds
    precs = ["fp32"] if args.smoke else args.precisions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    try:
        torch.linalg.eigh(torch.eye(4, device=device, dtype=torch.bfloat16))
        EIGH_BF16_NATIVE = "ok"
    except Exception as e:
        EIGH_BF16_NATIVE = f"unsupported: {type(e).__name__}: {str(e)[:100]}"
    print(f"native bf16 eigh on {device}: {EIGH_BF16_NATIVE}", flush=True)
    print(f"protocol: {'grid search' if FIXED_LAMBDA is None else f'fixed lambda={FIXED_LAMBDA:g}'}"
          f"; factored rows: {WITH_FACTORED}", flush=True)
    train, val, test = L.get_cifar10(device)

    for seed in seeds:
        print(f"\n=========== Seed {seed} ===========", flush=True)
        model, _ = L.train_or_load_map(seed, train, device)
        for prec in precs:
            for basis in args.bases:
                run_block(basis, prec, seed, model, train, val, test, device, args.force)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summarize(seeds)


if __name__ == "__main__":
    main()
