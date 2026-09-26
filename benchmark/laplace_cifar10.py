"""
benchmark/laplace_cifar10.py

Laplace approximation experiment for §5.8.  Trains ResNet-18 on CIFAR-10
with IFKFAC, captures the Fisher at end of training, builds the
Laplace posterior, then evaluates calibration metrics for four cells:

    (Classic | IFKFAC)  ×  (fp32 | bf16)

Pipeline per cell:
  1. Train ResNet-18 to convergence with the cell's optimizer + precision
  2. Capture K-FAC Fisher at MAP via one extra pass over the training data
  3. Grid-search Tikhonov damping λ on a held-out validation set (NLL)
  4. Build LaplacePosterior at the chosen λ
  5. Evaluate test-set metrics: accuracy / NLL / ECE / Brier / predictive entropy

Output per cell: JSON with all metrics + chosen λ + grid-search trace.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.metrics_calibration import all_metrics, reliability_diagram_data


# ---- Config -----------------------------------------------------------------

CELLS = [
    ("classic", "fp32"),
    ("classic", "bf16"),
    ("ifkfac",   "fp32"),
    ("ifkfac",   "bf16"),
]
SEEDS = [42, 43, 44]              # multi-seed for §5.8 paper claim
TRAIN_EPOCHS  = 20
FISHER_PASSES = 1                 # one extra pass over train_loader for Fisher
LAPLACE_SAMPLES = 30
# Lambda grid spans a wide range; with proper N-scaling on the Fisher,
# the optimum typically lands around 1-1000 for CIFAR-10/ResNet (Daxberger 2021).
LAMBDA_GRID = [1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
BATCH_SIZE = 128
N_TRAIN    = 45000        # CIFAR-10 train size after 5k val split

# When set (via --lambda CLI flag), skip grid search and use this fixed
# value for ALL cells.  Useful for the K-FAC × precision comparison: if
# the chosen λ is in the regime where the Fisher matters, Classic bf16's
# corrupted Fisher should fail visibly while IFKFAC bf16 still works.
FIXED_LAMBDA: float = None

KFAC_LR              = 1e-2
# CONTROL EXPERIMENT (task #90): IFKFAC's hook stores R such that
# R^T R = A + λ_train · I (training damping baked into the factor).  Classic
# K-FAC stores A directly.  This asymmetry means d_A_ifkfac = d_A_true + λ_train,
# adding a small effective damping on the smallest-eigenvalue axes that may
# explain part of the §5.8 NLL gap independent of the κ²/κ mechanism.
#
# Setting λ_train ≈ 0 here so IFKFAC's R ≈ QR(X) — disentangles the precision-
# stability effect from the damping-bias effect.  Was 1e-3 for the original
# §5.8 sweep (recorded in memory the project notes).
KFAC_DAMPING_TRAIN   = 1e-8       # Tikhonov used during TRAINING (not Laplace)
KFAC_MOMENTUM        = 0.9
KFAC_GAMMA           = 0.9
KFAC_FREQ            = 10
GRAD_CLIP            = 100.0
KFAC_MAX_DIM         = 4096


# ---- Model: ResNet-18 for CIFAR-10 -----------------------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, out_c, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)


# ---- Data: CIFAR-10 ---------------------------------------------------------

def get_cifar10(device):
    from torchvision import datasets, transforms
    data_root = ROOT / "data" / "cifar10"
    data_root.mkdir(parents=True, exist_ok=True)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    # Try with download=True first, fall back to manual placement if blocked.
    # If the http download hangs, place cifar-10-python.tar.gz at data_root by hand
    # and set CIFAR10_NO_DOWNLOAD=1 in the environment to force download=False.
    import os
    auto_dl = (os.environ.get("CIFAR10_NO_DOWNLOAD", "").lower() not in ("1", "true", "yes"))
    try:
        train_full = datasets.CIFAR10(str(data_root), train=True,  download=auto_dl, transform=train_tf)
        test       = datasets.CIFAR10(str(data_root), train=False, download=auto_dl, transform=test_tf)
    except Exception as e:
        # If automatic download failed, give the user a clear next-step message.
        if auto_dl:
            print(f"\n[get_cifar10] download failed: {e}", flush=True)
            print(f"[get_cifar10] Manual fix: download cifar-10-python.tar.gz to "
                  f"{data_root}\\cifar-10-python.tar.gz, then re-run with "
                  f"$env:CIFAR10_NO_DOWNLOAD='1' to skip the download attempt.",
                  flush=True)
        raise
    # Split 45k train / 5k val for damping selection
    train, val = torch.utils.data.random_split(
        train_full, [45000, 5000], generator=torch.Generator().manual_seed(0),
    )
    return train, val, test


# ---- Optimizer factory ------------------------------------------------------

def build_kfac(method, model, precision="fp32"):
    """K-FAC for Phase 2 Fisher capture only.  Phase 1 MAP training uses
    AdamW (standard BDL practice — Daxberger 2021).  K-FAC on Conv layers
    is unstable as a primary optimizer and the literature doesn't use it
    that way for Laplace experiments.

    For bf16 cells:
      - IFKFAC: use_true_bf16=True stores R factors as bf16 (κ¹ noise regime).
      - Classic: bf16 quantization of the Gram matrices is applied in
        capture_fisher() — the Cholesky inversion path amplifies the bf16
        noise quadratically in κ (the central claim).
    """
    if method == "classic":
        from optimizer.classic_kfac import ClassicKFAC
        return ClassicKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_TRAIN,
            factor_update_freq=KFAC_FREQ, decomp_update_freq=KFAC_FREQ,
            weight_decay=5e-4, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
    if method == "ifkfac":
        from optimizer.ifkfac_kfac import IFKFAC
        return IFKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_TRAIN,
            factor_update_freq=KFAC_FREQ, weight_decay=5e-4,
            momentum=KFAC_MOMENTUM, grad_clip=GRAD_CLIP,
            gamma=KFAC_GAMMA, max_out_dim=KFAC_MAX_DIM,
            deferred_qr=True,
            use_true_bf16=(precision == "bf16"),  # κ¹ bf16 storage path
        )
    if method == "asdl_classic":
        # ASDL reference Classic K-FAC as the κ² baseline.  It forms the Gram
        # factors A = E[xxᵀ] and B = E[δδᵀ] explicitly (just like our homebrew
        # ClassicKFAC), so its captured factors collapse at bf16 the same way.
        # Capture always runs in fp32; the bf16 precision regime is applied
        # later in capture_fisher() by quantising A/B before eigh — identical
        # to the homebrew classic path.  curvature_upd_interval=1 so EVERY
        # capture batch contributes to the Fisher estimate.
        from optimizer.asdl_classic_kfac import AsdlClassicKFAC
        return AsdlClassicKFAC(
            model, lr=KFAC_LR, damping=KFAC_DAMPING_TRAIN,
            factor_update_freq=1, momentum=KFAC_MOMENTUM,
            grad_clip=GRAD_CLIP, gamma=KFAC_GAMMA,
            use_bf16_factors=False,
        )
    raise ValueError(method)


def build_adamw_for_map(model):
    """AdamW for Phase 1 MAP training.  Standard BDL hyperparameters
    (Daxberger 2021 §B): lr=1e-3, weight_decay=5e-4, cosine schedule."""
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)


# ---- Training (Phase 1) -----------------------------------------------------

def train_to_map(model, opt, train_loader, device, epochs):
    model.train()
    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            loss = F.cross_entropy(model(x), y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:>2d}/{epochs}: loss={loss.item():.3e}", flush=True)


# ---- Fisher capture (Phase 2) -----------------------------------------------

def capture_fisher(model, opt, train_loader, device, n_passes=1, precision="fp32"):
    """Capture per-layer K-FAC Laplace factors in Daxberger 2021 EVD form.

    Both paths produce eigendecompositions (U_A, d_A, U_G, d_G) per layer
    such that A = U_A diag(d_A) U_A^T and G = U_G diag(d_G) U_G^T.  But the
    paths differ structurally:

    Classic K-FAC (EKFAC / Daxberger 2021 §4):
        1. Form A = E[xx^T] and G = E[δδ^T] via K-FAC hooks (cuSOLVER eigh).
        2. At bf16, quantize A and G — bf16 noise on the GRAM has effective
           conditioning κ²(X) due to the squaring in A = X^T X.
        3. eigh(A) and eigh(G) — at bf16 the smallest eigenvalues can go
           negative (the κ² catastrophe).

    IFKFAC (κ¹-stable path, our contribution for §5.8):
        1. Form R via QR(X) — K-FAC hook output is the upper-triangular R.
        2. At bf16, R inherits only κ¹(X) quantization noise.
        3. svd(R) → singular values S and right singular vectors V.
        4. Eigenvalues of A: d_A = S²;  eigenvectors of A: U_A = V.
           Never form A = X^T X — preserves κ¹ stability.
    """
    # ASDL reference Classic K-FAC stores curvature internally (no `hooks`
    # attribute); dispatch to the dedicated extractor.
    from optimizer.asdl_classic_kfac import AsdlClassicKFAC
    if isinstance(opt, AsdlClassicKFAC):
        return _capture_fisher_asdl(model, opt, train_loader, device,
                                    n_passes=n_passes, precision=precision)

    model.eval()
    opt.hooks.clear()
    # Daxberger 2021 §3.1: capture the TRUE Fisher (gradient of log-likelihood
    # w.r.t. MODEL-SAMPLED labels), not the empirical Fisher (gradient w.r.t.
    # TRUE labels).  At MAP convergence, empirical gradients δ = softmax − y_true
    # collapse to ≈ 0 (the model fits the training labels), making G essentially
    # zero across all layers — the posterior then degenerates to the prior 1/λ.
    # MC-sampled labels keep δ non-degenerate even at convergence.
    for _ in range(n_passes):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1)
                y_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            loss = F.cross_entropy(logits, y_sampled)
            model.zero_grad(set_to_none=True)
            loss.backward()

    if not (hasattr(opt, "hooks") and hasattr(opt.hooks, "get_factors")):
        raise RuntimeError("Optimizer doesn't expose hooks.get_factors()")
    factors_raw = opt.hooks.get_factors()

    from optimizer.raw_activation_hooks import RawActivationHooks
    factors_evd = {}

    # Diagnostic: track eigenvalue stats across all layers — answers
    # "what is the actual conditioning of ResNet-18's K-FAC factors?"
    eig_stats = {"d_A_min": [], "d_A_max": [], "d_G_min": [], "d_G_max": [],
                 "n_neg_A": 0, "n_neg_G": 0, "n_layers": 0}

    if isinstance(opt.hooks, RawActivationHooks):
        # ---- IFKFAC path: SVD of R (κ¹ stable) ----
        for mod, (R_A, R_G) in factors_raw.items():
            # Quantize is already applied via use_true_bf16=True in build_kfac.
            # Compute SVD on the κ¹-stable R factors directly — never form
            # X^T X, so the κ² catastrophe doesn't manifest.
            _, S_A, Vh_A = torch.linalg.svd(R_A.float(), full_matrices=False)
            _, S_G, Vh_G = torch.linalg.svd(R_G.float(), full_matrices=False)
            # Eigenvalues of A = R^T R are squared singular values of R.
            # Eigenvectors of A are right singular vectors of R (V^T row → eigenvectors are V columns).
            U_A, d_A = Vh_A.t().contiguous(), S_A ** 2
            U_G, d_G = Vh_G.t().contiguous(), S_G ** 2
            if precision == "bf16":
                # Round-trip storage to bf16 to honor the precision regime.
                U_A = U_A.to(torch.bfloat16).to(torch.float32)
                U_G = U_G.to(torch.bfloat16).to(torch.float32)
                d_A = d_A.to(torch.bfloat16).to(torch.float32)
                d_G = d_G.to(torch.bfloat16).to(torch.float32)
            eig_stats["d_A_min"].append(float(d_A.min())); eig_stats["d_A_max"].append(float(d_A.max()))
            eig_stats["d_G_min"].append(float(d_G.min())); eig_stats["d_G_max"].append(float(d_G.max()))
            eig_stats["n_neg_A"] += int((d_A < 0).sum()); eig_stats["n_neg_G"] += int((d_G < 0).sum())
            eig_stats["n_layers"] += 1
            factors_evd[mod] = (U_A, d_A, U_G, d_G)
        _report_eig_stats(eig_stats, precision)
        return factors_evd

    # ---- Classic K-FAC path: eigh(A), eigh(G) (κ² unstable at bf16) ----
    return _evd_from_gram(factors_raw, precision)


def _evd_from_gram(factors_raw, precision):
    """Classic-K-FAC EVD path: eigendecompose the explicit Gram factors A, G.

    Shared by the homebrew ClassicKFAC path and the ASDL reference-Classic
    path — both form A = E[xxᵀ] and G = E[δδᵀ] explicitly, so both collapse at
    bf16 the same way.  At bf16 this is where the κ² catastrophe lands:
    quantising the Gram carries effective conditioning κ²(X) (the X^T X
    squaring), so eigh can drive the smallest eigenvalues negative.

    Args:
        factors_raw: dict module → (A, G) fp32 Gram matrices.
        precision:   "fp32" or "bf16" — bf16 quantises A/G before eigh and
                     round-trips the EVD outputs through bf16 storage.
    """
    factors_evd = {}
    eig_stats = {"d_A_min": [], "d_A_max": [], "d_G_min": [], "d_G_max": [],
                 "n_neg_A": 0, "n_neg_G": 0, "n_layers": 0}
    for mod, (A, G) in factors_raw.items():
        if precision == "bf16":
            # ★ κ² catastrophe lands here — bf16 noise on the Gram has
            # effective conditioning κ²(X) due to A = X^T X squaring.
            A = A.to(torch.bfloat16).to(torch.float32)
            G = G.to(torch.bfloat16).to(torch.float32)
        # eigh expects symmetric — Gram matrices are by construction, but
        # round-tripping through bf16 may introduce asymmetry.  Symmetrize.
        A = 0.5 * (A + A.t())
        G = 0.5 * (G + G.t())
        # eigh returns eigenvalues ascending and eigenvectors columns.
        d_A, U_A = torch.linalg.eigh(A)
        d_G, U_G = torch.linalg.eigh(G)
        # Track # negatives BEFORE clamping — this is the κ² catastrophe signature.
        eig_stats["n_neg_A"] += int((d_A < 0).sum()); eig_stats["n_neg_G"] += int((d_G < 0).sum())
        # Clamp eigenvalues to >= 0 — bf16 noise can produce small negatives
        # in the smallest eigenvalues (the κ² collapse signature).  Without
        # this clamp the subsequent sqrt(d + λ) would yield NaN.
        d_A = d_A.clamp(min=0.0)
        d_G = d_G.clamp(min=0.0)
        if precision == "bf16":
            U_A = U_A.to(torch.bfloat16).to(torch.float32)
            U_G = U_G.to(torch.bfloat16).to(torch.float32)
            d_A = d_A.to(torch.bfloat16).to(torch.float32)
            d_G = d_G.to(torch.bfloat16).to(torch.float32)
        eig_stats["d_A_min"].append(float(d_A.min())); eig_stats["d_A_max"].append(float(d_A.max()))
        eig_stats["d_G_min"].append(float(d_G.min())); eig_stats["d_G_max"].append(float(d_G.max()))
        eig_stats["n_layers"] += 1
        factors_evd[mod] = (U_A, d_A, U_G, d_G)
    _report_eig_stats(eig_stats, precision)
    return factors_evd


def _capture_fisher_asdl(model, opt, train_loader, device, n_passes=1,
                         precision="fp32"):
    """Fisher capture using the ASDL reference Classic K-FAC.

    ASDL owns the forward/backward (its layer hooks capture activations and
    output gradients), so we drive it with opt.accumulate_curvature() — which
    updates the per-layer Kronecker factors WITHOUT stepping the weights, so
    the model stays at the MAP.  We then read the explicit Gram factors
    A = module.fisher.kron.A and G = module.fisher.kron.B and run the exact
    same EVD path as the homebrew Classic K-FAC (κ² collapse at bf16).

    Capture is always fp32; the bf16 regime is applied in _evd_from_gram by
    quantising A/G before eigh — identical to the homebrew classic path, so
    Classic-homebrew vs ASDL is apples-to-apples.
    """
    model.eval()
    # MC-sampled labels keep the output-gradient covariance G non-degenerate
    # at MAP convergence (Daxberger 2021 §3.1) — same rationale as the
    # homebrew path above.
    for _ in range(n_passes):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                probs = torch.softmax(model(x).float(), dim=-1)
                y_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            opt.accumulate_curvature(x, y_sampled, loss_fn=F.cross_entropy)

    # Extract the explicit Gram factors ASDL stored on each supported module.
    factors_raw = {}
    for module in model.modules():
        fisher = getattr(module, "fisher", None)
        if fisher is None:
            continue
        kron = getattr(fisher, "kron", None)
        if kron is None:
            continue
        A = getattr(kron, "A", None)
        B = getattr(kron, "B", None)   # ASDL's B == our G (output-grad cov)
        if A is None or B is None:
            continue
        factors_raw[module] = (A.detach().float(), B.detach().float())

    if not factors_raw:
        raise RuntimeError(
            "ASDL captured no Kronecker factors — check that the model has "
            "supported (Linear/Conv2d) layers and ignore_modules is sane: "
            f"{getattr(opt, 'ignore_modules', None)}"
        )

    # No hook cleanup needed: ASDL installs its curvature hooks via a
    # `with no_centered_cov(...)` context manager inside each
    # forward_and_backward and removes them on block exit (asdl/fisher.py).
    # Nothing persists, so Phase-3 MC-predictive forwards stay cheap.
    return _evd_from_gram(factors_raw, precision)


def _report_eig_stats(s, precision):
    """One-line summary of K-FAC eigenvalue distributions across layers."""
    import statistics as _st
    da_min = min(s["d_A_min"]); da_max = max(s["d_A_max"])
    dg_min = min(s["d_G_min"]); dg_max = max(s["d_G_max"])
    # Per-layer condition numbers — median across layers
    kappas_A = [mx / max(mn, 1e-30) for mn, mx in zip(s["d_A_min"], s["d_A_max"])]
    kappas_G = [mx / max(mn, 1e-30) for mn, mx in zip(s["d_G_min"], s["d_G_max"])]
    print(f"  [eig stats / {precision:>4s}]  "
          f"d_A∈[{da_min:.2e}, {da_max:.2e}]  d_G∈[{dg_min:.2e}, {dg_max:.2e}]  "
          f"med κ(A)={_st.median(kappas_A):.1e}  med κ(G)={_st.median(kappas_G):.1e}  "
          f"neg eigs: A={s['n_neg_A']} G={s['n_neg_G']}", flush=True)


# ---- Predictive evaluation (Phase 3) ----------------------------------------

@torch.no_grad()
def evaluate_predictive(posterior, loader, device, n_samples=LAPLACE_SAMPLES):
    """Return (probs, labels) for the full loader, using MC predictive averaging.

    Uses LaplacePosterior.predictive_full_loader which resamples weights ONCE
    per MC iteration (not per batch).  ~10× faster than per-batch resampling
    for ResNet-18 + CIFAR-10 val set (40 batches × 10 samples).
    """
    return posterior.predictive_full_loader(loader, device, n_samples=n_samples)


# ---- Damping grid search ---------------------------------------------------

GRID_SEARCH_SAMPLES = 3     # MC samples used PER λ during grid search
                              # 3 is enough for stable RANKING of λ values
                              # (full eval uses LAPLACE_SAMPLES=30 once at end)


def select_lambda(model, factors, val_loader, device, n_samples=GRID_SEARCH_SAMPLES):
    """Grid-search Tikhonov λ minimising val-NLL.  Uses fewer MC samples
    during search (5 instead of 30) — the relative ranking of λ values is
    stable at 5 samples; we only need precise NLL for the final test eval."""
    from optimizer.laplace import LaplacePosterior
    from benchmark.metrics_calibration import negative_log_likelihood
    best_lam, best_nll = LAMBDA_GRID[0], float("inf")
    trace = []
    for idx, lam in enumerate(LAMBDA_GRID):
        t_lam = time.perf_counter()
        print(f"      [{idx+1}/{len(LAMBDA_GRID)}] λ={lam:>6.3g}  building posterior...",
              flush=True)
        post = LaplacePosterior(model, factors, prior_precision=lam, dataset_size=N_TRAIN)
        print(f"        posterior built in {time.perf_counter() - t_lam:.1f}s, "
              f"running {n_samples}-sample MC predictive on val set...", flush=True)
        t_eval = time.perf_counter()
        probs, labels = evaluate_predictive(post, val_loader, device, n_samples=n_samples)
        eval_s = time.perf_counter() - t_eval
        nll = negative_log_likelihood(probs, labels)
        trace.append({"lambda": lam, "val_nll": nll,
                       "build_s": time.perf_counter() - t_lam - eval_s,
                       "eval_s": eval_s})
        winner = "  ←  winner so far" if nll < best_nll else ""
        print(f"      [{idx+1}/{len(LAMBDA_GRID)}] λ={lam:>6.3g}  "
              f"val_nll={nll:.4f}  ({time.perf_counter()-t_lam:.1f}s){winner}",
              flush=True)
        if nll < best_nll:
            best_nll, best_lam = nll, lam
    return best_lam, best_nll, trace


# ---- Main -------------------------------------------------------------------

def train_or_load_map(seed, train, device):
    """Train ResNet-18 with AdamW ONCE per seed, cache to disk.

    Phase 1 (MAP training) is independent of which K-FAC variant we use
    later for Fisher capture — AdamW produces the same MAP regardless of
    whether we'll Classic-bf16 or IFKFAC-fp32 it later.  So we train once
    per (architecture, seed) and load for all 4 (method × precision) cells.
    """
    ckpt_path = ROOT / "benchmark" / "results" / f"laplace_map_resnet18_seed{seed}.pt"
    if ckpt_path.exists():
        print(f"  Phase 1: loading cached MAP checkpoint from "
              f"{ckpt_path.name}", flush=True)
        model = ResNet18CIFAR(num_classes=10).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model, 0.0   # zero wall-time since cached

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = ResNet18CIFAR(num_classes=10).to(device)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    map_opt = build_adamw_for_map(model)
    print(f"  Phase 1: AdamW MAP training {TRAIN_EPOCHS} epochs "
          f"(one-time, shared across all cells for this seed)...", flush=True)
    t0 = time.perf_counter()
    train_to_map(model, map_opt, train_loader, device, epochs=TRAIN_EPOCHS)
    t_train = time.perf_counter() - t0
    print(f"  Phase 1 done in {t_train/60:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"  cached MAP checkpoint → {ckpt_path.name}", flush=True)
    return model, t_train


def run_cell(method, precision, seed, model, train, val, test, device):
    """Run Phase 2/3/4 for ONE cell — assumes a trained MAP `model` is passed in."""
    print(f"\n=== Cell: {method}/{precision}/seed{seed} ===", flush=True)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Re-enable per-layer profiling for the first sample of each cell
    from optimizer.laplace import LaplacePosterior as _LP
    _LP._diag_first_call = True

    # Engage bf16 precision regime
    if precision == "bf16":
        from benchmark.kfac_bf16_compare import (
            enable_bf16, enable_classic_bf16,
            disable_bf16, disable_classic_bf16,
        )
        if method == "ifkfac":
            enable_bf16(wgso=False)
        else:
            enable_classic_bf16()

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(
        val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    t_train = 0.0   # MAP was trained separately (or loaded from cache)

    # Phase 2: build the K-FAC instance, then capture the Fisher.
    # K-FAC's *method* (Classic / IFKFAC) and *precision* (fp32 / bf16) here
    # determine which Fisher we estimate — that's the actual experiment.
    opt = build_kfac(method, model, precision=precision)

    print(f"  Phase 2: K-FAC Fisher capture ({FISHER_PASSES} pass over train)...", flush=True)
    t0 = time.perf_counter()
    factors = capture_fisher(model, opt, train_loader, device,
                                n_passes=FISHER_PASSES, precision=precision)
    t_fisher = time.perf_counter() - t0

    # CRITICAL: detach K-FAC hooks so they don't fire during Phase 3.
    # Without this, every MC-predictive forward pass triggers factor
    # accumulation + streaming TSQR, which is 100× the cost of a plain
    # forward.  This was the "3-hour grid search" bug.
    try:
        opt.hooks.remove()
        print(f"  hooks detached", flush=True)
    except Exception as e:
        print(f"  hook detach failed: {e}", flush=True)
    print(f"  Phase 2 done in {t_fisher/60:.1f} min  ({len(factors)} layers captured)", flush=True)

    # Phase 3a: either grid-search λ on val set, or use a fixed value
    # passed in via the global FIXED_LAMBDA (set by CLI --lambda).
    if FIXED_LAMBDA is not None:
        print(f"  Phase 3a: SKIPPED — using fixed λ = {FIXED_LAMBDA:g}", flush=True)
        best_lam = FIXED_LAMBDA
        best_val_nll = float("nan")
        trace = [{"lambda": FIXED_LAMBDA, "val_nll": float("nan"), "fixed": True}]
        t_grid = 0.0
    else:
        print(f"  Phase 3a: grid-search Tikhonov λ on val set...", flush=True)
        t0 = time.perf_counter()
        best_lam, best_val_nll, trace = select_lambda(model, factors, val_loader, device)
        t_grid = time.perf_counter() - t0
        print(f"  Phase 3a done in {t_grid/60:.1f} min.  Chose λ = {best_lam:g}", flush=True)

    print(f"  Phase 3b: test-set evaluation with chosen λ ({LAPLACE_SAMPLES} samples)...", flush=True)
    from optimizer.laplace import LaplacePosterior
    posterior = LaplacePosterior(model, factors, prior_precision=best_lam, dataset_size=N_TRAIN)
    t0 = time.perf_counter()
    probs, labels = evaluate_predictive(posterior, test_loader, device, n_samples=LAPLACE_SAMPLES)
    t_test = time.perf_counter() - t0
    metrics = all_metrics(probs, labels)
    print(f"  Phase 3b done in {t_test/60:.1f} min", flush=True)
    print(f"  RESULT: acc={metrics['accuracy']*100:.2f}%  "
          f"NLL={metrics['nll']:.4f}  ECE={metrics['ece']*100:.2f}%  "
          f"Brier={metrics['brier']:.4f}", flush=True)

    if precision == "bf16":
        if method == "ifkfac":
            disable_bf16()
        else:
            disable_classic_bf16()

    out = {
        "method": method, "precision": precision, "seed": seed,
        "chosen_lambda": best_lam, "best_val_nll": best_val_nll,
        "lambda_trace": trace,
        "metrics": metrics,
        "reliability": {k: v.tolist() if hasattr(v, "tolist") else v
                          for k, v in reliability_diagram_data(probs, labels).items()},
        "wall_s_train": t_train, "wall_s_fisher": t_fisher,
        "wall_s_grid": t_grid,   "wall_s_test": t_test,
    }
    out_path = ROOT / "benchmark" / "results" / f"laplace_cifar10_{method}_{precision}_seed{seed}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"  saved {out_path.name}", flush=True)

    # Cleanup
    try:
        opt.cleanup()
    except Exception:
        pass
    del model, opt, posterior
    import gc as _gc; _gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    global FIXED_LAMBDA
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help="single (method, precision, seed) cell only")
    parser.add_argument("--lambda", dest="fixed_lambda", type=float, default=None,
                         help="if set, skip grid search and use this λ for ALL cells "
                              "(allows fair K-FAC × precision comparison in a regime "
                              "where the Fisher actually contributes)")
    args = parser.parse_args()
    if args.fixed_lambda is not None:
        FIXED_LAMBDA = args.fixed_lambda
        print(f"Using FIXED λ = {FIXED_LAMBDA:g} (grid search bypassed)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"loading CIFAR-10...", flush=True)
    train, val, test = get_cifar10(device)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}", flush=True)

    # Smoke = single-seed sweep across all 4 (method, precision) cells when
    # we have a fixed λ (so we can do the comparison), otherwise just one cell.
    if args.smoke:
        if FIXED_LAMBDA is not None:
            cells = [(m, p, 42) for (m, p) in CELLS]   # 4 cells, single seed
        else:
            cells = [("ifkfac", "fp32", 42)]            # 1 cell, validate pipeline
    else:
        cells = [(m, p, s) for (m, p) in CELLS for s in SEEDS]

    # Group by seed so we train the MAP model once per seed and reuse it
    # across all (method, precision) cells for that seed.
    from collections import defaultdict
    by_seed = defaultdict(list)
    for (m, p, s) in cells:
        by_seed[s].append((m, p))

    for seed, mp_pairs in by_seed.items():
        print(f"\n=========== Seed {seed} ===========", flush=True)
        # Phase 1: train MAP once (or load from cache)
        model, t_train = train_or_load_map(seed, train, device)
        # Phase 2-4 for each (method, precision) cell
        for method, precision in mp_pairs:
            run_cell(method, precision, seed, model, train, val, test, device)

    # Summary table
    print("\n=== Laplace CIFAR-10 summary ===", flush=True)
    print(f"  {'method':>9}  {'prec':>5}  {'seed':>5}  {'acc%':>7}  {'NLL':>7}  "
          f"{'ECE%':>7}  {'Brier':>7}  {'λ*':>8}", flush=True)
    for method, precision, seed in cells:
        p = ROOT / "benchmark" / "results" / f"laplace_cifar10_{method}_{precision}_seed{seed}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        m = d["metrics"]
        print(f"  {method:>9}  {precision:>5}  {seed:>5}  "
              f"{m['accuracy']*100:>7.2f}  {m['nll']:>7.4f}  "
              f"{m['ece']*100:>7.2f}  {m['brier']:>7.4f}  "
              f"{d['chosen_lambda']:>8.3g}", flush=True)


if __name__ == "__main__":
    main()
