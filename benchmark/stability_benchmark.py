"""
benchmark/stability_benchmark.py

Numerical-stability comparison of three K-FAC variants on the SmallGPT /
WikiText-2 from-scratch pretraining task:

    ClassicKFAC   torch.linalg.inv               error scaling ~ k(A)^4 * eps
    OlsSMKFAC     Cholesky solve  (no inverse)   error scaling ~ k(A)^2 * eps
    IFKFAC     QR triangular solves           error scaling ~ k(A)^1 * eps

The hypothesis: better numerical stability lets a variant tolerate (a) a
higher learning rate before the natural-gradient update diverges, and (b)
once stable, converge to the same target perplexity in fewer steps because
the curvature estimate is less corrupted by inversion noise.

Two phases:

  Phase 1 - Max stable LR sweep, two momentum settings
    For each (variant, momentum, lr) triple, run PROBE_STEPS=1000 steps at
    fixed damping=1e-3.  Each momentum value uses its own LR grid because
    the stability frontier shifts ~3-5x with momentum:
        momentum=0.0  LRs in [1.2e-2, 4e-1]   (no buffer to absorb noise -
                                                tolerates higher per-step LR)
        momentum=0.9  LRs in [3e-3, 8e-2]     (deployment optimum ~8e-3)
    Both grids share the LR range 1.2e-2 to 8e-2 for direct cross-momentum
    comparison at those overlap points.

    A probe is classified as:

        diverged  if  train_loss is NaN or inf at any step
                  or  train_loss exceeds RELATIVE_DIVERGE_FACTOR x running min
                      for RELATIVE_DIVERGE_WINDOW consecutive steps (after
                      WARMUP_STEPS_BEFORE_CHECK to avoid anchoring on the
                      huge step-1 loss from SmallGPT's wide default init)
        stable    otherwise

    Two momentum values are tested:
        momentum=0.0   isolates inversion stability (no buffer averaging);
                       direct test of the k(A)^1 vs k(A)^2 vs k(A)^4 scaling
        momentum=0.9   deployment regime; momentum amplifies effective step
                       ~10x and accumulates a moving average of nat-grad
                       estimates, which partially masks per-step inversion
                       noise. Tells us whether the hierarchy survives in the
                       actual training configuration.

    Records the highest-LR stable run per (variant, momentum).

  Phase 2 - Convergence at best stable LR
    For each variant: run PHASE2_STEPS=5000 steps at its highest stable LR
    found in Phase 1 at PHASE2_MOMENTUM (default 0.9, the deployment value).
    Records validation perplexity curve, wall-time-to-target-ppl, and median
    per-step natural-gradient norm (proxy for inversion noise).

Important design choice - decoupling the K-FAC LR from the embedding LR.
SmallGPT uses a dual-optimizer setup: K-FAC for Linear layers, AdamW for
embeddings + LayerNorm + LM head.  In the main gpu_benchmark.py the
embedding AdamW shares the K-FAC LR.  At K-FAC LR=8e-2 that would diverge
AdamW regardless of K-FAC stability and confound the experiment.  This
benchmark pins the embedding AdamW LR at EMB_LR (5e-4) for every probe so
divergence is unambiguously the K-FAC variant's.

Output files (all under benchmark/results/):
    stability_phase1_sweep.json    per-LR stability table
    stability_phase2_runs.json     full convergence runs at max stable LR
    stability_summary.csv          combined summary line per (variant, lr)
    stability_lr_frontier.png      phase 1 plot
    stability_convergence.png      phase 2 plot

Usage:
    python benchmark/stability_benchmark.py --phase all
    python benchmark/stability_benchmark.py --phase 1
    python benchmark/stability_benchmark.py --phase 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo root importable so we can reach optimizer/ and reuse
# SmallGPT + helpers from benchmark/gpu_benchmark.py.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.gpu_benchmark import (
    SmallGPT,
    get_device,
    get_hardware_info,
)

OUT = ROOT / "benchmark" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================================
#  Experimental constants
# ============================================================================

# Two momentum settings - the heart of this benchmark.
#   0.0  - inversion-only stability test (no buffer averaging masks noise)
#   0.9  - deployment regime; what the main benchmark uses
MOMENTUM_GRID = [0.0, 0.9]

# LR grids must be centered on each regime's stability frontier, otherwise
# the sweep wastes probes in regions where every variant is trivially
# stable or trivially diverged.
#
# Momentum amplifies the effective step ~ 1/(1-momentum) = 10x in steady
# state, but warmup (200 steps from 0.1*lr) and cosine decay mitigate this.
# Empirically the ceiling shifts ~3-5x lower with momentum=0.9 vs 0.0.
#
# Each grid is 8 log-spaced points spanning ~1.5 decades, designed so that
# all three variants' frontiers fall well inside the range.  Both grids
# share the LR range 1.2e-2 to 8e-2 so cross-momentum comparison at the
# same LR is possible at those points.
LR_GRIDS: Dict[float, List[float]] = {
    # Momentum=0.0: extended DOWN with [1e-3, 3e-3, 5e-3, 8e-3] because with
    # GRAD_CLIP=1000 (effectively unbounded), ClassicKFAC's max-stable LR
    # collapses below 0.012.  Need lower probes to find Classic's true
    # frontier so the kappa-scaling hierarchy can be measured directly
    # ("Classic stable up to LR=X, OlsSM up to Y, IFKFAC up to Z").
    # Stuck/diverged probes early-exit fast (~30s) so the extra 4 LRs
    # cost ~30 min total for the full Phase 1 sweep.
    0.0: [1e-3, 3e-3, 5e-3, 8e-3, 1.2e-2, 2e-2, 3e-2, 5e-2, 8e-2, 1.5e-1],
    # Momentum=0.9: also extended down with [1e-3, 2e-3] to find Classic's
    # mom=0.9 frontier with no clipping.
    0.9: [1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 1.2e-2, 2e-2, 3e-2, 5e-2, 8e-2],
}

# Phase 2 picks ONE momentum to do its 5000-step convergence runs at.
# Default is 0.9 (deployment-relevant). Override via CLI --phase2-momentum.
PHASE2_MOMENTUM = 0.9

# Fixed for the whole sweep.  All three variants use damping=1e-3 in the
# main gpu_benchmark.py transformer task, so this matches.
DAMPING = 1e-3

# Phase 1 probe length - long enough that slow-divergers show up but short
# enough that a 24-probe sweep finishes in ~30-40 minutes.
PROBE_STEPS = 1000

# Phase 2 convergence run - matches gpu_benchmark.py transformer steps so
# results are directly comparable to the main benchmark numbers.
PHASE2_STEPS = 5000

# Embedding optimizer LR - decoupled from K-FAC LR.  AdamW is robust at this
# value for SmallGPT pretraining; never the cause of divergence.
EMB_LR = 5e-4

# Divergence detection.
#
# We deliberately do NOT use an absolute loss threshold.  SmallGPT's default
# PyTorch init (nn.Embedding -> N(0,1), tied output head) produces initial
# loss ~ 160 on WikiText-2 even before any optimizer step - far above the
# ln(50257)~10.8 of uniform-random output.  An absolute threshold would
# trigger on step 1 of every probe regardless of optimizer behaviour.
#
# Instead, divergence is detected from any of three signals:
#   1. NaN/inf in the loss     (catastrophic numerical failure)
#   2. Sustained drift upward  (loss > RELATIVE_DIVERGE_FACTOR * running min
#                                for RELATIVE_DIVERGE_WINDOW consecutive steps)
#   3. Val perplexity pinned   (val_ppl >= PPL_CAP for PPL_CAP_CONSECUTIVE
#                                evaluations - means val cross-entropy > 9.21,
#                                worse than uniform random).
#
# WARMUP_STEPS_BEFORE_CHECK must be past the LR scheduler's warmup (200 steps)
# so running_min isn't anchored to artificially-low warmup-phase losses.
# Without this, a probe that briefly trains during low-LR warmup then explodes
# at full LR slips past detection because its 'min' is the warmup low.
WARMUP_STEPS_BEFORE_CHECK = 250
RELATIVE_DIVERGE_FACTOR   = 2.0     # 2x running_min means training is going backwards
RELATIVE_DIVERGE_WINDOW   = 30      # consecutive steps required (~7s of training)

# Val-perplexity cap detector.  evaluate_ppl() returns min(exp(loss), 9999)
# so a value at the cap means val loss > ln(9999) ~= 9.21 - worse than the
# 10.83 of uniform-random output.
#
# Trigger requires BOTH:
#   1. PPL_CAP_CONSECUTIVE evals in a row at cap
#   2. running_min_loss > PPL_CAP_LOSS_FLOOR  (training didn't progress past
#      "barely better than random uniform")
#
# The dual condition prevents false positives on slow-but-honest training
# (e.g., very tight grad_clip).  At grad_clip=0.3 the natural gradient is
# heavily throttled, so val_ppl can sit at the cap for several evals while
# train loss legitimately drops to ~10.5 - that's slow training, not
# divergence.  Only flag as diverged if loss is also stuck near random init.
PPL_CAP                 = 9990.0
PPL_CAP_CONSECUTIVE     = 10        # ~10 evals = ~1500 steps - effectively only
                                     # triggers on probes that NEVER break the cap
PPL_CAP_LOSS_FLOOR      = 9.5       # train loss must also be near-random

# Targets used for time-to-ppl reporting in Phase 2.
PPL_TARGETS = [1000.0, 500.0, 400.0]

VARIANTS = ["ClassicKFAC", "OlsSMKFAC", "IFKFAC"]
KFAC_MAX_DIM = 4096   # excludes LM head from K-FAC hooks (vocab=50257 G matrix would OOM)


# ============================================================================
#  Data prep (replicated from gpu_benchmark.py to keep this file self-contained)
# ============================================================================

def build_data(device: torch.device, batch_size: int = 64,
               seq_len: int = 128) -> Tuple[Callable, torch.utils.data.DataLoader, int]:
    """Tokenize WikiText-2 and return (train_loader_factory, val_loader, vocab_size).

    train_loader_factory is a function that returns a fresh DataLoader each
    time it's called - we do this because each probe re-shuffles training
    data with a fresh seed, and re-using a single iterator across probes
    would couple their data orderings.
    """
    from torch.utils.data import DataLoader
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    import datasets as hf_ds

    print(f"  Loading WikiText-2 + GPT-2 tokenizer (seq_len={seq_len}, B={batch_size}) ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(batch):
        flat = []
        for text in batch["text"]:
            if isinstance(text, str) and text.strip():
                enc = tokenizer(text, truncation=False, add_special_tokens=False)
                flat.extend(enc["input_ids"])
        chunks = [flat[i:i + seq_len + 1]
                  for i in range(0, len(flat) - seq_len, seq_len)]
        return {"input_ids": chunks}

    tokenized = {}
    for split in ("train", "validation"):
        ds = raw[split]
        tok = tokenize({"text": ds["text"]})
        tokenized[split] = hf_ds.Dataset.from_dict(tok)
        tokenized[split].set_format("torch")

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])
        return ids[:, :-1], ids[:, 1:]

    # num_workers=0 (no subprocesses) for two reasons:
    #   1. On Windows, DataLoader workers spawn child processes that must
    #      pickle collate_fn. A nested-function collate (defined inside
    #      build_data) is unpicklable -> the AttributeError we hit.
    #   2. WikiText-2 is small and already tokenized in RAM. Worker overhead
    #      is larger than the savings from parallel loading on this dataset.
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size,
                            collate_fn=collate, num_workers=0)

    def make_train_loader():
        return DataLoader(tokenized["train"], batch_size=batch_size,
                          shuffle=True, collate_fn=collate, num_workers=0)

    return make_train_loader, val_loader, tokenizer.vocab_size


def evaluate_ppl(model: nn.Module, loader, device: torch.device,
                 pad_id: int, max_batches: Optional[int] = None) -> float:
    """Validation perplexity.  Capped at 9999 so plots don't blow up."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=pad_id)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    if total_tokens == 0:
        return 9999.0
    return min(float(math.exp(total_loss / total_tokens)), 9999.0)


# ============================================================================
#  Optimizer factory
# ============================================================================

def make_optimizers(variant: str, model: nn.Module, kfac_lr: float,
                    damping: float, momentum: float,
                    grad_clip: Optional[float] = None,
                    gamma: Optional[float] = None,
                    factor_update_freq: int = 20,
                    ) -> Tuple[torch.optim.Optimizer,
                               torch.optim.Optimizer,
                               List[int]]:
    """Build (kfac_opt, emb_opt, kfac_covered_param_ids) for a variant.

    kfac_opt   - second-order optimizer for Linear layers with out_dim <= KFAC_MAX_DIM
    emb_opt    - AdamW at EMB_LR for everything else (Embedding, LayerNorm, LM head)

    momentum is applied to kfac_opt only; emb_opt always uses AdamW's defaults.
    Decoupling K-FAC LR (and momentum) from embedding LR ensures high-LR probes
    don't fail because AdamW diverged on the embeddings.
    """
    # Identify K-FAC-covered parameters
    kfac_covered_ids = set()
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            out_dim = (mod.out_features if isinstance(mod, nn.Linear)
                       else mod.out_channels)
            if out_dim <= KFAC_MAX_DIM:
                for p in mod.parameters():
                    kfac_covered_ids.add(id(p))

    other_params = []
    seen = set()
    for p in model.parameters():
        if id(p) not in kfac_covered_ids and id(p) not in seen:
            other_params.append(p)
            seen.add(id(p))

    emb_opt = torch.optim.AdamW(other_params, lr=EMB_LR, weight_decay=0.01)

    # IMPORTANT: grad_clip set HIGH (1000) across all three variants for the
    # stability benchmark.  Rationale:
    #   - Equal across variants -> isolates the inversion method (was 10/20/20
    #     in gpu_benchmark.py, a confound for stability comparison).
    #   - Set to 1000 (not 10) because at clip=10 all three variants produce
    #     essentially identical natural-gradient updates - the clip dominates
    #     and hides the kappa(X)^1 vs kappa(X)^2 vs kappa(X)^4 numerical
    #     differences this benchmark is meant to expose.
    #   - 1000 is effectively unbounded for typical operation but still catches
    #     genuine NaN-equivalent meltdowns.
    # Expect: at high LR / low damping, ClassicKFAC may genuinely diverge while
    # OlsSM/IFKFAC survive - which is the actual stability hierarchy in action.
    DEFAULT_GRAD_CLIP = 100.0
    GRAD_CLIP = grad_clip if grad_clip is not None else DEFAULT_GRAD_CLIP

    # Per-variant default gamma if not overridden
    DEFAULT_GAMMA = {"OlsSMKFAC": 0.5, "ClassicKFAC": 0.5, "IFKFAC": 0.7}
    GAMMA = gamma if gamma is not None else DEFAULT_GAMMA[variant]

    if variant == "OlsSMKFAC":
        from optimizer.olssm_kfac import OlsSMKFAC
        # decomp_update_freq=20 matches ClassicKFAC for apples-to-apples
        # speed comparison.  gpu_benchmark.py uses decomp_update_freq=5 for
        # OlsSMKFAC ("EVD is fast on GPU"), but for SmallGPT every layer is
        # <= 1024 dim and routes through the Cholesky path, not EVD - so the
        # 4x-more-frequent refresh just makes OlsSM look slower than Classic.
        kfac_opt = OlsSMKFAC(
            model, lr=kfac_lr, damping=damping,
            factor_update_freq=factor_update_freq,
            decomp_update_freq=factor_update_freq,
            adaptive=True, adaptive_min_n=4096,
            momentum=momentum, grad_clip=GRAD_CLIP, gamma=GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
    elif variant == "ClassicKFAC":
        from optimizer.classic_kfac import ClassicKFAC
        kfac_opt = ClassicKFAC(
            model, lr=kfac_lr, damping=damping,
            factor_update_freq=factor_update_freq,
            decomp_update_freq=factor_update_freq,
            momentum=momentum, grad_clip=GRAD_CLIP, gamma=GAMMA,
            max_gram_dim=KFAC_MAX_DIM,
        )
    elif variant == "IFKFAC":
        from optimizer.ifkfac_kfac import IFKFAC
        kfac_opt = IFKFAC(
            model, lr=kfac_lr, damping=damping,
            factor_update_freq=factor_update_freq,
            momentum=momentum, grad_clip=GRAD_CLIP, gamma=GAMMA,
            max_out_dim=KFAC_MAX_DIM,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return kfac_opt, emb_opt, list(kfac_covered_ids)


# ============================================================================
#  Probe runner (used by both phases)
# ============================================================================

def measure_natgrad_norm(kfac_opt) -> Optional[float]:
    """Median L2 norm of the natural-gradient updates applied this step.

    Reads from the optimizer's momentum buffers if available - they hold the
    most recent applied update.  Returns None if not available.
    """
    bufs = getattr(kfac_opt, "_momentum_buffers", None)
    if not bufs:
        return None
    norms = [b.norm().item() for b in bufs.values() if b is not None]
    if not norms:
        return None
    return float(np.median(norms))


class KFACConditionTracker:
    """Track condition numbers of A and G Gram matrices over training.

    Quantifies the kappa(A)^1 vs kappa(A)^2 vs kappa(A)^4 stability story:
    if all three variants see the same kappa(A), kappa(G) (because the data
    is identical), then any difference in convergence/stability comes purely
    from the inversion method, not from input conditioning.

    Logging cadence is controlled by log_every; track() is a no-op on steps
    that are not a multiple of log_every.  Set log_every=100 for 1000-step
    probes (10 measurements/probe), log_every=200 for 5000-step Phase 2 runs
    (25 measurements).
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.step = 0
        # layer_name -> list of (step, kappa_A, kappa_G)
        self.history: Dict[str, List[Tuple[int, float, float]]] = {}

    def track(self, layer_name: str, A: torch.Tensor, G: torch.Tensor):
        if self.step % self.log_every != 0:
            return
        kappa_A = self._condition_number(A)
        kappa_G = self._condition_number(G)
        if layer_name not in self.history:
            self.history[layer_name] = []
        self.history[layer_name].append((self.step, kappa_A, kappa_G))

    @staticmethod
    def _condition_number(M: torch.Tensor) -> float:
        # A and G are symmetric PSD; eigvalsh is faster than svd.
        eigvals = torch.linalg.eigvalsh(M).abs()   # ascending
        sigma_min = eigvals[0].clamp(min=1e-10)
        sigma_max = eigvals[-1]
        return (sigma_max / sigma_min).item()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregate stats per layer: mean/median/max kappa over training."""
        out: Dict[str, Dict[str, float]] = {}
        for layer, recs in self.history.items():
            if not recs:
                continue
            ka = np.array([r[1] for r in recs])
            kg = np.array([r[2] for r in recs])
            out[layer] = {
                "mean_kappa_A":   float(ka.mean()),
                "median_kappa_A": float(np.median(ka)),
                "max_kappa_A":    float(ka.max()),
                "mean_kappa_G":   float(kg.mean()),
                "median_kappa_G": float(np.median(kg)),
                "max_kappa_G":    float(kg.max()),
                "n_samples":      len(recs),
            }
        return out


def extract_factors(kfac_opt, variant: str) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Return {layer_idx: (A, G)} from any variant's internal state.

    Classic / OlsSM store Gram matrices directly in _factors.
    IFKFAC stores R factors (R_X, R_G) from QR; reconstruct as A = R_X.T @ R_X
    and G = R_G.T @ R_G to get the equivalent Gram matrix the inversion
    would see.
    """
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    factors = getattr(kfac_opt, "_factors", None)
    if not factors:
        return out
    layers = list(getattr(kfac_opt.hooks, "linear_layers",
                          getattr(kfac_opt.hooks, "_linear_layers", [])))
    for module, pair in factors.items():
        try:
            idx = layers.index(module)
        except ValueError:
            continue
        if variant == "IFKFAC":
            R_X, R_G = pair
            A = R_X.T @ R_X
            G = R_G.T @ R_G
        else:
            A, G = pair
        out[idx] = (A, G)
    return out


def run_probe(
    variant: str,
    kfac_lr: float,
    damping: float,
    momentum: float,
    max_steps: int,
    vocab_size: int,
    train_loader_factory: Callable,
    val_loader,
    pad_id: int,
    device: torch.device,
    seed: int = 42,
    record_natgrad: bool = False,
    record_condition: bool = False,
    condition_log_every: int = 100,
    eval_every_samples: int = 10_000,
    print_progress: bool = True,
    grad_clip: Optional[float] = None,    # override the optimizer's clip
    gamma: Optional[float] = None,        # override the optimizer's EMA gamma
    lr_schedule: str = "cosine_max_steps",  # see scheduler block below
    factor_update_freq: int = 20,         # K-FAC factor refresh cadence
    warmup_steps: Optional[int] = None,   # override the hardcoded 200; None = 200
) -> Dict:
    """Run a single (variant, lr, damping, momentum) probe; return metrics + status.

    Status is one of:
        "stable"     - completed max_steps without triggering divergence
        "diverged"   - hit NaN/inf or relative drift trigger

    record_condition: if True, snapshot kappa(A) and kappa(G) per layer every
    condition_log_every steps via KFACConditionTracker.  Adds a few hundred ms
    of overhead total per probe (eigvalsh on small Gram matrices is fast).
    """
    torch.manual_seed(seed)
    model = SmallGPT(vocab_size=vocab_size).to(device)
    kfac_opt, emb_opt, _ = make_optimizers(variant, model, kfac_lr, damping,
                                            momentum, grad_clip=grad_clip,
                                            gamma=gamma,
                                            factor_update_freq=factor_update_freq)

    if print_progress:
        # Dump every parameter that could affect the trajectory.  Added as
        # part of the 2026-05-12 regression investigation (post-fix screen
        # cells were ~2x worse than the historical reference at the same
        # step); comparing this block against the historical JSON's "config"
        # field is the fastest way to spot which knob drifted.
        n_params = sum(p.numel() for p in model.parameters())
        bs = getattr(val_loader, "batch_size", None)
        try:
            sample_x, _ = next(iter(val_loader))
            seq_len = int(sample_x.size(1))
        except Exception:
            seq_len = None
        print()
        print("  ----- run_probe parameters -----")
        print(f"    variant:              {variant}")
        print(f"    kfac_lr (arg):        {kfac_lr}")
        print(f"    damping (arg):        {damping}")
        print(f"    momentum (arg):       {momentum}")
        print(f"    gamma (arg):          {gamma!r}   "
              f"(None -> DEFAULT_GAMMA[{variant}])")
        print(f"    grad_clip (arg):      {grad_clip!r}   "
              f"(None -> DEFAULT_GRAD_CLIP=100.0)")
        print(f"    max_steps:            {max_steps}")
        print(f"    seed:                 {seed}")
        print(f"    vocab_size:           {vocab_size}")
        print(f"    pad_id:               {pad_id}")
        print(f"    EMB_LR (const):       {EMB_LR}")
        print(f"    KFAC_MAX_DIM (const): {KFAC_MAX_DIM}")
        print(f"    record_natgrad:       {record_natgrad}")
        print(f"    record_condition:     {record_condition}   "
              f"(log_every={condition_log_every})")
        print(f"    eval_every_samples:   {eval_every_samples}")
        print(f"    lr_schedule:          {lr_schedule!r}")
        print(f"    factor_update_freq:   {factor_update_freq}")
        print(f"    warmup_steps (arg):   {warmup_steps!r}   "
              f"(None -> 200)")
        print(f"    data: batch_size={bs}  seq_len={seq_len}")
        print(f"    model: {type(model).__name__}  params={n_params:,}")
        print(f"    ----- optimizer state ({type(kfac_opt).__name__}) -----")
        # Print every attribute we expect to find on a K-FAC variant.
        # Missing ones print as "<absent>" so a silent default change is visible.
        for attr in ("lr", "damping", "momentum", "gamma", "grad_clip",
                     "factor_update_freq", "decomp_update_freq",
                     "adaptive", "adaptive_min_n",
                     "max_out_dim", "max_gram_dim"):
            val = getattr(kfac_opt, attr, "<absent>")
            # param_groups stash lr/momentum on torch optimizers; surface those too
            if val == "<absent>" and hasattr(kfac_opt, "param_groups"):
                pg = kfac_opt.param_groups[0] if kfac_opt.param_groups else {}
                if attr in pg:
                    val = f"{pg[attr]}  (from param_groups[0])"
            print(f"      {attr:22s} {val}")
        print(f"    ----- scheduler shape ({lr_schedule}) -----")
        _warmup = warmup_steps if warmup_steps is not None else 200
        if lr_schedule == "cosine_max_steps":
            _cosine = max(1, max_steps - _warmup)
            print(f"      kfac:  warmup={_warmup}  cosine_T_max={_cosine}  "
                  f"eta_min={kfac_lr * 0.0015:.3e}")
            print(f"      emb:   cosine_T_max={max_steps}  "
                  f"eta_min={EMB_LR * 0.01:.3e}")
        else:  # constant_warmup
            print(f"      kfac:  warmup={_warmup}  then constant at "
                  f"{kfac_lr:.3e} for {max_steps - _warmup} steps")
            print(f"      emb:   constant at {EMB_LR:.3e} "
                  f"for all {max_steps} steps")
        print(f"    --------------------------------")
        print()

    # Schedulers.  Two options:
    #
    #   "cosine_max_steps" (default, original behavior):
    #     Linear warmup (200 steps) -> cosine decay over (max_steps - warmup)
    #     -> eta_min = kfac_lr * 0.0015.  Mirrors gpu_benchmark.py's transformer
    #     schedule.  PROBLEM for sweeps: the decay shape depends on max_steps,
    #     so a "lr=8e-3" cell at max_steps=1000 trains very differently from
    #     the same nominal lr at max_steps=5000 (the 1000-step run decays ~6x
    #     faster).  The lr label stops being meaningful.
    #
    #   "constant_warmup":
    #     Linear warmup (200 steps) -> constant kfac_lr forever.  Used by the
    #     2D screen and single-cell diagnostic so a cell labeled lr=X actually
    #     trains at LR=X after warmup, independent of max_steps.  Emb LR is
    #     also held constant at EMB_LR (AdamW is robust enough at this LR
    #     that warmup is not required).
    warmup = warmup_steps if warmup_steps is not None else 200
    if lr_schedule == "cosine_max_steps":
        cosine_steps = max(1, max_steps - warmup)
        scheduler = torch.optim.lr_scheduler.SequentialLR(kfac_opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                kfac_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                kfac_opt, T_max=cosine_steps, eta_min=kfac_lr * 0.0015),
        ], milestones=[warmup])
        emb_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            emb_opt, T_max=max_steps, eta_min=EMB_LR * 0.01)
    elif lr_schedule == "constant_warmup":
        scheduler = torch.optim.lr_scheduler.SequentialLR(kfac_opt, schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                kfac_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.ConstantLR(
                kfac_opt, factor=1.0, total_iters=max_steps),
        ], milestones=[warmup])
        emb_scheduler = torch.optim.lr_scheduler.ConstantLR(
            emb_opt, factor=1.0, total_iters=max_steps)
    else:
        raise ValueError(
            f"Unknown lr_schedule: {lr_schedule!r}.  "
            f"Expected 'cosine_max_steps' or 'constant_warmup'."
        )

    train_loader = train_loader_factory()
    data_iter = iter(train_loader)

    train_losses: List[float] = []
    natgrad_norms: List[float] = []
    val_steps: List[int] = []
    val_samples: List[int] = []
    val_times: List[float] = []
    val_ppls: List[float] = []
    step_times: List[float] = []
    samples_seen = 0
    running_min_loss = float("inf")
    grow_run = 0  # consecutive steps where loss > 2x running min

    tracker = (KFACConditionTracker(log_every=condition_log_every)
               if record_condition else None)

    status = "stable"
    diverge_step: Optional[int] = None
    diverge_loss: Optional[float] = None

    t0 = time.perf_counter()
    next_eval = eval_every_samples

    for step in range(1, max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        t_step = time.perf_counter()
        model.train()
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=pad_id)
        loss_val = loss.item()
        train_losses.append(loss_val)

        # Divergence detection (relative-only - see constants section).
        # 1. NaN/inf: always catastrophic, trigger immediately.
        if not math.isfinite(loss_val):
            status = "diverged"
            diverge_step = step
            diverge_loss = loss_val
            if print_progress:
                print(f"      [nan]      step={step:5d}  loss={loss_val}")
            break

        # 2. Relative drift: only after warmup so the huge step-1 loss
        #    (from SmallGPT's wide default init) doesn't anchor running_min.
        if step > WARMUP_STEPS_BEFORE_CHECK:
            running_min_loss = min(running_min_loss, loss_val)
            if loss_val > RELATIVE_DIVERGE_FACTOR * running_min_loss:
                grow_run += 1
                if grow_run >= RELATIVE_DIVERGE_WINDOW:
                    status = "diverged"
                    diverge_step = step
                    diverge_loss = loss_val
                    if print_progress:
                        print(f"      [drift]    step={step:5d}  "
                              f"loss={loss_val:.2f}  "
                              f"min={running_min_loss:.2f}  "
                              f"({RELATIVE_DIVERGE_FACTOR}x for "
                              f"{RELATIVE_DIVERGE_WINDOW} steps)")
                    break
            else:
                grow_run = 0

        model.zero_grad()
        loss.backward()
        kfac_opt.step()
        emb_opt.step()
        scheduler.step()
        emb_scheduler.step()

        if record_natgrad:
            ng = measure_natgrad_norm(kfac_opt)
            if ng is not None:
                natgrad_norms.append(ng)

        if tracker is not None:
            tracker.step = step
            with torch.no_grad():
                for layer_idx, (A, G) in extract_factors(kfac_opt, variant).items():
                    tracker.track(f"layer_{layer_idx:02d}", A, G)

        step_times.append(time.perf_counter() - t_step)
        samples_seen += x.size(0)

        # Validation perplexity at sample milestones
        if samples_seen >= next_eval and step != max_steps:
            ppl = evaluate_ppl(model, val_loader, device, pad_id)
            wall = time.perf_counter() - t0
            val_steps.append(step); val_samples.append(samples_seen)
            val_times.append(wall);  val_ppls.append(ppl)
            next_eval += eval_every_samples
            if print_progress:
                print(f"      step={step:5d}  samples={samples_seen:7,}  "
                      f"loss={loss_val:.4f}  val_ppl={ppl:7.1f}  "
                      f"wall={wall/60:.1f}m")

            # Val-perplexity cap detector: requires BOTH val_ppl pinned at
            # cap for PPL_CAP_CONSECUTIVE consecutive evals AND
            # running_min_loss above PPL_CAP_LOSS_FLOOR (i.e. training never
            # progressed past "barely better than random uniform").
            # The dual check avoids false-positives on slow-but-honest
            # training (e.g. very tight grad_clip).
            if (len(val_ppls) >= PPL_CAP_CONSECUTIVE
                    and all(p >= PPL_CAP for p in val_ppls[-PPL_CAP_CONSECUTIVE:])
                    and math.isfinite(running_min_loss)
                    and running_min_loss > PPL_CAP_LOSS_FLOOR):
                status = "diverged"
                diverge_step = step
                diverge_loss = loss_val
                if print_progress:
                    print(f"      [pinned]   step={step:5d}  val_ppl pinned at "
                          f">={PPL_CAP:.0f} for {PPL_CAP_CONSECUTIVE} evals "
                          f"with running_min_loss={running_min_loss:.2f} "
                          f"(model never progressed past random)")
                break

    # Final eval if we survived
    final_ppl: Optional[float] = None
    if status == "stable":
        final_ppl = evaluate_ppl(model, val_loader, device, pad_id)
        wall = time.perf_counter() - t0
        val_steps.append(step); val_samples.append(samples_seen)
        val_times.append(wall);  val_ppls.append(final_ppl)
        if print_progress:
            print(f"      [done]     step={step:5d}  final_ppl={final_ppl:.1f}  "
                  f"wall={wall/60:.1f}m")

    # Snapshot tracker state BEFORE cleanup wipes the optimizer state
    condition_history = tracker.history if tracker is not None else {}
    condition_summary = tracker.summary() if tracker is not None else {}

    # Cleanup K-FAC hooks before next probe
    if hasattr(kfac_opt, "cleanup"):
        kfac_opt.cleanup()
    del kfac_opt, emb_opt, model
    torch.cuda.empty_cache()

    return {
        "variant":         variant,
        "kfac_lr":         kfac_lr,
        "damping":         damping,
        "momentum":        momentum,
        "grad_clip":       grad_clip,
        "status":          status,
        "diverge_step":    diverge_step,
        "diverge_loss":    diverge_loss,
        "steps_completed": len(train_losses),
        "samples_seen":    samples_seen,
        "wall_s":          time.perf_counter() - t0,
        "running_min_loss": (None if not math.isfinite(running_min_loss)
                             else running_min_loss),
        "final_ppl":       final_ppl,
        "median_step_ms":  float(np.median(step_times) * 1000) if step_times else None,
        "median_natgrad_norm": (float(np.median(natgrad_norms))
                                  if natgrad_norms else None),
        "natgrad_norms":   natgrad_norms,
        "train_losses":    train_losses,
        "val_steps":       val_steps,
        "val_samples":     val_samples,
        "val_times":       val_times,
        "val_ppls":        val_ppls,
        "condition_history":  condition_history,   # {layer: [(step, kappa_A, kappa_G)]}
        "condition_summary":  condition_summary,   # {layer: {mean/median/max kappa}}
    }


# ============================================================================
#  Phase 1 - LR sweep
# ============================================================================

def phase1_sweep(
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
) -> Dict:
    """Sweep momentum x LR x variant.

    Outer loop over momentum so the early sanity-check momentum (0.0) results
    print first, before the longer momentum=0.9 runs which accumulate noise
    in the buffer and can drag stable probes out to full PROBE_STEPS length.

    Returns nested structure:
        runs[variant][momentum_str] -> list[probe_summary]
        max_stable[variant][momentum_str] -> highest_stable_lr or None
    """
    n_probes = len(VARIANTS) * sum(len(LR_GRIDS[m]) for m in MOMENTUM_GRID)
    print("\n" + "=" * 70)
    print(f"  PHASE 1 - LR x momentum sweep   "
          f"({n_probes} probes x {PROBE_STEPS} steps)")
    for m in MOMENTUM_GRID:
        print(f"    momentum={m}: {len(LR_GRIDS[m])} LRs in "
              f"[{min(LR_GRIDS[m]):.0e}, {max(LR_GRIDS[m]):.0e}]")
    print(f"  Damping={DAMPING}  EmbLR={EMB_LR}  "
          f"Diverge: NaN/inf or {RELATIVE_DIVERGE_FACTOR}x running-min for "
          f"{RELATIVE_DIVERGE_WINDOW} steps")
    print("=" * 70)

    out_path = OUT / "stability_phase1_sweep.json"

    # ------------------------------------------------------------------
    # Resume support: if a partial JSON exists from a prior crashed run,
    # re-load the completed (variant, momentum, lr) probes and skip them
    # this time.  Avoids re-running 30+ minutes of work after a BSOD.
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict[str, List[Dict]]] = {
        v: {f"{m}": [] for m in MOMENTUM_GRID} for v in VARIANTS
    }
    completed: Set[Tuple[str, float, float]] = set()    # (variant, momentum, lr)
    def _is_obsolete_probe(r: Dict) -> bool:
        """Detect probes from the OLD divergence detector that are no longer
        valid under the current rules.

        The old detector triggered on absolute loss > 50, often firing at step 1
        from SmallGPT's wide default init (loss ~163).  The current detector
        only triggers on NaN/inf or relative drift after WARMUP_STEPS_BEFORE_CHECK.
        Any saved probe that reports diverging in the warmup window with a
        finite loss can't come from the new detector and should be re-run.
        """
        if r.get("status") != "diverged":
            return False
        ds = r.get("diverge_step")
        dl = r.get("diverge_loss")
        if ds is None or dl is None:
            return False
        try:
            return (int(ds) <= WARMUP_STEPS_BEFORE_CHECK
                    and math.isfinite(float(dl)))
        except (TypeError, ValueError):
            return False

    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            n_obsolete = 0
            for v, by_m in (prior.get("runs") or {}).items():
                if v not in all_results:
                    continue
                for mkey, runs in by_m.items():
                    try:
                        m = float(mkey)
                    except ValueError:
                        continue
                    if mkey not in all_results[v]:
                        continue
                    for r in runs:
                        if _is_obsolete_probe(r):
                            n_obsolete += 1
                            continue   # discard - will be re-run
                        all_results[v][mkey].append(r)
                        completed.add((v, m, float(r["kfac_lr"])))
            if n_obsolete:
                print(f"  Discarded {n_obsolete} obsolete probes from prior "
                      f"detector (diverged in warmup window with finite loss)")
            if completed:
                print(f"  Resuming from {out_path.name}: "
                      f"skipping {len(completed)} already-completed probes")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  WARN: could not parse existing {out_path.name} "
                  f"({e}); starting fresh.")
            all_results = {v: {f"{m}": [] for m in MOMENTUM_GRID}
                           for v in VARIANTS}
            completed = set()

    def _compute_max_stable() -> Dict[str, Dict[str, Optional[float]]]:
        ms: Dict[str, Dict[str, Optional[float]]] = {}
        for variant in VARIANTS:
            ms[variant] = {}
            for m in MOMENTUM_GRID:
                stable_lrs = [r["kfac_lr"]
                              for r in all_results[variant][f"{m}"]
                              if r["status"] == "stable"]
                ms[variant][f"{m}"] = max(stable_lrs) if stable_lrs else None
        return ms

    def _save():
        """Atomic write of the current partial sweep state.  Called after every
        probe so a crash loses at most one probe of work, not the whole sweep.
        Atomicity via write-to-tmp + os.replace; survives a kill mid-write."""
        out = {
            "hw":            hw,
            "lr_grids":      {f"{m}": LR_GRIDS[m] for m in MOMENTUM_GRID},
            "momentum_grid": MOMENTUM_GRID,
            "damping":       DAMPING,
            "probe_steps":   PROBE_STEPS,
            "emb_lr":        EMB_LR,
            "max_stable":    _compute_max_stable(),
            "runs":          all_results,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, out_path)
        return out

    for momentum in MOMENTUM_GRID:
        lr_grid = LR_GRIDS[momentum]
        print(f"\n  ===== momentum = {momentum}  "
              f"(LRs: {[f'{x:.0e}' for x in lr_grid]}) =====")
        for variant in VARIANTS:
            print(f"\n  -- {variant}  (momentum={momentum}) --")
            for lr in lr_grid:
                if (variant, momentum, lr) in completed:
                    print(f"    probe lr={lr:.0e}  (already done, skipping)")
                    continue
                print(f"    probe lr={lr:.0e} ...")
                res = run_probe(
                    variant=variant, kfac_lr=lr, damping=DAMPING,
                    momentum=momentum,
                    max_steps=PROBE_STEPS, vocab_size=vocab_size,
                    train_loader_factory=train_loader_factory,
                    val_loader=val_loader, pad_id=pad_id, device=device,
                    seed=42, record_natgrad=False, print_progress=False,
                )
                summary = {k: v for k, v in res.items()
                           if k not in ("train_losses", "natgrad_norms")}
                all_results[variant][f"{momentum}"].append(summary)
                tag = "stable  " if res["status"] == "stable" else "DIVERGED"
                if res["final_ppl"] is not None:
                    ppl_str = f"final_ppl={res['final_ppl']:.0f}"
                else:
                    div_loss = res.get("diverge_loss")
                    ppl_str = (f"@step {res['diverge_step']} "
                               f"loss={div_loss:.2f}" if div_loss is not None
                               else f"@step {res['diverge_step']}")
                step_ms = res.get("median_step_ms")
                step_ms_str = f"{step_ms:.1f}" if step_ms is not None else "n/a"
                min_loss = res.get("running_min_loss")
                min_loss_str = f"{min_loss:.3f}" if min_loss is not None else "n/a"
                print(f"      -> {tag}  {ppl_str}  "
                      f"step_ms={step_ms_str}  "
                      f"min_train_loss={min_loss_str}")
                _save()    # incremental save after every probe

    # Final save with the full max_stable table computed and printed
    out = _save()
    max_stable = out["max_stable"]

    print("\n  Phase 1 frontier  (max stable LR per variant):")
    print(f"    {'variant':<14}  {'mom=0.0':>12}  {'mom=0.9':>12}  ratio")
    for v in VARIANTS:
        lr0 = max_stable[v][f"{MOMENTUM_GRID[0]}"]
        lr1 = max_stable[v][f"{MOMENTUM_GRID[1]}"]
        ratio = (f"{lr0/lr1:.1f}x" if lr0 and lr1 else "n/a")
        s0 = f"{lr0:.0e}" if lr0 else "NONE"
        s1 = f"{lr1:.0e}" if lr1 else "NONE"
        print(f"    {v:<14}  {s0:>12}  {s1:>12}  {ratio}")

    print(f"\n  Saved -> {out_path}")
    return out


# ============================================================================
#  Phase 2 - convergence at max stable LR
# ============================================================================

def time_to_ppl(val_ppls: List[float], val_times: List[float],
                target: float) -> Optional[float]:
    """Wall seconds at which val_ppl first dropped below target.  None if never."""
    for ppl, t in zip(val_ppls, val_times):
        if ppl <= target:
            return t
    return None


def phase2_convergence(
    max_stable: Dict[str, Dict[str, Optional[float]]],
    train_loader_factory: Callable,
    val_loader,
    vocab_size: int,
    pad_id: int,
    device: torch.device,
    hw: dict,
    momentum: float = PHASE2_MOMENTUM,
) -> Dict:
    """Run PHASE2_STEPS at each variant's max stable LR for the chosen momentum."""
    print("\n" + "=" * 70)
    print(f"  PHASE 2 - Convergence at max stable LR (momentum={momentum})   "
          f"({PHASE2_STEPS} steps)")
    print("=" * 70)

    mom_key = f"{momentum}"
    out_path = OUT / "stability_phase2_runs.json"

    # Resume: load existing per-variant convergence runs and skip them.
    runs: Dict[str, Dict] = {}
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            if (prior.get("momentum") == momentum
                    and prior.get("phase2_steps") == PHASE2_STEPS):
                for v, r in (prior.get("runs") or {}).items():
                    if (v in VARIANTS
                            and r.get("status") not in (None, "no_stable_lr")
                            and r.get("steps_completed", 0) >= PHASE2_STEPS):
                        runs[v] = r
                if runs:
                    print(f"  Resuming from {out_path.name}: "
                          f"{len(runs)} variant(s) already complete: "
                          f"{list(runs.keys())}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  WARN: could not parse existing {out_path.name} "
                  f"({e}); starting fresh.")

    def _save_phase2():
        out = {
            "hw":           hw,
            "damping":      DAMPING,
            "emb_lr":       EMB_LR,
            "momentum":     momentum,
            "phase2_steps": PHASE2_STEPS,
            "max_stable":   max_stable,
            "ppl_targets":  PPL_TARGETS,
            "runs":         runs,
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=str))
        os.replace(tmp, out_path)
        return out

    for variant in VARIANTS:
        if variant in runs:
            r = runs[variant]
            print(f"\n  -- {variant}: already complete "
                  f"(final_ppl={r.get('final_ppl', 'n/a')}), skipping --")
            continue
        per_variant = max_stable.get(variant) or {}
        lr = per_variant.get(mom_key)
        if lr is None:
            print(f"\n  -- {variant}: SKIPPED (no stable LR @ momentum={momentum}) --")
            runs[variant] = {"variant": variant, "status": "no_stable_lr",
                             "momentum": momentum}
            continue
        print(f"\n  -- {variant} @ lr={lr:.0e}, momentum={momentum} --")
        res = run_probe(
            variant=variant, kfac_lr=lr, damping=DAMPING,
            momentum=momentum,
            max_steps=PHASE2_STEPS, vocab_size=vocab_size,
            train_loader_factory=train_loader_factory,
            val_loader=val_loader, pad_id=pad_id, device=device,
            seed=42,
            record_natgrad=True,
            record_condition=True,    # log kappa(A), kappa(G) per layer
            condition_log_every=100,  # 50 measurements over 5000 steps
            print_progress=True,
        )
        ttt = {f"ppl<={int(t)}": time_to_ppl(res["val_ppls"], res["val_times"], t)
               for t in PPL_TARGETS}
        res["time_to_target"] = ttt
        runs[variant] = res

        # Aggregate kappa across layers for a one-line summary
        cs = res.get("condition_summary") or {}
        if cs:
            avg_ka = float(np.mean([v["mean_kappa_A"] for v in cs.values()]))
            avg_kg = float(np.mean([v["mean_kappa_G"] for v in cs.values()]))
            max_ka = float(max(v["max_kappa_A"] for v in cs.values()))
            max_kg = float(max(v["max_kappa_G"] for v in cs.values()))
            kappa_str = (f"  avg_kappa(A)={avg_ka:.1e}  avg_kappa(G)={avg_kg:.1e}  "
                         f"max_kappa(A)={max_ka:.1e}  max_kappa(G)={max_kg:.1e}")
        else:
            kappa_str = ""
        print(f"    final_ppl={res['final_ppl']:.1f}  "
              f"wall={res['wall_s']/60:.1f}m  "
              f"median_natgrad_norm={res['median_natgrad_norm']}{kappa_str}")
        _save_phase2()    # incremental save after every variant completes

    out = _save_phase2()
    print(f"\n  Saved -> {out_path}")
    return out


# ============================================================================
#  Plotting
# ============================================================================

def make_plots(phase1: Optional[Dict], phase2: Optional[Dict]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping plots.")
        return

    # ---- Phase 1: LR frontier per momentum -----
    if phase1 is not None:
        moms = phase1.get("momentum_grid", MOMENTUM_GRID)
        fig, axes = plt.subplots(1, len(moms), figsize=(7 * len(moms), 5),
                                  sharey=True)
        if len(moms) == 1:
            axes = [axes]
        colors = {"ClassicKFAC": "tab:red",
                  "OlsSMKFAC":   "tab:blue",
                  "IFKFAC":   "tab:green"}
        for ax, momentum in zip(axes, moms):
            mkey = f"{momentum}"
            for variant, by_mom in phase1["runs"].items():
                runs = by_mom.get(mkey, [])
                xs_stable, ys_stable = [], []
                xs_div,    ys_div    = [], []
                for r in runs:
                    lr = r["kfac_lr"]
                    if r["status"] == "stable":
                        ppl = r["final_ppl"] if r["final_ppl"] else 9999.0
                        xs_stable.append(lr); ys_stable.append(ppl)
                    else:
                        xs_div.append(lr); ys_div.append(9999.0)
                c = colors.get(variant, "gray")
                ax.plot(xs_stable, ys_stable, "o-", color=c,
                        label=f"{variant} stable")
                ax.plot(xs_div, ys_div, "x", color=c, markersize=10,
                        label=f"{variant} diverged")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("K-FAC learning rate")
            ax.set_title(f"momentum = {momentum}")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel(
            f"Validation perplexity at step {PROBE_STEPS}  (9999 = diverged)")
        fig.suptitle("Phase 1: max stable LR per variant, by momentum")
        fig.tight_layout()
        out_path = OUT / "stability_lr_frontier.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved -> {out_path}")

    # ---- Phase 2: convergence curves -----
    if phase2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_steps, ax_wall = axes
        for variant, r in phase2["runs"].items():
            if r.get("status") == "no_stable_lr":
                continue
            ax_steps.plot(r["val_samples"], r["val_ppls"], "o-",
                          label=f"{variant} (lr={r['kfac_lr']:.0e})")
            ax_wall.plot(r["val_times"], r["val_ppls"], "o-",
                          label=f"{variant} (lr={r['kfac_lr']:.0e})")
        for ax, xl in [(ax_steps, "Samples seen"), (ax_wall, "Wall-clock seconds")]:
            ax.set_yscale("log")
            ax.set_ylabel("Validation perplexity")
            ax.set_xlabel(xl)
            ax.legend(fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
        ax_steps.set_title("Phase 2: convergence by samples")
        ax_wall.set_title("Phase 2: convergence by wall time")
        fig.tight_layout()
        out_path = OUT / "stability_convergence.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved -> {out_path}")


def write_summary_csv(phase1: Optional[Dict], phase2: Optional[Dict]):
    csv_path = OUT / "stability_summary.csv"
    lines = ["phase,variant,momentum,kfac_lr,damping,status,steps,final_ppl,"
             "wall_s,median_step_ms,min_train_loss,median_natgrad_norm"]
    if phase1:
        for variant, by_mom in phase1["runs"].items():
            for mkey, runs in by_mom.items():
                for r in runs:
                    lines.append(",".join(str(x) for x in [
                        "1", variant, mkey, r["kfac_lr"], r["damping"],
                        r["status"], r["steps_completed"],
                        r.get("final_ppl") or "",
                        f"{r['wall_s']:.1f}", r.get("median_step_ms") or "",
                        r.get("running_min_loss") or "",
                        r.get("median_natgrad_norm") or "",
                    ]))
    if phase2:
        p2_mom = phase2.get("momentum", "")
        for variant, r in phase2["runs"].items():
            if r.get("status") == "no_stable_lr":
                continue
            lines.append(",".join(str(x) for x in [
                "2", variant, p2_mom, r["kfac_lr"], r["damping"],
                r["status"], r["steps_completed"],
                r.get("final_ppl") or "",
                f"{r['wall_s']:.1f}", r.get("median_step_ms") or "",
                r.get("running_min_loss") or "",
                r.get("median_natgrad_norm") or "",
            ]))
    csv_path.write_text("\n".join(lines))
    print(f"  Saved -> {csv_path}")


# ============================================================================
#  Main
# ============================================================================

def main():
    # Declare globals FIRST - Python requires `global` before any reference
    # in the function, and argparse's default= reads these names below.
    global PROBE_STEPS, PHASE2_STEPS, VARIANTS, PHASE2_MOMENTUM

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="Comma-separated variants to include "
                             "(default: ClassicKFAC,OlsSMKFAC,IFKFAC)")
    parser.add_argument("--probe-steps", type=int, default=PROBE_STEPS)
    parser.add_argument("--phase2-steps", type=int, default=PHASE2_STEPS)
    parser.add_argument("--phase2-momentum", type=float, default=PHASE2_MOMENTUM,
                        help="Which momentum value's max-stable LRs to use for "
                             "Phase 2 convergence runs. Must be one of "
                             "MOMENTUM_GRID. Default: 0.9 (deployment).")
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants
    PROBE_STEPS = args.probe_steps
    PHASE2_STEPS = args.phase2_steps
    PHASE2_MOMENTUM = args.phase2_momentum
    if PHASE2_MOMENTUM not in MOMENTUM_GRID:
        print(f"  ERROR: --phase2-momentum {PHASE2_MOMENTUM} must be one of "
              f"MOMENTUM_GRID = {MOMENTUM_GRID}")
        sys.exit(1)
    VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in VARIANTS if v not in ("ClassicKFAC", "OlsSMKFAC", "IFKFAC")]
    if invalid:
        print(f"  ERROR: unknown variants {invalid}")
        sys.exit(1)

    print("\nStability Benchmark - K-FAC numerical stability comparison")
    print("=" * 70)

    device = get_device()
    hw = get_hardware_info()
    print(f"  Hardware: {hw.get('gpu_name')}  |  CUDA {hw.get('cuda_version')}  "
          f"|  torch {hw.get('torch_version')}")

    train_loader_factory, val_loader, vocab_size = build_data(device)
    pad_id = vocab_size - 1   # GPT2 uses eos as pad

    phase1_out = None
    phase2_out = None

    if args.phase in ("1", "all"):
        phase1_out = phase1_sweep(
            train_loader_factory, val_loader, vocab_size, pad_id, device, hw,
        )
        max_stable = phase1_out["max_stable"]
    else:
        # Phase 2 only - load Phase 1 results if available
        p1_path = OUT / "stability_phase1_sweep.json"
        if not p1_path.exists():
            print("  ERROR: --phase 2 requires existing Phase 1 results "
                  f"at {p1_path}. Run --phase 1 first or use --phase all.")
            sys.exit(2)
        phase1_out = json.loads(p1_path.read_text())
        # Re-coerce nested {variant: {mom_str: lr_or_None}} from JSON
        max_stable = {}
        for variant, by_mom in phase1_out["max_stable"].items():
            max_stable[variant] = {
                mkey: (float(v) if v is not None else None)
                for mkey, v in by_mom.items()
            }
        print(f"  Loaded Phase 1 results from {p1_path}")
        print(f"  Max stable LR table: {max_stable}")

    if args.phase in ("2", "all"):
        phase2_out = phase2_convergence(
            max_stable, train_loader_factory, val_loader,
            vocab_size, pad_id, device, hw,
            momentum=PHASE2_MOMENTUM,
        )

    print("\n" + "=" * 70)
    print("  Writing summary + plots")
    print("=" * 70)
    write_summary_csv(phase1_out, phase2_out)
    make_plots(phase1_out, phase2_out)
    print("\nDone.")


if __name__ == "__main__":
    main()
