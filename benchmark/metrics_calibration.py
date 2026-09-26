"""
benchmark/metrics_calibration.py

Standard calibration / probabilistic-prediction-quality metrics for
Bayesian-deep-learning evaluation.  All metrics take predictive probabilities
(post-MC-averaging) and ground-truth labels.

Metrics:
  - accuracy
  - negative log-likelihood (NLL)
  - expected calibration error (ECE, 15-bin)
  - Brier score (multi-class)
  - predictive entropy (per-input scalar; used for OOD-detection AUROC)

References:
  Guo et al. 2017 ("On Calibration of Modern Neural Networks") — ECE.
  Brier 1950 — Brier score.
  Hendrycks & Gimpel 2017 — predictive entropy for OOD detection.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import numpy as np


@torch.no_grad()
def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy.  probs: (N, C), labels: (N,)."""
    preds = probs.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def negative_log_likelihood(
    probs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12,
) -> float:
    """Mean NLL = -1/N sum_i log p(y_i | x_i).  probs: (N, C), labels: (N,)."""
    safe = probs.clamp(min=eps)
    return float(-safe.gather(1, labels.unsqueeze(1)).log().mean().item())


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15,
) -> float:
    """ECE via equal-width bins on top-1 confidence.

    For each bin: |average confidence − empirical accuracy| weighted by bin size.
    Standard 15-bin protocol per Guo et al. 2017.
    """
    confidences, preds = probs.max(dim=-1)                     # (N,)
    correct = (preds == labels).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    n = labels.shape[0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = confidences[in_bin].mean().item()
        bin_acc  = correct[in_bin].mean().item()
        ece += abs(bin_conf - bin_acc) * (in_bin.sum().item() / n)
    return float(ece)


@torch.no_grad()
def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class Brier score: mean sum_c (p_c - 1[y=c])^2.

    Range [0, 2].  Lower is better.  Brier 1950.
    """
    n, c = probs.shape
    one_hot = torch.nn.functional.one_hot(labels, num_classes=c).float()
    return float(((probs - one_hot) ** 2).sum(dim=-1).mean().item())


@torch.no_grad()
def predictive_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-input predictive entropy H[p̂] = -sum_c p̂_c log p̂_c.

    Returns (N,) tensor — used as the OOD-detection score.
    """
    safe = probs.clamp(min=eps)
    return -(safe * safe.log()).sum(dim=-1)


def reliability_diagram_data(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """Return arrays for a reliability diagram.

    Returns dict with keys:
      bin_centers : (n_bins,) midpoint confidence per bin
      bin_acc     : (n_bins,) empirical accuracy per bin
      bin_conf    : (n_bins,) average confidence per bin
      bin_count   : (n_bins,) number of samples per bin
    """
    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).cpu().numpy()
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        bin_acc[i]   = float(correct[in_bin].mean().item())
        bin_conf[i]  = float(confidences[in_bin].mean().item())
        bin_count[i] = int(in_bin.sum().item())
    return {
        "bin_centers": bin_centers,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
    }


def all_metrics(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15,
) -> Dict[str, float]:
    """Compute all four headline metrics at once."""
    return {
        "accuracy": accuracy(probs, labels),
        "nll":      negative_log_likelihood(probs, labels),
        "ece":      expected_calibration_error(probs, labels, n_bins=n_bins),
        "brier":    brier_score(probs, labels),
    }
