"""
optimizer/laplace.py

K-FAC Laplace approximation — Daxberger et al. 2021 ("Laplace Redux") formulation.

The posterior over weights is approximated as
    p(w | data) ≈ N(w*, Σ),    Σ = (N · F + λI)^{-1}
where F is the K-FAC-approximated Fisher: per layer  F_l ≈ A_l ⊗ G_l.

Daxberger 2021 (§4 and Appendix A) uses **eigendecomposition** of the K-FAC
factors, not Cholesky.  For each layer:
    A = U_A diag(d_A) U_A^T,      G = U_G diag(d_G) U_G^T
The posterior covariance per layer is the Kronecker structure
    Σ_l = (N · A + λI)^{-1} ⊗ (N · G + λI)^{-1}
     = U_A diag(1/(√N·d_A + √λ)²) U_A^T  ⊗  U_G diag(1/(√N·d_G + √λ)²) U_G^T
(when we distribute the per-factor √λ damping; this is K-FAC's standard
approximate Kronecker factorization of the prior — Daxberger §A.2).

Sampling W_l ~ N(0, Σ_l) uses the closed-form Kronecker square root:
    W_l = U_G · diag(1/√(√N·d_G + √λ)) · Ξ · diag(1/√(√N·d_A + √λ)) · U_A^T
where Ξ has iid N(0, 1) entries of shape (n_out × n_in).

Two flavors of K-FAC Laplace factor capture:
  • Classic K-FAC: eigh(A) directly.  At bf16, the κ² catastrophe shows up
    in the smallest eigenvalues (which can go negative) — same κ² stability
    issue as inv() because they're factorizations of the same κ²-conditioned A.
  • IFKFAC: svd(R) where R is the QR factor of X.  Singular values of
    R are sqrt of eigenvalues of A, and R inherits only κ¹ stability.  This
    is the path that survives bf16 at typical NN conditioning.

References
----------
- Daxberger et al. 2021. "Laplace Redux — Effortless Bayesian Deep Learning."
  arXiv:2106.14806.  §4 (K-FAC Laplace) and Appendix A.
- George et al. 2018. "Fast approximate natural gradient descent in a
  Kronecker-factored eigenbasis."  arXiv:1806.03884.  (EKFAC — same EVD).
- Ritter et al. 2018. "Online structured Laplace approximations for
  overcoming catastrophic forgetting."  arXiv:1805.07810.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# A per-layer K-FAC Laplace factor is the eigendecomposition (U_A, d_A, U_G, d_G).
# U_A is (n_in, n_in) orthogonal; d_A is (n_in,) eigenvalues.
# U_G is (n_out, n_out) orthogonal; d_G is (n_out,) eigenvalues.
EVDFactor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class LaplacePosterior:
    """K-FAC Laplace posterior — Daxberger 2021 §4 + Appendix A.

    Parameters
    ----------
    model            : trained nn.Module (weights at MAP)
    factors_evd      : dict layer → (U_A, d_A, U_G, d_G) eigendecomposition
                        of the K-FAC factors per layer.
    prior_precision  : Tikhonov damping λ.
    dataset_size     : Number of training samples N.  Eigenvalues are scaled
                        by √N so the posterior precision matches the full-data
                        Fisher (N · F + λI), not the per-sample average F + λI.
    """

    # Class-level diagnostic flag — see benchmark/laplace_cifar10.py for use.
    _diag_first_call: bool = True

    def __init__(
        self,
        model: nn.Module,
        factors_evd: Dict[nn.Module, EVDFactor],
        prior_precision: float = 1.0,
        dataset_size: int = 1,
    ):
        self.model = model
        self.prior_precision = prior_precision
        self.dataset_size = dataset_size

        # Snapshot the MAP weights so sample_in_place can restore between draws.
        self._w_map: Dict[nn.Module, torch.Tensor] = {}
        for mod in factors_evd:
            self._w_map[mod] = mod.weight.data.clone()

        # Pre-compute per-layer 1/√(√N·d + √λ) — the diagonal scaling that
        # turns iid-Normal Ξ into a draw from the per-layer posterior.
        # Per-factor √λ damping is K-FAC's standard Kronecker-factored prior
        # (Daxberger 2021 §A.2; Martens & Grosse 2015 §6.2).
        sqrt_N    = float(dataset_size) ** 0.5
        sqrt_lam  = float(prior_precision) ** 0.5
        self._factors: Dict[nn.Module, EVDFactor] = {}
        for mod, (U_A, d_A, U_G, d_G) in factors_evd.items():
            # Eigenvalues of √N·A + √λ·I are √N·d_A + √λ (uniform shift in
            # the eigenbasis).  Posterior covariance eigenvalues per side
            # are 1/(√N·d + √λ); sampling uses the square root.
            sqrt_inv_d_A = 1.0 / torch.sqrt(sqrt_N * d_A + sqrt_lam)
            sqrt_inv_d_G = 1.0 / torch.sqrt(sqrt_N * d_G + sqrt_lam)
            self._factors[mod] = (
                U_A.to(torch.float32),
                sqrt_inv_d_A.to(torch.float32),
                U_G.to(torch.float32),
                sqrt_inv_d_G.to(torch.float32),
            )

    @torch.no_grad()
    def sample_in_place(self, generator: Optional[torch.Generator] = None):
        """Draw one weight sample from the per-layer posterior and overwrite
        model in place.

        For each layer with EVD factors:
            W_sample = W_map + U_G · diag(sqrt_inv_d_G) · Ξ · diag(sqrt_inv_d_A) · U_A^T
        where Ξ ~ N(0, I) of shape (n_out × n_in).

        Layers without EVD factors (BatchNorm, etc.) keep their MAP values.
        """
        import time
        diag = LaplacePosterior._diag_first_call
        if diag:
            print(f"      [sample_in_place] PROFILING FIRST CALL — "
                  f"{len(self._factors)} layers", flush=True)
        for mod, (U_A, sqrt_inv_d_A, U_G, sqrt_inv_d_G) in self._factors.items():
            if diag:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
            n_in = U_A.shape[0]
            n_out = U_G.shape[0]
            # Restore MAP
            mod.weight.data.copy_(self._w_map[mod])
            # Draw standard-normal Ξ on the same device, fp32
            xi = torch.randn(
                n_out, n_in, device=mod.weight.device,
                dtype=torch.float32, generator=generator,
            )
            # Element-wise scale Ξ by the per-axis sqrt(1/(√N·d + √λ))
            xi = xi * sqrt_inv_d_G.unsqueeze(1)   # left axis (n_out)
            xi = xi * sqrt_inv_d_A.unsqueeze(0)   # right axis (n_in)
            # Rotate into the original basis: W = U_G · Ξ_scaled · U_A^T
            delta_W = U_G @ xi @ U_A.t()
            if mod.weight.ndim > 2:
                # Conv2d: reshape (C_out, C_in*kH*kW) → original shape
                delta_W = delta_W.reshape(mod.weight.data.shape)
            mod.weight.data.add_(delta_W.to(mod.weight.dtype))
            if diag:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"        layer {type(mod).__name__}  n_in={n_in:>5}  "
                      f"n_out={n_out:>5}  → {elapsed:>7.1f} ms", flush=True)
        if diag:
            LaplacePosterior._diag_first_call = False
            print(f"      [sample_in_place] profiling done", flush=True)

    @torch.no_grad()
    def restore_map(self):
        """Reset weights to the MAP point."""
        for mod in self._factors:
            mod.weight.data.copy_(self._w_map[mod])

    @torch.no_grad()
    def predictive(
        self, x: torch.Tensor, n_samples: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Single-batch MC predictive.  Resamples weights per MC iteration.

        Prefer :meth:`predictive_full_loader` for efficient eval over a
        whole DataLoader — that variant resamples weights only N times
        instead of N × batches times.
        """
        was_training = self.model.training
        self.model.eval()
        probs_sum = None
        for _ in range(n_samples):
            self.sample_in_place(generator=generator)
            logits = self.model(x)
            p = torch.softmax(logits.float(), dim=-1)
            probs_sum = p if probs_sum is None else probs_sum + p
        self.restore_map()
        if was_training:
            self.model.train()
        return probs_sum / n_samples

    @torch.no_grad()
    def predictive_full_loader(
        self, loader, device, n_samples: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficient MC predictive over an entire DataLoader.

        Resamples weights `n_samples` times total (not `n_samples × n_batches`).
        ~10× faster than per-batch :meth:`predictive`.

        Returns
        -------
        probs  : (N_total, n_classes) — predictive class probabilities
        labels : (N_total,) — ground-truth labels
        """
        was_training = self.model.training
        self.model.eval()
        probs_sum = None
        labels_all = None
        for s in range(n_samples):
            self.sample_in_place(generator=generator)
            batch_probs = []
            batch_labels = [] if labels_all is None else None
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                logits = self.model(x)
                batch_probs.append(torch.softmax(logits.float(), dim=-1).cpu())
                if batch_labels is not None:
                    batch_labels.append(y)
            all_probs = torch.cat(batch_probs)
            probs_sum = all_probs if probs_sum is None else probs_sum + all_probs
            if labels_all is None:
                labels_all = torch.cat(batch_labels)
        self.restore_map()
        if was_training:
            self.model.train()
        return probs_sum / n_samples, labels_all
