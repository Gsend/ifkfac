"""
K-FAC optimizer with dual inversion paths (EVD and Cholesky).

Natural gradient update:  ΔW = G⁻¹ · ∇L · A⁻¹
where A = E[xxᵀ] (input Gram) and G = E[δδᵀ] (gradient Gram).

Two inversion strategies, routed per layer by Gram matrix dimension n:

  Cholesky-solve path  (n ≤ lu_max_dim, both A and G)
      Stores L = chol(Gram + λI) and solves at apply time via
      torch.cholesky_solve — never forms the explicit inverse, so
      condition number stays cond(A + λI) instead of cond(A)².
      2× cheaper than LU and numerically ideal for symmetric PSD matrices.
      CPU fallback uses the olssm Rust backend (faer SIMD).

  EVD path  (n > lu_max_dim or mixed dimensions)
      Stores (Q, diag(1/(λᵢ+δ))) from torch.linalg.eigh (cuSOLVER on GPU).
      CPU path routes through the olssm Rust backend (eigh_f32 /
      randomized_eigh_f32) for faer-accelerated decomposition.
"""

import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import logging

from optimizer.backend import eigh_f32, apply_kfac_eigen_f32, eigh_topk_f32, randomized_eigh_f32
from optimizer.errors import ConfigurationError
from optimizer.gram_estimator import GramMatrixEstimator
from optimizer.hooks import KFACHooks

logger = logging.getLogger(__name__)


class OlsSMKFAC(torch.optim.Optimizer):
    """K-FAC optimizer with olssm LU backend.

    Parameters
    ----------
    model : nn.Module
        The model whose Linear layers will be preconditioned.
    lr : float
        Learning rate. Default: 1e-3.
    damping : float
        Tikhonov damping λ added to Gram matrices before inversion.
        Lower values → more faithful curvature (olssm enables this).
        Default: 1e-2.
    factor_update_freq : int
        How often (in steps) to recompute Gram matrices A, G.
        Default: 10.
    decomp_update_freq : int
        How often (in steps) to recompute cached inverses A⁻¹, G⁻¹.
        Default: 10.
    weight_decay : float
        L2 regularisation coefficient. Default: 0.
    momentum : float
        SGD-style momentum on the preconditioned gradient. Default: 0.9.
    rank : int or None
        If set, use a rank-k approximation of A and G instead of the full
        eigen basis.  Reduces apply cost from O(4·n·d_out·d_in) to
        O((rank_g + rank_a)·d_out·d_in) — genuinely faster for rank << n.
        None (default) uses the full eigen basis.  Ignored when adaptive=True.
    randomized : bool
        When True (default) and rank is set (or adaptive=True), use randomized
        EVD to find the top-k eigenvectors in O(k·n²) instead of O(n³).
        Has no effect when neither rank nor adaptive is set.
    n_power_iter : int
        Number of power-iteration passes for the randomized EVD (default 1).
        More passes = more accurate but more expensive.  1 is sufficient for
        K-FAC Gram matrices whose eigenvalues decay rapidly.
    adaptive : bool
        When True, automatically choose the rank for each Gram matrix based on
        its size rather than applying a single global rank.  Rules applied
        per matrix:
          - n < adaptive_min_n  →  full EVD (no low-rank approximation)
          - n >= adaptive_min_n →  k = min(adaptive_rank_budget, n)
        This avoids the overhead of low-rank EVD on small layers (where it
        can actually be slower) while still speeding up large layers.
        Default: False.  Overrides the ``rank`` parameter when True.
    adaptive_min_n : int
        Minimum Gram matrix dimension to apply low-rank approximation.
        Matrices smaller than this use full EVD regardless of rank budget.
        Default: 256 (empirically safe — layers with n < 256 are too small
        to benefit from truncation at typical batch sizes).
    adaptive_rank_budget : int
        Maximum rank k to use when adaptive=True and n >= adaptive_min_n.
        Should match roughly the batch size used for training (the effective
        rank of a Gram matrix from a batch of B samples is at most B).
        Default: 64.
    lu_max_dim : int
        Gram matrices whose dimension n <= lu_max_dim use a Cholesky-
        based solve instead of EVD.  Default 4096 covers all standard BERT
        and transformer layer sizes (768, 3072) so the paper path fires for
        every layer by default.

        Chol solve: stores L = chol(A+λI) and solves via cholesky_solve —
                    never forms
                    the explicit inverse, so condition number stays cond(A + λI)
                    rather than squaring to cond(A)².  Faster than EVD for
                    small n and more numerically stable near singularity.
        EVD/random: better for large n where the randomized range-finder gives
                    a cheap low-rank approximation.

        Rule of thumb: LU wins for n <= 512 (O(n³) ≈ 134M FLOPs, sub-ms on
        GPU).  Set to 0 to disable (pure EVD everywhere).  Default: 512.
    gamma : float
        EMA decay for Kronecker factors: A ← γ·A_old + (1−γ)·A_batch.
        0.0 (default) disables EMA — factors are replaced each update.
        Higher γ smooths out per-batch noise at the cost of slower adaptation.
        OlsSMKFAC can safely use γ=0.95 because its EVD handles the
        near-singular matrices that aggressive smoothing can produce; ClassicKFAC
        should use a lower γ (≤0.9) since direct inversion is less stable.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        damping: float = 1e-2,
        factor_update_freq: int = 10,
        decomp_update_freq: int = 10,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        rank: Optional[int] = None,
        randomized: bool = True,
        n_power_iter: int = 1,
        adaptive: bool = False,
        adaptive_min_n: int = 256,
        adaptive_rank_budget: int = 64,
        grad_clip: Optional[float] = None,
        gamma: float = 0.0,
        max_gram_dim: int = 0,
        lu_max_dim: int = 4096,
        gram_estimator: Optional[GramMatrixEstimator] = None,
    ):
        # ── Parameter validation ─────────────────────────────────────────────
        if rank is not None and adaptive:
            raise ConfigurationError(
                "rank and adaptive=True are mutually exclusive. "
                "Use adaptive=True to auto-select rank per layer, "
                "or rank=k to use a fixed global rank — not both."
            )
        if damping <= 0:
            raise ConfigurationError(f"damping must be > 0, got {damping}.")
        logger.debug(
            "OlsSMKFAC init: factor_update_freq=%d  decomp_update_freq=%d  "
            "rank=%s  adaptive=%s",
            factor_update_freq, decomp_update_freq, rank, adaptive,
        )

        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay,
                        momentum=momentum)
        # Collect Linear and Conv2d layer parameters
        params = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params.append({"params": module.parameters()})
        super().__init__(params, defaults)

        self.model = model
        self.damping = damping
        self.factor_update_freq = factor_update_freq
        self.decomp_update_freq = decomp_update_freq
        self.rank = rank
        self.randomized = randomized      # use randomized EVD when rank is set
        self.n_power_iter = n_power_iter  # power-iteration passes for randomized EVD
        self.adaptive = adaptive
        self.adaptive_min_n = adaptive_min_n
        self.adaptive_rank_budget = adaptive_rank_budget
        self.grad_clip  = grad_clip  # max L2 norm per natural-gradient matrix (None=off)
        self.gamma      = gamma      # EMA decay for Gram matrices (0 = disabled)
        self.lu_max_dim = lu_max_dim # layers with n <= this use Cholesky solve instead of EVD

        # Per-layer rank choices recorded during _update_inverses for inspection.
        # Keys are module objects; values are (k_a, k_g) — None means full EVD,
        # "lu" means Cholesky-solve path (key kept as "lu" for backward compat).
        self.layer_ranks_: Dict[nn.Module, Tuple] = {}

        # Hook infrastructure — accept injected estimator or create default
        if gram_estimator is not None:
            self.hooks: GramMatrixEstimator = gram_estimator
        else:
            self.hooks = KFACHooks(model, max_gram_dim=max_gram_dim)
        self.hooks.enable()

        # Cached factors and decompositions.
        # _inverses  : EVD path   — stores (Q_A, inv_λ_A, Q_G, inv_λ_G) f32 tensors
        # _lu_factors: Chol path  — stores (L_A, L_G) lower Cholesky factors f32 tensors
        #   At apply time we call torch.cholesky_solve(grad, L_G) and
        #   torch.cholesky_solve(result.T, L_A).T — stays on GPU, no explicit inverse.
        self._factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._inverses: Dict[nn.Module, Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._lu_factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._momentum_buffers: Dict[nn.Module, torch.Tensor] = {}

        # Step counter
        self._step_count = 0

        # Timing instrumentation — bounded deques prevent unbounded memory growth
        _maxlen = 1000
        self.timing = {
            "factor_compute": deque(maxlen=_maxlen),
            "inversion":      deque(maxlen=_maxlen),
            "precondition":   deque(maxlen=_maxlen),
            "total_step":     deque(maxlen=_maxlen),
        }

    def _update_factors(self):
        """Recompute Gram matrix factors A, G, optionally EMA-smoothed.

        When gamma > 0, each new batch estimate is blended with the running
        average:  A ← γ·A_old + (1−γ)·A_batch.  This smooths per-batch noise
        at the cost of slower adaptation — critical on harder tasks (CIFAR-10,
        BERT) where single-batch Gram matrices are too noisy to precondition well.
        """
        t0 = time.perf_counter()
        new_factors = self.hooks.get_factors()
        self.hooks.clear()
        if self.gamma > 0.0 and self._factors:
            for module, (A_new, G_new) in new_factors.items():
                if module in self._factors:
                    A_old, G_old = self._factors[module]
                    new_factors[module] = (
                        self.gamma * A_old + (1.0 - self.gamma) * A_new,
                        self.gamma * G_old + (1.0 - self.gamma) * G_new,
                    )
        self._factors = new_factors
        self.timing["factor_compute"].append(time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Rank-selection helpers
    # ------------------------------------------------------------------

    def _effective_rank(self, n: int) -> Optional[int]:
        """Return the rank k to use for an n×n Gram matrix, or None for full EVD.

        Priority:
          1. adaptive=True   → per-matrix rule based on n
          2. rank is not None → fixed global rank (clamped to n)
          3. fallback        → None (full EVD)
        """
        if self.adaptive:
            if n < self.adaptive_min_n:
                return None  # small matrix: full EVD avoids overhead
            return min(self.adaptive_rank_budget, n)
        if self.rank is not None:
            return min(self.rank, n)
        return None  # full EVD

    def _decompose_torch(
        self, mat: torch.Tensor, k: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EVD entirely on the tensor's device (GPU or CPU).

        Replaces the old numpy/Rust path so that on CUDA the decomposition
        runs via cuSOLVER (torch.linalg.eigh) with zero host–device transfers.

        Parameters
        ----------
        mat : (n, n) symmetric float32 tensor on target device
        k   : None → full EVD (n×n Q); int → low-rank (n×k Q)

        Returns
        -------
        Q       : (n, n) or (n, k) float32 — eigenvectors as columns
        inv_lam : (n,)  or (k,)  float32 — 1 / (λ + damping)
        """
        # Ensure symmetry (numerical drift can break eigh)
        mat = (mat + mat.T) * 0.5
        is_cpu = mat.device.type == "cpu"

        if k is None or k >= mat.shape[0]:
            if is_cpu:
                # CPU: olssm Rust backend — faer SIMD-accelerated full EVD.
                # Falls back to numpy.linalg.eigh when Rust is unavailable.
                q_np, inv_lam_np = eigh_f32(mat.numpy(), self.damping)
                return torch.from_numpy(q_np), torch.from_numpy(inv_lam_np)
            # GPU: cuSOLVER via torch.linalg.eigh (zero host-device transfer)
            eigenvalues, Q = torch.linalg.eigh(mat)
            inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
            return Q, inv_lam

        if self.randomized:
            if is_cpu:
                # CPU: olssm Rust backend — randomized EVD (Halko-Martinsson-Tropp)
                q_np, inv_lam_np = randomized_eigh_f32(
                    mat.numpy(), k, self.n_power_iter, self.damping)
                return torch.from_numpy(q_np), torch.from_numpy(inv_lam_np)
            # GPU: Halko-Martinsson-Tropp randomized EVD — stays on GPU
            n = mat.shape[0]
            Omega = torch.randn(n, k, device=mat.device, dtype=mat.dtype)
            Y = mat @ Omega
            for _ in range(self.n_power_iter):
                Y = mat @ (mat @ Y)
            Q_basis, _ = torch.linalg.qr(Y)          # (n, k) orthonormal
            B = Q_basis.T @ mat @ Q_basis              # (k, k) small sketch
            B = (B + B.T) * 0.5
            eigenvalues, V = torch.linalg.eigh(B)
            Q = Q_basis @ V                            # (n, k) back to full space
            inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
            return Q, inv_lam

        # Deterministic top-k
        if is_cpu:
            # CPU: olssm Rust backend — top-k EVD via faer
            q_np, inv_lam_np = eigh_topk_f32(mat.numpy(), k, self.damping)
            return torch.from_numpy(q_np), torch.from_numpy(inv_lam_np)
        # GPU: full eigh then slice (cuSOLVER is fast enough that slicing wins
        # over a Python-side truncated solve for typical K-FAC dimensions)
        eigenvalues, Q_full = torch.linalg.eigh(mat)
        eigenvalues = eigenvalues[-k:]
        Q = Q_full[:, -k:]
        inv_lam = 1.0 / (eigenvalues + self.damping).clamp(min=1e-8)
        return Q, inv_lam

    def _use_lu(self, n: int) -> bool:
        """Return True if an n×n Gram matrix should use the LU path."""
        return self.lu_max_dim > 0 and n <= self.lu_max_dim

    def _decompose_lu(
        self, mat: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Cholesky factor of a damped Gram matrix (Cholesky approach).

        Instead of computing (A + λI)⁻¹ explicitly — which squares the
        condition number — we store the lower Cholesky factor L satisfying
        (A + λI) = L Lᵀ, then call torch.cholesky_solve at apply time.

        Cholesky is preferred over LU for symmetric PSD Gram matrices:
          - 2× fewer FLOPs (exploits symmetry, no pivoting)
          - Numerically ideal for PSD inputs; pivoting never needed
          - Condition stays cond(A + λI) rather than cond(A)²

        Parameters
        ----------
        mat : (n, n) symmetric float32 tensor on target device

        Returns
        -------
        L : (n, n) lower-triangular float32 — Cholesky factor of (mat + λI)
        """
        mat = (mat + mat.T) * 0.5          # enforce exact symmetry
        n   = mat.shape[0]
        eye_n = torch.eye(n, device=mat.device, dtype=mat.dtype)

        # Sanity check: if the input has any NaN/inf, all downstream paths
        # will fail.  Return identity-Cholesky immediately - this step's
        # update for this layer will be effectively SGD-like, and the
        # divergence detector upstream will catch the broken model state.
        if not torch.isfinite(mat).all():
            logger.warning(
                "Cholesky input has non-finite values; returning identity "
                "(this layer's update reduces to SGD this step)"
            )
            return torch.linalg.cholesky(eye_n * (1.0 + self.damping))

        # Progressive retry: try Cholesky with increasing damping multipliers.
        # Some Gram matrices accumulated during chaotic training have small
        # negative eigenvalues from numerical drift in the streaming sums;
        # the original "10x damping" retry isn't enough when high grad_clip
        # lets large gradients flow through.
        for damp_mult in (1.0, 10.0, 100.0, 1000.0):
            mat_damp = mat + (self.damping * damp_mult) * eye_n
            try:
                return torch.linalg.cholesky(mat_damp)
            except torch.linalg.LinAlgError:
                continue

        # Final fallback: project to nearest-PSD via eigendecomposition.
        # Wrapped in try/except so even a pathological eigh failure cannot
        # crash the whole benchmark - identity-Cholesky becomes the
        # last-ditch fallback.
        try:
            logger.warning(
                "Cholesky failed at all damping multipliers; falling back to "
                "EVD-based PSD projection (matrix may be severely ill-conditioned)"
            )
            eigvals, eigvecs = torch.linalg.eigh(mat)
            eigvals_clamped = eigvals.clamp(min=max(self.damping, 1e-8))
            mat_psd = (eigvecs * eigvals_clamped) @ eigvecs.T
            mat_psd = (mat_psd + mat_psd.T) * 0.5
            return torch.linalg.cholesky(mat_psd)
        except (torch.linalg.LinAlgError, RuntimeError) as e:
            logger.warning(
                f"EVD-based fallback also failed ({type(e).__name__}: {e}); "
                f"returning identity Cholesky as last resort"
            )
            return torch.linalg.cholesky(eye_n * (1.0 + self.damping))

    # ------------------------------------------------------------------

    def _update_inverses(self):
        """Recompute cached decompositions for all layers.

        Routing logic per layer (both A and G must qualify for LU path):
          n_a <= lu_max_dim AND n_g <= lu_max_dim  →  Cholesky solve path
          otherwise                                →  EVD / randomized-EVD path

        Chol path: stores (L_A, L_G) — lower Cholesky factors of (A+λI), (G+λI).
                   torch.cholesky_solve at apply time; never forms explicit inverse.
        EVD path:  stores (Q_A, inv_λ_A, Q_G, inv_λ_G) — 4-matmul apply.
                   CPU uses olssm Rust backend (faer SIMD); GPU uses cuSOLVER.

        Both paths run entirely on the model's device (GPU via cuSOLVER/cuBLAS
        when available) with no CPU/numpy roundtrip.
        """
        t0 = time.perf_counter()
        for module, (A, G) in self._factors.items():
            dtype = module.weight.dtype

            # Cast to float32 for numerical stability; keep on device
            A_f = A.to(dtype=torch.float32)
            G_f = G.to(dtype=torch.float32)

            # Guard: skip corrupt Gram matrices (NaN/inf from diverged training)
            if not (torch.isfinite(A_f).all() and torch.isfinite(G_f).all()):
                continue

            n_a, n_g = A_f.shape[0], G_f.shape[0]

            if self._use_lu(n_a) and self._use_lu(n_g):
                # ── Cholesky path ───────────────────────────────────
                # Compute Cholesky factor L of (Gram + λI) once; solve via
                # cholesky_solve at apply time.  Re-factored every
                # decomp_update_freq steps so gradual damping decay is picked up.
                L_A = self._decompose_lu(A_f).to(dtype=dtype)
                L_G = self._decompose_lu(G_f).to(dtype=dtype)
                self._lu_factors[module] = (L_A, L_G)
                # Remove any stale EVD cache for this layer
                self._inverses.pop(module, None)
                self.layer_ranks_[module] = ("lu", "lu")

            else:
                # ── EVD / randomized-EVD path ──────────────────────────────
                k_a = self._effective_rank(n_a)
                k_g = self._effective_rank(n_g)

                Q_A, inv_lam_A = self._decompose_torch(A_f, k_a)
                Q_G, inv_lam_G = self._decompose_torch(G_f, k_g)

                self._inverses[module] = (
                    Q_A.to(dtype=dtype),
                    inv_lam_A.to(dtype=dtype),
                    Q_G.to(dtype=dtype),
                    inv_lam_G.to(dtype=dtype),
                )
                # Remove any stale LU cache for this layer
                self._lu_factors.pop(module, None)
                self.layer_ranks_[module] = (k_a, k_g)

        self.timing["inversion"].append(time.perf_counter() - t0)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single K-FAC optimisation step.

        The natural gradient update for each Linear layer is:
            ΔW = G⁻¹ · ∇L_W · A⁻¹
            Δb = G⁻¹ · ∇L_b   (bias uses only gradient preconditioning)
        """
        t_total = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Update factors periodically
        if self._step_count % self.factor_update_freq == 1 or self.factor_update_freq == 1:
            self._update_factors()

        # Update inverses periodically
        if self._step_count % self.decomp_update_freq == 1 or self.decomp_update_freq == 1:
            if self._factors:
                self._update_inverses()

        # Apply preconditioned update to each Linear layer
        t_precond = time.perf_counter()
        for module in self.hooks.linear_layers:
            has_evd = module in self._inverses
            has_lu  = module in self._lu_factors

            if not has_evd and not has_lu:
                # No preconditioning available yet — fall back to SGD
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    grad = p.grad
                    for group in self.param_groups:
                        if any(p is pp for pp in group["params"]):
                            wd = group["weight_decay"]
                            lr = group["lr"]
                            mom = group["momentum"]
                            break
                    if wd > 0:
                        grad = p.grad.add(p.data, alpha=wd)
                    p.data.add_(grad, alpha=-lr)
                continue

            # Get hyperparams for this layer
            for group in self.param_groups:
                if any(p is module.weight for p in group["params"]):
                    lr  = group["lr"]
                    wd  = group["weight_decay"]
                    mom = group["momentum"]
                    break

            # Conv2d: flatten weight gradient to 2D, reshape back after
            is_conv  = isinstance(module, nn.Conv2d)

            # ── Weight update ─────────────────────────────────────────────
            if module.weight.grad is not None:
                raw_grad = module.weight.grad
                if is_conv:
                    raw_grad = raw_grad.view(module.weight.shape[0], -1)
                grad_w = raw_grad.add(module.weight.data.view_as(raw_grad), alpha=wd) \
                         if wd > 0 else raw_grad

                if has_lu:
                    # ── Cholesky path ────────────────────────────────
                    # ΔW = (G + λI)⁻¹ · ∇W · (A + λI)⁻¹
                    #
                    # L_G, L_A are lower Cholesky factors: G+λI = L_G L_Gᵀ etc.
                    # Step 1: cholesky_solve(∇W,  L_G) → C  = (G+λI)⁻¹ ∇W
                    # Step 2: cholesky_solve(Cᵀ,  L_A) → ΔWᵀ = (A+λI)⁻¹ Cᵀ
                    # Never forms explicit inverse — condition stays cond(A+λI).
                    L_A, L_G = self._lu_factors[module]
                    C        = torch.cholesky_solve(grad_w, L_G)         # (d_out, d_in)
                    nat_grad = torch.cholesky_solve(C.T, L_A).T          # (d_out, d_in)

                else:
                    # ── EVD path ───────────────────────────────────────────
                    # ΔW = Q_G d_G Q_Gᵀ ∇W Q_A d_A Q_Aᵀ
                    Q_A, inv_lam_A, Q_G, inv_lam_G = self._inverses[module]
                    tmp      = Q_G.T @ grad_w @ Q_A
                    tmp      = tmp * (inv_lam_G.unsqueeze(1) * inv_lam_A.unsqueeze(0))
                    nat_grad = Q_G @ tmp @ Q_A.T

                # Gradient clipping (guards early steps with rank-deficient Grams)
                if self.grad_clip is not None:
                    grad_norm = nat_grad.norm()
                    if grad_norm > self.grad_clip:
                        nat_grad = nat_grad * (self.grad_clip / grad_norm)

                # Momentum
                if mom > 0:
                    if module not in self._momentum_buffers:
                        self._momentum_buffers[module] = torch.zeros_like(nat_grad)
                    buf = self._momentum_buffers[module]
                    buf.mul_(mom).add_(nat_grad)
                    nat_grad = buf

                if is_conv:
                    nat_grad = nat_grad.view_as(module.weight)
                module.weight.data.add_(nat_grad, alpha=-lr)

            # ── Bias update ───────────────────────────────────────────────
            # Δb = (G + λI)⁻¹ · ∇b  (A not involved — bias has no input dim)
            if module.bias is not None and module.bias.grad is not None:
                grad_b = module.bias.grad.add(module.bias.data, alpha=wd) \
                         if wd > 0 else module.bias.grad

                if has_lu:
                    L_A, L_G = self._lu_factors[module]
                    nat_grad_b = torch.cholesky_solve(grad_b.unsqueeze(1), L_G).squeeze(1)
                else:
                    Q_A, inv_lam_A, Q_G, inv_lam_G = self._inverses[module]
                    tmp_b      = Q_G.T @ grad_b
                    tmp_b      = tmp_b * inv_lam_G
                    nat_grad_b = Q_G @ tmp_b

                module.bias.data.add_(nat_grad_b, alpha=-lr)

        self.timing["precondition"].append(time.perf_counter() - t_precond)
        self.timing["total_step"].append(time.perf_counter() - t_total)

        return loss

    def print_layer_ranks(self):
        """Print a summary of the inversion method chosen for each layer.

        Example output (lu_max_dim=4096, adaptive=True)::

            Layer inversion methods (last _update_inverses call):
              Linear(784→512)   A(784×784): Chol-solve  G(512×512): Chol-solve
              Linear(512→256)   A(512×512): Chol-solve  G(256×256): Chol-solve
              Linear(768→768)   A(768×768): Chol-solve  G(768×768): Chol-solve
              Linear(768→3072)  A(768×768): Chol-solve  G(3072×3072): Chol-solve
        """
        if not self.layer_ranks_:
            print("No layer ranks recorded yet — call step() at least once.")
            return
        print("Layer inversion methods (last _update_inverses call):")
        for module, ranks in self.layer_ranks_.items():
            if isinstance(module, nn.Conv2d):
                kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) \
                          else (module.kernel_size, module.kernel_size)
                n_a = module.in_channels * kH * kW
                n_g = module.out_channels
                label = f"Conv2d({module.in_channels}→{n_g}, k={kH}×{kW})"
            else:
                n_a = module.weight.shape[1]
                n_g = module.weight.shape[0]
                label = f"Linear({n_a}→{n_g})"
            k_a, k_g = ranks
            ka_str = "Chol-solve" if k_a == "lu" else (f"EVD k={k_a}" if k_a is not None else "EVD full")
            kg_str = "Chol-solve" if k_g == "lu" else (f"EVD k={k_g}" if k_g is not None else "EVD full")
            print(f"  {label:<28}  A({n_a}×{n_a}): {ka_str:<12}  G({n_g}×{n_g}): {kg_str}")

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Return timing statistics for profiling."""
        stats = {}
        for key, times in self.timing.items():
            if times:
                arr = np.array(times)
                stats[key] = {
                    "mean_ms": float(arr.mean() * 1000),
                    "p50_ms": float(np.percentile(arr, 50) * 1000),
                    "p99_ms": float(np.percentile(arr, 99) * 1000),
                    "total_s": float(arr.sum()),
                    "count": len(times),
                }
        return stats

    def kfac_state_dict(self) -> dict:
        """Serialize K-FAC curvature state for warm-start checkpointing.

        Saves Gram matrices (_factors), EVD inverses (_inverses), and Cholesky
        factors (_lu_factors) keyed by layer index (pickle-safe).

        Returns
        -------
        dict with keys:
            "step_count" : int
            "factors"    : {layer_idx: (A_cpu, G_cpu)}
            "inverses"   : {layer_idx: (Q_A_cpu, il_A_cpu, Q_G_cpu, il_G_cpu)}
            "lu_factors" : {layer_idx: (L_A_cpu, L_G_cpu)}  — lower Cholesky factors
        """
        mod_to_idx = {mod: i for i, mod in enumerate(self.hooks.linear_layers)}
        state: dict = {
            "step_count": self._step_count,
            "factors":    {},
            "inverses":   {},
            "lu_factors": {},
        }
        for mod, (A, G) in self._factors.items():
            if mod in mod_to_idx:
                state["factors"][mod_to_idx[mod]] = (A.cpu(), G.cpu())
        for mod, (Q_A, il_A, Q_G, il_G) in self._inverses.items():
            if mod in mod_to_idx:
                state["inverses"][mod_to_idx[mod]] = (
                    Q_A.cpu(), il_A.cpu(), Q_G.cpu(), il_G.cpu())
        for mod, (L_A, L_G) in self._lu_factors.items():
            if mod in mod_to_idx:
                state["lu_factors"][mod_to_idx[mod]] = (L_A.cpu(), L_G.cpu())
        return state

    def load_kfac_state_dict(self, state: dict, device=None):
        """Restore K-FAC curvature state from a checkpoint.

        Parameters
        ----------
        state  : dict returned by kfac_state_dict()
        device : torch.device or None — defaults to the model's current device
        """
        if device is None:
            device = next(self.model.parameters()).device
        idx_to_mod = {i: mod for i, mod in enumerate(self.hooks.linear_layers)}

        self._step_count = int(state.get("step_count", 0))

        self._factors = {}
        for i, (A, G) in state.get("factors", {}).items():
            mod = idx_to_mod.get(int(i))
            if mod is not None:
                self._factors[mod] = (A.to(device), G.to(device))

        self._inverses = {}
        for i, (Q_A, il_A, Q_G, il_G) in state.get("inverses", {}).items():
            mod = idx_to_mod.get(int(i))
            if mod is not None:
                self._inverses[mod] = (
                    Q_A.to(device), il_A.to(device),
                    Q_G.to(device), il_G.to(device),
                )

        self._lu_factors = {}
        for i, (L_A, L_G) in state.get("lu_factors", {}).items():
            mod = idx_to_mod.get(int(i))
            if mod is not None:
                self._lu_factors[mod] = (L_A.to(device), L_G.to(device))

    def cleanup(self):
        """Remove hooks and free cached state."""
        self.hooks.remove()
        self._factors.clear()
        self._inverses.clear()
        self._lu_factors.clear()
        self._momentum_buffers.clear()
        self.layer_ranks_.clear()

    def __repr__(self) -> str:
        rank_str = (
            f"adaptive(min_n={self.adaptive_min_n}, budget={self.adaptive_rank_budget})"
            if self.adaptive else
            (f"rank={self.rank}" if self.rank is not None else "full")
        )
        lu_str = f"chol_max_dim={self.lu_max_dim}" if self.lu_max_dim > 0 else "chol=off"
        return (
            f"OlsSMKFAC("
            f"damping={self.damping}, "
            f"factor_update_freq={self.factor_update_freq}, "
            f"decomp_update_freq={self.decomp_update_freq}, "
            f"rank={rank_str}, "
            f"{lu_str}, "
            f"gamma={self.gamma}, "
            f"n_layers={len(self.hooks.linear_layers)})"
        )
