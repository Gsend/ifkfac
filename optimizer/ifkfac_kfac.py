"""
IFKFAC optimizer — QR-based second-order optimiser.

Overview
--------
Classic K-FAC approximates the Fisher information matrix as

    F ≈ G ⊗ A

where A = E[xᵀx] and G = E[δᵀδ] are Kronecker factors (Gram matrices).
Applying the preconditioner requires inverting A and G, which amplifies
condition-number error as κ(X)⁴ for the classical LU path.

IFKFAC replaces Gram-matrix inversion with QR-based triangular solves:

    X = Q R_X   →   A = R_Xᵀ R_X  (without forming A explicitly)
    δ = Q R_G   →   G = R_Gᵀ R_G

The preconditioned gradient is then:

    ΔW = G⁻¹ ∇L A⁻¹
       = (R_Gᵀ R_G)⁻¹ ∇L (R_Xᵀ R_X)⁻¹

computed via four triangular solves — never forming G⁻¹ or A⁻¹.

Error scaling: κ(X)¹ ε  (vs κ(X)⁴ ε for Classic, κ(X)² for Cholesky).

Memory
------
RawActivationHooks runs streaming TSQR in the hooks, keeping only the
O(n²) R factor per layer — same memory as KFACHooks.  Raw activations are
discarded as soon as each chunk is processed.

p ≥ n requirement
-----------------
QR produces a full-rank R only when p ≥ n (more rows than columns).  For
transformer Linear layers p = B × T × factor_update_freq.  If a layer
violates p < n, IFKFAC falls back silently to Classic K-FAC inversion
(using the Gram matrix accumulated by the same hooks via finalize_R) and
logs a warning.

GPU path
--------
The streaming TSQR in the hooks calls torch.linalg.qr (cuSOLVER Householder)
on GPU — no custom kernels required.  The four triangular solves in apply_ifkfac
map to cuBLAS TRSM calls (already GPU-accelerated in PyTorch).

Usage
-----
    optimizer = IFKFAC(model, lr=1e-3, damping=1e-2, factor_update_freq=10)
    # training loop
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    optimizer.cleanup()
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from optimizer.raw_activation_hooks import RawActivationHooks, IFKFACRankError
from optimizer.dtype_check import expect, expect_all
from optimizer.sgso import apply_ifkfac, apply_ifkfac_bias, apply_ifkfac_batched

logger = logging.getLogger(__name__)


class IFKFAC(torch.optim.Optimizer):
    """K-FAC optimizer using QR factorisation (no explicit matrix inverse).

    Parameters
    ----------
    model : nn.Module
        The model whose Linear / Conv2d layers will be preconditioned.
    lr : float
        Learning rate.  Default: 1e-3.
    damping : float
        Tikhonov damping λ applied via ridge augmentation before QR.
        Default: 1e-2.
    factor_update_freq : int
        How often (in optimizer steps) to re-run streaming TSQR and
        refresh the stored R factors.  Default: 10.
    weight_decay : float
        L2 regularisation.  Default: 0.
    momentum : float
        SGD-style momentum on the preconditioned gradient.  Default: 0.9.
    grad_clip : float or None
        If set, clip the natural gradient norm to this value before the
        weight update.  Useful for stabilising early training.
        Default: None (disabled).
    gamma : float
        EMA decay for R factors.  0.0 (default) disables EMA.
        When > 0, old R factors are blended with new ones via:
            R_new ← gamma · R_old + (1 − gamma) · R_fresh
        Note: EMA on triangular factors is an approximation — it does
        not preserve the exact QR relationship, but is useful in practice
        for smoothing noisy curvature estimates.
    max_out_dim : int
        Skip layers whose output dimension exceeds this value (e.g. LM head).
        0 = disabled.  Default: 0.
    augment_bias : bool
        If True, append a ones column to input activations for layers with
        bias so the bias term is folded into the A factor.  Default: False.
        Leave False to match ClassicKFAC's bias handling (apply_ifkfac_bias
        with only the G factor, treating A=1 for the bias), which avoids
        the centred-covariance contamination that augmentation introduces.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        damping: float = 1e-2,
        factor_update_freq: int = 10,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        grad_clip: Optional[float] = None,
        gamma: float = 0.0,
        max_out_dim: int = 0,
        augment_bias: bool = False,
        max_conv_rows: int = 512,
        max_seq_rows: Optional[int] = None,
        batched_qr: bool = False,
        deferred_qr: bool = False,
        use_true_bf16: bool = False,
        pure_bf16: Optional[bool] = None,
        kfac_dtype: Optional[torch.dtype] = None,
    ):
        logger.debug(
            "IFKFAC init: factor_update_freq=%d  damping=%.2e  augment_bias=%s  batched_qr=%s  deferred_qr=%s  true_bf16=%s",
            factor_update_freq, damping, augment_bias, batched_qr, deferred_qr, use_true_bf16,
        )
        # When True, R_X and R_G are stored as bfloat16 between refreshes and
        # upcast (losslessly) to fp32 for the four cuSOLVER triangular solves
        # (cuSOLVER has no bf16 triangular solve).
        # The natural gradient is cast back to fp32 before the weight update
        # so master weights stay at fp32 (standard mixed-precision pattern).
        self.use_true_bf16 = use_true_bf16

        # Collect all linear-layer parameters for the base optimizer
        params = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params.append({"params": list(module.parameters())})

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

        self.model = model
        self.damping = damping
        self.factor_update_freq = factor_update_freq
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.augment_bias = augment_bias

        # Pure-bf16 mode: the model itself is bf16 (weights, activations,
        # gradients), and every stage here keeps bf16 - hooks buffer bf16
        # chunks, the QRs / ridge / moving-average blend run in the bf16
        # Householder kernel, the solves in the bf16 blocked triangular
        # solver, momentum and the weight update in bf16.  Detected from the
        # layer weights' dtype unless forced.
        w_dtypes = {m.weight.dtype for m in model.modules()
                    if isinstance(m, (nn.Linear, nn.Conv2d))}
        # kfac_dtype=torch.bfloat16 selects the same bf16 pipeline for a
        # mixed-precision model (fp32 master weights, bf16 autocast compute):
        # captured activations / output gradients and the weight gradient are
        # cast to bf16 on entry, everything K-FAC computes is bf16, and only
        # the final add into the fp32 master weight happens in fp32.
        if kfac_dtype is not None and kfac_dtype != torch.bfloat16:
            raise ValueError("kfac_dtype must be None or torch.bfloat16")
        if pure_bf16 is None:
            pure_bf16 = (w_dtypes == {torch.bfloat16}) or kfac_dtype == torch.bfloat16
        if pure_bf16 and w_dtypes != {torch.bfloat16} and kfac_dtype != torch.bfloat16:
            raise ValueError(f"pure_bf16=True needs a bf16 model or kfac_dtype=bf16, got {w_dtypes}")
        self.pure_bf16 = bool(pure_bf16)       # True = bf16 K-FAC pipeline
        self._pure_buckets: Dict = {}
        self.pure_stats = {"refreshes": 0, "skipped_partial_rank": 0}

        # Hooks — streaming TSQR accumulation of R factors.  In pure-bf16
        # mode the hooks only buffer the bf16 chunks of each refresh window;
        # the QRs run once per window, batched over layers of equal width.
        self.hooks = RawActivationHooks(
            model,
            damping=damping,
            max_out_dim=max_out_dim,
            augment_bias=augment_bias,
            max_conv_rows=max_conv_rows,
            max_seq_rows=max_seq_rows,
            batched=(batched_qr and not self.pure_bf16),
            deferred=(deferred_qr or self.pure_bf16),
            deferred_window=(10 ** 9 if self.pure_bf16 else 5),
        )
        if self.pure_bf16:
            self.hooks.compute_dtype = torch.bfloat16
        # a bf16 model must hand over bf16 activations, output gradients and
        # weight gradients (checked on every call, optimizer/dtype_check.py)
        self._model_bf16 = (w_dtypes == {torch.bfloat16})
        if self._model_bf16:
            self.hooks.require_input_dtype = torch.bfloat16
        self.hooks.enable()

        # Cached R factors: module → (R_X, R_G)
        # Updated every factor_update_freq steps.
        self._factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Layers that fell back to Classic K-FAC (p < n violation)
        # For these we store explicit inverses from the Gram approximation.
        self._classic_fallback: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Momentum buffers: module → tensor (same shape as weight.grad)
        self._momentum_buffers: Dict[nn.Module, torch.Tensor] = {}

        self._step_count = 0

        # Timing instrumentation
        _maxlen = 1000
        self.timing: Dict[str, deque] = {
            "factor_compute":  deque(maxlen=_maxlen),
            "precondition":    deque(maxlen=_maxlen),
            "total_step":      deque(maxlen=_maxlen),
        }

    # ------------------------------------------------------------------
    # Factor update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pure-bf16 path
    # ------------------------------------------------------------------

    PURE_QR_BUDGET = 1 << 28      # max elements (512 MB of bf16) per batched QR

    @classmethod
    def _bucket_qr(cls, mats):
        """R factors of a list of bf16 matrices (m_i, n_i): batched bf16
        Householder QRs, one per distinct n (rows zero-padded to the bucket
        max, which leaves R unchanged), split so that no batch exceeds
        PURE_QR_BUDGET elements.  Consumes the list: each input is released
        once copied into its batch, and the batch is factored in place."""
        from optimizer.bf16_linalg import qr_r_bf16
        out = [None] * len(mats)
        by_n: Dict[int, list] = {}
        for i, M in enumerate(mats):
            by_n.setdefault(M.shape[1], []).append(i)
        for n, idx in by_n.items():
            m_max = max(mats[i].shape[0] for i in idx)
            per = max(1, cls.PURE_QR_BUDGET // (m_max * n))
            for s in range(0, len(idx), per):
                sub = idx[s:s + per]
                stack = torch.zeros(len(sub), m_max, n, dtype=mats[sub[0]].dtype,
                                    device=mats[sub[0]].device)
                for k, i in enumerate(sub):
                    stack[k, :mats[i].shape[0]] = mats[i]
                    mats[i] = None
                R = qr_r_bf16(stack, overwrite=True)
                for k, i in enumerate(sub):
                    out[i] = R[k].clone()
                del stack, R
        return out

    def _update_factors_pure(self):
        """Refresh: window QR -> scale 1/sqrt(rows) -> ridge [R; sqrt(lam) I]
        -> moving-average blend [sqrt(g) R_old; sqrt(1-g) R_new], all bf16."""
        import math
        t0 = time.perf_counter()
        h = self.hooks
        mods, mats, rows = [], [], []
        for m in h.linear_layers:
            cx, cg = h._raw_X.get(m, []), h._raw_G.get(m, [])
            if not cx or not cg:
                continue
            X = cx[0] if len(cx) == 1 else torch.cat(cx, dim=0)
            G = cg[0] if len(cg) == 1 else torch.cat(cg, dim=0)
            expect("ifkfac.window_rows", X, G)
            if X.shape[0] < X.shape[1] or G.shape[0] < G.shape[1]:
                self.pure_stats["skipped_partial_rank"] += 1
                continue
            mods.append(m); mats += [X, G]; rows += [X.shape[0], G.shape[0]]
        h.clear()
        if not mods:
            self.timing["factor_compute"].append(time.perf_counter() - t0)
            return
        expect_all("ifkfac.window_rows_full_rank", mats)   # the windows that go into the QR
        R = self._bucket_qr(mats)          # consumes mats
        del mats
        expect_all("ifkfac.R_window", R)
        R = [r / math.sqrt(p) for r, p in zip(R, rows)]
        sl = math.sqrt(self.damping)
        R = self._bucket_qr([torch.cat([r, sl * torch.eye(r.shape[0], dtype=r.dtype, device=r.device)], 0)
                             for r in R])
        expect_all("ifkfac.R_damped", R)
        new = {m: (R[2 * i], R[2 * i + 1]) for i, m in enumerate(mods)}
        if self.gamma > 0.0:
            sg, s1g = math.sqrt(self.gamma), math.sqrt(1.0 - self.gamma)
            keys = [(m, j) for m in mods if m in self._factors for j in (0, 1)]
            if keys:
                B = self._bucket_qr([torch.cat([sg * self._factors[m][j], s1g * new[m][j]], 0)
                                     for m, j in keys])
                expect_all("ifkfac.R_blended", B)
                for (m, j), r in zip(keys, B):
                    pair = list(new[m]); pair[j] = r; new[m] = tuple(pair)
        expect_all("ifkfac.factors", [r for p in new.values() for r in p])
        self._factors.update(new)
        self._build_pure_buckets()
        self.pure_stats["refreshes"] += 1
        self.pure_stats["factor_dtypes"] = sorted({str(r.dtype) for p in new.values() for r in p})
        self.timing["factor_compute"].append(time.perf_counter() - t0)

    def _build_pure_buckets(self):
        """Group layers by (n_in, n_out); stack their R's once per refresh and
        precompute the blocked triangular solvers (inverted diagonal blocks)."""
        from optimizer.bf16_linalg import TriSolver
        groups: Dict = {}
        for m in self.hooks.linear_layers:
            if m not in self._factors:
                continue
            R_X, R_G = self._factors[m]
            groups.setdefault((R_X.shape[0], R_G.shape[0]), []).append(m)
        self._pure_buckets = {}
        for key, ms in groups.items():
            RX = torch.stack([self._factors[m][0] for m in ms])
            RG = torch.stack([self._factors[m][1] for m in ms])
            SX, SG = TriSolver(RX), TriSolver(RG)
            expect_all("ifkfac.solver_blocks", [SX.U, SG.U] + SX.Dinv + SG.Dinv)
            self._pure_buckets[key] = (ms, SX, SG)

    def _natural_gradients_pure(self):
        """bf16 natural gradient G^-1 g A^-1 of every factored layer (bucketed
        bf16 solves); the gradients are cast to bf16 on entry.  Returns
        [(module, nat_weight_bf16, nat_bias_bf16 or None)]."""
        out = []
        for (n_in, n_out), (ms, SX, SG) in self._pure_buckets.items():
            live = [m for m in ms if m.weight.grad is not None]
            if not live:
                continue
            if len(live) != len(ms):          # rare: rebuild a sub-bucket
                from optimizer.bf16_linalg import TriSolver
                SX = TriSolver(torch.stack([self._factors[m][0] for m in live]))
                SG = TriSolver(torch.stack([self._factors[m][1] for m in live]))
            if self._model_bf16:
                expect_all("ifkfac.weight_grad_from_model", [m.weight.grad for m in live])
            g = torch.stack([m.weight.grad.to(torch.bfloat16).reshape(n_out, n_in) for m in live])
            expect("ifkfac.natgrad_input", g)
            T2 = SG.solve(SG.solve_t(g))
            nat = SX.solve(SX.solve_t(T2.transpose(1, 2).contiguous())).transpose(1, 2)
            expect("ifkfac.natgrad_W", T2, nat)
            has_b = [m.bias is not None and m.bias.grad is not None for m in live]
            nat_b = None
            if any(has_b):
                gb = torch.stack([m.bias.grad.to(torch.bfloat16) if hb
                                  else torch.zeros(n_out, dtype=g.dtype, device=g.device)
                                  for m, hb in zip(live, has_b)]).unsqueeze(2)
                if self._model_bf16:
                    expect_all("ifkfac.bias_grad_from_model",
                               [m.bias.grad for m, hb in zip(live, has_b) if hb])
                expect("ifkfac.natgrad_input_bias", gb)
                nat_b = SG.solve(SG.solve_t(gb)).squeeze(2)
                expect("ifkfac.natgrad_b", nat_b)
            for i, m in enumerate(live):
                out.append((m, nat[i].reshape(m.weight.shape), nat_b[i] if has_b[i] else None))
        return out

    def _master_update(self, m, w, b):
        """Clip, momentum, decoupled weight decay and update in the parameter
        dtype: fp32 master weights under mixed precision (as SINGD and AMP
        optimizers do), bf16 for a bf16 model."""
        lr, wd, mom = self._get_module_hp(m)
        if lr is None:
            return
        w = w.to(m.weight.dtype)
        if self.grad_clip is not None:
            gnorm = w.norm()
            if gnorm > self.grad_clip:
                w = w * (self.grad_clip / gnorm)
        if mom > 0:
            if m not in self._momentum_buffers:
                self._momentum_buffers[m] = torch.zeros_like(w)
            buf = self._momentum_buffers[m]
            buf.mul_(mom).add_(w)
            w = buf
        if wd > 0:
            m.weight.data.mul_(1.0 - lr * wd)
        m.weight.data.add_(w, alpha=-lr)
        if b is not None:
            b = b.to(m.bias.dtype)
            if wd > 0:
                m.bias.data.mul_(1.0 - lr * wd)
            m.bias.data.add_(b, alpha=-lr)

    def _apply_pure(self):
        """Per step: bf16 natural gradients for all factored layers, then the
        update of the (master) weights; layers without factors yet get the
        plain-SGD fallback of the fp32 path."""
        done = set()
        for m, w, b in self._natural_gradients_pure():
            self._master_update(m, w, b)
            done.add(m)
        for m in self.hooks.linear_layers:
            if m in done or m in self._factors or m.weight.grad is None:
                continue
            lr, wd, mom = self._get_module_hp(m)
            if lr is None:
                continue
            self._sgd_fallback(m.weight, lr, wd)

    def _update_factors(self):
        """Refresh R factors from the hooks' streaming TSQR accumulators."""
        if self.pure_bf16:
            return self._update_factors_pure()
        t0 = time.perf_counter()
        logger.debug("IFKFAC._update_factors() called at step %d", self._step_count)

        try:
            new_factors = self.hooks.get_factors()
        except IFKFACRankError as e:
            # Partial failure for at least one layer (typically the layer with
            # the largest n_in hasn't seen enough rows yet — e.g. AE first
            # layer at n_in=784 with batch_size=256 needs 4+ forward passes).
            # BUG FIX: do NOT call self.hooks.clear() here.  The previous
            # behaviour wiped successfully accumulated R factors for ALL
            # layers (including the ones that would have factored fine), AND
            # wiped the deferred buffer's accumulated rows, forcing a fresh
            # restart every freq steps — leading to a perpetual fail loop on
            # any model whose layers' n_in exceeds the per-step row count.
            # Instead: keep the buffer + running R intact and let the next
            # factor-update attempt see the additional rows.
            logger.warning("IFKFAC: factor update skipped — %s", e)
            self.timing["factor_compute"].append(time.perf_counter() - t0)
            return

        # Successful factor update — reset the streaming TSQR window for the
        # next factor-update cycle (K-FAC factors are based on recent activations).
        self.hooks.clear()

        # Validate finite values
        clean: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module, (R_X, R_G) in new_factors.items():
            if not (torch.isfinite(R_X).all() and torch.isfinite(R_G).all()):
                logger.warning(
                    "IFKFAC: non-finite R for layer %s — skipping.",
                    type(module).__name__,
                )
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "IFKFAC._update_factors() [%s]: R_X=%s cond≈%.2g  R_G=%s cond≈%.2g",
                    type(module).__name__,
                    tuple(R_X.shape),
                    (R_X.diag().max() / (R_X.diag().min() + 1e-30)).item(),
                    tuple(R_G.shape),
                    (R_G.diag().max() / (R_G.diag().min() + 1e-30)).item(),
                )
            clean[module] = (R_X, R_G)

        # Optional EMA blending — augmented-QR (exact) instead of linear-on-R.
        #
        # The naive blend `γ·R_old + (1-γ)·R_new` is NOT a valid R factor of
        # the blended Gram: cross-terms γ(1-γ)·(R_old^T R_new + R_new^T R_old)
        # contaminate the implied Gram by 5-20× relative error at typical
        # gamma values (verified in tests/test_kfac_equivalence.py:
        # test_ema_blend_approximation_error).  This was the bottleneck
        # masking IFKFAC's kappa^1 stability advantage in training.
        #
        # Exact blend: stack [sqrt(gamma)·R_old; sqrt(1-gamma)·R_new] and
        # re-QR. The resulting R satisfies R^T R = gamma·A_old + (1-gamma)·A_new
        # exactly — no cross-term contamination.
        if self.gamma > 0.0 and self._factors:
            import math
            sqrt_g  = math.sqrt(self.gamma)
            sqrt_1g = math.sqrt(1.0 - self.gamma)
            blended: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
            for module, (R_X_new, R_G_new) in clean.items():
                if module in self._factors:
                    R_X_old, R_G_old = self._factors[module]
                    # Exact augmented-QR blend for both factors
                    def _blend(R_old, R_new):
                        aug = torch.cat(
                            [sqrt_g * R_old, sqrt_1g * R_new], dim=0
                        )
                        _, R_b = torch.linalg.qr(aug, mode="reduced")
                        # Positive-diagonal sign normalisation
                        diag_signs = torch.sign(torch.diagonal(R_b))
                        diag_signs[diag_signs == 0] = 1.0
                        return R_b * diag_signs.unsqueeze(1)
                    blended[module] = (
                        _blend(R_X_old, R_X_new),
                        _blend(R_G_old, R_G_new),
                    )
                else:
                    blended[module] = (R_X_new, R_G_new)
            clean = blended

        # If use_true_bf16 is enabled, downcast factors to bf16 BEFORE storage.
        # apply_ifkfac (see optimizer/sgso.py) dispatches on R.dtype and upcasts
        # bf16-stored factors to fp32 for the triangular solves
        # (fp32 cuSOLVER; storage-only bf16).  The natural-gradient output is cast back to
        # fp32 (master-weights precision) before the weight update.
        if self.use_true_bf16:
            clean = {m: (R_X.to(torch.bfloat16), R_G.to(torch.bfloat16))
                      for m, (R_X, R_G) in clean.items()}
            expect_all("ifkfac.storage.R_stored", [r for p in clean.values() for r in p])

        self._factors = clean
        self.timing["factor_compute"].append(time.perf_counter() - t0)

        logger.debug(
            "IFKFAC: factors updated for %d layers (step %d)  true_bf16=%s",
            len(self._factors), self._step_count, self.use_true_bf16,
        )

    # ------------------------------------------------------------------
    # Preconditioner apply
    # ------------------------------------------------------------------

    def _get_module_hp(self, module):
        """Return (lr, wd, mom) for the param-group containing module.weight."""
        for group in self.param_groups:
            if any(p is module.weight for p in group["params"]):
                return group["lr"], group["weight_decay"], group["momentum"]
        return None, None, None

    def _apply_preconditioner_bucketed(self):
        """Phase 2: bucket layers by (n_in, n_out) shape, batched apply per bucket.

        Layers that need the per-layer slow path (Conv2d with multi-dim weight,
        augment_bias-trimmed R_X, modules without cached factors, classic
        fallback) are processed by the original per-layer routine.  This keeps
        Phase 2 semantically identical to baseline while harvesting the trsm
        batching win on the common case (transformer Linear with consistent
        shapes).
        """
        if self.pure_bf16:
            return self._apply_pure()
        if self.use_true_bf16 and self._factors:
            # bf16-storage runs: the R factors applied below must be the bf16-stored ones
            expect_all("ifkfac.storage.R_applied", [r for p in self._factors.values() for r in p])
        # First pass: classify each module.  Per-layer fallback handles the
        # tricky cases; the fast path handles plain 2-D Linear.
        fast = {}   # (n_in, n_out, dtype, device) → list of (module, lr, wd, mom)
        slow_modules = []

        for module in self.hooks.linear_layers:
            if module.weight.grad is None:
                continue
            lr, wd, mom = self._get_module_hp(module)
            if lr is None:
                continue

            if module not in self._factors:
                # Plain SGD fallback for unprecondtioned layers.
                slow_modules.append((module, lr, wd, mom))
                continue

            R_X, R_G = self._factors[module]

            # Fast path requirements: 2-D Linear weight, no bias augmentation
            # trim needed.  Conv2d and augment_bias take the slow path.
            weight = module.weight
            is_linear_2d = (isinstance(module, nn.Linear) and weight.dim() == 2)
            needs_trim = self.augment_bias and module.bias is not None
            if not is_linear_2d or needs_trim:
                slow_modules.append((module, lr, wd, mom))
                continue

            key = (R_X.shape[0], R_G.shape[0], weight.dtype, weight.device)
            fast.setdefault(key, []).append((module, lr, wd, mom))

        # ---- Slow path: per-layer apply for fallback + irregular modules ----
        for module, lr, wd, mom in slow_modules:
            self._apply_preconditioner(module, lr=lr, weight_decay=wd, momentum=mom)

        # ---- Fast path: batched apply per shape bucket ----
        for (n_in, n_out, _, _), entries in fast.items():
            B = len(entries)
            modules = [e[0] for e in entries]

            # Stack grad_W — do NOT apply weight_decay here.  Decoupled
            # weight decay (AdamW style) is applied AFTER the natural-gradient
            # operation, by scaling the weights separately.  Adding wd*W to the
            # gradient before preconditioning lets the natural-gradient operator
            # amplify the wd*W term in low-curvature directions, which diverges
            # (observed in the wd sweep: ppl jumped from 668 to 5345 at wd=0.1).
            grad_list = [m.weight.grad for m, lr, wd, mom in entries]
            grad_stack = torch.stack(grad_list, dim=0)              # (B, n_out, n_in)

            R_X_stack = torch.stack([self._factors[m][0] for m in modules], dim=0)
            R_G_stack = torch.stack([self._factors[m][1] for m in modules], dim=0)

            try:
                nat_grad_stack = apply_ifkfac_batched(
                    grad_stack, R_X_stack, R_G_stack
                )
            except Exception as e:
                logger.warning(
                    "IFKFAC: apply_ifkfac_batched failed for bucket "
                    "(n_in=%d, n_out=%d, B=%d): %s — falling back to per-layer.",
                    n_in, n_out, B, e,
                )
                for m, lr, wd, mom in entries:
                    self._apply_preconditioner(m, lr=lr, weight_decay=wd, momentum=mom)
                continue

            # Per-layer post-processing: grad clip, momentum, decoupled weight
            # decay, weight update, bias.  No way to fully batch this since
            # lr/mom/wd can differ across param groups.
            for i, (m, lr, wd, mom) in enumerate(entries):
                nat_grad_w = nat_grad_stack[i]

                if self.grad_clip is not None:
                    gnorm = nat_grad_w.norm()
                    if gnorm > self.grad_clip:
                        nat_grad_w = nat_grad_w * (self.grad_clip / gnorm)

                if mom > 0:
                    if m not in self._momentum_buffers:
                        self._momentum_buffers[m] = torch.zeros_like(nat_grad_w)
                    buf = self._momentum_buffers[m]
                    buf.mul_(mom).add_(nat_grad_w)
                    nat_grad_w = buf

                # Decoupled weight decay (AdamW style): scale weights, then
                # add preconditioned gradient.
                if wd > 0:
                    m.weight.data.mul_(1.0 - lr * wd)
                m.weight.data.add_(nat_grad_w, alpha=-lr)

                # Bias gradient — bias has no Kronecker preconditioner on its
                # input side (treat A=I for bias).  Apply natgrad through R_G
                # only, then decoupled wd.
                if m.bias is not None and m.bias.grad is not None:
                    R_G = self._factors[m][1]
                    nat_grad_b = apply_ifkfac_bias(m.bias.grad, R_G)
                    if wd > 0:
                        m.bias.data.mul_(1.0 - lr * wd)
                    m.bias.data.add_(nat_grad_b, alpha=-lr)

    @staticmethod
    def _sgd_fallback(weight, lr, weight_decay):
        """Plain SGD (decoupled wd) for a layer that has no factors yet."""
        if weight_decay > 0:
            weight.data.mul_(1.0 - lr * weight_decay)
        weight.data.add_(weight.grad, alpha=-lr)

    def _apply_preconditioner(
        self,
        module: nn.Module,
        lr: float,
        weight_decay: float,
        momentum: float,
    ):
        """Apply IFKFAC preconditioner to module.weight.grad (and bias if present).

        Falls back to plain SGD for this module if:
          - No R factors have been computed yet.
          - The module is in the classic_fallback set.
        """
        weight = module.weight
        if weight.grad is None:
            return

        # ---- Select R factors ----
        if module not in self._factors:
            # No preconditioner yet — plain SGD step with decoupled wd
            self._sgd_fallback(weight, lr, weight_decay)
            return

        R_X, R_G = self._factors[module]

        # ---- Weight gradient ----
        # Decoupled weight decay (AdamW style): scale weights AFTER computing
        # the natural-gradient step.  Adding wd*W to the gradient before
        # preconditioning would let the natural-gradient operator amplify the
        # wd*W term in low-curvature directions, which diverges.
        grad_w = weight.grad

        # Bias augmentation: if augment_bias is on, R_X has shape (n_in+1, n_in+1).
        # grad_w has shape (n_out, n_in) — we need to drop the last column of R_X
        # for the weight apply (the +1 column corresponds to the bias term).
        if self.augment_bias and module.bias is not None:
            R_X_w = R_X[:-1, :-1].contiguous()   # (n_in, n_in) — weight block only
        else:
            R_X_w = R_X

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_apply_preconditioner [%s]: grad_W=%s  R_X_w=%s  R_G=%s",
                type(module).__name__,
                tuple(grad_w.shape), tuple(R_X_w.shape), tuple(R_G.shape),
            )

        # Conv2d weights are 4D (C_out, C_in, kH, kW).
        # The R factors are 2D (hooks use im2col to flatten spatial dims),
        # so reshape to (C_out, C_in·kH·kW) before the triangular solves,
        # then restore the original shape.
        orig_shape = grad_w.shape
        if grad_w.dim() > 2:
            grad_w = grad_w.reshape(orig_shape[0], -1)

        try:
            nat_grad_w = apply_ifkfac(grad_w, R_X_w, R_G)
        except Exception as e:
            logger.warning(
                "IFKFAC: apply_ifkfac failed for %s (%s) — using raw gradient.",
                type(module).__name__, e,
            )
            nat_grad_w = grad_w

        nat_grad_w = nat_grad_w.reshape(orig_shape)

        # Gradient clipping
        if self.grad_clip is not None:
            gnorm = nat_grad_w.norm()
            if gnorm > self.grad_clip:
                nat_grad_w = nat_grad_w * (self.grad_clip / gnorm)

        # Momentum
        if momentum > 0:
            if module not in self._momentum_buffers:
                self._momentum_buffers[module] = torch.zeros_like(nat_grad_w)
            buf = self._momentum_buffers[module]
            buf.mul_(momentum).add_(nat_grad_w)
            nat_grad_w = buf

        # Decoupled weight decay (AdamW style)
        if weight_decay > 0:
            weight.data.mul_(1.0 - lr * weight_decay)
        weight.data.add_(nat_grad_w, alpha=-lr)

        # ---- Bias gradient ----
        # Matches ClassicKFAC: apply only the G factor (treat A=1 for bias).
        # nat_grad_b = G⁻¹ · grad_b = (R_Gᵀ R_G)⁻¹ · grad_b
        if module.bias is not None and module.bias.grad is not None:
            nat_grad_b = apply_ifkfac_bias(module.bias.grad, R_G)
            if weight_decay > 0:
                module.bias.data.mul_(1.0 - lr * weight_decay)
            module.bias.data.add_(nat_grad_b, alpha=-lr)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one IFKFAC optimisation step.

        Every factor_update_freq steps, finalises the streaming TSQR
        accumulators into damped R factors and caches them.  On every step,
        applies those R factors to the current gradients via triangular solves
        and updates parameters.
        """
        t_total = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        _verbose = self._step_count <= 20

        if _verbose:
            print(f"    [IFKFAC step={self._step_count}] entry", flush=True)

        # Drain batched-mode buffers (no-op if not batched).  This converts
        # the per-step ~144 cuSOLVER launches into ~6-12 batched launches.
        self.hooks.flush()
        if _verbose:
            print(f"    [IFKFAC step={self._step_count}] hooks.flush done", flush=True)

        # Refresh R factors on schedule.
        # BUG FIX: previously fired at step 1 (% freq == 1), but at step 1
        # only one forward pass has run, so the deferred buffer holds only
        # one batch worth of rows.  For layers whose n_in exceeds the batch
        # size (e.g. AE first layer n_in=784 vs batch=256), this caused a
        # IFKFACRankError every factor-update cycle.  Wait until we have at
        # least one full factor_update_freq window of forward passes before
        # the first factor update.
        if self.factor_update_freq == 1:
            self._update_factors()
        elif (self._step_count >= self.factor_update_freq and
              self._step_count % self.factor_update_freq == 0):
            self._update_factors()

        logger.debug("IFKFAC.step() step=%d  factors_cached=%d",
                     self._step_count, len(self._factors))

        # Apply preconditioner and update weights.  Phase 2: layers are
        # bucketed by (n_in, n_out) shape and apply_ifkfac_batched runs once
        # per bucket (4 batched trsm calls instead of 4 per layer).
        t_precond = time.perf_counter()
        self._apply_preconditioner_bucketed()

        self.timing["precondition"].append(time.perf_counter() - t_precond)
        self.timing["total_step"].append(time.perf_counter() - t_total)

        return loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-operation timing statistics (same format as ClassicKFAC)."""
        stats: Dict[str, Dict[str, float]] = {}
        for key, times in self.timing.items():
            if times:
                arr = np.array(list(times))
                stats[key] = {
                    "mean_ms":  float(arr.mean() * 1000),
                    "p50_ms":   float(np.percentile(arr, 50) * 1000),
                    "p99_ms":   float(np.percentile(arr, 99) * 1000),
                    "total_s":  float(arr.sum()),
                    "count":    len(times),
                }
        return stats

    # ------------------------------------------------------------------
    # Factor-capture-mode forwarding (see GramMatrixEstimator.capture).
    # Lets callers write `with optimizer.capture():` for multi-pass autograd
    # training loops (PINNs, MAML, WGAN-GP, contrastive learning, influence
    # functions) where exactly one forward+backward should populate the
    # Kronecker factors.
    # ------------------------------------------------------------------
    def capture(self):
        """Context manager that enables factor capture for one pass.

        Use this when the training step contains multiple forward passes
        through the network or any ``autograd.grad`` calls that traverse
        the K-FAC-instrumented layers (PINN derivative computation, MAML
        inner loop, WGAN gradient penalty, etc.).  Wrap exactly one
        forward+backward (or forward + ``autograd.grad`` for the dominant
        loss term) in ``with kfac.capture():`` to populate the per-layer
        Kronecker factors from that designated pass.
        """
        return self.hooks.capture()

    def pause_capture(self):
        """Suppress factor capture without detaching hooks."""
        self.hooks.pause()

    def resume_capture(self):
        """Re-enable factor capture."""
        self.hooks.resume()

    def cleanup(self):
        """Remove hooks and free all cached state."""
        self.hooks.remove()
        self._factors.clear()
        self._classic_fallback.clear()
        self._momentum_buffers.clear()

    def __repr__(self) -> str:
        return (
            f"IFKFAC("
            f"damping={self.damping}, "
            f"factor_update_freq={self.factor_update_freq}, "
            f"gamma={self.gamma}, "
            f"augment_bias={self.augment_bias}, "
            f"n_layers={len(self.hooks.linear_layers)})"
        )
