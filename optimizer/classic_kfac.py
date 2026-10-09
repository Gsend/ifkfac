"""
Classical K-FAC optimizer using torch.linalg.inv for Gram matrix inversion.

This is the CONTROL implementation — identical to OlsSMKFAC in every way
except the inversion method. This ensures the benchmark compares inversion
strategies, not implementation differences.

Natural gradient update:  ΔW = G⁻¹ · ∇L · A⁻¹
"""

import logging
import time
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from optimizer.errors import ConfigurationError
from optimizer.hooks import KFACHooks
from optimizer.dtype_check import expect

logger = logging.getLogger(__name__)

class ClassicKFAC(torch.optim.Optimizer):
    """K-FAC optimizer with classical torch.linalg.inv backend.

    Same interface and logic as OlsSMKFAC — only the inversion differs.

    Parameters
    ----------
    model : nn.Module
        The model whose Linear layers will be preconditioned.
    lr : float
        Learning rate. Default: 1e-3.
    damping : float
        Tikhonov damping λ. Default: 1e-2.
    factor_update_freq : int
        How often to recompute Gram matrices. Default: 10.
    decomp_update_freq : int
        How often to recompute cached inverses. Default: 10.
    weight_decay : float
        L2 regularisation. Default: 0.
    momentum : float
        Momentum coefficient. Default: 0.9.
    gamma : float
        EMA decay for Kronecker factors: A ← γ·A_old + (1−γ)·A_batch.
        0.0 (default) disables EMA. Keep γ ≤ 0.9 for ClassicKFAC since
        direct matrix inversion is less numerically stable than EVD when
        matrices are nearly singular (which high γ can produce).
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
        grad_clip: Optional[float] = None,
        gamma: float = 0.0,
        max_gram_dim: int = 0,
        kfac_dtype: Optional[torch.dtype] = None,
    ):
        logger.debug(
            "ClassicKFAC init: factor_update_freq=%d  decomp_update_freq=%d",
            factor_update_freq, decomp_update_freq,
        )
        defaults = dict(lr=lr, damping=damping, weight_decay=weight_decay,
                        momentum=momentum)
        params = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params.append({"params": list(module.parameters())})
        super().__init__(params, defaults)

        self.model = model
        self.damping = damping
        self.factor_update_freq = factor_update_freq
        self.decomp_update_freq = decomp_update_freq
        self.grad_clip = grad_clip
        self.gamma = gamma

        self.hooks = KFACHooks(model, max_gram_dim=max_gram_dim)
        # kfac_dtype=torch.bfloat16: activations and output gradients are cast
        # to bf16 on capture (bf16 Grams, bf16 Cholesky inverses), and the
        # weight gradient is cast to bf16 before G^-1 g A^-1 - the whole K-FAC
        # computation is bf16 even for a mixed-precision model; only the final
        # add into an fp32 master weight is fp32.
        self.kfac_dtype = kfac_dtype
        self.hooks.compute_dtype = kfac_dtype
        # A bf16 model: the K-FAC computation is bf16 as well, and the model
        # must hand over bf16 activations, output gradients and weight
        # gradients (checked on every call, optimizer/dtype_check.py).
        w_dtypes = {m.weight.dtype for m in model.modules()
                    if isinstance(m, (nn.Linear, nn.Conv2d))}
        if w_dtypes == {torch.bfloat16}:
            if self.kfac_dtype is None:
                self.kfac_dtype = torch.bfloat16
                self.hooks.compute_dtype = torch.bfloat16
            self.hooks.require_input_dtype = torch.bfloat16
        self.hooks.enable()

        self._factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._inverses: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._momentum_buffers: Dict[nn.Module, torch.Tensor] = {}

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
        """Recompute Gram matrix factors, optionally EMA-smoothed."""
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

    def _update_inverses(self):
        """Recompute cached inverses using torch.linalg.inv (classical approach).

        bf16 factors (a bf16 model: bf16 activations and gradients give bf16
        Grams) are inverted by the pure-bf16 Cholesky of
        optimizer/bf16_linalg.py - bf16 in, bf16 out, no pivot clamping.  If
        a damped factor is not positive definite in bf16, that layer keeps
        its last good inverse (plain SGD if it has none yet) and the failure
        is counted in self.bf16_chol_failures / self.bf16_chol_log.
        """
        t0 = time.perf_counter()
        if self.kfac_dtype is not None:
            # a bf16 K-FAC run must never fall through to the fp32 inverse below
            for A, G in self._factors.values():
                expect("classic.factors", A, G, dtype=self.kfac_dtype)
        if self._factors and next(iter(self._factors.values()))[0].dtype == torch.bfloat16:
            self._update_inverses_bf16()
            self.timing["inversion"].append(time.perf_counter() - t0)
            return
        for module, (A, G) in self._factors.items():
            # Guard: skip corrupt Gram matrices (NaN/inf from diverged model)
            # Use torch.isfinite — stays on device, no CPU transfer
            if not (torch.isfinite(A).all() and torch.isfinite(G).all()):
                continue

            d_in = A.shape[0]
            d_out = G.shape[0]

            # Add damping
            A_damped = A + self.damping * torch.eye(d_in, device=A.device, dtype=A.dtype)
            G_damped = G + self.damping * torch.eye(d_out, device=G.device, dtype=G.dtype)

            # Classical explicit inversion
            A_inv = torch.linalg.inv(A_damped)
            G_inv = torch.linalg.inv(G_damped)

            self._inverses[module] = (A_inv, G_inv)

        self.timing["inversion"].append(time.perf_counter() - t0)

    # Set by benchmark/kfac_bf16_compare.enable_classic_bf16() for the
    # bf16-storage runs: called with every (A_inv, G_inv) pair before it is
    # applied, to verify that the applied inverses are the bf16-rounded ones.
    inverse_check = None

    def _sgd_fallback(self, module):
        """Plain SGD (decoupled wd) for a layer that has no inverses yet."""
        for p in module.parameters():
            if p.grad is None:
                continue
            lr = wd = mom = None
            for group in self.param_groups:
                if any(p is pp for pp in group["params"]):
                    wd = group["weight_decay"]
                    lr = group["lr"]
                    mom = group["momentum"]
                    break
            if lr is None:
                continue   # param not in any group — skip
            # Decoupled weight decay (AdamW style): apply BEFORE the
            # gradient step, NOT through the gradient.  Avoids amplifying
            # the wd*W term by the (non-existent here) preconditioner.
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)
            p.data.add_(p.grad, alpha=-lr)

    def _master_update(self, module, nat_w, nat_b, lr, wd, mom):
        """Clip, momentum, decoupled weight decay and update, in the parameter
        dtype (fp32 master weights under mixed precision; the natural
        gradients arrive in the K-FAC dtype and are cast here)."""
        if nat_w is not None:
            nat_grad = nat_w.to(module.weight.dtype)

            # Clip to prevent divergence on early / rank-deficient steps
            if self.grad_clip is not None:
                grad_norm = nat_grad.norm()
                if grad_norm > self.grad_clip:
                    nat_grad = nat_grad * (self.grad_clip / grad_norm)

            if mom > 0:
                if module not in self._momentum_buffers:
                    self._momentum_buffers[module] = torch.zeros_like(nat_grad)
                buf = self._momentum_buffers[module]
                buf.mul_(mom).add_(nat_grad)
                nat_grad = buf

            # Apply weight decay decoupled, then preconditioned gradient
            if wd > 0:
                module.weight.data.mul_(1.0 - lr * wd)
            module.weight.data.add_(nat_grad, alpha=-lr)

        if nat_b is not None:
            nat_grad_b = nat_b.to(module.bias.dtype)
            if wd > 0:
                module.bias.data.mul_(1.0 - lr * wd)
            module.bias.data.add_(nat_grad_b, alpha=-lr)

    @staticmethod
    def _natgrad(A_inv, G_inv, grad_w):
        """G^-1 grad A^-1 in the inverses' dtype (the gradient is cast on entry)."""
        if grad_w.dtype != A_inv.dtype:
            grad_w = grad_w.to(A_inv.dtype)
        return G_inv @ grad_w @ A_inv

    @staticmethod
    def _natgrad_bias(G_inv, grad_b):
        return G_inv @ grad_b.to(G_inv.dtype)

    def _update_inverses_bf16(self):
        from optimizer.bf16_linalg import inv_spd_bf16_checked
        if not hasattr(self, "bf16_chol_failures"):
            self.bf16_chol_failures: Dict[str, int] = {}
            self.bf16_chol_attempts = 0
            self.bf16_chol_log = []
        names = {m: n for n, m in self.model.named_modules()} if hasattr(self, "model") else {}
        n_fail = 0
        for module, (A, G) in self._factors.items():
            A_d = A + self.damping * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            G_d = G + self.damping * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            expect("classic.damped_factors", A_d, G_d)
            A_inv, ok_a = inv_spd_bf16_checked(A_d)
            G_inv, ok_g = (inv_spd_bf16_checked(G_d) if ok_a else (None, False))
            self.bf16_chol_attempts += 1
            if ok_a and ok_g:
                expect("classic.inverses", A_inv, G_inv)
                self._inverses[module] = (A_inv, G_inv)
            else:
                n_fail += 1
                key = names.get(module, type(module).__name__)
                self.bf16_chol_failures[key] = self.bf16_chol_failures.get(key, 0) + 1
        self.bf16_chol_log.append({"step": self._step_count, "failed": n_fail,
                                   "layers": len(self._factors),
                                   "factor_dtypes": sorted({str(t.dtype) for p in self._factors.values()
                                                            for t in p}),
                                   "inverse_dtypes": sorted({str(t.dtype) for p in self._inverses.values()
                                                             for t in p})})

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single K-FAC optimisation step (classical inversion)."""
        t_total = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        if self._step_count % self.factor_update_freq == 1 or self.factor_update_freq == 1:
            self._update_factors()

        if self._step_count % self.decomp_update_freq == 1 or self.decomp_update_freq == 1:
            if self._factors:
                self._update_inverses()

        t_precond = time.perf_counter()
        for module in self.hooks.linear_layers:
            if module not in self._inverses:
                self._sgd_fallback(module)
                continue

            A_inv, G_inv = self._inverses[module]
            if self.inverse_check is not None:          # bf16-storage runs (benchmark/kfac_bf16_compare.py)
                self.inverse_check(module, A_inv, G_inv)

            lr = wd = mom = None
            for group in self.param_groups:
                if any(p is module.weight for p in group["params"]):
                    lr = group["lr"]
                    wd = group["weight_decay"]
                    mom = group["momentum"]
                    break

            if lr is None:
                continue   # module not in any param group — skip

            # --- Natural gradients, in the K-FAC dtype ---
            nat_w = nat_b = None
            if module.weight.grad is not None:
                # Decoupled weight decay: do NOT add wd*W to the gradient
                # before preconditioning — that would let the natural-gradient
                # operator amplify the wd*W term in low-curvature directions,
                # causing divergence (observed in the wd sweep).  Instead, scale
                # the weights AFTER computing the preconditioned gradient.
                grad_w = module.weight.grad

                # Conv2d weights are 4D (C_out, C_in, kH, kW).
                # The Kronecker factors are 2D (the hooks use im2col to flatten
                # the spatial dims), so we reshape to (C_out, C_in·kH·kW),
                # apply G⁻¹ · grad · A⁻¹, then restore the original shape.
                if self.hooks.require_input_dtype is not None:
                    expect("classic.weight_grad_from_model", grad_w, dtype=self.hooks.require_input_dtype)
                orig_shape = grad_w.shape
                if grad_w.dim() > 2:
                    grad_w = grad_w.reshape(orig_shape[0], -1)

                if self.kfac_dtype is not None:
                    grad_w = grad_w.to(self.kfac_dtype)
                    expect("classic.natgrad_inputs", grad_w, A_inv, G_inv, dtype=self.kfac_dtype)
                nat_w = self._natgrad(A_inv, G_inv, grad_w).reshape(orig_shape)
                if self.kfac_dtype is not None:
                    expect("classic.natgrad_W", nat_w, dtype=self.kfac_dtype)
            if module.bias is not None and module.bias.grad is not None:
                grad_b = module.bias.grad
                if self.kfac_dtype is not None:
                    if self.hooks.require_input_dtype is not None:
                        expect("classic.bias_grad_from_model", grad_b, dtype=self.hooks.require_input_dtype)
                    grad_b = grad_b.to(self.kfac_dtype)
                    expect("classic.natgrad_inputs_bias", grad_b, G_inv, dtype=self.kfac_dtype)
                nat_b = self._natgrad_bias(G_inv, grad_b)
                if self.kfac_dtype is not None:
                    expect("classic.natgrad_b", nat_b, dtype=self.kfac_dtype)

            # --- Update of the (master) weights, in the parameter dtype ---
            self._master_update(module, nat_w, nat_b, lr, wd, mom)

        self.timing["precondition"].append(time.perf_counter() - t_precond)
        self.timing["total_step"].append(time.perf_counter() - t_total)

        return loss

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
        """Remove hooks and free cached state."""
        self.hooks.remove()
        self._factors.clear()
        self._inverses.clear()
        self._momentum_buffers.clear()

    def __repr__(self) -> str:
        return (
            f"ClassicKFAC("
            f"damping={self.damping}, "
            f"factor_update_freq={self.factor_update_freq}, "
            f"decomp_update_freq={self.decomp_update_freq}, "
            f"gamma={self.gamma}, "
            f"n_layers={len(self.hooks.linear_layers)})"
        )
