"""
Abstract interface for Gram matrix estimators used by K-FAC optimizers.

Defining a formal interface decouples KFACHooks from the optimizer classes,
enabling alternative implementations (online EVD, mini-batch averaging,
per-layer EMA with different decay rates, etc.) without modifying optimizer code.

Usage
-----
The default implementation is KFACHooks.  To inject a custom estimator::

    class MyEstimator(GramMatrixEstimator):
        ...

    opt = OlsSMKFAC(model, gram_estimator=MyEstimator(model))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GramMatrixEstimator(ABC):
    """Abstract base class for K-FAC Gram matrix estimators.

    An estimator registers hooks on a model's layers, accumulates
    input-activation (A) and output-gradient (G) statistics per batch,
    and exposes averaged (A, G) pairs on demand.

    Lifecycle::

        estimator.enable()                  # attach hooks
        for step in range(update_freq):
            loss = model(x); loss.backward()  # hooks accumulate
        factors = estimator.get_factors()   # read averages
        estimator.clear()                   # reset for next window
        # ...
        estimator.remove()                  # detach hooks, free memory
    """

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def enable(self) -> None:
        """Attach forward and backward hooks to all tracked layers."""

    @abstractmethod
    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return averaged Gram matrices (A, G) for each tracked layer.

        Returns
        -------
        dict mapping ``nn.Module`` → ``(A, G)`` where:
            A : (d_in,  d_in)  float tensor — input covariance  E[xᵀx]
            G : (d_out, d_out) float tensor — gradient covariance E[δᵀδ]

        Only layers that have received at least one forward+backward pass
        are included.  Call ``clear()`` after reading to reset accumulators.
        """

    @abstractmethod
    def clear(self) -> None:
        """Reset all accumulator tensors.

        Call after ``get_factors()`` to start a fresh accumulation window.
        """

    @abstractmethod
    def remove(self) -> None:
        """Detach all hooks from the model and free cached tensors."""

    @property
    @abstractmethod
    def linear_layers(self) -> List[nn.Module]:
        """Ordered list of all tracked ``nn.Linear`` / ``nn.Conv2d`` layers."""

    # ------------------------------------------------------------------
    # Optional methods — subclasses may override
    # ------------------------------------------------------------------

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        """Return the number of samples accumulated since the last ``clear()``.

        Default implementation returns 0 — override to provide real counts.

        Parameters
        ----------
        module : nn.Module or None
            If None, returns the count for the first tracked layer as a
            proxy for the whole network.
        """
        return 0

    @property
    def is_enabled(self) -> bool:
        """True if hooks are currently attached."""
        return False

    # ------------------------------------------------------------------
    # Factor-capture-mode API (workshop paper, IFKFAC §6 Future Work)
    # ------------------------------------------------------------------
    # When a training step has multiple forward passes through the K-FAC-
    # instrumented network (PINNs with autograd.grad for higher-order
    # derivatives; MAML inner+outer loops; WGAN-GP gradient penalty;
    # contrastive learning with siamese forward passes; influence functions),
    # the default "hooks fire on every forward/backward" behaviour overwrites
    # the per-layer X and δ buffers, leaving the K-FAC factor accumulator
    # filled with mismatched (X_from_pass_k, δ_from_pass_m) pairs.  See the
    # PINN diagnosis diagram in the workshop paper for the full failure mode.
    #
    # Resolution: subclasses already check ``self._enabled`` at the top of
    # both forward and backward hook callbacks.  We expose pause/resume and
    # a context manager so callers can designate exactly which forward+grad
    # block is the "factor capture pass".  Typical usage::
    #
    #     kfac.hooks.enable()        # attach + capture on (the default)
    #     kfac.hooks.pause()         # turn capture off while we set things up
    #     ...                        # arbitrary multi-pass autograd here
    #     with kfac.hooks.capture():
    #         u = model(x_capture)              # fwd hook fires once → X
    #         grad_for_factors = torch.autograd.grad(
    #             loss_branch, params, retain_graph=True)  # bwd hook fires once → δ
    #     # ... rest of loss / gradient assembly with hooks paused ...
    #     kfac.step()                # uses the (X, δ) captured in the block
    #
    # Concrete subclasses inherit these methods unchanged.  They only need
    # to expose a writeable ``_enabled`` boolean attribute, which both
    # KFACHooks and RawActivationHooks already do.

    def pause(self) -> None:
        """Temporarily suppress activation/gradient capture without
        detaching hooks.  Hook callbacks still fire but become no-ops.
        Use when subsequent forward/backward passes should not contribute
        to the K-FAC factor estimates (e.g. ``autograd.grad`` calls for
        higher-order derivatives, auxiliary-loss forwards, teacher-network
        forwards in distillation)."""
        self._enabled = False

    def resume(self) -> None:
        """Re-enable activation/gradient capture (the inverse of
        :meth:`pause`).  Hooks must already be attached via :meth:`enable`."""
        self._enabled = True

    @contextmanager
    def capture(self):
        """Context manager: enable capture for the duration of the ``with``
        block, restore the previous state on exit.

        Designed for training loops with multiple forward passes per step.
        Wrap exactly one forward+backward (or forward+``autograd.grad``)
        pair in ``with kfac.hooks.capture():`` to populate the per-layer
        Kronecker factors from that single, designated pass.  All other
        model evaluations and gradient computations should run with hooks
        paused so they cannot pollute the factor estimate.

        Restores the previous ``_enabled`` value on exit, so this can be
        used while hooks are already paused (it temporarily re-enables
        them) or while they're active (it's a no-op on entry but allows
        nested code to ``pause`` without confusing the caller).
        """
        previous = getattr(self, "_enabled", False)
        self._enabled = True
        try:
            yield self
        finally:
            self._enabled = previous
