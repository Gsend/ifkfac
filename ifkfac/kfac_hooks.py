"""
Forward and backward hooks for capturing activation and gradient statistics.

Shared infrastructure used by both OlsSMKFAC and ClassicKFAC.
Registers hooks on nn.Linear and nn.Conv2d layers to capture:
  - Input activations x (forward hook)  → accumulates A_sum = XᵀX
  - Output gradients δ (backward hook)  → accumulates G_sum = δᵀδ

Conv2d support (im2col):
  For a Conv2d(C_in, C_out, k) layer the weight gradient is equivalent to
  a linear layer operating on unfolded input patches:
    x̃  : (B·H_out·W_out,  C_in·kH·kW)   via F.unfold
    δ̃  : (B·H_out·W_out,  C_out)          via permute+reshape
  The Gram matrices A = x̃ᵀx̃ and G = δ̃ᵀδ̃ are then computed identically
  to the Linear case.  This is the standard K-FAC-for-CNNs formulation
  (KFAC-Reduce: spatially averaged G, see Grosse & Martens 2016).

Transformer / sequence model support (KFAC-Reduce for Linear layers):
  For a transformer's nn.Linear layer the input is (B, seq_len, d_in).
  Naively flattening to (B·seq_len, d_in) inflates the outer-product cost
  by seq_len (128× for BERT) — this is why opt_ms was ~1 500 ms/step
  regardless of decomp_update_freq, since the hooks ran every step.

  Fix: randomly subsample at most KFACHooks._SEQ_SUBSAMPLE rows from the
  flattened (B·seq_len, d) tensor before computing the outer product.  This
  keeps the estimator unbiased (n_A / n_G track actual rows used) while
  making hook cost O(_SEQ_SUBSAMPLE · d²) instead of O(B·seq_len · d²).
  At the default cap of 512, BERT hook cost drops by ~128×.

Design: accumulate directly into Gram sums each step
---------------------------------------------------------
The previous implementation stored the raw activation/gradient tensors and
only computed Gram matrices when get_factors() was called.  This meant:

  - Only the *last* step's batch was used (overwrite bug)
  - Raw tensors from all accumulated steps were kept in memory

The correct K-FAC approach accumulates running outer-product sums in the
hooks themselves:

    A_sum += xᵀx   (every forward pass)
    G_sum += δᵀδ   (every backward pass)

get_factors() then just divides by the sample count and returns.
Memory cost is O(d_in² + d_out²) per layer — two fixed matrices, not growing
with the number of accumulation steps.
"""

from typing import Dict, List, Tuple, Optional
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from ifkfac.gram_estimator import GramMatrixEstimator

# PyTorch emits this UserWarning every backward pass for layers whose *input*
# doesn't require grad (e.g. the first layer, which receives raw data).
# The hook still fires correctly — the warning is pure noise in K-FAC use.
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)


class KFACHooks(GramMatrixEstimator):
    """Manages forward/backward hooks on Linear/Conv2d layers for K-FAC.

    Implements :class:`GramMatrixEstimator`.  Accumulates per-layer Gram
    matrices incrementally every step:

        A = E[xᵀx]  of shape (d_in,  d_in)   — input activation covariance
        G = E[δᵀδ]  of shape (d_out, d_out)  — output gradient covariance

    Call ``get_factors()`` to read the running averages, then ``clear()``
    to reset for the next accumulation window.

    Usage::

        hooks = KFACHooks(model)
        hooks.enable()
        for step in range(update_freq):
            loss = model(x)
            loss.backward()           # triggers hooks → A_sum, G_sum updated
        factors = hooks.get_factors() # dict: module → (A, G)
        hooks.clear()                 # reset for next window
        hooks.remove()                # when training is done
    """

    def __init__(self, model: nn.Module, max_gram_dim: int = 0):
        """
        Parameters
        ----------
        model : nn.Module
        max_gram_dim : int
            If > 0, skip K-FAC hooks on any Linear/Conv2d layer whose output
            dimension (out_features / out_channels) exceeds this value.
            Those layers fall back to vanilla gradient descent.
            Useful for LM head layers with vocab-sized output (e.g. 50 257)
            whose G Gram matrix would be ~10 GB and cause OOM.
            Default: 0 (disabled — all layers are tracked).
        """
        self.model = model
        self.max_gram_dim = max_gram_dim
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        # Running Gram sums, accumulated across steps.
        # Keyed by module; values are (d_in × d_in) and (d_out × d_out) tensors.
        self._A_sum: Dict[nn.Module, torch.Tensor] = {}
        self._G_sum: Dict[nn.Module, torch.Tensor] = {}

        # Total number of *samples* (not steps) accumulated since last clear().
        self._n_A: Dict[nn.Module, int] = {}
        self._n_G: Dict[nn.Module, int] = {}

        self._enabled = False
        self._linear_layers: List[nn.Module] = []   # nn.Linear + nn.Conv2d

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if max_gram_dim > 0:
                    # Check output dimension
                    out_dim = (module.out_features if isinstance(module, nn.Linear)
                               else module.out_channels)
                    if out_dim > max_gram_dim:
                        continue   # skip — G would be out_dim² → OOM
                self._linear_layers.append(module)

    @property
    def linear_layers(self) -> List[nn.Module]:
        """All tracked layers: nn.Linear and nn.Conv2d."""
        return self._linear_layers

    @property
    def is_enabled(self) -> bool:
        """True if hooks are currently attached and accumulating."""
        return self._enabled

    def enable(self):
        """Register hooks on all Linear and Conv2d layers."""
        if self._enabled:
            return
        for module in self._linear_layers:
            h_fwd = module.register_forward_hook(self._forward_hook)
            self._handles.append(h_fwd)
            h_bwd = module.register_full_backward_hook(self._backward_hook)
            self._handles.append(h_bwd)
        self._enabled = True

    @staticmethod
    def _unfold_conv_input(x: torch.Tensor, module: nn.Conv2d) -> torch.Tensor:
        """Unfold Conv2d input into patch matrix via im2col.

        Returns x̃ of shape (B·H_out·W_out, C_in·kH·kW), so that the weight
        gradient ∇W = δ̃ᵀ x̃ matches the linear-layer formulation exactly.
        """
        x_unf = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride,
        )  # (B, C_in·kH·kW, L)  where L = H_out · W_out
        # Rearrange to (B·L, C_in·kH·kW)
        B, CkkL, L = x_unf.shape[0], x_unf.shape[1], x_unf.shape[2]
        return x_unf.permute(0, 2, 1).reshape(B * L, CkkL)

    @staticmethod
    def _reshape_conv_grad(delta: torch.Tensor) -> torch.Tensor:
        """Reshape Conv2d output gradient to (B·H_out·W_out, C_out)."""
        # delta: (B, C_out, H_out, W_out)
        B, C_out, H_out, W_out = delta.shape
        return delta.permute(0, 2, 3, 1).reshape(B * H_out * W_out, C_out)

    # Maximum number of token rows used for the outer-product accumulation when
    # the input is a 3-D sequence tensor (batch, seq_len, d).  Capping at this
    # value gives KFAC-Reduce behaviour for transformer Linear layers: cost is
    # O(_SEQ_SUBSAMPLE · d²) regardless of seq_len, and the estimator stays
    # unbiased because n_A / n_G track the actual row count that was used.
    # For BERT (B=512, seq_len=128 → 65 536 rows) this is a 128× speedup on
    # the hook outer products, which were the dominant cost (~1 500 ms/step).
    #
    # Bumped from 512 to 2048 to match RawActivationHooks._SEQ_SUBSAMPLE so
    # Classic/OlsSM and IFKFAC use the same sample budget per step.  Without
    # this match, IFKFAC would have 4x more activation rows per Gram estimate
    # than Classic/OlsSM, confounding the inversion-method comparison.
    _SEQ_SUBSAMPLE: int = 2048

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """Accumulate A_sum += x̃ᵀx̃ from this step's input activations.

        For nn.Linear: x̃ = x reshaped to (N, d_in).
        For nn.Conv2d: x̃ = im2col(x) of shape (B·L, C_in·kH·kW).
        For 3-D sequence tensors (B, seq_len, d): KFAC-Reduce — subsample at
        most _SEQ_SUBSAMPLE rows so cost is independent of seq_len.
        """
        if not self._enabled:
            return
        x = input[0].detach()
        if isinstance(module, nn.Conv2d):
            x = self._unfold_conv_input(x, module)   # (B·L, C_in·kH·kW)
        elif x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])            # (B·seq_len, d_in)
            if x.shape[0] > self._SEQ_SUBSAMPLE:
                idx = torch.randperm(x.shape[0], device=x.device)[: self._SEQ_SUBSAMPLE]
                x = x[idx]                            # (≤_SEQ_SUBSAMPLE, d_in)

        gram_a = x.t().mm(x)                          # (d_in, d_in)
        if module in self._A_sum:
            self._A_sum[module].add_(gram_a)
            self._n_A[module] += x.shape[0]
        else:
            self._A_sum[module] = gram_a
            self._n_A[module] = x.shape[0]

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ):
        """Accumulate G_sum += δ̃ᵀδ̃ from this step's output gradients.

        For nn.Linear: δ̃ = delta reshaped to (N, d_out).
        For nn.Conv2d: δ̃ = delta reshaped to (B·H_out·W_out, C_out).
        For 3-D sequence tensors (B, seq_len, d): KFAC-Reduce — subsample at
        most _SEQ_SUBSAMPLE rows so cost is independent of seq_len.
        """
        if not self._enabled:
            return
        delta = grad_output[0].detach()
        if isinstance(module, nn.Conv2d):
            delta = self._reshape_conv_grad(delta)    # (B·L, C_out)
        elif delta.ndim > 2:
            delta = delta.reshape(-1, delta.shape[-1])  # (B·seq_len, d_out)
            if delta.shape[0] > self._SEQ_SUBSAMPLE:
                idx = torch.randperm(delta.shape[0], device=delta.device)[: self._SEQ_SUBSAMPLE]
                delta = delta[idx]                     # (≤_SEQ_SUBSAMPLE, d_out)

        gram_g = delta.t().mm(delta)                   # (d_out, d_out)
        if module in self._G_sum:
            self._G_sum[module].add_(gram_g)
            self._n_G[module] += delta.shape[0]
        else:
            self._G_sum[module] = gram_g
            self._n_G[module] = delta.shape[0]

    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return averaged Gram matrices (A, G) for each tracked layer.

        A = A_sum / n_A     shape (d_in,  d_in)
        G = G_sum / n_G     shape (d_out, d_out)

        Call clear() afterwards to reset accumulators for the next window.
        Returns only layers that have received at least one forward+backward pass.
        """
        factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module in self._linear_layers:
            if module not in self._A_sum or module not in self._G_sum:
                continue
            A = self._A_sum[module] / self._n_A[module]
            G = self._G_sum[module] / self._n_G[module]
            factors[module] = (A, G)
        return factors

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        """Return how many samples have been accumulated since last clear().

        If module is None, returns the count for the first layer (as a proxy
        for the whole network — all layers see the same batch).
        """
        target = module or (self._linear_layers[0] if self._linear_layers else None)
        if target is None:
            return 0
        return self._n_A.get(target, 0)

    def clear(self):
        """Reset all Gram accumulators.  Call after get_factors() to start a
        fresh accumulation window."""
        self._A_sum.clear()
        self._G_sum.clear()
        self._n_A.clear()
        self._n_G.clear()

    def remove(self):
        """Remove all hooks from the model and free cached state."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        self.clear()
