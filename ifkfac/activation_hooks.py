"""
Raw-activation hooks for IFKFAC.

Key difference from KFACHooks
------------------------------
KFACHooks accumulates  A_sum += xᵀx  in each forward hook — it discards the
raw activation x immediately after forming the outer product.  This is
memory-efficient (O(n²) per layer) but squares the condition number.

RawActivationHooks instead runs a streaming TSQR update in each hook:

    running_R_X ← streaming_tsqr_update(running_R_X, x_chunk)

Only the O(n²) upper-triangular factor is retained; the raw x rows are
discarded as soon as each chunk is processed.  The final R satisfies:

    Rᵀ R  ≈  Σ xᵢᵀ xᵢ  =  Xᵀ X

with condition number κ(R) = κ(X)  (vs κ(X²) for the Gram approach).

At get_factors() time, Tikhonov damping is applied via ridge augmentation:
    R_damped = tsqr([R_undamped ; √λ · I_n])
so that R_dampedᵀ R_damped = XᵀX + λI without ever touching X directly.

p >= n requirement
------------------
IFKFAC requires the total accumulated batch rows p >= n_in (or n_out).
For layers where this is not satisfied, get_factors() raises IFKFACRankError
and the optimizer falls back to Classic K-FAC for that layer.

Conv2d support
--------------
Conv2d layers use the same im2col unfolding as KFACHooks, giving an
effective (B·L, C_in·kH·kW) activation matrix.  Typically B·L >> C_in·kH·kW
so the p >= n constraint is easy to satisfy.

Bias handling
-------------
When module.bias is not None, the input activation x is augmented with a
column of ones — [x, 1] — so that n_in effectively becomes n_in + 1 and the
bias Kronecker factor is handled implicitly.  Disabled by augment_bias=False.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _module_tag(module: nn.Module) -> str:
    """Short human-readable identifier for a module, used in log messages."""
    if isinstance(module, nn.Linear):
        return f"Linear({module.in_features}→{module.out_features})"
    if isinstance(module, nn.Conv2d):
        return (f"Conv2d({module.in_channels}→{module.out_channels}, "
                f"k={module.kernel_size})")
    return type(module).__name__
import torch.nn.functional as F

from ifkfac.gram_estimator import GramMatrixEstimator
from ifkfac.triangular import (
    streaming_tsqr_update,
    finalize_R,
    batched_streaming_tsqr_update,
)

# Suppress the benign PyTorch warning about backward hooks on layers
# whose inputs don't require grad (e.g. the first layer).
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)


class IFKFACRankError(RuntimeError):
    """Raised when accumulated p < n for a layer, making QR rank-deficient."""


class RawActivationHooks(GramMatrixEstimator):
    """Forward/backward hooks that maintain streaming TSQR R-factors.

    Implements GramMatrixEstimator so IFKFAC can share the same
    hook lifecycle interface as OlsSMKFAC / ClassicKFAC.

    get_factors() returns (R_X, R_G) — upper-triangular factors — rather than
    the Gram matrices (A, G) returned by KFACHooks.  IFKFAC calls
    apply_ifkfac() which consumes these R factors directly via triangular solves.

    Parameters
    ----------
    model : nn.Module
    damping : float
        Tikhonov damping λ applied at finalization (ridge augmentation).
    max_out_dim : int
        Skip layers whose output dimension exceeds this value (e.g. LM heads).
        0 = disabled (all layers tracked).
    augment_bias : bool
        If True, append a column of ones to input activations for layers with
        bias, so the bias Kronecker factor is handled implicitly.
        Default: False — matches ClassicKFAC's bias handling and avoids the
        centred-covariance contamination the augmentation introduces.
    """

    # Maximum number of token rows fed into the streaming TSQR when the input
    # is a 3-D sequence tensor (B, T, d).  Mirrors KFACHooks._SEQ_SUBSAMPLE so
    # the three K-FAC variants see the same sample budget per step on
    # transformer Linear layers (KFAC-Reduce / "Reduce" approximation).
    # Without this, BERT/SmallGPT layers feed B*T rows (e.g. 64*128 = 8192)
    # into TSQR every step - 16x more work than KFACHooks does.
    # Set to 0 to disable (use all rows).
    # Bumped from 512 to 2048 to satisfy TSQR's p>=n requirement on SmallGPT
    # FFN layers (n_out=1024).  At 512 the first factor-update window had only
    # 512 rows accumulated, less than the 1024-dim FFN, so the layer fell
    # back to Classic K-FAC (silently, with a warning) for that window.  At
    # 2048 there are always >= 2 layers worth of headroom even on transformer
    # tasks with d_ff up to 2048.  Costs ~4x more rows per QR but eliminates
    # the warning and lets IFKFAC's preconditioning work from step 1.
    _SEQ_SUBSAMPLE: int = 2048

    def __init__(
        self,
        model: nn.Module,
        damping: float = 1e-2,
        max_out_dim: int = 0,
        augment_bias: bool = False,
        max_conv_rows: int = 512,
        max_seq_rows: Optional[int] = None,
        batched: bool = False,
        deferred: bool = False,
        deferred_window: int = 5,
    ):
        self.model = model
        self.damping = damping
        self.augment_bias = augment_bias
        self.max_conv_rows = max_conv_rows  # cap Conv2d patch rows per batch (0 = no cap)
        # cap 3D-sequence rows per batch (None = use class default _SEQ_SUBSAMPLE,
        # 0 = no cap)
        self.max_seq_rows = (self._SEQ_SUBSAMPLE if max_seq_rows is None
                              else max_seq_rows)
        # When True, hooks buffer chunks per layer; the optimizer must call
        # flush() before reading factors.  flush() does batched cuSOLVER QR
        # per (n)-bucket, reducing ~128 launches/step to ~6-12.
        self.batched = batched
        # When True, hooks store raw chunks across (up to deferred_window)
        # steps then merge via streaming_tsqr_update.  Bigger chunks → fewer
        # cuSOLVER launches, better tall-skinny QR utilisation.  Trades extra
        # GPU memory for ~1.5-2x wall-time speedup vs streaming TSQR (still
        # kappa^1 stable — never forms X^T X).  Mutually exclusive with `batched`.
        #
        # deferred_window caps in-flight raw chunks per layer.  At freq=20 and
        # window=5, each layer holds at most 5 chunks ≈ 5x memory of streaming
        # (vs 20x for full-window deferred which can OOM on wide models).
        self.deferred = deferred
        self.deferred_window = deferred_window
        # Optional per-chunk transform applied to x/delta inside the hooks
        # after subsample/augment, BEFORE buffering or streaming TSQR.  This
        # is the right place to inject row-weighting schemes (e.g. WGSO):
        # the transform sees the same per-step chunk regardless of which
        # accumulation mode is active.
        self.chunk_transform_X = None
        self.chunk_transform_G = None
        if batched and deferred:
            raise ValueError("batched and deferred modes are mutually exclusive")

        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._enabled = False

        # Running upper-triangular R factors, updated each hook call.
        # None means no data accumulated yet for this layer.
        self._R_X: Dict[nn.Module, Optional[torch.Tensor]] = {}
        self._R_G: Dict[nn.Module, Optional[torch.Tensor]] = {}

        # Total accumulated rows (for p >= n validation).
        self._n_rows_X: Dict[nn.Module, int] = {}
        self._n_rows_G: Dict[nn.Module, int] = {}

        # Batched-mode pending buffers: per-layer list of chunks to flush.
        # Cleared on each flush() call.
        self._pending_X: Dict[nn.Module, List[torch.Tensor]] = {}
        self._pending_G: Dict[nn.Module, List[torch.Tensor]] = {}

        # Deferred-mode raw chunk buffers: kept across the whole refresh
        # window, drained inside get_factors().
        self._raw_X: Dict[nn.Module, List[torch.Tensor]] = {}
        self._raw_G: Dict[nn.Module, List[torch.Tensor]] = {}

        self._linear_layers: List[nn.Module] = []

        for module in model.modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            if max_out_dim > 0:
                out_dim = (
                    module.out_features
                    if isinstance(module, nn.Linear)
                    else module.out_channels
                )
                if out_dim > max_out_dim:
                    continue
            self._linear_layers.append(module)
            self._R_X[module] = None
            self._R_G[module] = None
            self._n_rows_X[module] = 0
            self._n_rows_G[module] = 0
            self._pending_X[module] = []
            self._pending_G[module] = []
            self._raw_X[module] = []
            self._raw_G[module] = []

    # ------------------------------------------------------------------
    # GramMatrixEstimator interface
    # ------------------------------------------------------------------

    @property
    def linear_layers(self) -> List[nn.Module]:
        return self._linear_layers

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        """Register forward and backward hooks on all tracked layers."""
        if self._enabled:
            return
        for module in self._linear_layers:
            h_fwd = module.register_forward_hook(self._forward_hook)
            h_bwd = module.register_full_backward_hook(self._backward_hook)
            self._handles.extend([h_fwd, h_bwd])
        self._enabled = True

    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return damped R factors (R_X, R_G) for each layer with data.

        Applies ridge augmentation for damping before returning.  Raises
        IFKFACRankError if any layer has p < n (caller should fall back to
        Classic K-FAC for that layer).

        Returns
        -------
        dict : module → (R_X, R_G)
            R_X : (n_in,  n_in)  upper-triangular  (input factor)
            R_G : (n_out, n_out) upper-triangular  (gradient factor)
        """
        logger.debug("get_factors() called: %d tracked layers", len(self._linear_layers))

        # Deferred mode: drain raw chunk buffers via one big leaf QR per
        # layer, then fall through to the regular damping/normalisation path.
        if self.deferred:
            self._drain_deferred()

        factors: Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]] = {}
        for module in self._linear_layers:
            R_X_raw = self._R_X.get(module)
            R_G_raw = self._R_G.get(module)
            if R_X_raw is None or R_G_raw is None:
                logger.debug("get_factors() [%s]: skipped — no data yet",
                             _module_tag(module))
                continue   # no data yet

            # When p < n, torch.linalg.qr(mode='reduced') returns a non-square
            # R of shape (p, n) instead of (n, n).  Detect this by checking
            # whether the stored R is square — if not, p < n was violated.
            # In batched mode the chunks are zero-padded to (B, max_p, n) so
            # R is always (n, n); use the row-count tally instead.
            rows_X = self._n_rows_X[module]
            rows_G = self._n_rows_G[module]

            # BUG FIX (was: raise IFKFACRankError).
            # If R is non-square, this layer hasn't yet accumulated enough
            # rows for XᵀX to be full-rank.  Previously we raised, which the
            # caller handled by hooks.clear() — wiping ALL layers' progress,
            # not just this one.  That caused the perpetual-fail loop on
            # models with wide first layers (e.g. AE n_in=784, batch=256).
            # New behaviour: skip this layer for this round.  Its R_X stays
            # partial-rank and will accumulate more rows on subsequent
            # forward passes.  Other layers that DID accumulate enough still
            # get their factors returned.
            if R_X_raw.shape[0] != R_X_raw.shape[1]:
                logger.debug(
                    "get_factors() [%s]: skipped — X factor partial-rank "
                    "(%d rows accumulated, need >= n_in=%d)",
                    _module_tag(module), rows_X, R_X_raw.shape[1],
                )
                continue
            if R_G_raw.shape[0] != R_G_raw.shape[1]:
                logger.debug(
                    "get_factors() [%s]: skipped — G factor partial-rank "
                    "(%d rows accumulated, need >= n_out=%d)",
                    _module_tag(module), rows_G, R_G_raw.shape[1],
                )
                continue
            if self.batched:
                if rows_X < R_X_raw.shape[1]:
                    raise IFKFACRankError(
                        f"Layer {module}: accumulated {rows_X} rows for X, "
                        f"but need >= n_in={R_X_raw.shape[1]} (batched mode)."
                    )
                if rows_G < R_G_raw.shape[1]:
                    raise IFKFACRankError(
                        f"Layer {module}: accumulated {rows_G} rows for δ, "
                        f"but need >= n_out={R_G_raw.shape[1]} (batched mode)."
                    )

            n_in  = R_X_raw.shape[0]
            n_out = R_G_raw.shape[0]

            logger.debug(
                "get_factors() [%s]: rows_X=%d rows_G=%d  "
                "R_X_raw=%s  R_G_raw=%s  damping=%.4g",
                _module_tag(module), rows_X, rows_G,
                tuple(R_X_raw.shape), tuple(R_G_raw.shape), self.damping,
            )

            # BUG FIX: normalize by sqrt(n_rows) so that RᵀR ≈ XᵀX/n (per-sample
            # mean), matching ClassicKFAC which divides A_sum by n_rows in
            # get_factors().  Without this, RᵀR = XᵀX (sum), making the
            # preconditioner n_rows times too small and the effective lr ~1280×
            # too small for typical KFAC_FREQ=20, batch=64 settings.
            R_X_scaled = R_X_raw / math.sqrt(rows_X)
            R_G_scaled = R_G_raw / math.sqrt(rows_G)

            R_X = finalize_R(R_X_scaled, self.damping)
            R_G = finalize_R(R_G_scaled, self.damping)

            logger.debug(
                "get_factors() [%s]: done → R_X=%s  R_G=%s",
                _module_tag(module), tuple(R_X.shape), tuple(R_G.shape),
            )
            factors[module] = (R_X, R_G)

        logger.debug("get_factors() returning %d factor pairs", len(factors))
        return factors

    # ------------------------------------------------------------------
    # Deferred drain — one big leaf QR per layer at refresh time
    # ------------------------------------------------------------------

    def _drain_deferred(self):
        """Drain any chunks still buffered at refresh into the running R.

        In-window merges happen inside the hooks once a layer's buffer hits
        ``deferred_window`` chunks.  This call handles only the residual
        buffer (≤ deferred_window - 1 chunks per layer) at get_factors() time.
        """
        for module in self._linear_layers:
            chunks_X = self._raw_X.get(module, [])
            chunks_G = self._raw_G.get(module, [])
            if chunks_X:
                X = chunks_X[0] if len(chunks_X) == 1 else torch.cat(chunks_X, dim=0)
                self._R_X[module] = streaming_tsqr_update(self._R_X.get(module), X)
                self._raw_X[module] = []
            if chunks_G:
                G = chunks_G[0] if len(chunks_G) == 1 else torch.cat(chunks_G, dim=0)
                self._R_G[module] = streaming_tsqr_update(self._R_G.get(module), G)
                self._raw_G[module] = []

    # ------------------------------------------------------------------
    # Batched flush — drain pending chunks via 2 batched QRs per bucket
    # ------------------------------------------------------------------

    def flush(self):
        """Drain pending chunks (batched mode) into the running R factors.

        Groups pending chunks by ``n`` (the trailing dim).  For each bucket:
          - concatenates each layer's pending chunks along the row axis
          - pads to the bucket's max row count with zeros (inert in QR)
          - stacks running R factors into (B, n, n), using zeros for layers
            that have no prior R
          - calls ``batched_streaming_tsqr_update`` (2 cuSOLVER launches)
          - writes results back to ``_R_X`` / ``_R_G``

        No-op when batched mode is off, or when buffers are empty.
        """
        if not self.batched:
            return
        if any(self._pending_X.values()):
            self._flush_bucket(self._pending_X, self._R_X)
        if any(self._pending_G.values()):
            self._flush_bucket(self._pending_G, self._R_G)

    def _flush_bucket(
        self,
        pending: Dict[nn.Module, List[torch.Tensor]],
        running: Dict[nn.Module, Optional[torch.Tensor]],
    ):
        """Bucket pending chunks by trailing dim n and drain each bucket."""
        # Bucket modules by n_in/n_out (already includes bias augmentation
        # for X if augment_bias is on).
        buckets: Dict[int, List[nn.Module]] = {}
        for module, chunks in pending.items():
            if not chunks:
                continue
            n = chunks[0].shape[1]
            buckets.setdefault(n, []).append(module)

        for n, modules in buckets.items():
            self._drain_one_bucket(modules, n, pending, running)

        # Reset pending buffers.
        for module in pending:
            pending[module] = []

    def _drain_one_bucket(
        self,
        modules: List[nn.Module],
        n: int,
        pending: Dict[nn.Module, List[torch.Tensor]],
        running: Dict[nn.Module, Optional[torch.Tensor]],
    ):
        """Run two batched QRs for one (n)-bucket of layers."""
        B = len(modules)
        if B == 0:
            return

        # Concatenate each layer's pending chunks, find bucket max rows.
        merged_chunks: List[torch.Tensor] = []
        for m in modules:
            if len(pending[m]) == 1:
                merged_chunks.append(pending[m][0])
            else:
                merged_chunks.append(torch.cat(pending[m], dim=0))

        # Use first chunk's device/dtype as bucket reference.
        ref = merged_chunks[0]
        device, dtype = ref.device, ref.dtype
        max_p = max(c.shape[0] for c in merged_chunks)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("_drain_one_bucket: n=%d B=%d max_p=%d", n, B, max_p)

        # Pad each chunk to (max_p, n) with zeros, then stack -> (B, max_p, n).
        # Keep the chunks' own dtype here (bf16 under AMP) — the promotion to
        # fp32 happens once inside the QR, so this staging buffer stays half-size.
        padded = torch.zeros(B, max_p, n, device=device, dtype=dtype)
        for i, c in enumerate(merged_chunks):
            padded[i, :c.shape[0]] = c

        # Build running_R stack: (B, n, n) with zeros for first-call layers.
        # The running factors are fp32 accumulators (see triangular._qr), so this
        # buffer must match them — allocating it at the chunk dtype would round
        # every prior R back down to bf16 on assignment.
        r_dtype = torch.float32
        for m in modules:
            if running[m] is not None:
                r_dtype = running[m].dtype
                break
        running_stack = torch.zeros(B, n, n, device=device, dtype=r_dtype)
        for i, m in enumerate(modules):
            R_old = running[m]
            if R_old is not None:
                running_stack[i] = R_old

        # Two batched cuSOLVER QR calls do all B layers at once.
        R_out = batched_streaming_tsqr_update(running_stack, padded)

        # Write merged Rs back to the running dict.
        for i, m in enumerate(modules):
            running[m] = R_out[i]

    def clear(self):
        """Reset all R-factor accumulators."""
        for module in self._linear_layers:
            self._R_X[module] = None
            self._R_G[module] = None
            self._n_rows_X[module] = 0
            self._n_rows_G[module] = 0
            self._pending_X[module] = []
            self._pending_G[module] = []
            self._raw_X[module] = []
            self._raw_G[module] = []

    def remove(self):
        """Detach all hooks and free state."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._enabled = False
        self.clear()

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        target = module or (self._linear_layers[0] if self._linear_layers else None)
        if target is None:
            return 0
        return self._n_rows_X.get(target, 0)

    # ------------------------------------------------------------------
    # Conv2d helpers (same as KFACHooks — im2col unfolding)
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_out_hw(module: nn.Conv2d, hw) -> Optional[Tuple[int, int]]:
        """Spatial output size of ``module`` for an input of size ``hw``.

        Returns None when ``padding`` is a string ('same' / 'valid'), where the
        arithmetic below does not apply; callers then fall back to capping rows
        after the unfold.
        """
        if isinstance(module.padding, str):
            return None
        out = []
        for i in range(2):
            k, s = module.kernel_size[i], module.stride[i]
            p, d = module.padding[i], module.dilation[i]
            out.append((hw[i] + 2 * p - d * (k - 1) - 1) // s + 1)
        return out[0], out[1]

    @staticmethod
    def _spatial_subsample_factor(total_rows: int, max_rows: int) -> int:
        """Stride multiplier that brings ``total_rows`` under ``max_rows``.

        Both spatial axes are subsampled, so rows scale as 1/s².
        """
        if max_rows <= 0 or total_rows <= max_rows:
            return 1
        return max(1, math.ceil(math.sqrt(total_rows / max_rows)))

    @classmethod
    def _unfold_conv_input(cls, x: torch.Tensor, module: nn.Conv2d,
                           max_rows: int = 0) -> torch.Tensor:
        """im2col of a conv input, subsampled *before* materialisation.

        F.unfold expands (B, C_in, H, W) into (B, C_in·kH·kW, L), and the
        permute + reshape that follows copies it again.  For a UNet encoder that
        is multiple GB per layer per step (64 imgs × 576 cols × 65536 locations
        in bf16 = 4.8 GB), so capping rows *after* the unfold — as this used to
        do — still paid the full allocation on every forward pass.

        Subsampling by raising the unfold stride to a multiple of the conv's own
        stride yields an exact subset of the true patches (locations at multiples
        of s·stride are a subset of those at multiples of stride), never
        materialises the large tensor, and keeps every image in the batch
        represented — which slicing whole images off the front would not.
        """
        stride = tuple(module.stride)
        out_hw = cls._conv_out_hw(module, x.shape[-2:])
        if max_rows > 0 and out_hw is not None:
            total = x.shape[0] * out_hw[0] * out_hw[1]
            s = cls._spatial_subsample_factor(total, max_rows)
            if s > 1:
                stride = tuple(v * s for v in stride)

        x_unf = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=stride,
        )   # (B, C_in·kH·kW, L)
        B, C_kk, L = x_unf.shape
        return x_unf.permute(0, 2, 1).reshape(B * L, C_kk)

    @classmethod
    def _reshape_conv_grad(cls, delta: torch.Tensor,
                           max_rows: int = 0) -> torch.Tensor:
        """Flatten (B, C_out, H, W) gradients into (B·H·W, C_out) rows.

        Same allocation problem as _unfold_conv_input: permute + reshape on a
        non-contiguous tensor copies the whole thing, so the subsampling has to
        happen first.  Each row is one spatial location's C_out vector, so
        striding the H/W axes selects an exact subset of rows taken from every
        image in the batch.
        """
        B, C_out, H_out, W_out = delta.shape
        s = cls._spatial_subsample_factor(B * H_out * W_out, max_rows)
        if s > 1:
            delta = delta[:, :, ::s, ::s]
            B, C_out, H_out, W_out = delta.shape
        return delta.permute(0, 2, 3, 1).reshape(B * H_out * W_out, C_out)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """Streaming TSQR update for input activations X."""
        if not self._enabled:
            return
        x = input[0].detach()
        raw_shape = tuple(x.shape)

        if isinstance(module, nn.Conv2d):
            # Spatial patches are highly correlated — subsample to cap the leaf
            # QR size.  Without this, a CIFAR-10 conv layer produces 8192 rows
            # per batch (128 images × 64 spatial locations), making streaming
            # TSQR the per-step bottleneck even on GPU.  The subsampling runs
            # inside the unfold so the im2col intermediate is never allocated at
            # full size; the randperm below only mops up the ceil() slack.
            x = self._unfold_conv_input(x, module, self.max_conv_rows)
            if self.max_conv_rows > 0 and x.shape[0] > self.max_conv_rows:
                idx = torch.randperm(x.shape[0], device=x.device)[:self.max_conv_rows]
                x = x[idx]
        elif x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])            # (B·T, d_in)
            # KFAC-Reduce: subsample to cap leaf-QR rows for transformer Linear
            # layers.  Same policy as KFACHooks._SEQ_SUBSAMPLE so IFKFAC uses the
            # same sample budget per step as Classic/OlsSM.
            if self.max_seq_rows > 0 and x.shape[0] > self.max_seq_rows:
                idx = torch.randperm(x.shape[0], device=x.device)[:self.max_seq_rows]
                x = x[idx]

        # Bias augmentation: append column of ones so bias is handled implicitly
        if self.augment_bias and isinstance(module, nn.Linear) and module.bias is not None:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=1)           # (p, n_in + 1)

        # Optional per-step row transform (e.g. WGSO weighting).  Applied
        # exactly once per chunk regardless of streaming/batched/deferred mode.
        if self.chunk_transform_X is not None:
            x = self.chunk_transform_X(x)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_forward_hook [%s]: raw_shape=%s → chunk=%s  "
                "total_rows_X=%d  running_R_X=%s",
                _module_tag(module), raw_shape, tuple(x.shape),
                self._n_rows_X[module] + x.shape[0],
                "None" if self._R_X[module] is None
                else str(tuple(self._R_X[module].shape)),
            )

        if self.deferred:
            # Buffer chunks; merge in larger batches via streaming_tsqr_update
            # once the per-layer buffer hits deferred_window entries.
            self._raw_X[module].append(x)
            self._n_rows_X[module] += x.shape[0]
            if len(self._raw_X[module]) >= self.deferred_window:
                Xc = (self._raw_X[module][0] if len(self._raw_X[module]) == 1
                       else torch.cat(self._raw_X[module], dim=0))
                self._R_X[module] = streaming_tsqr_update(self._R_X[module], Xc)
                self._raw_X[module] = []
            return

        if self.batched:
            # Buffer for later flush — no QR launched here.
            self._pending_X[module].append(x)
            self._n_rows_X[module] += x.shape[0]
            return

        # Streaming TSQR update
        self._R_X[module] = streaming_tsqr_update(self._R_X[module], x)
        self._n_rows_X[module] += x.shape[0]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_forward_hook [%s]: R_X updated → shape=%s  "
                "diag_min=%.4g diag_max=%.4g",
                _module_tag(module),
                tuple(self._R_X[module].shape),
                self._R_X[module].diag().min().item(),
                self._R_X[module].diag().max().item(),
            )

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ):
        """Streaming TSQR update for gradient signals δ."""
        if not self._enabled:
            return
        delta = grad_output[0].detach()
        raw_shape = tuple(delta.shape)

        if isinstance(module, nn.Conv2d):
            # As in the forward hook: subsample spatially before the reshape so
            # the (B·H·W, C_out) copy is never made at full size.
            delta = self._reshape_conv_grad(delta, self.max_conv_rows)
            if self.max_conv_rows > 0 and delta.shape[0] > self.max_conv_rows:
                idx = torch.randperm(delta.shape[0],
                                     device=delta.device)[:self.max_conv_rows]
                delta = delta[idx]
        elif delta.ndim > 2:
            delta = delta.reshape(-1, delta.shape[-1])  # (B·T, d_out)
            # KFAC-Reduce on output gradients - mirror the forward-hook subsample.
            if self.max_seq_rows > 0 and delta.shape[0] > self.max_seq_rows:
                idx = torch.randperm(delta.shape[0], device=delta.device)[:self.max_seq_rows]
                delta = delta[idx]

        # Optional per-step row transform (e.g. WGSO weighting on gradients).
        if self.chunk_transform_G is not None:
            delta = self.chunk_transform_G(delta)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_backward_hook [%s]: raw_shape=%s → chunk=%s  "
                "total_rows_G=%d  running_R_G=%s",
                _module_tag(module), raw_shape, tuple(delta.shape),
                self._n_rows_G[module] + delta.shape[0],
                "None" if self._R_G[module] is None
                else str(tuple(self._R_G[module].shape)),
            )

        if self.deferred:
            self._raw_G[module].append(delta)
            self._n_rows_G[module] += delta.shape[0]
            if len(self._raw_G[module]) >= self.deferred_window:
                Gc = (self._raw_G[module][0] if len(self._raw_G[module]) == 1
                       else torch.cat(self._raw_G[module], dim=0))
                self._R_G[module] = streaming_tsqr_update(self._R_G[module], Gc)
                self._raw_G[module] = []
            return

        if self.batched:
            self._pending_G[module].append(delta)
            self._n_rows_G[module] += delta.shape[0]
            return

        self._R_G[module] = streaming_tsqr_update(self._R_G[module], delta)
        self._n_rows_G[module] += delta.shape[0]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "_backward_hook [%s]: R_G updated → shape=%s  "
                "diag_min=%.4g diag_max=%.4g",
                _module_tag(module),
                tuple(self._R_G[module].shape),
                self._R_G[module].diag().min().item(),
                self._R_G[module].diag().max().item(),
            )
