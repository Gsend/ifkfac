"""
optimizer/asdl_classic_kfac.py — Adapter wrapping Kazuki Osawa's `asdl`
library's reference K-FAC implementation so it matches the same constructor
signature as our homebrew `optimizer.classic_kfac.ClassicKFAC`, with one
unavoidable API difference: the `.step()` method takes a **closure** that
returns the loss (like `torch.optim.LBFGS`), because ASDL owns the
forward + backward pass internally.

Discovered via Phase 0 (benchmark/test_asdl_install.py) against asdl 0.1.0
installed from https://github.com/kazukiosawa/asdl:

- `from asdl.precondition import KfacGradientMaker, PreconditioningConfig`
- Constructor: `KfacGradientMaker(model, config)` — both positional.
- Flow:
    maker.setup_model_call(model, x)
    maker.setup_loss_call(loss_fn, model_output, y)
    maker.do_update_curvature      = (step % freq == 0)
    maker.do_update_preconditioner = (step % freq == 0)
    maker.forward_and_backward()   # runs forward+backward, captures
                                    # curvature, preconditions .grad
    loss = maker.loss
    base_opt.step()                 # applies the preconditioned .grad

Usage in a training loop:

    opt = AsdlClassicKFAC(
        model, lr=2e-3, damping=1e-4,
        factor_update_freq=20, momentum=0.7,
        grad_clip=300, use_bf16_factors=False,
    )

    for x, y in loader:
        def closure():
            opt.zero_grad()
            return F.cross_entropy(model(x), y)
        loss = opt.step(closure, inputs=x, targets=y)

The `inputs` and `targets` kwargs are how the adapter feeds ASDL's
`setup_model_call`/`setup_loss_call`.  The closure itself is only used as a
fallback for diagnostic logging — ASDL re-runs the forward+backward via its
own hooks.

This adapter is honest about the API difference: it cannot pretend to be a
drop-in `torch.optim.Optimizer.step()` because ASDL needs to OWN the forward
pass (it installs PyTorch hooks that capture activations + gradients per
layer).  Section runners in `benchmark/run_asdl_sweep.py` adapt the existing
training loops to feed inputs/targets explicitly.
"""
from __future__ import annotations
import logging
import warnings
import inspect
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

def _import_asdl():
    """Locate the Osawa K-FAC library.  Rejects the stdlib ASDL grammar
    parser (Python's `asdl` for AST manipulation) by checking exports."""
    last_err: Exception | None = None
    for module_name in ["asdl", "asdfghjkl"]:
        try:
            top = __import__(module_name)
            exports = set(x for x in dir(top) if not x.startswith("_"))
            stdlib_markers = {"ASDLParser", "ASDLSyntaxError", "VisitorBase",
                              "TokenKind", "tokenize_asdl"}
            if stdlib_markers.issubset(exports):
                last_err = ImportError(
                    f"'{module_name}' is the Python stdlib ASDL grammar parser, "
                    f"not the Osawa K-FAC library."
                )
                continue
            precond_mod = __import__(
                f"{module_name}.precondition",
                fromlist=["KfacGradientMaker", "PreconditioningConfig"],
            )
            return (
                getattr(precond_mod, "KfacGradientMaker"),
                getattr(precond_mod, "PreconditioningConfig"),
            )
        except ImportError as e:
            last_err = e
            continue
    raise ImportError(
        "Could not locate the Osawa K-FAC library.  Install with:\n"
        "    pip install git+https://github.com/kazukiosawa/asdl.git\n"
        f"Last error: {last_err}"
    ) from last_err


def _autodetect_unsupported_modules(model: nn.Module) -> list[str]:
    """Walk the model and return the names of any modules ASDL's K-FAC
    cannot handle.  ASDL K-FAC supports nn.Linear and nn.Conv2d only.

    Container modules (nn.Sequential, nn.ModuleList, etc.) and the model
    itself (name="") are not unsupported — they don't have learnable
    parameters of their own, so ASDL skips them naturally.
    """
    SUPPORTED = (nn.Linear, nn.Conv2d)
    CONTAINERS = (nn.Sequential, nn.ModuleList, nn.ModuleDict, nn.Module)

    skip = []
    for name, mod in model.named_modules():
        if name == "":
            continue   # the top-level model itself
        if isinstance(mod, SUPPORTED):
            continue
        # Some modules have children but no own parameters — skip if they
        # have no own .weight/.bias.  This catches the activation modules
        # (Tanh, ReLU, etc.) AND containers (Sequential, ModuleList).
        own_params = [p for p in mod.parameters(recurse=False)]
        if not own_params:
            # Activations / containers / dropout — but only skip if it's
            # NOT a supported parameterised layer. Some layers like
            # BatchNorm2d have learnable params but ASDL doesn't support
            # them either, so they fall through here too.
            if not any(p.requires_grad for p in mod.parameters(recurse=False)):
                continue
        # Has own params and is not supported — mark to ignore
        skip.append(name)
    return skip


def _build_precond_config(PreconditioningConfig, *, damping, factor_update_freq,
                            ema_decay, ignore_modules=None):
    """Construct a PreconditioningConfig, populating fields the installed
    asdl version actually accepts.  Discovered via dump_asdl_loss_api.py:

        fields = ['num_total_steps', 'preconditioner_upd_interval',
                  'preconditioner_warmup_steps', 'preconditioner_upd_ratio',
                  'preconditioner_warmup_ratio', 'preconditioner_interval_type',
                  'curvature_upd_interval', 'curvature_warmup_steps',
                  'curvature_upd_ratio', 'curvature_warmup_ratio',
                  'data_size', 'damping', 'ema_decay', 'ignore_modules']
    """
    sig = inspect.signature(PreconditioningConfig.__init__)
    fields = set(sig.parameters.keys()) - {"self"}

    candidate = {}
    if "damping" in fields:
        candidate["damping"] = damping
    if "data_size" in fields:
        candidate["data_size"] = 1
    if "curvature_upd_interval" in fields:
        candidate["curvature_upd_interval"] = factor_update_freq
    if "preconditioner_upd_interval" in fields:
        candidate["preconditioner_upd_interval"] = factor_update_freq
    if "ema_decay" in fields:
        candidate["ema_decay"] = ema_decay
    if "ignore_modules" in fields and ignore_modules:
        candidate["ignore_modules"] = list(ignore_modules)

    try:
        return PreconditioningConfig(**candidate)
    except TypeError as e:
        warnings.warn(
            f"PreconditioningConfig(**{candidate}) failed: {e}.  Falling back to "
            f"the default constructor."
        )
        return PreconditioningConfig()


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class AsdlClassicKFAC(torch.optim.Optimizer):
    """ASDL K-FAC preconditioner + SGD base optimiser.

    Constructor signature matches our homebrew ClassicKFAC.  The `.step()`
    method requires a closure that returns the loss, and additionally takes
    `inputs` and `targets` kwargs so ASDL can run forward_and_backward
    internally with its layer hooks engaged."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-3,
        damping: float = 1e-4,
        factor_update_freq: int = 20,
        momentum: float = 0.7,
        grad_clip: float | None = 300.0,
        weight_decay: float = 0.0,
        use_bf16_factors: bool = False,
        gamma: float = 0.9,
        ignore_modules: list[str] | None = None,
        fisher_type: str = "fisher_mc",
        loss_type: str = "cross_entropy",
        base_params: list | None = None,
        **extra: Any,
    ):
        """ASDL Classic K-FAC wrapper.

        Args:
            ignore_modules: list of named-module substrings to skip during
                K-FAC curvature computation.  Use this for transformer /
                CNN models that contain modules ASDL doesn't support
                (LayerNorm, Embedding, MultiheadAttention, Dropout, etc.).

                Example for SmallGPT:
                    ignore_modules=["embed", "ln", "attn", "drop"]

                Example for ResNet:
                    ignore_modules=["bn", "downsample"]

                If not given, auto-detects by walking the model and listing
                every named_module that isn't nn.Linear or nn.Conv2d.
            fisher_type: which Fisher ASDL estimates — "fisher_mc" (default;
                MC-samples labels from the model's predictive distribution;
                correct for a categorical/softmax head, e.g. §5.8 CIFAR-10),
                "fisher_exact", or "fisher_emp" (empirical Fisher = E[δδᵀ] using
                the REAL targets' loss gradient, no output-distribution
                assumption).  For the §5.7 MNIST autoencoder the head is 784
                independent Bernoulli pixels (BCE), which ASDL's MC path cannot
                model (it only knows cross_entropy / mse), so use "fisher_emp"
                there — that also matches our homebrew ClassicKFAC, whose G is
                built from the real BCE backward.
            loss_type: ASDL's loss family for the Fisher — "cross_entropy" or
                "mse".  Ignored when fisher_type="fisher_emp".
        """
        KfacGradientMaker, PreconditioningConfig = _import_asdl()

        defaults = dict(
            lr=lr, damping=damping,
            factor_update_freq=factor_update_freq,
            momentum=momentum, weight_decay=weight_decay,
        )
        super().__init__(model.parameters(), defaults)

        self.model = model
        self.lr = lr
        self.damping = damping
        self.factor_update_freq = factor_update_freq
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_bf16_factors = use_bf16_factors
        self.gamma = gamma
        self._step_count = 0

        # Auto-detect unsupported modules if no explicit list given.
        # ASDL K-FAC only supports nn.Linear and nn.Conv2d.
        if ignore_modules is None:
            ignore_modules = _autodetect_unsupported_modules(model)
        self.ignore_modules = ignore_modules
        if ignore_modules:
            logger.info(
                "AsdlClassicKFAC: ignoring %d module(s) ASDL doesn't support: %s",
                len(ignore_modules), ignore_modules,
            )

        # --- Build ASDL's preconditioner ---
        config = _build_precond_config(
            PreconditioningConfig,
            damping=damping,
            factor_update_freq=factor_update_freq,
            ema_decay=gamma,
            ignore_modules=ignore_modules,
        )
        # fisher_type / loss_type are constructor kwargs on KfacGradientMaker
        # (not PreconditioningConfig fields).  fisher_emp ignores loss_type.
        self.fisher_type = fisher_type
        self.loss_type = loss_type
        maker_kwargs = {"fisher_type": fisher_type}
        if fisher_type != "fisher_emp":
            maker_kwargs["loss_type"] = loss_type
        try:
            self._maker = KfacGradientMaker(model, config, **maker_kwargs)
        except TypeError:
            # Some versions take (config, model) order or lack these kwargs.
            try:
                self._maker = KfacGradientMaker(model, config)
            except TypeError:
                self._maker = KfacGradientMaker(config, model)

        # --- Base SGD ---
        # By default the internal SGD steps EVERY model parameter.  For models
        # with non-K-FAC layers (transformer embeddings/LayerNorm/attention,
        # CNN BatchNorm) ASDL only preconditions the Linear/Conv2d params, so
        # the rest would be stepped with raw SGD here.  To match our homebrew
        # ClassicKFAC setup — which trains non-K-FAC params with a secondary
        # AdamW (see comparison_4way_multiseed.make_optimizers) — pass
        # `base_params` = just the K-FAC-eligible params, and run your own
        # secondary optimiser for the remainder.  When None, step all params.
        sgd_params = base_params if base_params is not None else list(
            model.parameters())
        self._base_sgd = torch.optim.SGD(
            sgd_params,
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )

        # bf16 regime: patch ASDL's Gram inversion so each Kron factor (A, B) is
        # quantised to bf16 *before* cholesky_inv — the κ² injection point.
        self._bf16_patched = False
        if use_bf16_factors:
            self._install_bf16_gram_patch()

        logger.info(
            "AsdlClassicKFAC ready: lr=%.2e damping=%.2e freq=%d mom=%.2f "
            "wd=%.2e bf16=%s",
            lr, damping, factor_update_freq, momentum, weight_decay,
            use_bf16_factors,
        )

    # -----------------------------------------------------------------
    # bf16 κ² injection — quantise the Gram before ASDL inverts it.
    # -----------------------------------------------------------------
    def _install_bf16_gram_patch(self) -> None:
        """Monkeypatch ASDL's cholesky_inv so the Gram factor is round-tripped
        through bf16 *before* the inversion.

        This is the faithful analogue of our homebrew classic bf16 path
        (kfac_bf16_compare._classic_step_bf16, which bf16-quantises the cached
        inverses) and of §5.8's _evd_from_gram (which bf16-quantises A/G before
        eigh).  bf16 noise on the Gram A = XᵀX carries effective conditioning
        κ²(X) — the squaring catastrophe — so the resulting inverse (and hence
        the natural-gradient preconditioner) is corrupted exactly where the
        paper says it should be.

        symmatrix.py does `from .utils import cholesky_inv`, binding its own
        module-level name, so we patch `asdl.symmatrix.cholesky_inv` (the
        reference ASDL actually calls), not `asdl.utils.cholesky_inv`.
        """
        if self._bf16_patched:
            return
        import asdl.symmatrix as _sym
        self._sym_mod = _sym
        self._orig_cholesky_inv = _sym.cholesky_inv

        def _bf16_gram_cholesky_inv(X, damping=1e-7, _orig=_sym.cholesky_inv):
            # ★ κ² catastrophe lands here: quantise the Gram, then invert.
            Xb = X.to(torch.bfloat16).to(torch.float32)
            return _orig(Xb, damping)

        _sym.cholesky_inv = _bf16_gram_cholesky_inv
        self._bf16_patched = True

    def _remove_bf16_gram_patch(self) -> None:
        if getattr(self, "_bf16_patched", False):
            self._sym_mod.cholesky_inv = self._orig_cholesky_inv
            self._bf16_patched = False

    def cleanup(self) -> None:
        """Restore any monkeypatches.  Called by the sweep driver's finally
        block so a bf16 cell can't leak its cholesky_inv patch into the next."""
        self._remove_bf16_gram_patch()

    # -----------------------------------------------------------------
    # The closure-style step
    # -----------------------------------------------------------------
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        *,
        inputs: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Run one ASDL-K-FAC step.

        Args:
            closure  : optional, only used to *initially* compute the loss
                       so the caller can log it; ASDL re-runs the forward
                       internally via setup_model_call/setup_loss_call.
            inputs   : input tensor to feed to model(inputs).
            targets  : target tensor to feed to loss_fn(output, targets).
            loss_fn  : the loss function (default: cross_entropy).
        """
        loss = self._setup_and_fwd_bwd(inputs, targets, loss_fn)

        # bf16 regime: handled by the cholesky_inv patch installed in __init__
        # (it quantises each Gram to bf16 *before* inversion — the κ² injection
        # point).  No post-hoc round-trip here: quantising the cached factors
        # AFTER forward_and_backward already used the fp32 inverses has ~zero
        # effect on the natural gradient (ASDL recomputes the inverse in fp32 at
        # the next update), which is why the old _bf16_roundtrip never produced
        # a collapse.  See _install_bf16_gram_patch.

        # Grad clip + base SGD
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip,
            )
        self._base_sgd.step()
        self._step_count += 1

        return loss

    def _setup_and_fwd_bwd(self, inputs, targets, loss_fn):
        """Register model+loss callables with ASDL and run its
        forward+backward (which updates curvature/preconditioner per the
        configured cadence and writes the preconditioned natural gradient
        into each parameter's .grad).  Returns ASDL's reported loss.

        Shared by step() (which then applies the SGD weight update) and
        accumulate_curvature() (which does NOT — used for Fisher capture at a
        fixed point, e.g. §5.8 Laplace, where moving the weights is wrong)."""
        if inputs is None or targets is None:
            raise ValueError(
                "AsdlClassicKFAC needs explicit inputs and targets — ASDL must "
                "own the forward pass to engage its hooks.  See the module "
                "docstring for the recommended training-loop pattern."
            )
        if loss_fn is None:
            import torch.nn.functional as F
            loss_fn = F.cross_entropy

        # ASDL scales the captured Fisher by 1/data_size (see asdl/fisher.py:
        # `scale /= data_size`).  data_size must therefore be the number of
        # samples the curvature is averaged over — i.e. this minibatch's batch
        # size.  Leaving it at 1 makes the Fisher a SUM rather than a MEAN of
        # per-sample outer products, inflating it ~batch_size× and shrinking
        # the natural gradient by the same factor (→ SGD-slow convergence).
        # It also un-scales the Fisher-loop loss on curvature steps.
        self._maker.config.data_size = int(inputs.shape[0])

        # setup_model_call returns a DummyObject placeholder for the model
        # output.  That placeholder is then passed as the FIRST positional
        # arg to setup_loss_call, which ASDL resolves to the real model
        # output at call time.  Example from the ASDL docstring:
        #     dummy_y = grad_maker.setup_model_call(model, x)
        #     grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t)
        dummy_y = self._maker.setup_model_call(self.model, inputs)
        self._maker.setup_loss_call(loss_fn, dummy_y, targets)

        # ASDL's forward_and_backward() takes no step kwarg — the cadence is
        # controlled by curvature_upd_interval / preconditioner_upd_interval in
        # PreconditioningConfig set at construction.
        try:
            self._maker.forward_and_backward()
        except Exception as e:
            raise RuntimeError(
                f"asdl.KfacGradientMaker.forward_and_backward() failed: {e}.  "
                f"Check optimizer/asdl_classic_kfac.py."
            ) from e
        return getattr(self._maker, "loss", None)

    @torch.enable_grad()
    def accumulate_curvature(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Run ASDL's forward+backward to update the K-FAC curvature WITHOUT
        applying a weight update.  Used for Fisher capture at a fixed point
        (§5.8 Laplace): the model must stay at the MAP while we accumulate the
        per-layer Kronecker factors.  Read them afterwards from
        `module.fisher.kron.A` / `.B`."""
        loss = self._setup_and_fwd_bwd(inputs, targets, loss_fn)
        # Drop the preconditioned grad so it can't leak into a later step.
        self.zero_grad(set_to_none=True)
        self._step_count += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._base_sgd.zero_grad(set_to_none=set_to_none)

    # -----------------------------------------------------------------
    # bf16 round-trip — best-effort.  Walks the maker's attribute tree
    # looking for fp32 tensors and round-trips them through bf16.
    # -----------------------------------------------------------------
    def _bf16_roundtrip(self) -> None:
        seen: set[int] = set()
        def walk(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.Tensor) and obj.dtype == torch.float32:
                obj.data.copy_(obj.to(torch.bfloat16).to(torch.float32))
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)
                return
            if hasattr(obj, "__dict__"):
                for v in vars(obj).values():
                    walk(v)
        walk(self._maker)


def build_asdl_classic_kfac(model, precision="fp32", **kwargs):
    """Drop-in for the benchmark dispatch."""
    return AsdlClassicKFAC(
        model, use_bf16_factors=(precision == "bf16"), **kwargs,
    )
