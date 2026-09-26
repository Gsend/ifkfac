"""
Phase 3 — Full ASDL Classic K-FAC sweep with incremental saving and resume.

Re-runs all the cells that used Classic K-FAC in the paper, with ASDL as the
κ² baseline.  Each cell's result JSON is written to disk the moment the cell
finishes — if the script crashes or you kill it, only the in-progress cell
is lost.  Re-launching the script skips cells whose result JSON already
exists.

Run:
    # Full sweep (~8 GPU-hours total but resumable):
    python benchmark/run_asdl_sweep.py

    # Specific section only:
    python benchmark/run_asdl_sweep.py --section 5.6
    python benchmark/run_asdl_sweep.py --section 5.7
    python benchmark/run_asdl_sweep.py --section 5.8
    python benchmark/run_asdl_sweep.py --section 5.2

    # Re-run a specific cell (delete its JSON first):
    rm benchmark/results/asdl_classic/§5.6_transformer_bf16_seed42.json
    python benchmark/run_asdl_sweep.py --section 5.6 --seed 42

Result layout:
    benchmark/results/asdl_classic/
    ├── §5.2_smallgpt_small_bf16_seed42.json
    ├── §5.6_transformer_fp32_seed42.json
    ├── §5.7_mnist_ae_bf16_seed42.json
    ├── §5.8_laplace_classic_fp32_seed42.json
    └── summary.json              ← rebuilt after every cell completes
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmark" / "results" / "asdl_classic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# §5.6 ASDL ResNet-34 hyperparameters.  ASDL's K-FAC natural gradient is scaled
# differently from our homebrew (homebrew's lr=2e-3 barely trains ASDL), so the
# CNN gets its own screened lr.  damping=1e-4 matches homebrew's §5.6 config —
# low enough that the bf16 κ² collapse (non-PD Gram) actually manifests rather
# than being absorbed by the ridge (see _run_5_6_cell and §5.8 disambiguation).
ASDL_CNN_LR = 3e-2
ASDL_CNN_DAMPING = 1e-4
# factor_update_freq=5 (not homebrew's 20) is the single biggest fp32 lever: the
# stale-preconditioner bottleneck capped ASDL at 41%; freq=5 + momentum=0.9 lifts
# fp32 to ~66% at 1000 steps (still climbing).  See the tuning-decider screen.
ASDL_CNN_FREQ = 5
ASDL_CNN_MOMENTUM = 0.9

# §5.6 transformer (SmallGPT) ASDL hyperparameters.  ASDL's fisher_mc handles the
# (B,T,V) categorical head; same cadence lesson as the CNN (freq=5 ≫ homebrew's
# 20).  Starting point screened below; refined if the perplexity trajectory needs
# it.  damping=1e-4 matches homebrew §5.6 (low enough for the bf16 collapse).
ASDL_TFM_LR = 3e-2
ASDL_TFM_DAMPING = 1e-4
ASDL_TFM_FREQ = 5
ASDL_TFM_MOMENTUM = 0.9


def _asdl_hp(arch):
    """Arch-aware ASDL default (lr, damping, freq, momentum)."""
    if arch == "cnn":
        return ASDL_CNN_LR, ASDL_CNN_DAMPING, ASDL_CNN_FREQ, ASDL_CNN_MOMENTUM
    return ASDL_TFM_LR, ASDL_TFM_DAMPING, ASDL_TFM_FREQ, ASDL_TFM_MOMENTUM


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------

def all_cells():
    """Yield (section, cell_name, runner_fn) tuples for every cell.

    SCOPE (user decision 2026-06-26): ASDL is used as a κ² baseline in two
    places — the §5.8 Laplace factor capture (where the disambiguation probe
    demonstrates the factor-level collapse) and §5.6 ResNet-34 CNN *training*
    (fisher_mc, the one training section where ASDL is well-behaved enough for a
    fair comparison).  The §5.2 SmallGPT, §5.7 MNIST-AE, and §5.6 transformer
    ASDL-training cells are descoped (ASDL's natural-gradient scale differs
    ~1000× from homebrew and would need per-arch retuning); homebrew Classic
    remains the κ² training baseline there.  Their runners are kept but not
    enumerated here.
    """
    SEEDS_3SEED = [42, 43, 44]

    # §5.6 CNN (ResNet-34) × fp32 + bf16 × 3 seeds — fp32 trains, bf16 collapses.
    for precision in ["fp32", "bf16"]:
        for seed in SEEDS_3SEED:
            yield (
                "5.6",
                f"§5.6_cnn_{precision}_seed{seed}",
                lambda p=precision, s=seed: _run_5_6_cell("cnn", p, s),
            )

    # §5.8 Laplace Classic capture × fp32 + bf16 × 3 seeds (already complete).
    for precision in ["fp32", "bf16"]:
        for seed in SEEDS_3SEED:
            yield (
                "5.8",
                f"§5.8_laplace_classic_{precision}_seed{seed}",
                lambda p=precision, s=seed: _run_5_8_cell(p, s),
            )


# ---------------------------------------------------------------------------
# Per-section runners — STUBS.  Each one wires the ASDL adapter into the
# existing training script that section uses.  Most are thin wrappers: the
# core training loops already exist in other benchmark scripts; we just
# import them and pass `method="asdl_classic"`.
#
# These stubs raise NotImplementedError so the script reports clearly which
# section still needs hand-wiring after Phase 2 succeeds.
# ---------------------------------------------------------------------------

def _run_5_2_cell(arch: str, precision: str, seed: int) -> dict:
    """SmallGPT/WikiText-2 main-result cell."""
    # TODO: import and adapt benchmark.kfac_bf16_multiseed.run_cell or
    # whatever your §5.2 runner is.  Replace method="classic" with the
    # ASDL adapter.  Pseudocode:
    #
    # from benchmark.kfac_bf16_multiseed import run_cell
    # from optimizer.asdl_classic_kfac import AsdlClassicKFAC
    #
    # def build_opt(model, **kw):
    #     return AsdlClassicKFAC(model, **kw)
    #
    # result = run_cell(arch=arch, precision=precision, seed=seed,
    #                   optimizer_builder=build_opt)
    # return {"section": "5.2", "arch": arch, ...}
    raise NotImplementedError(
        "§5.2 runner not yet wired — edit benchmark/run_asdl_sweep.py "
        "and connect to your §5.2 training script."
    )


_C4_CTX = {}


def _get_4way_ctx(arch, device):
    """Build (and cache) the per-arch data context from comparison_4way."""
    if arch not in _C4_CTX:
        from benchmark import comparison_4way_multiseed as c4
        _C4_CTX[arch] = c4.get_data(arch, device)
    return _C4_CTX[arch]


# Layers whose Kronecker factor would exceed this dimension are too big for
# K-FAC and go to the secondary optimizer instead — matching homebrew's
# KFAC_MAX_DIM.  Critically, this excludes a transformer's vocab-projection head
# (out ≈ 50257): forming its output Gram B (≈50257²) blows up GPU memory and
# triggers a CUDA illegal-memory-access inside ASDL's cholesky_inv.
KFAC_MAX_DIM = 4096


def _kfac_eligible_split(model, max_dim=None):
    """Split model params into (kfac_eligible, other, ignore_module_names).

    ASDL K-FAC only preconditions nn.Linear / nn.Conv2d, so those params get the
    natural-gradient SGD step; everything else (embeddings, LayerNorm, BatchNorm,
    attention projections) is trained by a secondary AdamW — mirroring our
    homebrew make_optimizers.

    `max_dim` additionally excludes Linear/Conv2d layers whose largest Kronecker
    factor dimension exceeds it (e.g. the LM head): those join the ignore list
    and the secondary AdamW.  `ignore_module_names` (the non-K-FAC modules plus
    the oversized ones) is passed to the adapter so ASDL never tries to form
    their curvature.  For the ResNet-34 CNN (max layer dim 512) nothing is
    excluded by max_dim, so CNN behaviour is unchanged."""
    import torch.nn as nn
    from optimizer.asdl_classic_kfac import _autodetect_unsupported_modules
    ignore_names = set(_autodetect_unsupported_modules(model))
    kfac_ids = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            if isinstance(mod, nn.Linear):
                in_dim, out_dim = mod.in_features, mod.out_features
            else:
                in_dim, out_dim = mod.in_channels, mod.out_channels
            if max_dim is not None and max(in_dim, out_dim) > max_dim:
                ignore_names.add(name)   # too big for K-FAC → secondary AdamW
                continue
            for p in mod.parameters(recurse=False):
                kfac_ids.add(id(p))
    kfac = [p for p in model.parameters() if id(p) in kfac_ids]
    other = [p for p in model.parameters() if id(p) not in kfac_ids]
    return kfac, other, sorted(ignore_names)


def _run_5_6_cell(arch: str, precision: str, seed: int,
                  max_steps: int | None = None,
                  lr: float | None = None,
                  damping: float | None = None,
                  freq: int | None = None,
                  momentum: float | None = None) -> dict:
    """§5.6 ResNet-34 CNN and SmallGPT transformer trained with ASDL reference
    Classic K-FAC as the κ² baseline.

    SCOPE (2026-06-26, final): the **CNN** is the paper's ASDL training row —
    tuned config (freq=5, mom=0.9) reaches fp32 65 ± 3% (up from 41% at homebrew's
    freq=20), bf16 collapses at step 1.  The **transformer** path is wired and
    mechanically correct (the KFAC_MAX_DIM filter below excludes the vocab head,
    whose 50257² output Gram otherwise CUDA-crashes), but is **descoped from the
    paper**: ASDL's K-FAC natural gradient is numerically unstable as a
    from-scratch SmallGPT optimizer in our harness — every screened
    (lr × freq × secondary-lr) config either diverges (train-CE 35–56 nats, far
    past random's 10.8) or stalls at random.  This is an ASDL-specific
    instability orthogonal to the κ² story; homebrew Classic remains the
    transformer/§5.2 κ² baseline.  The §5.7 autoencoder is likewise descoped (its
    Bernoulli/BCE head forces the rank-deficient empirical Fisher).  The
    transformer branch is kept here for reproducibility but not enumerated in
    all_cells().

    Both archs use ASDL's MC Fisher (fisher_type="fisher_mc", loss_type=
    "cross_entropy") — full-rank PD (no cholesky crash) and the correct Fisher
    for a softmax head.  Non-K-FAC params (CNN BatchNorm; transformer
    embeddings/LayerNorm/attention) train under a secondary AdamW, mirroring
    homebrew's make_optimizers.

    lr/damping/freq/momentum override ASDL's hyperparameters; defaults are
    arch-aware (_asdl_hp) because ASDL's natural gradient is scaled differently
    from homebrew and needs its own — notably freq=5 for the preconditioner
    cadence.
    """
    if arch not in ("cnn", "transformer"):
        raise NotImplementedError(
            f"§5.6 ASDL training is wired for cnn/transformer (got arch={arch!r}); "
            "the autoencoder path stays descoped — see docstring."
        )
    import math
    import torch.optim.lr_scheduler as sched_mod
    import torch.nn.functional as F
    from benchmark import comparison_4way_multiseed as c4
    from optimizer.asdl_classic_kfac import AsdlClassicKFAC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _get_4way_ctx(arch, device)
    steps = max_steps if max_steps is not None else c4.MAX_STEPS

    # NOTE: we deliberately do NOT call c4.engage_bf16("classic") here — that
    # only monkeypatches our homebrew ClassicKFAC.step, which ASDL never uses.
    # ASDL's bf16 regime is entirely self-contained in use_bf16_factors below
    # (the adapter round-trips ASDL's stored Kron factors through bf16).

    opt = sec = model = None
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = (c4.make_transformer(ctx["vocab"]) if arch == "transformer"
                 else c4.make_cnn()).to(device)

        kfac_params, other_params, ignore_names = _kfac_eligible_split(
            model, max_dim=KFAC_MAX_DIM)
        d_lr, d_damp, d_freq, d_mom = _asdl_hp(arch)
        use_lr = lr if lr is not None else d_lr
        use_damp = damping if damping is not None else d_damp
        use_freq = freq if freq is not None else d_freq
        use_mom = momentum if momentum is not None else d_mom
        opt = AsdlClassicKFAC(
            model, lr=use_lr, damping=use_damp,
            factor_update_freq=use_freq, momentum=use_mom,
            grad_clip=c4.GRAD_CLIP, gamma=c4.KFAC_GAMMA, weight_decay=0.0,
            use_bf16_factors=(precision == "bf16"),
            fisher_type="fisher_mc", loss_type="cross_entropy",
            base_params=kfac_params, ignore_modules=ignore_names,
        )
        # Secondary AdamW for the non-K-FAC params (matches homebrew emb opt).
        sec = (torch.optim.AdamW(other_params, lr=1e-3, weight_decay=0.0)
               if other_params else None)
        print(f"  ASDL K-FAC params={len(kfac_params)}  "
              f"secondary-AdamW params={len(other_params)}  "
              f"ignore_modules={len(opt.ignore_modules)}", flush=True)

        # LR warmup on the K-FAC base SGD (constant after warmup, as in §5.6).
        scheduler = None
        if steps > c4.WARMUP:
            base = opt._base_sgd
            scheduler = sched_mod.SequentialLR(
                base, schedulers=[
                    sched_mod.LinearLR(base, start_factor=0.1, end_factor=1.0,
                                       total_iters=c4.WARMUP),
                    sched_mod.ConstantLR(base, factor=1.0, total_iters=steps),
                ], milestones=[c4.WARMUP])

        # ASDL loss_fn: must expose `reduction`.  For the transformer, reshape
        # (B,T,V)->(B*T,V) and honour the pad ignore_index inside the closure.
        if arch == "transformer":
            pad = ctx["pad"]
            def asdl_loss_fn(logits, target, reduction="mean"):
                return F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                    ignore_index=pad, reduction=reduction)
        else:
            asdl_loss_fn = F.cross_entropy   # has `reduction`; (B,10) logits

        train_iter = iter(ctx["tlf"]())
        recs = []
        nan_abort = False
        collapsed = False           # bf16 κ² collapse: Gram lost pos-definiteness
        collapse_reason = None
        collapse_step = None
        print(f"  training {arch} with ASDL/{precision} for {steps} steps "
              f"(fisher_mc, lr={use_lr:g}, damping={use_damp:g})...", flush=True)
        t0 = time.perf_counter()
        for step in range(1, steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(ctx["tlf"]())
                batch = next(train_iter)
            x, y = (t.to(device, non_blocking=True) for t in batch)

            model.train()
            if sec is not None:
                sec.zero_grad(set_to_none=True)
            try:
                loss = opt.step(closure=None, inputs=x, targets=y,
                                loss_fn=asdl_loss_fn)
            except RuntimeError as e:
                # The κ² catastrophe: the bf16-quantised Gram is no longer
                # positive-definite, so ASDL's cholesky_inv raises.  This is the
                # collapse itself (the reference-library analogue of our homebrew
                # classic bf16 NaN-abort), not an integration bug — record it as
                # a result rather than letting the whole cell "fail".
                msg = str(e)
                if "positive-definite" in msg or "cholesky" in msg.lower():
                    collapsed = True
                    collapse_reason = "cholesky_not_PD"
                    collapse_step = step
                    print(f"    step={step} kappa^2 COLLAPSE: Gram not positive-"
                          f"definite after bf16 quantisation (cholesky failed)",
                          flush=True)
                    break
                raise
            if sec is not None:
                sec.step()          # raw-grad AdamW on non-K-FAC params
            if scheduler is not None:
                scheduler.step()

            lv = float(loss.detach()) if loss is not None else float("nan")
            if not math.isfinite(lv):
                print(f"    step={step} NaN: kappa^2 collapse (non-finite loss)",
                      flush=True)
                nan_abort = True
                collapsed = True
                collapse_reason = "nan_loss"
                collapse_step = step
                break
            recs.append({"step": step, "loss": lv})
            if step % max(1, steps // 10) == 0 or step <= 5:
                print(f"    step {step}: loss={lv:.3e}", flush=True)

        wall = time.perf_counter() - t0

        eval_metric = {}
        try:
            if arch == "transformer":
                from benchmark.stability_benchmark import evaluate_ppl
                eval_metric = {"final_ppl": evaluate_ppl(
                    model, ctx["vl"], device, ctx["pad"])}
            else:
                from benchmark.models_cnn import evaluate_acc
                eval_metric = {"final_acc": evaluate_acc(model, ctx["vl"], device)}
        except Exception as e:
            print(f"    eval failed: {e}", flush=True)

        out_json = (ROOT / "benchmark" / "results"
                    / f"per_step_4way_{arch}_{precision}_asdl_classic_seed{seed}.json")
        out_json.write_text(json.dumps({
            "arch": arch, "precision": precision, "method": "asdl_classic",
            "seed": seed, "wall_s": wall, "nan_abort": nan_abort,
            "collapsed": collapsed, "collapse_reason": collapse_reason,
            "collapse_step": collapse_step,
            "lr": use_lr, "damping": use_damp, "fisher_type": "fisher_mc",
            "steps_run": len(recs), "per_step": recs, **eval_metric,
        }, indent=2, default=str))

        return {
            "section": "5.6", "arch": arch, "method": "asdl_classic",
            "precision": precision, "seed": seed,
            "nan_abort": nan_abort, "collapsed": collapsed,
            "collapse_reason": collapse_reason, "collapse_step": collapse_step,
            "steps_run": len(recs),
            "summary": {**eval_metric, "collapsed": collapsed,
                        "collapse_step": collapse_step},
            "source_json": out_json.name,
        }
    finally:
        try:
            if opt is not None and hasattr(opt, "cleanup"):
                opt.cleanup()
        except Exception:
            pass
        del opt, sec, model
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_AE_CTX = {}


def _get_mnist_cached(device):
    """Load MNIST once and reuse across §5.7 cells."""
    if "ctx" not in _AE_CTX:
        from benchmark.autoencoder_mnist import get_mnist
        tlf, Xte = get_mnist(device)
        _AE_CTX["ctx"] = {"tlf": tlf, "Xte": Xte}
    return _AE_CTX["ctx"]


def _run_5_7_cell(precision: str, seed: int, max_steps: int | None = None) -> dict:
    """§5.7 Hinton-Salakhutdinov MNIST autoencoder, trained with ASDL reference
    Classic K-FAC as the κ² baseline.

    The AE head is 784 independent Bernoulli pixels (BCE), which ASDL's MC
    Fisher (cross_entropy / mse only) cannot model — so we drive ASDL with the
    EMPIRICAL Fisher (fisher_type="fisher_emp"), whose G = E[δδᵀ] is built from
    the real BCE backward.  That is exactly what our homebrew ClassicKFAC does,
    making ASDL-Classic vs homebrew-Classic apples-to-apples.

    bf16 collapse mechanism: use_bf16_factors=True round-trips ASDL's stored
    Kron factors through bf16 between steps (the adapter's _bf16_roundtrip), so
    the κ² damage lands in the preconditioner the same way the homebrew classic
    bf16 path quantises A/G.
    """
    import torch.optim.lr_scheduler as sched_mod
    from benchmark import autoencoder_mnist as ae
    from optimizer.asdl_classic_kfac import AsdlClassicKFAC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = _get_mnist_cached(device)
    tlf, Xte = ctx["tlf"], ctx["Xte"]

    steps = max_steps if max_steps is not None else ae.MAX_STEPS

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ae.DeepAutoencoder().to(device)

    opt = AsdlClassicKFAC(
        model, lr=ae.KFAC_LR, damping=ae.KFAC_DAMPING_CLASSIC,
        factor_update_freq=ae.KFAC_FREQ, momentum=ae.KFAC_MOMENTUM,
        grad_clip=ae.GRAD_CLIP, gamma=ae.KFAC_GAMMA,
        weight_decay=0.0,
        use_bf16_factors=(precision == "bf16"),
        fisher_type="fisher_emp",   # AE is Bernoulli/BCE — empirical Fisher only
    )

    # Three-phase LR schedule matching autoencoder_mnist.run_one, but built on
    # the base SGD the adapter actually steps (the AsdlClassicKFAC param_groups
    # are separate and inert).  Skip the schedule for short smoke runs.
    scheduler = None
    if steps > ae.WARMUP + ae.CONST_PHASE:
        base = opt._base_sgd
        decay_steps = steps - ae.WARMUP - ae.CONST_PHASE
        scheduler = sched_mod.SequentialLR(
            base, schedulers=[
                sched_mod.LinearLR(base, start_factor=0.1, end_factor=1.0,
                                   total_iters=ae.WARMUP),
                sched_mod.ConstantLR(base, factor=1.0, total_iters=ae.CONST_PHASE),
                sched_mod.CosineAnnealingLR(base, T_max=decay_steps,
                                            eta_min=ae.KFAC_LR * 1e-3),
            ],
            milestones=[ae.WARMUP, ae.WARMUP + ae.CONST_PHASE],
        )

    import math
    import torch.nn.functional as F  # noqa: F401  (kept for parity / debugging)

    train_iter = iter(tlf())
    recs = []
    nan_abort = False
    print(f"  training AE with ASDL/{precision} for {steps} steps "
          f"(fisher_emp)...", flush=True)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            x, target = next(train_iter)
        except StopIteration:
            train_iter = iter(tlf())
            x, target = next(train_iter)

        model.train()
        # ASDL owns the forward/backward; it computes the empirical-Fisher
        # curvature from ae.ae_bce_loss and writes the preconditioned natural
        # gradient into .grad, then the adapter clips + SGD-steps.
        loss = opt.step(closure=None, inputs=x, targets=target,
                        loss_fn=ae.ae_bce_loss)
        if scheduler is not None:
            scheduler.step()

        # ASDL's reported loss is graph-attached; detach before float().
        lv = float(loss.detach()) if loss is not None else float("nan")
        if not math.isfinite(lv):
            print(f"    step={step} NaN — aborting cell", flush=True)
            nan_abort = True
            break
        recs.append({"step": step, "loss": lv})
        if step % max(1, steps // 10) == 0 or step <= 5:
            print(f"    step {step}: loss={lv:.3e}", flush=True)

    wall = time.perf_counter() - t0
    final_recon = None
    try:
        final_recon = ae.evaluate_recon(model, Xte)
    except Exception as e:
        print(f"    eval failed: {e}", flush=True)

    # Persist a per-cell JSON alongside the homebrew AE results so the §5.7
    # aggregation in autoencoder_mnist can pick it up if desired.
    out_json = (ROOT / "benchmark" / "results"
                / f"ae_mnist_{precision}_asdl_classic_seed{seed}.json")
    out_json.write_text(json.dumps({
        "benchmark": "ae_mnist", "precision": precision,
        "method": "asdl_classic", "seed": seed,
        "wall_s": wall, "final_recon_bce": final_recon,
        "nan_abort": nan_abort, "steps_run": len(recs),
        "per_step": recs,
    }, indent=2, default=str))

    return {
        "section": "5.7",
        "method": "asdl_classic",
        "precision": precision,
        "seed": seed,
        "final_recon_bce": final_recon,
        "nan_abort": nan_abort,
        "steps_run": len(recs),
        "summary": {"final_recon_bce": final_recon, "nan_abort": nan_abort},
        "source_json": out_json.name,
    }


_CIFAR_CACHE = {}


def _get_cifar10_cached(device):
    """Load CIFAR-10 once and reuse across §5.8 cells (the splits are
    deterministic, so sharing them is safe)."""
    if "data" not in _CIFAR_CACHE:
        from benchmark.laplace_cifar10 import get_cifar10
        _CIFAR_CACHE["data"] = get_cifar10(device)
    return _CIFAR_CACHE["data"]


def _run_5_8_cell(precision: str, seed: int) -> dict:
    """§5.8 Laplace K-FAC Fisher-capture cell using the ASDL reference
    Classic K-FAC as the κ² baseline.

    The MAP model is trained with AdamW (cached per seed on disk by
    train_or_load_map); ASDL is used only for the Fisher capture, which is
    where the κ² bf16 collapse manifests in the Gram-factor eigendecomposition
    (see laplace_cifar10._capture_fisher_asdl / _evd_from_gram).
    """
    from benchmark import laplace_cifar10 as lap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = _get_cifar10_cached(device)

    # Phase 1: AdamW MAP (loaded from the per-seed checkpoint cache).
    model, _ = lap.train_or_load_map(seed, train, device)

    # Phase 2-4: capture ASDL factors, grid-search λ, evaluate on test.
    # run_cell writes laplace_cifar10_asdl_classic_<precision>_seed<seed>.json.
    lap.run_cell("asdl_classic", precision, seed, model, train, val, test, device)

    src = (ROOT / "benchmark" / "results"
           / f"laplace_cifar10_asdl_classic_{precision}_seed{seed}.json")
    d = json.loads(src.read_text())
    return {
        "section": "5.8",
        "method": "asdl_classic",
        "precision": precision,
        "seed": seed,
        "chosen_lambda": d.get("chosen_lambda"),
        "summary": d.get("metrics", {}),
        "source_json": src.name,
    }


# ---------------------------------------------------------------------------
# Incremental-save driver
# ---------------------------------------------------------------------------

def run_cell_with_save(name: str, runner: Callable[[], dict]) -> dict:
    """Run one cell, save its result immediately, rebuild summary."""
    result_path = RESULTS_DIR / f"{name}.json"

    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text())
            # Only skip on a genuine success.  not_implemented / failed should
            # re-run so the user can iterate without manually deleting JSONs.
            if existing.get("status") == "success":
                print(f"  ✓ SKIP {name} (already done)")
                return existing
        except json.JSONDecodeError:
            print(f"  ⚠ {result_path.name} exists but is corrupt, re-running")

    print(f"\n=== {name} ===")
    t0 = time.perf_counter()
    try:
        result = runner()
        result["status"] = "success"
        result["wall_s"] = time.perf_counter() - t0
        result["name"] = name
    except NotImplementedError as e:
        result = {
            "status": "not_implemented",
            "name": name,
            "error": str(e),
            "wall_s": time.perf_counter() - t0,
        }
    except Exception as e:
        result = {
            "status": "failed",
            "name": name,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "wall_s": time.perf_counter() - t0,
        }
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")

    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  → saved {result_path.name} ({result['status']})")
    return result


def rebuild_summary():
    """Read every cell JSON and write a one-page summary."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("§*.json")):
        try:
            d = json.loads(f.read_text())
            rows.append({
                "name":    d.get("name", f.stem),
                "status":  d.get("status", "unknown"),
                "wall_s":  d.get("wall_s", None),
                "summary": d.get("summary", d.get("metrics", {})),
            })
        except Exception as e:
            rows.append({"name": f.stem, "status": "summary-read-failed",
                          "error": str(e)})

    summary = {
        "n_cells_total":      len(rows),
        "n_success":          sum(r["status"] == "success" for r in rows),
        "n_failed":           sum(r["status"] == "failed"  for r in rows),
        "n_not_implemented":  sum(r["status"] == "not_implemented" for r in rows),
        "cells":              rows,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Windows console defaults to cp1252; the eig-stats line and ✓/✗ glyphs
    # downstream use UTF-8.  Force it so long sweeps never crash on a print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=str, default=None,
                          choices=["5.2", "5.6", "5.7", "5.8"],
                          help="Run only cells from one section.")
    parser.add_argument("--seed", type=int, default=None,
                          help="Run only cells matching this seed.")
    parser.add_argument("--list", action="store_true",
                          help="List cells that would run, don't actually run.")
    args = parser.parse_args()

    cells = list(all_cells())
    if args.section:
        cells = [c for c in cells if c[0] == args.section]
    if args.seed is not None:
        cells = [c for c in cells if f"seed{args.seed}" in c[1]]

    print(f"\n{'=' * 70}")
    print(f"ASDL Classic K-FAC sweep — {len(cells)} cell(s) to consider")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"{'=' * 70}")

    if args.list:
        for section, name, _ in cells:
            print(f"  {section}  {name}")
        return

    overall_t0 = time.perf_counter()
    for section, name, runner in cells:
        run_cell_with_save(name, runner)
        # Rebuild summary after every cell (~milliseconds, worth it)
        rebuild_summary()

    overall = time.perf_counter() - overall_t0
    summary = rebuild_summary()

    print(f"\n{'=' * 70}")
    print(f"Sweep complete  ({overall / 60:.1f} min wall)")
    print(f"  {summary['n_success']}/{summary['n_cells_total']} cells succeeded")
    if summary["n_not_implemented"] > 0:
        print(f"  {summary['n_not_implemented']} cells need section runners wired "
              f"(edit benchmark/run_asdl_sweep.py)")
    if summary["n_failed"] > 0:
        print(f"  {summary['n_failed']} cells failed — see individual JSONs "
              f"for tracebacks")
    print(f"  summary: {RESULTS_DIR / 'summary.json'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
