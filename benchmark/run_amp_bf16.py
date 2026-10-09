"""
benchmark/run_amp_bf16.py  --  the paper's bf16 experiments in mixed precision,
with every K-FAC calculation in bf16.

Mixed precision (the standard bf16 training setup):
  * fp32 master weights; the forward and backward passes run under
    torch.autocast(bf16) for every method (AdamW, SINGD, Classic, IFKFAC);
    the model output is cast to fp32 and the loss is computed in fp32;
  * the K-FAC calculations are bf16 from the first operation to the natural
    gradient:
      Classic: activations and output gradients cast to bf16 on capture,
               bf16 Gram accumulation, bf16 moving average and damping, bf16
               Cholesky inverse (a failed factorization keeps the layer's last
               good inverse and is counted), bf16 G^-1 g A^-1 with the weight
               gradient cast to bf16 on entry
      IFKFAC:  activations and output gradients cast to bf16 on capture,
               bf16 Householder QR of each window, bf16 ridge [R; sqrt(lam) I],
               bf16 moving-average blend, bf16 blocked triangular solves with
               the weight gradient cast to bf16 on entry
      SINGD:   preconditioner_dtype=(bf16, bf16): bf16 K, C, their momenta and
               the H terms, bf16 natural gradient
  * the natural gradient is then cast to fp32 and applied to the fp32 master
    weights (clip, momentum, decoupled weight decay, update), as SINGD and
    every AMP optimizer do.

benchmark/bf16_guard.py checks the K-FAC part: `--smoke` trains every
(recipe, method) pair through at least one factor refresh with the guard
active inside the K-FAC capture hooks and the K-FAC optimizer step.  Any
operation there that produces a floating tensor that is not bf16 fails the
smoke, except (listed in the report, by function):
  _master_update / _sgd_fallback (Classic, IFKFAC) and _step (SINGD):
      the fp32 master-weight update, and the plain-SGD step a layer takes
      before its first factors exist;
  SINGD _compute_natural_gradient: the concatenation [grad_W, grad_b] of the
      two fp32 gradients (a copy, cast to bf16 right after) and the final
      cast of the bf16 natural gradient to the parameter dtype.
View / alias operations of an fp32 input (detach, reshape, transpose) produce
no data and are counted separately.

Every run (not only the smoke) also verifies dtypes as it goes
(optimizer/dtype_check.py, benchmark/bf16_checks.py): each K-FAC optimizer
must be built in bf16 mode; every tensor entering the K-FAC computation and
every statistic, factor, inverse, solver and natural gradient it produces must
be bf16, checked on every call; a mismatch stops the run at once.  The
per-site check counts are written to each result file ("dtype_checks"), and a
run in which a K-FAC method never reached its check sites is refused.  The forward/backward pass itself is
autocast's business (it runs LayerNorm, softmax and the loss in fp32 by
design) and is not guarded.

Recipes, seeds, schedules and hyperparameters are the original scripts',
imported and called unchanged; this runner only switches the precision
(patches in install()) and the result filenames.  One change for the MNIST
autoencoder, an identity in exact arithmetic: its last sigmoid moves into the
loss (fp32 BCE from logits), because under autocast the sigmoid would run in
bf16, where p > 0.998 rounds to 1.0 and the original 1 - 1e-7 clamp then
zeroes the gradient of those pixels.

Run
    python -m benchmark.run_amp_bf16 --smoke            # guard + timing, every recipe x method
    python -m benchmark.run_amp_bf16                    # §5.2, §5.7, §5.8
    python -m benchmark.run_amp_bf16 --only 5.5         # §5.5 damping curves (Classic, IFKFAC)
    python -m benchmark.run_amp_bf16 --summary
Resumable: finished runs are skipped.

Outputs (benchmark/results/)
    §5.2  per_step_bf16amp_{small|medium}_{classic|ifkfac|singd}_seed{42..46}_s1000.json
    §5.7  per_step_4way_{transformer|cnn}_bf16amp_{adamw|classic|ifkfac|singd}_seed{42..44}.json
    §5.8  ae_mnist_bf16amp_{adamw|classic|ifkfac}_seed{42..44}.json
    §5.5  per_step_bf16amp_{classic|ifkfac}_damp_d{d}_seed{42..44}_s1000.json
    smoke amp_bf16_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

import benchmark.kfac_bf16_compare as KC
import benchmark.kfac_bf16_multiseed as MS
import benchmark.comparison_4way_multiseed as C4
import benchmark.damping_sweep_multiseed as DS
import benchmark.autoencoder_mnist as AE
import benchmark.stability_benchmark as S
from benchmark.bf16_guard import Bf16Guard
from benchmark import bf16_checks as BC
import optimizer.dtype_check as DC
import optimizer.ifkfac_kfac as VK
import optimizer.classic_kfac as CK
import optimizer.hooks as KH
import optimizer.raw_activation_hooks as RH
import singd.optim.optimizer as SO

BF16 = torch.bfloat16
RES = ROOT / "benchmark" / "results"
PREFIX = "bf16amp"
TAG = {"precision": "bf16_amp",
       "note": "mixed precision: fp32 master weights, bf16 autocast forward/backward; "
               "K-FAC statistics, factors, factorizations and natural gradients in bf16 "
               "(Classic/IFKFAC kfac_dtype=bf16, SINGD preconditioner_dtype=(bf16,bf16)); "
               "checked by benchmark/bf16_guard.py (run_amp_bf16.py --smoke)"}
_OPTS: list = []              # K-FAC optimizer instances created during the current run

# the fp32 operations permitted inside the guarded K-FAC regions (see docstring)
ALLOW = [("_master_update", "*"), ("_sgd_fallback", "*"), ("_step", "*"),
         ("_compute_natural_gradient", "cat"), ("_compute_natural_gradient", "_to_copy")]
_GUARD = [None]               # the active region guard (smoke only)


def _noop(*a, **k):
    return None


# ---- model wrapper: bf16 autocast forward, fp32 output ------------------------------------

def _fp32_out(out):
    if isinstance(out, torch.Tensor):
        return out.float() if out.is_floating_point() else out
    if isinstance(out, (tuple, list)):
        return type(out)(_fp32_out(o) for o in out)
    if isinstance(out, dict):
        return {k: _fp32_out(v) for k, v in out.items()}
    return out


def to_amp(model: nn.Module) -> nn.Module:
    """In place: fp32 parameters stay; forward (and so backward) runs under
    torch.autocast(bf16); the output is returned in fp32."""
    fwd = model.forward

    def forward(*a, **k):
        dev = next(model.parameters()).device.type
        with torch.autocast(device_type=dev, dtype=BF16):
            out = fwd(*a, **k)
        return _fp32_out(out)
    model.forward = forward
    model._amp_bf16 = True
    return model


_OrigAE = AE.DeepAutoencoder


class LogitAE(_OrigAE):
    """Same network; the last sigmoid moves into the loss (logit output)."""

    def forward(self, x):
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = torch.sigmoid(x)
        return x


def ae_bce_logits(logits, target, reduction="mean"):
    """fp32 per-pixel BCE from logits; 'mean' = sum over pixels, mean over the batch."""
    per = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    return per.sum(dim=1).mean()


def _eval_recon_logits(model, Xte_bin):
    model.eval()
    with torch.no_grad():
        losses = [ae_bce_logits(model(Xte_bin[i:i + 1024]), Xte_bin[i:i + 1024], reduction="none").sum(dim=1)
                  for i in range(0, Xte_bin.size(0), 1024)]
        return float(torch.cat(losses).mean().item())


# ---- install ------------------------------------------------------------------------

def _guarded(fn):
    def w(*a, **k):
        g = _GUARD[0]
        if g is None:
            return fn(*a, **k)
        with g:
            return fn(*a, **k)
    w.__wrapped__ = fn
    return w


def install():
    """Switch the original recipes to mixed precision with bf16 K-FAC (idempotent)."""
    if getattr(install, "done", False):
        return
    # the storage-only bf16 patches must stay off
    for mod in (KC, MS, DS):
        for name in ("enable_bf16", "disable_bf16", "enable_classic_bf16", "disable_classic_bf16"):
            if hasattr(mod, name):
                setattr(mod, name, _noop)
    for mod in (C4, AE):
        mod.engage_bf16 = _noop
        mod.disengage_bf16 = _noop
        mod.need_autocast = lambda method, precision: False     # the model wrapper does it, for all methods
    # models: fp32 master weights, bf16 autocast forward/backward
    gpt = MS.SmallGPT
    MS.SmallGPT = DS.SmallGPT = lambda *a, **k: to_amp(gpt(*a, **k))
    mt, mc = C4.make_transformer, C4.make_cnn
    C4.make_transformer = lambda *a, **k: to_amp(mt(*a, **k))
    C4.make_cnn = lambda *a, **k: to_amp(mc(*a, **k))
    AE.DeepAutoencoder = lambda: to_amp(LogitAE())
    AE.ae_bce_loss = ae_bce_logits
    AE.evaluate_recon = _eval_recon_logits
    # K-FAC calculations in bf16
    for cls in (VK.IFKFAC, CK.ClassicKFAC):
        orig = cls.__init__

        def init(self, *a, __orig=orig, **k):
            k.setdefault("kfac_dtype", BF16)
            __orig(self, *a, **k)
            BC.verify_config(self)                          # bf16 K-FAC mode, or stop
            _OPTS.append(self)
        cls.__init__ = init
    orig_singd = SO.SINGD.__init__

    def singd_init(self, *a, **k):
        k.setdefault("preconditioner_dtype", (BF16, BF16))
        orig_singd(self, *a, **k)
        BC.verify_config(self)
        _OPTS.append(self)
    SO.SINGD.__init__ = singd_init
    BC.install_singd_checks()                               # SINGD tensors, checked on every call
    SO.Tensor = lambda data: torch.tensor(data, dtype=BF16)      # grad-scale placeholder 1.0 (read with .item())
    # the guard, active inside the K-FAC capture hooks (smoke only; a no-op otherwise)
    for cls in (KH.KFACHooks, RH.RawActivationHooks):
        cls._forward_hook = _guarded(cls._forward_hook)
        cls._backward_hook = _guarded(cls._backward_hook)
    SO.SINGD._accumulate_H_terms = _guarded(SO.SINGD._accumulate_H_terms)
    # result names
    MS.out_path = lambda arch, label, seed: S.OUT / f"per_step_{PREFIX}_{arch}_{label}_seed{seed}_s1000.json"
    C4.out_path = lambda arch, precision, method, seed: RES / f"per_step_4way_{arch}_{PREFIX}_{method}_seed{seed}.json"
    DS.new_path = lambda method, damping, seed: S.OUT / f"per_step_{PREFIX}_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json"
    AE.out_path = lambda precision, method, seed: RES / f"ae_mnist_{PREFIX}_{method}_seed{seed}.json"
    install.done = True


def _is_kfac(o):
    return isinstance(o, (VK.IFKFAC, CK.ClassicKFAC, SO.SINGD))


def _opt_stats():
    out = {}
    for o in _OPTS:
        if isinstance(o, VK.IFKFAC):
            out["ifkfac"] = {"bf16_kfac": o.pure_bf16, **getattr(o, "pure_stats", {})}
        elif isinstance(o, CK.ClassicKFAC):
            out["classic"] = {"kfac_dtype": str(o.kfac_dtype),
                              "factor_dtypes": sorted({d for r in getattr(o, "bf16_chol_log", [])
                                                       for d in r.get("factor_dtypes", [])}),
                              "cholesky_attempts": getattr(o, "bf16_chol_attempts", 0),
                              "cholesky_failures_by_layer": getattr(o, "bf16_chol_failures", {}),
                              "cholesky_failures_per_refresh": [r["failed"] for r in getattr(o, "bf16_chol_log", [])],
                              "inverse_dtypes": sorted({d for r in getattr(o, "bf16_chol_log", [])
                                                        for d in r.get("inverse_dtypes", [])})}
        elif isinstance(o, SO.SINGD):
            out["singd"] = {"preconditioner_dtypes": sorted({str(M.to_dense().dtype)
                                                             for D in (o.Ks, o.Cs) for M in D.values()})}
    return out


def _run(fn, path, *args):
    """Run one original-recipe cell, then tag its result file."""
    already = path.exists()
    _OPTS.clear()
    DC.reset()
    fn(*args)
    if path.exists() and not already:
        d = json.loads(path.read_text())
        d.update(TAG)
        d["optimizer_stats"] = _opt_stats()
        BC.record(path, d, _OPTS)                       # writes the file; raises if checks were missed


# ---- sections --------------------------------------------------------------------

def run_52(device, hw):
    tlf, vl, vocab = S.build_data(device)
    pad = vocab - 1
    cells = [c for c in MS.CELLS if c[0] in ("classic", "ifkfac", "singd")]
    for arch_label, arch_kwargs in MS.ARCHS:
        for label, variant, damping, wgso in cells:
            for seed in MS.SEEDS:
                _run(MS.run_one, MS.out_path(arch_label, label, seed),
                     arch_label, arch_kwargs, label, variant, damping, wgso,
                     seed, tlf, vl, vocab, pad, device, hw)


def run_57(device, hw):
    for arch in C4.ARCHS:
        ctx = C4.get_data(arch, device)
        for method in ("adamw", "classic", "ifkfac", "singd"):
            for seed in C4.SEEDS:
                _run(C4.run_one, C4.out_path(arch, "bf16", method, seed),
                     arch, "bf16", method, seed, ctx, device, hw)


def run_58(device, hw):
    tlf, Xte = AE.get_mnist(device)
    ctx = {"tlf": tlf, "Xte": Xte}
    for method in ("adamw", "classic", "ifkfac"):
        for seed in AE.SEEDS:
            _run(AE.run_one, AE.out_path("bf16", method, seed), "bf16", method, seed, ctx, device, hw)


def run_55(device, hw):
    tlf, vl, vocab = S.build_data(device)
    pad = vocab - 1
    for seed in DS.SEEDS:
        for method in ("classic", "ifkfac"):
            for d in DS.DAMPINGS:
                _run(DS.run_cell, DS.new_path(method, d, seed), method, d, seed, tlf, vl, vocab, pad, device, hw)


# ---- smoke: guard over the K-FAC code + timing, every recipe x method ------------------------

def _kfac_dtypes(opts):
    st = _opt_stats()
    out = {}
    if "classic" in st:
        out["classic_factors"] = st["classic"]["factor_dtypes"]
        out["classic_inverses"] = st["classic"]["inverse_dtypes"]
    if "ifkfac" in st:
        out["ifkfac_factors"] = st["ifkfac"].get("factor_dtypes")
    if "singd" in st:
        out["singd_preconditioners"] = st["singd"]["preconditioner_dtypes"]
    for o in opts:
        h = getattr(o, "hooks", None)
        if h is not None:
            out["hook_compute_dtype"] = str(getattr(h, "compute_dtype", None))
    return out


def _smoke_case(name, model, opts, batches_fn, loss_fn, freq, device):
    rec = {"case": name}
    DC.reset()
    it = batches_fn()

    def next_batch():
        nonlocal it
        try:
            batch = next(it)
        except StopIteration:
            it = batches_fn(); batch = next(it)
        return tuple(t.to(device, non_blocking=True) for t in batch)

    def step(batch, guard=None):
        model.train()
        loss = loss_fn(model, batch)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            if guard is not None and _is_kfac(o):
                with guard:
                    o.step()
            else:
                o.step()
        return loss

    kfac = any(_is_kfac(o) for o in opts)
    g = Bf16Guard(name, allow=ALLOW, ignore_views=True)
    _GUARD[0] = g if kfac else None
    try:
        for _ in range(freq + 2):          # through at least one factor refresh
            loss = step(next_batch(), guard=g if kfac else None)
    finally:
        _GUARD[0] = None
    rec["kfac"] = kfac
    rec["guard_ops"] = g.n_ops
    rec["guard_clean"] = g.clean and (g.n_ops > 0 or not kfac)
    rec["violations"] = {f"{op}->{dt}": {"count": n, "first_at": g.sites[(op, dt)]}
                         for (op, dt), n in g.violations.items()}
    rec["permitted"] = {f"{fn}: {op}->{dt}": n for (fn, op, dt), n in g.allowed.items()}
    rec["views_of_fp32_inputs"] = sum(g.views.values())
    rec["kfac_dtypes"] = _kfac_dtypes(opts)
    rec["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    rec["loss_after_guarded_steps"] = float(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(freq):
        loss = step(next_batch())
    if device.type == "cuda":
        torch.cuda.synchronize()
    rec["seconds_per_step"] = (time.perf_counter() - t0) / freq
    rec["optimizer_stats"] = _opt_stats()
    try:
        rec["dtype_checks"] = BC.run_summary(opts)
        rec["dtype_ok"] = True
    except DC.DtypeCheckError as e:
        rec["dtype_checks"] = {"all_passed": False, "error": str(e)}
        rec["dtype_ok"] = False
    if kfac:
        g.report()
        print("  " + (BC.line(rec["dtype_checks"]) if rec["dtype_ok"] else rec["dtype_checks"]["error"]), flush=True)
    print(f"  {name}: K-FAC bf16 clean={rec['guard_clean'] and rec['dtype_ok']}  dtypes={rec['kfac_dtypes']}  "
          f"{rec['seconds_per_step'] * 1000:.0f} ms/step  loss={rec['loss_after_guarded_steps']:.3f}", flush=True)
    return rec


def _free(opts, device):
    for o in opts:
        if hasattr(o, "cleanup"):
            o.cleanup()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def smoke(device, only=None):
    cases = []
    want = lambda sec: only is None or sec in only
    if want("5.2"):
        tlf, vl, vocab = S.build_data(device)
        ctx_t = {"pad": vocab - 1, "vocab": vocab}
        lm_loss = lambda m, b: C4.compute_loss_transformer(m, b, ctx_t)
        F_ = MS.CFG_BASE["factor_update_freq"]
        for arch_label, arch_kwargs in MS.ARCHS:
            for label, variant, damping, wgso in [c for c in MS.CELLS if c[0] in ("classic", "ifkfac", "singd")]:
                _OPTS.clear(); torch.manual_seed(42)
                model = MS.SmallGPT(vocab_size=vocab, **arch_kwargs).to(device)
                if variant == "SINGD":
                    opts = list(MS._build_singd_optimizers(model, damping))
                else:
                    k, e, _ = MS._build_optimizers(variant, model, MS.CFG_BASE["kfac_lr"], damping,
                                                   MS.CFG_BASE["momentum"], grad_clip=MS.CFG_BASE["grad_clip"],
                                                   gamma=MS.CFG_BASE["gamma"], factor_update_freq=F_)
                    opts = [k, e]
                cases.append(_smoke_case(f"5.2 {arch_label} {label}", model, opts, lambda: iter(tlf()),
                                         lm_loss, F_, device))
                _free(opts, device); del model, opts
    if want("5.7"):
        ctx = C4.get_data("cnn", device)
        for method in ("adamw", "classic", "ifkfac", "singd"):
            _OPTS.clear(); torch.manual_seed(42)
            model = C4.make_cnn().to(device)
            p, s2 = C4.build_optimizers(method, model, "cnn", ctx)
            opts = [p] + ([s2] if s2 is not None else [])
            cases.append(_smoke_case(f"5.7 cnn {method}", model, opts, lambda: iter(ctx["tlf"]()),
                                     lambda m, b: C4.compute_loss_cnn(m, b, ctx), C4.KFAC_FREQ, device))
            _free(opts, device); del model, opts
        ctx = C4.get_data("transformer", device)
        _OPTS.clear(); torch.manual_seed(42)
        model = C4.make_transformer(ctx["vocab"]).to(device)
        p, _ = C4.build_optimizers("adamw", model, "transformer", ctx)
        cases.append(_smoke_case("5.7 transformer adamw", model, [p], lambda: iter(ctx["tlf"]()),
                                 lambda m, b: C4.compute_loss_transformer(m, b, ctx), C4.KFAC_FREQ, device))
        del model
    if want("5.8"):
        tlf_ae, Xte = AE.get_mnist(device)
        for method in ("adamw", "classic", "ifkfac"):
            _OPTS.clear(); torch.manual_seed(42)
            model = AE.DeepAutoencoder().to(device)
            p, _ = AE.build_optimizers(method, model, "bf16")
            cases.append(_smoke_case(f"5.8 ae {method}", model, [p], lambda: iter(tlf_ae()),
                                     lambda m, b: AE.compute_loss(m, b, None), AE.KFAC_FREQ, device))
            _free([p], device); del model
    # training-time estimate
    steps = {"5.2": MS.CFG_BASE["max_steps"], "5.7": C4.MAX_STEPS, "5.8": AE.MAX_STEPS}
    est = 0.0
    for c in cases:
        sec = c["case"].split()[0]
        est += c["seconds_per_step"] * steps[sec] * (len(MS.SEEDS) if sec == "5.2" else 3)
    est += sum(c["seconds_per_step"] for c in cases if c["case"].startswith("5.2 medium")) * C4.MAX_STEPS * 3
    small = {c["case"].split()[-1]: c["seconds_per_step"] for c in cases if c["case"].startswith("5.2 small")}
    est_55 = ((small.get("classic", 0) + small.get("ifkfac", 0)) * DS.CFG["max_steps"] * len(DS.DAMPINGS)
              * len(DS.SEEDS)) if hasattr(DS, "CFG") and "max_steps" in DS.CFG else 0.0
    ok = all(c["guard_clean"] and c.get("dtype_ok", False) for c in cases)
    out = {"all_clean": ok, "cases": cases, "device": str(device), "allow": ALLOW,
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "torch": torch.__version__,
           "estimated_hours_main": est / 3600, "estimated_hours_5_5": est_55 / 3600}
    if only is None:
        (RES / "amp_bf16_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsmoke: K-FAC calculations {'ALL bf16' if ok else 'NOT all bf16'} over {len(cases)} cases; "
          f"estimated training time: §5.2+§5.7+§5.8 {est / 3600:.1f} h, §5.5 {est_55 / 3600:.1f} h "
          f"(results/amp_bf16_smoke.json)", flush=True)
    if not ok:
        sys.exit(1)
    return out


# ---- summary -------------------------------------------------------------------------

def _stats(paths, key, scale=1.0):
    vals = []
    for p in paths:
        if p.exists():
            v = json.loads(p.read_text()).get(key)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v) * scale)
    if not vals:
        return "--"
    s = f"{mean(vals):.4g}" + (f" ± {stdev(vals):.2g}" if len(vals) > 1 else "")
    return s + f" (n={len(vals)})"


def summary():
    R = S.OUT
    print("\n=== §5.2 SmallGPT, final ppl: mixed precision + bf16 K-FAC | bf16 storage only ===")
    for arch, _ in MS.ARCHS:
        for lab, st in (("classic", "bf16fix"), ("ifkfac", "bf16tb"), ("singd", "bf16")):
            f = lambda tag: [R / f"per_step_{tag}_{arch}_{lab}_seed{s}_s1000.json" for s in MS.SEEDS]
            print(f"  {arch:<7} {lab:<8} amp: {_stats(f(PREFIX), 'final_ppl'):<24} "
                  f"storage-only: {_stats(f(st), 'final_ppl')}")
    print("\n=== §5.7 (3 seeds): transformer ppl, CNN accuracy % ===")
    for arch, key, sc in (("transformer", "final_ppl", 1.0), ("cnn", "final_acc", 100.0)):
        for m in ("adamw", "classic", "ifkfac", "singd"):
            f = lambda tag: [RES / f"per_step_4way_{arch}_{tag}_{m}_seed{s}.json" for s in C4.SEEDS]
            print(f"  {arch:<11} {m:<8} amp: {_stats(f(PREFIX), key, sc):<24} fp32: {_stats(f('fp32'), key, sc)}")
    print("\n=== §5.8 autoencoder, test BCE (3 seeds) ===")
    for m in ("adamw", "classic", "ifkfac"):
        f = lambda tag: [RES / f"ae_mnist_{tag}_{m}_seed{s}.json" for s in AE.SEEDS]
        print(f"  {m:<8} amp: {_stats(f(PREFIX), 'final_recon_bce'):<24} fp32: {_stats(f('fp32'), 'final_recon_bce')}")
    rows = [(m, d) for m in ("classic", "ifkfac") for d in DS.DAMPINGS]
    if any((R / f"per_step_{PREFIX}_{m}_damp_d{d:.0e}_seed{s}_s1000.json").exists()
           for m, d in rows for s in DS.SEEDS):
        print("\n=== §5.5 damping, final ppl (3 seeds), mixed precision + bf16 K-FAC ===")
        for m, d in rows:
            f = [R / f"per_step_{PREFIX}_{m}_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS]
            print(f"  {m:<8} lambda={d:.0e}  {_stats(f, 'final_ppl')}")
    fails = []
    for p in sorted(RES.glob(f"*{PREFIX}*classic*.json")):
        st = json.loads(p.read_text()).get("optimizer_stats", {}).get("classic")
        if st:
            fails.append((p.name, sum(st.get("cholesky_failures_per_refresh", [])), st.get("cholesky_attempts", 0)))
    if fails:
        print("\n=== Classic bf16 Cholesky failures (failed / attempted layer inversions) ===")
        for n, f, a in fails:
            print(f"  {n:<60} {f} / {a}")
    BC.report(list(RES.glob(f"*{PREFIX}*.json")) + list(R.glob(f"*{PREFIX}*.json")))


def main():
    for s in (sys.stdout, sys.stderr):          # Windows consoles/pipes are cp1252: never crash on a symbol
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=["5.2", "5.7", "5.8", "5.5"])
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); return
    install()
    from benchmark.gpu_benchmark import get_device, get_hardware_info
    device = get_device(); hw = get_hardware_info()
    print(f"GPU: {hw.get('gpu_name')}", flush=True)
    if args.smoke:
        smoke(device); return
    for sec in ([args.only] if args.only else ["5.2", "5.7", "5.8"]):
        {"5.2": run_52, "5.7": run_57, "5.8": run_58, "5.5": run_55}[sec](device, hw)
    summary()


if __name__ == "__main__":
    main()
