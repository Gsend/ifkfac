"""
benchmark/run_pure_bf16.py  --  the paper's bf16 experiments in pure bf16.

Pure bf16: the model's weights, activations and gradients, every optimizer
state and every stage of the K-FAC pipelines are bf16 tensors.  The dtype is
set once, on the model and the input batches, and flows from stage to stage:

  data loader -> bf16 batch -> bf16 model (bf16 LayerNorm / BatchNorm, bf16 loss)
  -> bf16 activations and output gradients captured by the K-FAC hooks
  -> IFKFAC: bf16 Householder QR of each window, bf16 ridge [R; sqrt(lam) I],
             bf16 moving-average blend, bf16 blocked triangular solves
     Classic: bf16 Gram accumulation, bf16 damping, bf16 Cholesky inverse
             (a failed factorization keeps the layer's last good inverse and
             is counted), bf16 G^-1 g A^-1
     SINGD:  preconditioner in the parameter dtype (bf16)
     AdamW:  optimizer/bf16_adamw.py (bf16 moments, integer step count)
  -> bf16 momentum, bf16 weight update.

No stage upcasts.  benchmark/bf16_guard.py enforces it: `--smoke` trains
every (recipe, method) pair through at least one factor refresh under the
guard and fails if any operation produced a floating tensor that is not bf16.
Every run also checks dtypes as it goes (optimizer/dtype_check.py,
benchmark/bf16_checks.py): each K-FAC optimizer must be built in bf16 mode,
and every tensor entering the K-FAC computation and every statistic, factor,
inverse, solver and natural gradient it produces must be bf16, on every call;
a mismatch stops the run.  Per-site counts go into each result file
("dtype_checks").
Inside a single GPU kernel, bf16 matmuls and reductions accumulate in fp32 and
round the result to bf16 (as every bf16 kernel does); no fp32 tensor exists.
Metrics (validation perplexity, accuracy, test BCE) are computed from the
bf16 model outputs and averaged in Python.

Recipes, seeds, schedules and hyperparameters are the original scripts', which
are imported and called unchanged; this runner only swaps the precision
(patches listed in install()) and the result filenames.  Two numerically
motivated changes, both identities in exact arithmetic:
  * the MNIST autoencoder applies its last sigmoid inside the loss (BCE from
    logits, softplus form): the original clamps probabilities to 1 - 1e-7,
    which is exactly 1.0 in bf16 and gives log(0);
  * SINGD 0.0.5 creates Tensor([1.0]) (fp32) as its gradient-scale placeholder
    on every step; it is created as bf16 instead (same value, 1.0).

Run
    python -m benchmark.run_pure_bf16 --smoke            # guard + timing, every recipe x method (~20-30 min)
    python -m benchmark.run_pure_bf16                    # §5.2, §5.7, §5.8
    python -m benchmark.run_pure_bf16 --only 5.5         # §5.5 damping curves (Classic, IFKFAC)
    python -m benchmark.run_pure_bf16 --summary
Resumable: finished runs are skipped.

Outputs (benchmark/results/)
    §5.2  per_step_bf16pure_{small|medium}_{classic|ifkfac|singd}_seed{42..46}_s1000.json
    §5.7  per_step_4way_{transformer|cnn}_bf16pure_{adamw|classic|ifkfac|singd}_seed{42..44}.json
    §5.8  ae_mnist_bf16pure_{adamw|classic|ifkfac}_seed{42..44}.json
    §5.5  per_step_bf16pure_{classic|ifkfac}_damp_d{d}_seed{42..44}_s1000.json
    smoke pure_bf16_smoke.json
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

import benchmark.kfac_bf16_compare as KC
import benchmark.kfac_bf16_multiseed as MS
import benchmark.comparison_4way_multiseed as C4
import benchmark.damping_sweep_multiseed as DS
import benchmark.autoencoder_mnist as AE
import benchmark.stability_benchmark as S
from benchmark import pure_bf16 as P
from benchmark.bf16_guard import Bf16Guard
from benchmark import bf16_checks as BC
import optimizer.dtype_check as DC
from optimizer.bf16_adamw import BF16AdamW
import optimizer.ifkfac_kfac as VK
import optimizer.classic_kfac as CK

BF16 = torch.bfloat16
RES = ROOT / "benchmark" / "results"
TAG = {"precision": "bf16_pure",
       "note": "pure bf16: bf16 model, activations, gradients, optimizer state and K-FAC pipeline; "
               "checked by benchmark/bf16_guard.py (run_pure_bf16.py --smoke)"}
_OPTS: list = []              # optimizer instances created during the current run


def _noop(*a, **k):
    return None


# ---- models, data, losses, evaluation ------------------------------------------------

_OrigAE = AE.DeepAutoencoder


class PureAE(_OrigAE):
    """Same network; the last sigmoid moves into the loss (logit output)."""

    def forward(self, x):
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = torch.sigmoid(x)
        return x


def _bf16_batches(it):
    for batch in it:
        yield tuple(t.to(BF16) if t.is_floating_point() else t for t in batch)


class _Bf16Loader:
    """Wraps a DataLoader (or factory output) so floating tensors arrive as bf16."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return _bf16_batches(iter(self.loader))

    def __len__(self):
        return len(self.loader)


def _get_mnist_pure(device):
    from torchvision import datasets, transforms
    root = ROOT / "data" / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(root), train=True, download=True, transform=tf)
    test = datasets.MNIST(str(root), train=False, download=True, transform=tf)
    Xte_bin = torch.stack([(test[i][0].view(-1) > 0.5) for i in range(len(test))]).to(BF16).to(device)

    def train_loader_factory():
        loader = torch.utils.data.DataLoader(train, batch_size=AE.BATCH_SIZE, shuffle=True,
                                             drop_last=True, num_workers=0,
                                             pin_memory=(device.type == "cuda"))
        for x, _ in loader:
            xb = (x.view(x.size(0), -1) > 0.5).to(BF16).to(device, non_blocking=True)
            yield xb, xb
    return train_loader_factory, Xte_bin


def _eval_recon_pure(model, Xte_bin):
    """Mean per-image BCE (sum over pixels) from the bf16 logits; averaged in Python."""
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, Xte_bin.size(0), 1024):
            x = Xte_bin[i:i + 1024]
            per = P.bce_with_logits_bf16(model(x), x, reduction="none")        # bf16 (B, 784)
            for row in per.tolist():
                tot += math.fsum(row); n += 1
    return tot / max(n, 1)


def _eval_ppl_pure(model, loader, device, pad_id, max_batches=None):
    """stability_benchmark.evaluate_ppl, with the per-token bf16 losses averaged in Python."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            yf = y.reshape(-1)
            nll = P.cross_entropy_bf16(logits.reshape(-1, logits.size(-1)), yf,
                                       ignore_index=pad_id, reduction="none")
            vals = nll[yf != pad_id].tolist()
            batch_mean = math.fsum(vals) / max(len(vals), 1)
            total_loss += batch_mean * y.numel()
            total_tokens += y.numel()
    if total_tokens == 0:
        return 9999.0
    return min(float(math.exp(total_loss / total_tokens)), 9999.0)


def install():
    """Switch the original recipes to pure bf16 (idempotent)."""
    if getattr(install, "done", False):
        return
    # optimizers, losses, autocast
    torch.optim.AdamW = BF16AdamW
    P.install_bf16_losses()
    torch.autocast = P.NoAutocast
    import singd.optim.optimizer as SO
    SO.Tensor = lambda data: torch.tensor(data, dtype=BF16)   # gradient-scale placeholder
    # the storage-only bf16 patches must stay off
    for mod in (KC, MS, DS):
        for name in ("enable_bf16", "disable_bf16", "enable_classic_bf16", "disable_classic_bf16"):
            if hasattr(mod, name):
                setattr(mod, name, _noop)
    for mod in (C4, AE):
        mod.engage_bf16 = _noop
        mod.disengage_bf16 = _noop
        mod.need_autocast = lambda method, precision: False
    # models: bf16 parameters, bf16 norms, bf16 inputs
    gpt = MS.SmallGPT
    MS.SmallGPT = DS.SmallGPT = lambda *a, **k: P.to_pure_bf16(gpt(*a, **k))
    mt, mc = C4.make_transformer, C4.make_cnn
    C4.make_transformer = lambda *a, **k: P.to_pure_bf16(mt(*a, **k))
    C4.make_cnn = lambda *a, **k: P.to_pure_bf16(mc(*a, **k))
    AE.DeepAutoencoder = lambda: P.to_pure_bf16(PureAE())
    # data
    gd = C4.get_data

    def get_data(arch, device):
        ctx = gd(arch, device)
        if arch == "cnn":
            tlf = ctx["tlf"]
            ctx["tlf"] = lambda: _Bf16Loader(tlf())
            ctx["vl"] = _Bf16Loader(ctx["vl"])
        return ctx
    C4.get_data = get_data
    AE.get_mnist = _get_mnist_pure
    # losses and evaluation
    AE.ae_bce_loss = P.bce_with_logits_bf16
    AE.evaluate_recon = _eval_recon_pure
    S.evaluate_ppl = MS.evaluate_ppl = DS.evaluate_ppl = _eval_ppl_pure
    # record optimizer instances (pure-mode check + Classic failure counts) and
    # verify that each K-FAC optimizer is in bf16 mode
    for cls in (VK.IFKFAC, CK.ClassicKFAC, SO.SINGD):
        orig = cls.__init__

        def rec(self, *a, __orig=orig, **k):
            __orig(self, *a, **k)
            BC.verify_config(self)
            _OPTS.append(self)
        cls.__init__ = rec
    BC.install_singd_checks()
    # result names
    MS.out_path = lambda arch, label, seed: S.OUT / f"per_step_bf16pure_{arch}_{label}_seed{seed}_s1000.json"
    C4.out_path = lambda arch, precision, method, seed: RES / f"per_step_4way_{arch}_bf16pure_{method}_seed{seed}.json"
    DS.new_path = lambda method, damping, seed: S.OUT / f"per_step_bf16pure_{method}_damp_d{damping:.0e}_seed{seed}_s1000.json"
    AE.out_path = lambda precision, method, seed: RES / f"ae_mnist_bf16pure_{method}_seed{seed}.json"
    install.done = True


def _opt_stats():
    out = {}
    for o in _OPTS:
        if isinstance(o, VK.IFKFAC):
            out["ifkfac"] = {"pure_bf16": o.pure_bf16, **o.pure_stats}
        elif isinstance(o, CK.ClassicKFAC):
            out["classic"] = {"cholesky_attempts": getattr(o, "bf16_chol_attempts", 0),
                              "cholesky_failures_by_layer": getattr(o, "bf16_chol_failures", {}),
                              "cholesky_failures_per_refresh": [r["failed"] for r in getattr(o, "bf16_chol_log", [])],
                              "inverse_dtypes": sorted({d for r in getattr(o, "bf16_chol_log", [])
                                                        for d in r.get("inverse_dtypes", [])})}
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


# ---- smoke: guard + timing for every recipe x method ------------------------------------

def _smoke_case(name, model, opts, batches_fn, loss_fn, freq, device):
    """Guarded steps through one refresh, then an unguarded window for timing."""
    rec = {"case": name}
    DC.reset()
    it = batches_fn()

    def next_batch():
        # data decoding / collation happens here, outside the guard; the batch
        # handed to the model is already bf16 (integer token ids stay integer)
        nonlocal it
        try:
            batch = next(it)
        except StopIteration:
            it = batches_fn(); batch = next(it)
        return tuple(t.to(device, non_blocking=True) for t in batch)

    def step(batch):
        model.train()
        loss = loss_fn(model, batch)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        return loss

    n_guard = freq + 2
    g = Bf16Guard(name)
    for _ in range(n_guard):
        batch = next_batch()
        assert all(t.dtype in (torch.bfloat16, torch.int64, torch.int32, torch.bool) for t in batch), \
            f"{name}: batch dtypes {[t.dtype for t in batch]}"
        with g:
            loss = step(batch)
    rec["guard_ops"] = g.n_ops
    rec["guard_clean"] = g.clean
    rec["violations"] = {f"{op}->{dt}": {"count": n, "first_at": g.sites[(op, dt)]}
                         for (op, dt), n in g.violations.items()}
    rec["loss_after_guarded_steps"] = float(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(freq):
        loss = step(next_batch())
    if device.type == "cuda":
        torch.cuda.synchronize()
    rec["seconds_per_step"] = (time.perf_counter() - t0) / freq
    rec["params_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    rec["optimizer_stats"] = _opt_stats()
    try:
        rec["dtype_checks"] = BC.run_summary(opts)
        rec["dtype_ok"] = True
    except DC.DtypeCheckError as e:
        rec["dtype_checks"] = {"all_passed": False, "error": str(e)}
        rec["dtype_ok"] = False
    g.report()
    print("  " + (BC.line(rec["dtype_checks"]) if rec["dtype_ok"] else rec["dtype_checks"]["error"]), flush=True)
    print(f"  {name}: clean={g.clean and rec['dtype_ok']}  {rec['seconds_per_step'] * 1000:.0f} ms/step (incl. one refresh "
          f"per {freq} steps)  loss={rec['loss_after_guarded_steps']:.3f}", flush=True)
    return rec


def smoke(device):
    from benchmark.gpu_benchmark import SmallGPT  # noqa: F401 (import check)
    cases = []
    tlf, vl, vocab = S.build_data(device)
    pad = vocab - 1
    ctx_t = {"pad": pad, "vocab": vocab}
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
            for o in opts:
                if hasattr(o, "cleanup"):
                    o.cleanup()
            del model, opts
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for arch in ("cnn",):   # the §5.7 transformer is the §5.2 medium recipe; AdamW covered below
        ctx = C4.get_data(arch, device)
        for method in ("adamw", "classic", "ifkfac", "singd"):
            _OPTS.clear(); torch.manual_seed(42)
            model = (C4.make_cnn() if arch == "cnn" else C4.make_transformer(ctx["vocab"])).to(device)
            p, s2 = C4.build_optimizers(method, model, arch, ctx)
            opts = [p] + ([s2] if s2 is not None else [])
            lf = (lambda m, b: C4.compute_loss_cnn(m, b, ctx))
            cases.append(_smoke_case(f"5.7 {arch} {method}", model, opts, lambda: iter(ctx["tlf"]()),
                                     lf, C4.KFAC_FREQ, device))
            for o in opts:
                if hasattr(o, "cleanup"):
                    o.cleanup()
            del model, opts
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ctx = C4.get_data("transformer", device)
    _OPTS.clear(); torch.manual_seed(42)
    model = C4.make_transformer(ctx["vocab"]).to(device)
    p, _ = C4.build_optimizers("adamw", model, "transformer", ctx)
    cases.append(_smoke_case("5.7 transformer adamw", model, [p], lambda: iter(ctx["tlf"]()),
                             lambda m, b: C4.compute_loss_transformer(m, b, ctx), C4.KFAC_FREQ, device))
    del model
    tlf_ae, Xte = AE.get_mnist(device)
    for method in ("adamw", "classic", "ifkfac"):
        _OPTS.clear(); torch.manual_seed(42)
        model = AE.DeepAutoencoder().to(device)
        p, _ = AE.build_optimizers(method, model, "bf16")
        cases.append(_smoke_case(f"5.8 ae {method}", model, [p], lambda: iter(tlf_ae()),
                                 lambda m, b: AE.compute_loss(m, b, None), AE.KFAC_FREQ, device))
        if hasattr(p, "cleanup"):
            p.cleanup()
        del model
    # timing estimate for the full program
    steps = {"5.2": MS.CFG_BASE["max_steps"], "5.7": C4.MAX_STEPS, "5.8": AE.MAX_STEPS}
    est = 0.0
    for c in cases:
        sec = c["case"].split()[0]
        n_runs = len(MS.SEEDS) if sec == "5.2" else 3
        est += c["seconds_per_step"] * steps[sec] * n_runs
    tr_med = [c for c in cases if c["case"].startswith("5.2 medium")]
    est += sum(c["seconds_per_step"] for c in tr_med) * C4.MAX_STEPS * 3      # §5.7 transformer K-FAC runs
    small = {c["case"].split()[-1]: c["seconds_per_step"] for c in cases if c["case"].startswith("5.2 small")}
    est_55 = (small.get("classic", 0) + small.get("ifkfac", 0)) * DS.CFG["max_steps"] * len(DS.DAMPINGS) * len(DS.SEEDS) \
        if hasattr(DS, "CFG") and "max_steps" in DS.CFG else 0.0
    ok = all(c["guard_clean"] and c.get("dtype_ok", False) for c in cases)
    out = {"all_clean": ok, "cases": cases, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "estimated_hours_main": est / 3600, "estimated_hours_5_5": est_55 / 3600}
    (RES / "pure_bf16_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsmoke: {'ALL CLEAN' if ok else 'NOT CLEAN'} over {len(cases)} cases; "
          f"estimated training time: §5.2+§5.7+§5.8 {est / 3600:.1f} h, §5.5 {est_55 / 3600:.1f} h "
          f"(results/pure_bf16_smoke.json)", flush=True)
    if not ok:
        sys.exit(1)


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
    print("\n=== §5.2 SmallGPT, final ppl: pure bf16 | bf16 storage only | ===")
    for arch, _ in MS.ARCHS:
        for lab, st in (("classic", "bf16fix"), ("ifkfac", "bf16tb"), ("singd", "bf16")):
            f = lambda tag: [R / f"per_step_{tag}_{arch}_{lab}_seed{s}_s1000.json" for s in MS.SEEDS]
            print(f"  {arch:<7} {lab:<8} pure: {_stats(f('bf16pure'), 'final_ppl'):<24} "
                  f"storage-only: {_stats(f(st), 'final_ppl')}")
    print("\n=== §5.7 (3 seeds): transformer ppl, CNN accuracy % ===")
    for arch, key, sc in (("transformer", "final_ppl", 1.0), ("cnn", "final_acc", 100.0)):
        for m in ("adamw", "classic", "ifkfac", "singd"):
            f = lambda tag: [RES / f"per_step_4way_{arch}_{tag}_{m}_seed{s}.json" for s in C4.SEEDS]
            print(f"  {arch:<11} {m:<8} pure: {_stats(f('bf16pure'), key, sc):<24} fp32: {_stats(f('fp32'), key, sc)}")
    print("\n=== §5.8 autoencoder, test BCE (3 seeds) ===")
    for m in ("adamw", "classic", "ifkfac"):
        f = lambda tag: [RES / f"ae_mnist_{tag}_{m}_seed{s}.json" for s in AE.SEEDS]
        print(f"  {m:<8} pure: {_stats(f('bf16pure'), 'final_recon_bce'):<24} fp32: {_stats(f('fp32'), 'final_recon_bce')}")
    rows = [(m, d) for m in ("classic", "ifkfac") for d in DS.DAMPINGS]
    if any((R / f"per_step_bf16pure_{m}_damp_d{d:.0e}_seed{s}_s1000.json").exists()
           for m, d in rows for s in DS.SEEDS):
        print("\n=== §5.5 damping, final ppl (3 seeds), pure bf16 ===")
        for m, d in rows:
            f = [R / f"per_step_bf16pure_{m}_damp_d{d:.0e}_seed{s}_s1000.json" for s in DS.SEEDS]
            print(f"  {m:<8} lambda={d:.0e}  {_stats(f, 'final_ppl')}")
    fails = []
    for p in sorted(RES.glob("*bf16pure*classic*.json")):
        st = json.loads(p.read_text()).get("optimizer_stats", {}).get("classic")
        if st:
            fails.append((p.name, sum(st.get("cholesky_failures_per_refresh", [])), st.get("cholesky_attempts", 0)))
    if fails:
        print("\n=== Classic bf16 Cholesky failures (failed / attempted layer inversions) ===")
        for n, f, a in fails:
            print(f"  {n:<60} {f} / {a}")
    BC.report(list(RES.glob("*bf16pure*.json")) + list(S.OUT.glob("*bf16pure*.json")))


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
