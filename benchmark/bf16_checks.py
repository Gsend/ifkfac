"""
benchmark/bf16_checks.py  --  dtype verification for the bf16 benchmark runners.

Every bf16 run (benchmark/run_amp_bf16.py, run_pure_bf16.py,
run_amp_singd_tune.py) and every bf16-storage run (rerun_classic_bf16_fixed.py,
rerun_ifkfac_true_bf16.py) uses three layers of checking:

1. configuration, when each optimizer is built: Classic K-FAC and IFKFAC
   must run their K-FAC computation in bf16 (kfac_dtype / pure_bf16, hook
   compute dtype bf16), and SINGD's preconditioner dtype must resolve to
   (bf16, bf16) for every layer - verify_config();
2. tensors, on every step: optimizer/dtype_check.py checks each tensor that
   enters the K-FAC computation and each statistic, factor, inverse, solver
   and natural gradient it produces inside ClassicKFAC / IFKFAC and
   optimizer/bf16_linalg.py; SINGD (a library) is checked by the wrappers of
   install_singd_checks(): its H terms, Kronecker factors and their momenta
   after each update, and its factors and preconditioner dtype before each
   natural gradient;
3. coverage, after each run: run_summary() requires that the check sites of
   every K-FAC method in the run were actually reached (a check that never
   ran proves nothing), and returns the per-site counts that the runners
   write into the result file under "dtype_checks".  The natural-gradient
   sites are required only if the method produced a preconditioner at all
   (a Classic run whose every bf16 Cholesky failed, or an IFKFAC run whose
   every refresh window was rank-deficient, trains with plain SGD and never
   forms a natural gradient; such sites are listed under "not_applicable").

Any failed check raises optimizer.dtype_check.DtypeCheckError at once.
record() writes the counts into a result file; if the coverage check fails,
the file is renamed to *.dtype_failed.json (so a resumed launch repeats the
run) and the error is raised.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import optimizer.dtype_check as DC
from optimizer.dtype_check import DtypeCheckError, expect, expect_all

BF16 = torch.bfloat16

# sites that must have been reached in a run of each method (bf16 computation)
REQUIRED = {
    "classic": ["classic.hook.activations", "classic.hook.output_grads", "classic.factors",
                "classic.damped_factors", "bf16_linalg.cholesky.input", "classic.natgrad_inputs",
                "classic.natgrad_W"],
    "ifkfac": ["ifkfac.hook.activations", "ifkfac.hook.output_grads", "ifkfac.window_rows",
               "bf16_linalg.qr_r_bf16.input", "ifkfac.factors", "ifkfac.solver_blocks",
               "ifkfac.natgrad_input", "ifkfac.natgrad_W"],
    "singd": ["singd.H_terms", "singd.preconditioner", "singd.natgrad_factors"],
    # bf16-storage runs
    "classic_storage": ["classic.storage.inverse_rounded", "classic.storage.inverse_applied"],
    "ifkfac_storage": ["ifkfac.storage.R_stored", "ifkfac.storage.R_applied"],
}
# site -> site that must have been reached for it to be required
REQUIRED_IF = {
    "classic.natgrad_inputs": "classic.inverses",       # at least one bf16 Cholesky succeeded
    "classic.natgrad_W": "classic.inverses",
    # at least one layer's refresh window had full rank (rows >= columns);
    # IFKFAC skips rank-deficient windows and that layer keeps plain SGD
    "bf16_linalg.qr_r_bf16.input": "ifkfac.window_rows_full_rank",
    "ifkfac.factors": "ifkfac.window_rows_full_rank",
    "ifkfac.solver_blocks": "ifkfac.window_rows_full_rank",
    "ifkfac.natgrad_input": "ifkfac.solver_blocks",     # at least one layer has R factors
    "ifkfac.natgrad_W": "ifkfac.solver_blocks",
    "classic.storage.inverse_applied": "classic.storage.inverse_rounded",
}


def _method(o):
    import optimizer.classic_kfac as CK
    import optimizer.ifkfac_kfac as VK
    if isinstance(o, CK.ClassicKFAC):
        return "classic"
    if isinstance(o, VK.IFKFAC):
        return "ifkfac"
    try:
        import singd.optim.optimizer as SO
        if isinstance(o, SO.SINGD):
            return "singd"
    except ImportError:
        pass
    return None


def verify_config(o):
    """Raise unless optimizer `o` computes its K-FAC part in bf16."""
    m = _method(o)
    if m == "classic":
        if o.kfac_dtype != BF16 or o.hooks.compute_dtype != BF16:
            raise DtypeCheckError(f"ClassicKFAC is not in bf16 K-FAC mode "
                                  f"(kfac_dtype={o.kfac_dtype}, hooks={o.hooks.compute_dtype})")
    elif m == "ifkfac":
        if not o.pure_bf16 or o.hooks.compute_dtype != BF16:
            raise DtypeCheckError(f"IFKFAC is not in bf16 K-FAC mode "
                                  f"(pure_bf16={o.pure_bf16}, hooks={o.hooks.compute_dtype})")
    elif m == "singd":
        for module in o.module_names:
            (dk, dc), _ = o._get_preconditioner_dtypes_and_device(module)
            if (dk, dc) != (BF16, BF16):
                raise DtypeCheckError(f"SINGD preconditioner dtype for {o.module_names[module]} "
                                      f"is ({dk}, {dc}), not (bf16, bf16)")
    DC.CHECKS[f"config.{m}"] += 1


def _smat_tensors(M):
    if M is None:
        return []
    if isinstance(M, torch.Tensor):
        return [M]
    return [t for _, t in M.named_tensors()]


def install_singd_checks():
    """Wrap SINGD's H-term accumulation, preconditioner update and natural
    gradient with bf16 checks (idempotent)."""
    import singd.optim.optimizer as SO
    if getattr(SO.SINGD, "_bf16_checks_installed", False):
        return
    acc, upd, nat = SO.SINGD._accumulate_H_terms, SO.SINGD._update_preconditioner, SO.SINGD._compute_natural_gradient

    def _accumulate_H_terms(self, module, inputs, grad_output):
        (dk, dc), _ = self._get_preconditioner_dtypes_and_device(module)
        if (dk, dc) != (BF16, BF16):
            raise DtypeCheckError(f"SINGD casts H-term inputs to ({dk}, {dc}), not bf16")
        out = acc(self, module, inputs, grad_output)
        name = self.module_names[module]
        expect_all("singd.H_terms", _smat_tensors(self.H_Ks.get(name)) + _smat_tensors(self.H_Cs.get(name)))
        return out

    def _update_preconditioner(self, module):
        out = upd(self, module)
        name = self.module_names[module]
        ts = _smat_tensors(self.Ks[name]) + _smat_tensors(self.Cs[name])
        for d in (getattr(self, "m_Ks", {}), getattr(self, "m_Cs", {})):
            ts += _smat_tensors(d.get(name))
        expect_all("singd.preconditioner", ts)
        return out

    def _compute_natural_gradient(self, module):
        name = self.module_names[module]
        dk, dc = self._get_param_group_entry(module, "preconditioner_dtype")
        if (dk or module.weight.dtype, dc or module.weight.dtype) != (BF16, BF16):
            raise DtypeCheckError(f"SINGD natural gradient computed in ({dk}, {dc}), not bf16")
        expect_all("singd.natgrad_factors", _smat_tensors(self.Ks[name]) + _smat_tensors(self.Cs[name]))
        return nat(self, module)

    for f, n in ((_accumulate_H_terms, "_accumulate_H_terms"), (_update_preconditioner, "_update_preconditioner"),
                 (_compute_natural_gradient, "_compute_natural_gradient")):
        f.__wrapped__ = getattr(SO.SINGD, n)
        setattr(SO.SINGD, n, f)
    SO.SINGD._bf16_checks_installed = True


def run_summary(opts=(), storage: bool = False, methods=None):
    """Per-site check counts for one run; raises if a required site of a
    K-FAC method present in `opts` (or named in `methods`, for runners that do
    not record their optimizers) was never reached."""
    s = DC.summary()
    found = {m for m in (_method(o) for o in opts) if m}
    methods = sorted(found | set(methods or ()))
    missing, n_a = [], []
    for m in methods:
        key = f"{m}_storage" if storage and m in ("classic", "ifkfac") else m
        for site in REQUIRED.get(key, []):
            if DC.CHECKS.get(site, 0):
                continue
            pre = REQUIRED_IF.get(site)
            if pre is not None and DC.CHECKS.get(pre, 0) == 0:
                n_a.append(site)
            else:
                missing.append(site)
    if missing:
        raise DtypeCheckError(f"dtype checks never reached for {methods}: {missing}")
    s["methods"] = methods
    s["not_applicable"] = n_a
    s["all_passed"] = True
    return s


def record(path: Path, d: dict, opts=(), storage: bool = False, methods=None) -> dict:
    """Add run_summary() to result dict `d` and write it to `path`.  On a
    coverage failure the result is moved to *.dtype_failed.json and the error
    is raised (the run counts as not done)."""
    try:
        d["dtype_checks"] = run_summary(opts, storage=storage, methods=methods)
    except DtypeCheckError as e:
        d["dtype_checks"] = {"all_passed": False, "error": str(e), **DC.summary()}
        bad = path.with_name(path.stem + ".dtype_failed.json")
        path.write_text(json.dumps(d, indent=2, default=str))
        path.replace(bad)
        raise DtypeCheckError(f"{e}  (result moved to {bad.name}; a resumed launch repeats this run)") from None
    path.write_text(json.dumps(d, indent=2, default=str))
    print("  " + line(d["dtype_checks"]), flush=True)
    return d


def report(paths):
    """Print how many result files carry passed dtype checks (files made
    before the checks existed have none)."""
    paths = sorted(set(paths))
    ok, none, na, failed = [], [], [], []
    for p in paths:
        if p.name.endswith(".dtype_failed.json"):
            failed.append(p.name)
            continue
        try:
            dc = json.loads(p.read_text()).get("dtype_checks")
        except Exception:
            continue
        if not dc:
            none.append(p.name)
        elif dc.get("all_passed"):
            ok.append(p.name)
            if dc.get("not_applicable"):
                na.append(p.name)
        else:
            failed.append(p.name)
    print(f"\n=== dtype checks: {len(ok)} result files verified, {len(none)} without checks "
          f"(made before the checks existed), {len(failed)} failed ===")
    for n in na:
        print(f"  natural gradient never formed (plain SGD for the whole run): {n}")
    for n in failed:
        print(f"  FAILED: {n}")


def line(s):
    na = f"; not applicable: {', '.join(s['not_applicable'])}" if s.get("not_applicable") else ""
    return (f"dtype checks: {s['total']} passed over {len(s['sites'])} sites "
            f"({', '.join(s['methods']) or 'no K-FAC'}){na}")
