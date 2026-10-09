"""
benchmark/run_overnight.py  --  every pending paper run, in one command.

Runs each stage as a separate Python process (clean GPU memory between
stages), logs everything to benchmark/results/overnight_<timestamp>.log, keeps
going if a stage fails, and prints the summaries at the end.  Every stage is
resumable (finished runs are skipped), so if the night ends before the list
does, just run the same command again the next night.

Stages, in priority order (times for the RTX 3080 Laptop; finished runs are skipped):
  1 amp_smoke        mixed precision (fp32 master weights, bf16 autocast) with every K-FAC
                     calculation in bf16: every recipe x method through a factor refresh
                     with the bf16 guard inside the K-FAC hooks and optimizer step, + timing
                     (~15 min); skipped if it already passed and no covered source changed
  2 amp_52           §5.2 (Classic, IFKFAC, SINGD; small + medium)                    ~4 h
  3 amp_57           §5.7 (transformer + CNN; AdamW, Classic, IFKFAC, SINGD)           ~6.5 h
  4 amp_55           §5.5 (Classic, IFKFAC damping curves)                             ~4 h
  5 amp_58           §5.8 (autoencoder; AdamW, Classic, IFKFAC)                        ~14 h
     (the pure-bf16 runner, benchmark/run_pure_bf16.py, is no longer part of the list;
      its finished results stay in results/*bf16pure*)
  6 classic_smoke / classic_main     Classic bf16-storage reruns, fixed patch (autoencoder left)
  7 fp32_damping     §5.5  fp32 damping curves (3 runs left)
  8 classic_damping  §5.5  Classic bf16-storage damping curve
  9 tb_smoke / tb_main / tb_damping  IFKFAC bf16-storage reruns (§5.2/§5.7 done) + damping curve
 10 side_lr / side_lr_confirm        §5.7 K-FAC side-AdamW lr test
 11 plot_damping     §5.5  redraw Figure 2
 12 kappa_sweep      §4.5  synthetic kappa sweep (done; ~1 min)
 13 eig_audit        §5.9  factor-level eigenvalue audit (ASDL part still missing)    ~5 min
A stage whose smoke check failed is skipped.  Every bf16 run checks on every
step that the K-FAC computation is fed and produces bf16 tensors (bf16-valued
factors for the storage-only reruns) and stops on a mismatch; the check counts
are in each result file ("dtype_checks") and in the summaries.

Run:
    python -m benchmark.run_overnight                 # all stages, no time limit
    python -m benchmark.run_overnight --hours 10      # stop starting/continuing work after 10 h
    python -m benchmark.run_overnight --only fp32_damping side_lr
    python -m benchmark.run_overnight --list          # show stages and exit
With --hours, a stage still running at the deadline is stopped; its finished
runs are kept and the next launch resumes from the first unfinished run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"

STAGES = [
    ("amp_smoke",       ["-m", "benchmark.run_amp_bf16", "--smoke"]),
    ("amp_52",          ["-m", "benchmark.run_amp_bf16", "--only", "5.2"]),
    ("amp_57",          ["-m", "benchmark.run_amp_bf16", "--only", "5.7"]),
    ("amp_55",          ["-m", "benchmark.run_amp_bf16", "--only", "5.5"]),
    ("amp_58",          ["-m", "benchmark.run_amp_bf16", "--only", "5.8"]),
    ("classic_smoke",   ["-m", "benchmark.rerun_classic_bf16_fixed", "--smoke"]),
    ("classic_main",    ["-m", "benchmark.rerun_classic_bf16_fixed"]),
    ("fp32_damping",    ["-m", "benchmark.damping_sweep_fp32"]),
    ("classic_damping", ["-m", "benchmark.rerun_classic_bf16_fixed", "--only", "5.5"]),
    ("tb_smoke",        ["-m", "benchmark.rerun_ifkfac_true_bf16", "--smoke"]),
    ("tb_main",         ["-m", "benchmark.rerun_ifkfac_true_bf16"]),
    ("tb_damping",      ["-m", "benchmark.rerun_ifkfac_true_bf16", "--only", "5.5"]),
    ("side_lr",         ["-m", "benchmark.kfac_side_lr_sweep"]),
    ("side_lr_confirm", ["-m", "benchmark.kfac_side_lr_sweep", "--confirm"]),
    ("plot_damping",    ["benchmark/plot_damping_sweep.py"]),
    ("kappa_sweep",     ["tests/test_kappa_scaling.py"]),
    ("eig_audit",       ["-m", "benchmark.laplace_eig_audit"]),
]
NEEDS_SMOKE = {"classic_main": "classic_smoke", "classic_damping": "classic_smoke",
               "tb_main": "tb_smoke", "tb_damping": "tb_smoke",
               "amp_52": "amp_smoke", "amp_57": "amp_smoke",
               "amp_55": "amp_smoke", "amp_58": "amp_smoke"}
# Source files the bf16 check covers: if none changed since the last passing
# check (results/amp_bf16_smoke.json), the check is not repeated.
AMP_SMOKE_SOURCES = ["optimizer/*.py", "benchmark/run_amp_bf16.py",
                      "benchmark/bf16_guard.py", "benchmark/bf16_checks.py", "benchmark/stability_benchmark.py",
                      "benchmark/kfac_bf16_multiseed.py", "benchmark/comparison_4way_multiseed.py",
                      "benchmark/damping_sweep_multiseed.py", "benchmark/autoencoder_mnist.py",
                      "benchmark/gpu_benchmark.py", "benchmark/models_cnn.py"]


def amp_smoke_still_valid():
    """True if the last bf16 K-FAC check passed and no covered source changed since."""
    import json
    j = RESULTS / "amp_bf16_smoke.json"
    if not j.exists():
        return False
    try:
        if json.loads(j.read_text()).get("all_clean") is not True:
            return False
    except Exception:
        return False
    srcs = [p for pat in AMP_SMOKE_SOURCES for p in ROOT.glob(pat)]
    return bool(srcs) and j.stat().st_mtime > max(p.stat().st_mtime for p in srcs)

SUMMARIES = [
    ("mixed precision, bf16 K-FAC", ["-m", "benchmark.run_amp_bf16", "--summary"]),
    ("Classic bf16, fixed patch", ["-m", "benchmark.rerun_classic_bf16_fixed", "--summary"]),
    ("IFKFAC R-in-bf16 reruns", ["-m", "benchmark.rerun_ifkfac_true_bf16", "--summary"]),
    ("fp32 vs bf16 damping",    ["-m", "benchmark.damping_sweep_fp32", "--summary"]),
    ("side-AdamW lr",           ["-m", "benchmark.kfac_side_lr_sweep", "--summary"]),
]


REQUIRED_MODULES = ["torch", "torchvision", "scipy", "matplotlib", "datasets", "transformers", "singd", "asdl"]


def preflight():
    """Check that this Python has a CUDA torch and every package the stages import.
    Returns a list of problems (empty = ok)."""
    import importlib.util
    problems = [f"module '{m}' not installed" for m in REQUIRED_MODULES
                if importlib.util.find_spec(m) is None]
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append(f"torch {torch.__version__} has no CUDA GPU (CPU-only build or no driver)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"torch import failed: {e}")
    return problems


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()

    def raw(self, text):
        sys.stdout.write(text); sys.stdout.flush()
        self.f.write(text); self.f.flush()


def run_stage(args, log, deadline):
    """Run `python <args>` streaming output to console + log.  Returns exit code,
    or None if it was stopped at the deadline."""
    cmd = [sys.executable, "-u"] + args
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})   # child writes utf-8 (not cp1252)
    try:
        for line in proc.stdout:
            log.raw(line)
            if deadline and time.time() > deadline:
                log("deadline reached: stopping this stage (finished runs are kept)")
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return None
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise


def main():
    for s in (sys.stdout, sys.stderr):      # never crash on a symbol the console cannot show
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget for this launch")
    ap.add_argument("--only", nargs="+", choices=[s for s, _ in STAGES])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, a in STAGES:
            print(f"  {name:<16} python {' '.join(a)}")
        return

    problems = preflight()
    if problems:
        print(f"Pre-flight check failed for {sys.executable}:")
        for p in problems:
            print(f"  - {p}")
        venv = ROOT / ".venv" / "Scripts" / "python.exe"
        print("\nRun it with this repo's environment, e.g.:")
        print(f"  deactivate; {ROOT / '.venv' / 'Scripts' / 'Activate.ps1'}")
        print(f"  python -m benchmark.run_overnight {' '.join(sys.argv[1:])}".rstrip())
        print(f"or directly:  {venv} -m benchmark.run_overnight {' '.join(sys.argv[1:])}".rstrip())
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    log = Log(RESULTS / f"overnight_{dt.datetime.now():%Y%m%d_%H%M}.log")
    log(f"python: {sys.executable}")
    deadline = time.time() + args.hours * 3600 if args.hours else None
    stages = [(n, a) for n, a in STAGES if not args.only or n in args.only]
    log(f"starting {len(stages)} stage(s)" + (f", budget {args.hours:g} h" if args.hours else ""))

    status, smoke_ok = {}, {}
    for name, a in stages:
        if deadline and time.time() > deadline:
            status[name] = "not started (time budget used)"
            continue
        if name == "amp_smoke" and amp_smoke_still_valid():
            smoke_ok[name] = True
            status[name] = "skipped (already passed; no covered source changed)"
            log(f"=== {name}: {status[name]} ===")
            continue
        if smoke_ok.get(NEEDS_SMOKE.get(name)) is False:
            status[name] = "skipped (smoke check failed)"
            log(f"=== {name}: skipped because {NEEDS_SMOKE[name]} failed ===")
            continue
        log(f"=== {name}: python {' '.join(a)} ===")
        t0 = time.time()
        rc = run_stage(a, log, deadline)
        mins = (time.time() - t0) / 60
        if name in NEEDS_SMOKE.values():
            smoke_ok[name] = rc == 0
        status[name] = ("stopped at deadline" if rc is None else
                        "ok" if rc == 0 else f"FAILED (exit {rc})") + f", {mins:.0f} min"
        log(f"=== {name}: {status[name]} ===")

    log("")
    log("=== summaries ===")
    for title, a in SUMMARIES:
        log(f"--- {title} ---")
        run_stage(a, log, None)
    log("")
    log("=== stage status ===")
    for name, _ in stages:
        log(f"  {name:<16} {status.get(name, '-')}")
    log(f"log: {log.f.name}")


if __name__ == "__main__":
    main()
