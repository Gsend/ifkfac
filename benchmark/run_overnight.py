"""
benchmark/run_overnight.py  --  every pending paper run, in one command.

Runs each stage as a separate Python process (clean GPU memory between
stages), logs everything to benchmark/results/overnight_<timestamp>.log, keeps
going if a stage fails, and prints the summaries at the end.  Every stage is
resumable (finished runs are skipped), so if the night ends before the list
does, just run the same command again the next night.

Stages, in priority order (times for the RTX 3080 Laptop):
  1 eig_audit       §5.9  factor-level eigenvalue audit, seeds 42-44           ~15 min
  2 tb_smoke        §5.2/5.7  30-step check that IFKFAC stores R in bf16       ~2 min
  3 tb_main         §5.2 + §5.7  IFKFAC bf16 cells with R stored in bf16       ~6 h
  4 fp32_damping    §5.5  fp32 damping curves (Figure 2 control)               ~6 h
  5 tb_damping      §5.5  IFKFAC bf16 damping curve with R stored in bf16      ~4.5 h
  6 side_lr         §5.7  K-FAC side-AdamW lr grid, seed 42                     ~1.6 h
  7 side_lr_confirm §5.7  best side setting + AdamW, seeds 43-44                ~1 h
  8 plot_damping    §5.5  redraw Figure 2                                       seconds
Total about 19-20 h, i.e. two nights.  Stages 3 and 5 are skipped if the smoke
check (2) fails.

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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "results"

STAGES = [
    ("eig_audit",       ["-m", "benchmark.laplace_eig_audit"]),
    ("tb_smoke",        ["-m", "benchmark.rerun_ifkfac_true_bf16", "--smoke"]),
    ("tb_main",         ["-m", "benchmark.rerun_ifkfac_true_bf16"]),
    ("fp32_damping",    ["-m", "benchmark.damping_sweep_fp32"]),
    ("tb_damping",      ["-m", "benchmark.rerun_ifkfac_true_bf16", "--only", "5.5"]),
    ("side_lr",         ["-m", "benchmark.kfac_side_lr_sweep"]),
    ("side_lr_confirm", ["-m", "benchmark.kfac_side_lr_sweep", "--confirm"]),
    ("plot_damping",    ["benchmark/plot_damping_sweep.py"]),
]
NEEDS_SMOKE = {"tb_main", "tb_damping"}
SUMMARIES = [
    ("IFKFAC R-in-bf16 reruns", ["-m", "benchmark.rerun_ifkfac_true_bf16", "--summary"]),
    ("fp32 vs bf16 damping",    ["-m", "benchmark.damping_sweep_fp32", "--summary"]),
    ("side-AdamW lr",           ["-m", "benchmark.kfac_side_lr_sweep", "--summary"]),
]


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
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget for this launch")
    ap.add_argument("--only", nargs="+", choices=[s for s, _ in STAGES])
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, a in STAGES:
            print(f"  {name:<16} python {' '.join(a)}")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    log = Log(RESULTS / f"overnight_{dt.datetime.now():%Y%m%d_%H%M}.log")
    deadline = time.time() + args.hours * 3600 if args.hours else None
    stages = [(n, a) for n, a in STAGES if not args.only or n in args.only]
    log(f"starting {len(stages)} stage(s)" + (f", budget {args.hours:g} h" if args.hours else ""))

    status, smoke_ok = {}, None
    for name, a in stages:
        if deadline and time.time() > deadline:
            status[name] = "not started (time budget used)"
            continue
        if name in NEEDS_SMOKE and smoke_ok is False:
            status[name] = "skipped (smoke check failed)"
            log(f"=== {name}: skipped because the smoke check failed ===")
            continue
        log(f"=== {name}: python {' '.join(a)} ===")
        t0 = time.time()
        rc = run_stage(a, log, deadline)
        mins = (time.time() - t0) / 60
        if name == "tb_smoke":
            smoke_ok = rc == 0
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
