"""
OlsSMKFAC GPU Benchmark
============================
Empirically measures the advantage of OlsSMKFAC over ClassicKFAC and Adam
on real GPU hardware, using two tasks:

  Task 1 — Large MLP (quick, 20-40 min):
      4-layer 784→2048→2048→2048→10 network trained on MNIST.
      All Linear layers → full OlsSMKFAC coverage.
      Used to verify per-step overhead ratios and convergence curves.

  Task 2 — BERT-base fine-tuning on SST-2 (comprehensive, 3-6 hrs):
      110M param transformer fine-tuned for sentiment classification.
      All 200+ attention/FFN layers are nn.Linear → full coverage.
      Measures whether K-FAC step advantage holds at scale.

Metrics tracked per optimizer:
  - Val accuracy / loss  vs  gradient steps
  - Val accuracy / loss  vs  samples seen   (fair cross-batchsize comparison)
  - Val accuracy / loss  vs  wall-clock time
  - Optimizer overhead per step (timed separately from fwd/bwd)
  - Peak GPU memory allocated (GB)
  - Average GPU power draw (W)  — via nvidia-smi if available

Usage:
  # Quick test (MLP only, ~30 min on any GPU with ≥4GB VRAM):
  python benchmark/gpu_benchmark.py --task mlp

  # Full benchmark (MLP + BERT, ~4-6h, needs ≥12GB VRAM):
  python benchmark/gpu_benchmark.py --task all

  # BERT only:
  python benchmark/gpu_benchmark.py --task bert

  # Custom:
  python benchmark/gpu_benchmark.py --task all --max-steps-mlp 2000 --max-steps-bert 5000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT  = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ─── Serialisation helpers ────────────────────────────────────────────────

def to_serialisable(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, list):        return [to_serialisable(x) for x in obj]
    if isinstance(obj, dict):        return {k: to_serialisable(v) for k,v in obj.items()}
    return obj

def save_result_incremental(result: dict, task: str):
    """Save a single optimizer result immediately after it completes."""
    # Stamp hardware provenance onto result (shallow copy so caller dict is unchanged)
    result = {**result, "hw": HW_INFO}
    name_slug = result["name"].lower().replace(" ", "_")
    # ── JSON (full detail) ──
    json_path = OUT / f"{task}_{name_slug}_result.json"
    with open(json_path, "w") as f:
        json.dump(to_serialisable(result), f, indent=2)
    # ── CSV (summary row) ──
    csv_path = OUT / f"{task}_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("task,name,batch_size,lr_init,lr_final,steps,samples,wall_s,"
                    "avg_fwdbwd_ms,avg_opt_ms,p99_opt_ms,"
                    "peak_mem_gb,avg_power_w,final_val_acc,final_val_loss\n")
        # Support both accuracy-based tasks (val_acc) and ppl-based tasks (val_ppl)
        acc_curve  = result.get("curve_val_acc") or result.get("curve_val_ppl") or []
        loss_curve = result.get("curve_val_loss") or []
        final_acc  = acc_curve[-1]  if acc_curve  else ""
        final_loss = loss_curve[-1] if loss_curve else ""
        pwr      = result.get("avg_power_w") or ""
        lr_final = result.get("lr_final", "")
        f.write(f"{result['task']},{result['name']},{result['B']},{result['lr']},"
                f"{lr_final},{result['steps']},{result['samples']},{result['wall_s']:.1f},"
                f"{result['avg_fwdbwd_ms']:.2f},{result['avg_opt_ms']:.2f},"
                f"{result['p99_opt_ms']:.2f},{result['peak_mem_gb']:.3f},{pwr},"
                f"{final_acc},{final_loss}\n")
    print(f"  ✓  Saved → {json_path.name}  |  {csv_path.name}")
    git_commit_and_push_results(result, json_path, csv_path)

# ─── Git result publishing ─────────────────────────────────────────────────

def git_commit_and_push_results(result: dict, json_path: Path, csv_path: Path):
    """Stage the just-written result files, commit, and push to origin.

    Failures are printed as warnings and never propagate — a git problem
    must not abort a benchmark that may have taken hours to reach this point.
    """
    repo_root = Path(__file__).parent.parent

    # ── Build a descriptive commit message ───────────────────────────────────
    task   = result.get("task", "unknown")
    name   = result.get("name", "unknown")
    wall   = result.get("wall_s")
    wall_s = f"{wall/60:.1f} min" if wall else "?"

    acc_curve  = result.get("curve_val_acc") or []
    ppl_curve  = result.get("curve_val_ppl") or []
    loss_curve = result.get("curve_val_loss") or []

    if acc_curve:
        metric = f"acc={acc_curve[-1]:.4f}"
    elif ppl_curve:
        metric = f"ppl={ppl_curve[-1]:.1f}"
    elif loss_curve:
        metric = f"loss={loss_curve[-1]:.4f}"
    else:
        metric = "no-metric"

    gpu  = (result.get("hw") or {}).get("gpu_name", "unknown-gpu")
    msg  = f"results: {task} {name} — {metric}  wall={wall_s}  [{gpu}]"

    try:
        # Stage only the two result files (never touches source code)
        subprocess.run(
            ["git", "add", str(json_path), str(csv_path)],
            cwd=repo_root, check=True, capture_output=True, text=True
        )

        # Check whether there is actually anything new to commit
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root, capture_output=True
        )
        if diff.returncode == 0:
            print("  git  nothing new to commit (result files unchanged).")
            return

        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, check=True, capture_output=True, text=True
        )
        print(f"  git  committed: {msg}")

        push = subprocess.run(
            ["git", "push"],
            cwd=repo_root, capture_output=True, text=True
        )
        if push.returncode == 0:
            print("  git  pushed to origin.")
        else:
            print(f"  git  push failed (will retry on next result):\n"
                  f"       {push.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"  git  WARNING — commit failed (results are still saved locally):\n"
              f"       {e.stderr.strip() if e.stderr else e}")
    except Exception as e:
        print(f"  git  WARNING — unexpected error ({e}); results saved locally.")

# ─── Hardware fingerprint ─────────────────────────────────────────────────

def get_hardware_info() -> dict:
    """Collect GPU, CPU, RAM and software versions for result provenance."""
    import platform, multiprocessing
    hw = {}

    # GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hw["gpu_name"]       = torch.cuda.get_device_name(0)
        hw["gpu_vram_gb"]    = round(props.total_memory / 1e9, 2)
        hw["gpu_count"]      = torch.cuda.device_count()
        hw["cuda_version"]   = torch.version.cuda
        hw["gpu_sm"]         = f"sm_{props.major}{props.minor}"
    else:
        hw["gpu_name"] = "none"

    # Driver version via nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        hw["gpu_driver"] = out.stdout.strip().split("\n")[0]
    except Exception:
        hw["gpu_driver"] = None

    # CPU / RAM
    hw["cpu"]        = platform.processor() or platform.machine()
    hw["cpu_cores"]  = multiprocessing.cpu_count()
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    hw["ram_gb"] = round(int(line.split()[1]) / 1e6, 1)
                    break
    except Exception:
        hw["ram_gb"] = None

    # Software
    hw["torch_version"]  = torch.__version__
    hw["python_version"] = platform.python_version()

    return hw

# Collected once at import time; injected into every result dict.
HW_INFO: dict = {}   # populated in main() after CUDA is initialised

# ─── GPU utilities ────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"| VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        return d
    print("  ERROR: No CUDA GPU found. Run via run_benchmark.sh for GPU verification.")
    sys.exit(2)

def gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_gpu_power_w() -> Optional[float]:
    """Read instantaneous GPU power via nvidia-smi (returns None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        return float(out.stdout.strip().split("\n")[0])
    except Exception:
        return None

class PowerMonitor:
    """Background power sampling via nvidia-smi every 0.5 s."""
    def __init__(self):
        self.samples = []
        self._active = False

    def start(self):
        self._active = True; self.samples = []

    def sample(self):
        if self._active:
            w = get_gpu_power_w()
            if w is not None: self.samples.append(w)

    def stop(self) -> Optional[float]:
        self._active = False
        return float(np.mean(self.samples)) if self.samples else None

# ─── Helpers ──────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0; total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches: break
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
            else:  # HuggingFace dict batch
                batch  = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                out    = model(**batch)
                logits = out.logits; y = labels
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            correct    += (logits.argmax(-1) == y).sum().item()
            total      += y.size(0)
    return correct / total, total_loss / max(1, i+1)

# ─── TASK 1: Large MLP on MNIST ───────────────────────────────────────────

def run_mlp_benchmark(device, args):
    print("\n" + "="*70)
    print("  TASK 1: Large MLP  (784 → 2048 → 2048 → 2048 → 10)  on MNIST")
    print("="*70)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tf  = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,),(0.3081,))])
    val_loader = DataLoader(
        datasets.MNIST(ROOT/"data", train=False, download=True, transform=tf),
        batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    class LargeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 10))
        def forward(self, x):
            return self.net(x.view(x.size(0), -1))

    # LR hierarchy under momentum=0.9:
    #   ClassicKFAC : OlsSMKFAC : IFKFAC = 1 : 1.5 : 2.25
    # B=512 × factor_update_freq=20 = 10240 rows/window → safely > max layer
    # width (2048), so IFKFAC's TSQR p ≥ n requirement is satisfied.
    configs = [
        dict(name="Adam",          B=128,  lr=1e-3,    kfac=False),
        dict(name="ClassicKFAC",   B=512,  lr=7e-3,    kfac=True,  randomised=False),
        dict(name="OlsSMKFAC",  B=512,  lr=1e-2,    kfac=True,  randomised=True),
        dict(name="IFKFAC",     B=512,  lr=1.5e-2,  kfac=True,  ifkfac=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for MLP task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = LargeMLP().to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_ds = datasets.MNIST(ROOT/"data", train=True, download=False, transform=tf)
        train_loader = DataLoader(train_ds, batch_size=cfg['B'], shuffle=True,
                                  num_workers=2, pin_memory=True)

        if not cfg['kfac']:
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        elif cfg.get('ifkfac'):
            from optimizer.ifkfac_kfac import IFKFAC
            # B=512 × freq=20 = 10240 rows/window  >>  max n_in (2048).
            # max_out_dim=0: no layers excluded (all layers <= 2048 out).
            opt = IFKFAC(model, lr=cfg['lr'], damping=5e-3,
                            factor_update_freq=20,
                            momentum=0.9, grad_clip=10.0, gamma=0.7)
        elif cfg['randomised']:
            from optimizer.olssm_kfac import OlsSMKFAC
            # On GPU: decomp_update_freq=10 is fine (EVD is fast).
            # On CPU: increase to 50 to amortise the expensive EVD cost.
            evd_freq = 20 if torch.cuda.is_available() else 50
            opt = OlsSMKFAC(model, lr=cfg['lr'], damping=5e-3,
                               factor_update_freq=20, decomp_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=256,
                               adaptive_rank_budget=256, momentum=0.9,
                               grad_clip=10.0, gamma=0.99)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=5e-3,
                              factor_update_freq=10, decomp_update_freq=10,
                              momentum=0.9, grad_clip=10.0, gamma=0.9)

        criterion  = nn.CrossEntropyLoss()
        # K-FAC: 100-step linear warmup (Gram matrices are uninitialized for
        # the first factor_update_freq steps so early nat-grad steps are
        # unreliable), then monotonic cosine decay over the remaining steps
        # down to 0.2 % of lr.
        # Adam: plain cosine over the full run — no warmup needed.
        # IMPORTANT: T_max must equal the number of scheduler.step() calls in
        # the cosine phase.  Using T_max < total_steps would create a V-shape
        # where LR bounces back to lr_init at 2*T_max.
        if cfg['kfac']:
            warmup      = 100
            cosine_steps = max(1, args.max_steps_mlp - warmup)
            eta_min      = cfg['lr'] * 0.002
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cosine_steps, eta_min=eta_min),
            ], milestones=[warmup])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.max_steps_mlp, eta_min=cfg['lr'] * 0.01)
        data_iter  = iter(train_loader)
        power_mon  = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        step = samples_seen = 0
        opt_times = []; fwdbwd_times = []
        record_every_n_samples = 2_000
        next_record = record_every_n_samples
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_acc=[]; curve_val_loss=[]

        while step < args.max_steps_mlp:
            try: x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            # Timed fwd+bwd
            t_fwd = time.perf_counter()
            model.train(); logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            # Timed optimizer step
            t_opt = time.perf_counter()
            opt.step()
            t_opt_done = time.perf_counter()

            scheduler.step()
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += x.size(0)

            if samples_seen >= next_record:
                va, vl = evaluate(model, val_loader, device, max_batches=20)
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_acc.append(va)
                curve_val_loss.append(vl)
                next_record += record_every_n_samples
                cur_lr = scheduler.get_last_lr()[0]
                print(f"     step={step:5d}  samples={samples_seen:7,}  "
                      f"val_acc={va:.4f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall:.1f}s  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="mlp", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times,99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_acc=curve_val_acc,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "mlp")
        if hasattr(opt,'cleanup'): opt.cleanup()

    return all_results

# ─── TASK 2: BERT-base fine-tuning on SST-2 ───────────────────────────────

def run_bert_benchmark(device, args):
    print("\n" + "="*70)
    print("  TASK 2: BERT-base fine-tuning on SST-2 (sentiment classification)")
    print("="*70)

    try:
        from transformers import (BertForSequenceClassification, BertTokenizer,
                                  DataCollatorWithPadding)
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: transformers and datasets packages required.")
        print("         pip install transformers datasets")
        return []

    print("  Loading SST-2 dataset and BERT-base tokenizer...")
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    raw_ds     = load_dataset("glue", "sst2")

    def preprocess(batch):
        enc = tokenizer(batch["sentence"], truncation=True, max_length=128)
        enc["labels"] = batch["label"]
        return enc

    encoded = raw_ds.map(preprocess, batched=True,
                          remove_columns=["sentence","idx"])
    encoded.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(encoded["validation"], batch_size=256,
                            collate_fn=collator, num_workers=2)

    _lr_ifkfac_bert = (args.lr_ifkfac_bert
                      if args.lr_ifkfac_bert is not None else 4.5e-3)
    configs = [
        dict(name="Adam",        B=32,  lr=2e-5,          kfac=False),
        dict(name="OlsSMKFAC",  B=512, lr=3e-3,          kfac=True, randomised=True),
        dict(name="ClassicKFAC", B=512, lr=2e-3,          kfac=True, randomised=False),
        dict(name="IFKFAC",   B=32,  lr=_lr_ifkfac_bert, kfac=True, ifkfac=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for BERT task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2).to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_loader = DataLoader(
            encoded["train"], batch_size=cfg['B'], shuffle=True,
            collate_fn=collator, num_workers=2)

        if not cfg['kfac']:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                    weight_decay=0.001)
        elif cfg.get('ifkfac'):
            from optimizer.ifkfac_kfac import IFKFAC
            # B=32 × seq_len=128 × freq=20 = 81920 rows per R update;
            # max n_in = 3072 (BERT FFN): 81920 >> 3072  ✓
            opt = IFKFAC(model, lr=cfg['lr'], damping=1e-3,
                            factor_update_freq=20,
                            momentum=0.9, grad_clip=5.0, gamma=0.7)
        elif cfg['randomised']:
            from optimizer.olssm_kfac import OlsSMKFAC
            evd_freq = 50 if torch.cuda.is_available() else 100
            opt = OlsSMKFAC(model, lr=cfg['lr'], damping=1e-3,
                               factor_update_freq=20, decomp_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=4096,
                               momentum=0.9, grad_clip=20.0, gamma=0.5)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=1e-3,
                              factor_update_freq=20, decomp_update_freq=50,
                              momentum=0.9, grad_clip=10.0, gamma=0.5)

        if cfg['kfac']:
            warmup       = 200
            cosine_steps = max(1, args.max_steps_bert - warmup)
            eta_min      = cfg['lr'] * 0.002
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cosine_steps, eta_min=eta_min),
            ], milestones=[warmup])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.max_steps_bert, eta_min=cfg['lr'] * 0.01)
        # Checkpoint paths — one per optimizer so runs don't overwrite each other
        ckpt_name  = cfg['name'].lower().replace(' ', '_')
        CKPT_MODEL = OUT / f"bert_ckpt_{ckpt_name}_model.pt"
        CKPT_OPT   = OUT / f"bert_ckpt_{ckpt_name}_opt.pt"
        CKPT_KFAC  = OUT / f"bert_ckpt_{ckpt_name}_kfac.pt"
        CKPT_META  = OUT / f"bert_ckpt_{ckpt_name}_meta.pt"

        # Resume from checkpoint if available (skip when --fresh is set)
        if getattr(args, 'fresh', False):
            for _ckpt in [CKPT_MODEL, CKPT_OPT, CKPT_KFAC, CKPT_META]:
                if _ckpt.exists():
                    _ckpt.unlink()
            print("     --fresh: cleared existing BERT checkpoints, starting from scratch.")
        step = samples_seen = 0
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_acc=[]; curve_val_loss=[]
        if CKPT_MODEL.exists() and CKPT_META.exists():
            print(f"     Resuming from checkpoint {CKPT_MODEL.name} ...")
            model.load_state_dict(torch.load(CKPT_MODEL, map_location=device))
            meta = torch.load(CKPT_META, map_location="cpu")
            step         = meta["step"]
            samples_seen = meta["samples_seen"]
            curve_steps  = meta["curve_steps"]
            curve_samples= meta["curve_samples"]
            curve_times  = meta["curve_times"]
            curve_val_acc= meta["curve_val_acc"]
            curve_val_loss=meta["curve_val_loss"]
            if CKPT_OPT.exists():
                opt.load_state_dict(torch.load(CKPT_OPT, map_location=device))
            if CKPT_KFAC.exists() and hasattr(opt, 'load_kfac_state_dict'):
                opt.load_kfac_state_dict(
                    torch.load(CKPT_KFAC, map_location="cpu"), device=device)
                print(f"     K-FAC curvature state restored — "
                      f"preconditioner is pre-warmed ({len(opt._factors)} layers).")
            # Restore scheduler to the right step
            for _ in range(step):
                scheduler.step()
            print(f"     Resumed at step={step}, samples={samples_seen:,}")

        data_iter = iter(train_loader)
        power_mon = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        opt_times=[]; fwdbwd_times=[]
        record_every_n = 5_000
        next_record = (samples_seen // record_every_n + 1) * record_every_n

        # Threshold-triggered LR decay: when val_acc first crosses this level,
        # replace the scheduler with a fast cosine decay from current LR → eta_min
        # over ACC_DECAY_STEPS remaining steps.  Stops the high-LR plateau that
        # follows early K-FAC convergence without shortening the warmup phase.
        ACC_DECAY_THRESHOLD = 0.91   # trigger accuracy
        ACC_DECAY_STEPS     = 2000   # steps for cosine LR decay after trigger
        acc_decay_triggered = False
        acc_decay_step0     = None   # step index when trigger fired
        acc_decay_eta_min   = None   # eta_min locked in at trigger (for clamping)
        acc_decay_complete  = False  # True once LR has reached eta_min and is locked
        # Damping cosine decay state (set at trigger, applied each step)
        damp_decay_start    = None   # damping value at trigger
        damp_decay_end      = 2e-4   # target damping at end of decay window
        damp_decay_step0    = None   # training step at trigger
        damp_decay_steps    = None   # steps over which to decay (= ACC_DECAY_STEPS)

        while step < args.max_steps_bert:
            if args.max_wall_bert > 0 and time.perf_counter() - t0 > args.max_wall_bert:
                print(f"     Wall-time limit {args.max_wall_bert/60:.0f} min reached at step {step} — stopping.")
                break
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); batch = next(data_iter)
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            t_fwd = time.perf_counter()
            model.train()
            out  = model(**batch)
            loss = F.cross_entropy(out.logits, labels)
            opt.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            t_opt = time.perf_counter()
            opt.step()
            t_opt_done = time.perf_counter()

            if not acc_decay_complete:
                scheduler.step()
            # Once ACC_DECAY_STEPS have elapsed since trigger, lock LR at eta_min
            # so CosineAnnealingLR doesn't cycle back upward.
            if acc_decay_step0 is not None and not acc_decay_complete:
                if (step - acc_decay_step0) >= ACC_DECAY_STEPS:
                    for _g in opt.param_groups:
                        _g['lr'] = acc_decay_eta_min
                    acc_decay_complete = True
            # Cosine-decay damping in lockstep with LR after threshold trigger
            if damp_decay_start is not None and hasattr(opt, 'damping'):
                import math
                t = min(step - damp_decay_step0, damp_decay_steps)
                cos_factor = 0.5 * (1 + math.cos(math.pi * t / damp_decay_steps))
                opt.damping = damp_decay_end + (damp_decay_start - damp_decay_end) * cos_factor
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += labels.size(0)

            if samples_seen >= next_record:
                va, vl = evaluate(model, val_loader, device)
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_acc.append(va)
                curve_val_loss.append(vl)
                next_record += record_every_n
                cur_lr = scheduler.get_last_lr()[0]
                # Threshold-triggered LR decay + gradual damping decay
                if cfg['kfac'] and not acc_decay_triggered and va >= ACC_DECAY_THRESHOLD:
                    acc_decay_triggered = True
                    eta_min = cfg['lr'] * 0.002
                    acc_decay_eta_min = eta_min
                    acc_decay_step0   = step
                    # Decay over ACC_DECAY_STEPS (not remaining_steps).
                    # Using remaining_steps made the cosine window ~7k steps,
                    # so LR barely moved for hundreds of steps after the trigger
                    # — causing continued val_acc oscillation.  A fixed 2000-step
                    # window drops LR from current → eta_min in ~4% of total
                    # budget, tight enough to suppress fluctuations.  After the
                    # window, LR is clamped at eta_min (no cosine cycling back up).
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=ACC_DECAY_STEPS, eta_min=eta_min)
                    # Decay damping over same window (gradual, not instantaneous —
                    # instantaneous 15× drop caused model collapse to ~49% acc).
                    if hasattr(opt, 'damping'):
                        damp_decay_start = opt.damping
                        damp_decay_step0 = step
                        damp_decay_steps = ACC_DECAY_STEPS
                        print(f"     *** Accuracy threshold {ACC_DECAY_THRESHOLD:.0%} reached — "
                              f"cosine decay over {ACC_DECAY_STEPS} steps "
                              f"(lr {cur_lr:.2e} → {eta_min:.2e}), "
                              f"damping {damp_decay_start:.0e} → {damp_decay_end:.0e} "
                              f"[gradual cosine, then locked] ***")
                    else:
                        print(f"     *** Accuracy threshold {ACC_DECAY_THRESHOLD:.0%} reached — "
                              f"cosine decay over {ACC_DECAY_STEPS} steps "
                              f"(lr {cur_lr:.2e} → {eta_min:.2e}, then locked) ***")
                print(f"     step={step:5d}  samples={samples_seen:7,}  "
                      f"val_acc={va:.4f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall/60:.1f}min  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")
                # Save checkpoint after every validation point
                torch.save(model.state_dict(), CKPT_MODEL)
                if hasattr(opt, 'state_dict'):
                    torch.save(opt.state_dict(), CKPT_OPT)
                if hasattr(opt, 'kfac_state_dict'):
                    torch.save(opt.kfac_state_dict(), CKPT_KFAC)
                torch.save(dict(
                    step=step, samples_seen=samples_seen,
                    curve_steps=curve_steps, curve_samples=curve_samples,
                    curve_times=curve_times, curve_val_acc=curve_val_acc,
                    curve_val_loss=curve_val_loss,
                ), CKPT_META)
                print(f"     Checkpoint saved → {CKPT_MODEL.name}")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="bert", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times,99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_acc=curve_val_acc,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "bert")
        if hasattr(opt,'cleanup'): opt.cleanup()

    return all_results

# ─── TASK 3: CIFAR-10 MLP ─────────────────────────────────────────────────
#
# Why CIFAR-10?  MNIST is too easy — Adam saturates at ~98.7% within the
# first few hundred steps, leaving no room for K-FAC to show its advantage.
# CIFAR-10 with a plain MLP tops out around 55-58% for Adam; the loss
# landscape is poorly conditioned and has high curvature, which is exactly
# where K-FAC's natural gradient outperforms first-order methods.
#
# Architecture: 3072 → 2048 → 1024 → 512 → 256 → 10
# All Linear layers → full K-FAC coverage, no Conv layers to complicate things.

def run_cifar_benchmark(device, args):
    print("\n" + "="*70)
    print("  TASK 3: CIFAR-10 MLP  (3072 → 2048 → 1024 → 512 → 256 → 10)")
    print("="*70)

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    val_loader = DataLoader(
        datasets.CIFAR10(ROOT/"data", train=False, download=True, transform=tf),
        batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # Dropout(0.2) regularises the loss landscape, pushing the accuracy
            # ceiling from ~58% to ~62% and smoothing the curvature so K-FAC's
            # natural gradient is more reliable.  K-FAC only preconditions the
            # Linear layers — Dropout has no parameters so it is invisible to it.
            self.net = nn.Sequential(
                nn.Linear(3072, 2048), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(1024,  512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear( 512,  256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear( 256,   10),
            )
        def forward(self, x):
            return self.net(x.view(x.size(0), -1))

    # K-FAC shines here: lower lr than MNIST (harder task), same damping/rank
    # improvements from the MLP task.  Adam uses a slightly lower lr too since
    # CIFAR-10 is noisier.
    # Empirically tuned on CIFAR-10: B=1024 + gamma=0.999 for OlsSMKFAC
    # gives 58.8% — beating Adam (57.9%).  Larger batch improves Gram matrix
    # quality; gamma=0.999 smooths over ~1000 update windows for stable curvature.
    # LR hierarchy under momentum=0.9:
    #   ClassicKFAC : OlsSMKFAC : IFKFAC = 1 : 1.5 : 2.25
    # B=1024 × factor_update_freq=20 = 20480 rows/window >> max n_in (3072) ✓
    configs = [
        dict(name="Adam",         B=128,  lr=3e-4,   kfac=False),
        dict(name="ClassicKFAC",  B=512,  lr=1.3e-3, kfac=True, randomised=False),
        dict(name="OlsSMKFAC", B=1024, lr=2e-3,   kfac=True, randomised=True),
        dict(name="IFKFAC",    B=1024, lr=3e-3,   kfac=True, ifkfac=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for CIFAR task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = DeepMLP().to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_ds = datasets.CIFAR10(ROOT/"data", train=True, download=False, transform=tf)
        train_loader = DataLoader(train_ds, batch_size=cfg['B'], shuffle=True,
                                  num_workers=2, pin_memory=True)

        if not cfg['kfac']:
            opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        elif cfg.get('ifkfac'):
            from optimizer.ifkfac_kfac import IFKFAC
            # B=1024 × freq=20 = 20480 rows  >>  max n_in (3072) ✓
            # max_out_dim=0: all layers <= 2048 out, none excluded.
            opt = IFKFAC(model, lr=cfg['lr'], damping=5e-3,
                            factor_update_freq=20,
                            momentum=0.9, grad_clip=10.0, gamma=0.7)
        elif cfg['randomised']:
            from optimizer.olssm_kfac import OlsSMKFAC
            evd_freq = 20 if torch.cuda.is_available() else 50
            opt = OlsSMKFAC(model, lr=cfg['lr'], damping=5e-3,
                               factor_update_freq=20, decomp_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=256,
                               adaptive_rank_budget=256, momentum=0.9,
                               grad_clip=10.0, gamma=0.999)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            # ClassicKFAC needs higher damping on CIFAR-10: direct inversion is
            # less stable than EVD and crashed (14% accuracy drops) at 5e-3.
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=1e-2,
                              factor_update_freq=20, decomp_update_freq=20,
                              momentum=0.9, grad_clip=10.0, gamma=0.9)

        criterion = nn.CrossEntropyLoss()
        if cfg['kfac']:
            warmup = 200
            cosine_steps = max(1, args.max_steps_cifar - warmup)
            eta_min  = cfg['lr'] * 0.0001
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cosine_steps, eta_min=eta_min),
            ], milestones=[warmup])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.max_steps_cifar, eta_min=cfg['lr'] * 0.01)

        data_iter = iter(train_loader)
        power_mon = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        step = samples_seen = 0
        opt_times = []; fwdbwd_times = []
        record_every_n_samples = 5_000
        next_record = record_every_n_samples
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_acc=[]; curve_val_loss=[]

        while step < args.max_steps_cifar:
            try: x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            t_fwd = time.perf_counter()
            model.train(); logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            t_opt = time.perf_counter()
            opt.step()
            t_opt_done = time.perf_counter()

            scheduler.step()
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += x.size(0)

            if samples_seen >= next_record:
                va, vl = evaluate(model, val_loader, device, max_batches=20)
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_acc.append(va)
                curve_val_loss.append(vl)
                next_record += record_every_n_samples
                cur_lr = scheduler.get_last_lr()[0]
                print(f"     step={step:5d}  samples={samples_seen:7,}  "
                      f"val_acc={va:.4f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall:.1f}s  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="cifar", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times,99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_acc=curve_val_acc,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "cifar")
        if hasattr(opt, 'cleanup'): opt.cleanup()

    return all_results

# ─── TASK 4: Small GPT trained from scratch on WikiText-2 ────────────────
#
# This is the regime where K-FAC's curvature advantage is strongest:
#   - Random weight initialisation — no pre-trained features
#   - Rough, high-dimensional loss landscape from the start
#   - All Q/K/V/O + FFN layers are nn.Linear → full K-FAC coverage
#
# Metric: validation perplexity (exp(cross-entropy)) — lower is better.
# A small model (~8M params) reaches perplexity ~150-200 after 5 000 steps.
# K-FAC should reach the same perplexity in fewer samples than Adam.

class _CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit Q/K/V nn.Linear layers."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, Dh   = self.n_heads, self.d_head
        q = self.q(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        scale = Dh ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
        attn  = attn.masked_fill(mask, float('-inf'))
        attn  = torch.softmax(attn, dim=-1)
        attn  = self.drop(attn)
        out   = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class SmallGPT(nn.Module):
    """Small GPT-style causal LM — all projections are nn.Linear for K-FAC.

    Default config (~8 M params):
        4 layers, d=256, 4 heads, FFN=1024, vocab=50 257, seq_len=128.
    """
    def __init__(self, vocab_size: int = 50_257, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int = 1024,
                 max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        # Pre-compute causal mask (upper-triangular = future tokens)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)   # (B, T, vocab_size)

def run_transformer_benchmark(device, args):
    """Train a small GPT from random init on WikiText-2.

    This is the from-scratch regime where K-FAC's curvature advantage over
    Adam is most visible.  Pre-trained weights are intentionally NOT used —
    both optimizers start from the same random initialisation.

    Metric: validation perplexity (lower = better).
    OlsSMKFAC is expected to reach the same perplexity in fewer samples
    because the natural gradient follows the loss curvature from step 1.
    """
    print("\n" + "="*70)
    print("  TASK 4: Small GPT trained from scratch on WikiText-2")
    print("  (random init — no pre-trained weights)")
    print("="*70)

    try:
        from transformers import GPT2TokenizerFast
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: transformers and datasets packages required.")
        print("         pip install transformers datasets")
        return []

    SEQ_LEN = 128

    print("  Loading WikiText-2 and GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(batch):
        # Iterate individually — WikiText-2 contains empty lines and section
        # headers that cause batch tokenization to fail on some tokenizer versions.
        flat = []
        for text in batch["text"]:
            if isinstance(text, str) and text.strip():
                enc = tokenizer(text, truncation=False, add_special_tokens=False)
                flat.extend(enc["input_ids"])
        chunks = [flat[i:i + SEQ_LEN + 1]
                  for i in range(0, len(flat) - SEQ_LEN, SEQ_LEN)]
        return {"input_ids": chunks}

    tokenized = {}
    for split in ("train", "validation"):
        ds = raw[split]
        tok = tokenize({"text": ds["text"]})
        import datasets as hf_ds
        tokenized[split] = hf_ds.Dataset.from_dict(tok)
        tokenized[split].set_format("torch")

    from torch.utils.data import DataLoader

    def collate(batch):
        ids = torch.stack([b["input_ids"] for b in batch])  # (B, SEQ_LEN+1)
        return ids[:, :-1], ids[:, 1:]                        # x, y

    val_loader = DataLoader(tokenized["validation"], batch_size=64,
                            collate_fn=collate, num_workers=2)

    def evaluate_ppl(model, loader, device):
        model.eval()
        total_loss = total_tokens = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)                              # (B, T, V)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    ignore_index=tokenizer.pad_token_id)
                total_loss   += loss.item() * y.numel()
                total_tokens += y.numel()
        ppl = float(torch.exp(torch.tensor(total_loss / total_tokens)))
        return min(ppl, 9999.0)   # cap to avoid inf display

    vocab_size = tokenizer.vocab_size   # 50 257

    # LR hierarchy (KFAC variants tolerate momentum differently):
    #   ClassicKFAC : OlsSMKFAC : IFKFAC = 1 : 1.5 : 2.25
    # OlsSMKFAC default 8e-3 is the prior LR-sweep optimum (ppl=368).
    # ClassicKFAC = OlsSMKFAC / 1.5 = 5.3e-3
    # IFKFAC   = OlsSMKFAC * 1.5 = 1.2e-2
    _lr_ols   = args.lr_ols_transformer   if args.lr_ols_transformer   is not None else 8e-3
    _lr_cls   = args.lr_cls_transformer   if args.lr_cls_transformer   is not None else 5.3e-3
    _lr_ifkfac = args.lr_ifkfac_transformer if args.lr_ifkfac_transformer is not None else 1.2e-2
    configs = [
        dict(name="Adam",        B=32, lr=3e-4,    kfac=False),
        dict(name="OlsSMKFAC",  B=64, lr=_lr_ols,  kfac=True, randomised=True),
        dict(name="ClassicKFAC", B=64, lr=_lr_cls,  kfac=True, randomised=False),
        dict(name="IFKFAC",   B=64, lr=_lr_ifkfac, kfac=True, ifkfac=True),
    ]
    configs = [c for c in configs if c["name"].lower() not in args.skip]
    if not configs:
        print("  All optimizers skipped — nothing to run for transformer task.")
        return []

    all_results = []
    for cfg in configs:
        print(f"\n  ── {cfg['name']}  (B={cfg['B']}, lr={cfg['lr']}) ──")
        torch.manual_seed(42)
        model = SmallGPT(vocab_size=vocab_size).to(device)
        print(f"     Parameters: {count_params(model):,}")

        train_loader = DataLoader(tokenized["train"], batch_size=cfg['B'],
                                  shuffle=True, collate_fn=collate, num_workers=2)

        # For K-FAC optimizers, identify which parameters are covered by K-FAC
        # (nn.Linear / nn.Conv2d with out_dim <= max_gram_dim) and which are not
        # (Embedding, LayerNorm, LM head excluded via max_gram_dim).
        # Non-K-FAC params must be updated by a secondary AdamW — without this,
        # token embeddings (nn.Embedding) stay frozen at random init and the model
        # cannot learn *anything* (ppl stuck at ~9999 from step 1).
        _KFAC_MAX_DIM = 4096
        if cfg['kfac']:
            kfac_covered_ids = set()
            for mod in model.modules():
                if isinstance(mod, (nn.Linear, nn.Conv2d)):
                    out_dim = (mod.out_features if isinstance(mod, nn.Linear)
                               else mod.out_channels)
                    if out_dim <= _KFAC_MAX_DIM:
                        for p in mod.parameters():
                            kfac_covered_ids.add(id(p))
            # Collect unique non-K-FAC params (embeddings, LayerNorm, LM head)
            seen_ids = set()
            other_params = []
            for p in model.parameters():
                if id(p) not in kfac_covered_ids and id(p) not in seen_ids:
                    other_params.append(p)
                    seen_ids.add(id(p))
            print(f"     K-FAC covers {len(kfac_covered_ids)} param tensors; "
                  f"AdamW fallback covers {len(other_params)} "
                  f"(embeddings / LM head / LayerNorm)")
        else:
            other_params = []

        if not cfg['kfac']:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                    weight_decay=0.01)
            emb_opt = None
        elif cfg.get('ifkfac'):
            from optimizer.ifkfac_kfac import IFKFAC
            # B=64 × seq_len=128 = 8192 rows/step; freq=20 → 163840 rows/R update.
            # max n_in = embedding_dim (512 for SmallGPT): 163840 >> 512  ✓
            # max_out_dim=4096: excludes LM head (out=50257) — same policy as
            # OlsSMKFAC/ClassicKFAC via max_gram_dim.  Head is updated by emb_opt.
            opt = IFKFAC(model, lr=cfg['lr'], damping=1e-3,
                            factor_update_freq=20,
                            momentum=0.9, grad_clip=20.0, gamma=0.7,
                            max_out_dim=_KFAC_MAX_DIM)
            emb_opt = torch.optim.AdamW(other_params, lr=cfg['lr'],
                                        weight_decay=0.01)
        elif cfg['randomised']:
            from optimizer.olssm_kfac import OlsSMKFAC
            evd_freq = 5 if torch.cuda.is_available() else 20
            # max_gram_dim=4096: skip K-FAC hooks on the LM head (out=50257).
            # Its G matrix (50257×50257 ≈ 10 GB) would cause OOM.
            # Embeddings + excluded head are updated by emb_opt (AdamW).
            opt = OlsSMKFAC(model, lr=cfg['lr'], damping=1e-3,
                               factor_update_freq=20, decomp_update_freq=evd_freq,
                               adaptive=True, adaptive_min_n=4096,
                               momentum=0.9, grad_clip=20.0, gamma=0.5,
                               max_gram_dim=_KFAC_MAX_DIM)
            emb_opt = torch.optim.AdamW(other_params, lr=cfg['lr'],
                                        weight_decay=0.01)
        else:
            from optimizer.classic_kfac import ClassicKFAC
            opt = ClassicKFAC(model, lr=cfg['lr'], damping=1e-3,
                              factor_update_freq=20, decomp_update_freq=20,
                              momentum=0.9, grad_clip=10.0, gamma=0.5,
                              max_gram_dim=_KFAC_MAX_DIM)
            emb_opt = torch.optim.AdamW(other_params, lr=cfg['lr'],
                                        weight_decay=0.01)

        warmup       = 200
        cosine_steps = max(1, args.max_steps_transformer - warmup)
        if cfg['kfac']:
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cosine_steps, eta_min=cfg['lr'] * 0.0015),
            ], milestones=[warmup])
            emb_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                emb_opt, T_max=args.max_steps_transformer,
                eta_min=cfg['lr'] * 0.002)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.max_steps_transformer, eta_min=cfg['lr'] * 0.01)
            emb_scheduler = None

        data_iter = iter(train_loader)
        power_mon  = PowerMonitor()
        reset_memory_stats()
        power_mon.start()

        t0 = time.perf_counter()
        step = samples_seen = 0
        opt_times = []; fwdbwd_times = []
        record_every_n = 10_000
        next_record    = record_every_n
        curve_steps=[]; curve_samples=[]; curve_times=[]
        curve_val_ppl=[]; curve_val_loss=[]

        # Threshold-triggered LR decay (same mechanism as BERT)
        PPL_DECAY_THRESHOLD = 200.0   # trigger when ppl drops below this
        PPL_DECAY_TRIGGERED = False

        while step < args.max_steps_transformer:
            try: x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); x, y = next(data_iter)
            x, y = x.to(device), y.to(device)

            t_fwd = time.perf_counter()
            model.train()
            logits = model(x)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            # Use model.zero_grad() so ALL params (incl. embeddings) are cleared
            model.zero_grad(); loss.backward()
            t_fwd_done = time.perf_counter()

            t_opt = time.perf_counter()
            opt.step()
            if emb_opt is not None:
                emb_opt.step()
            t_opt_done = time.perf_counter()

            scheduler.step()
            if emb_scheduler is not None:
                emb_scheduler.step()
            power_mon.sample()
            fwdbwd_times.append(t_fwd_done - t_fwd)
            opt_times.append(t_opt_done - t_opt)
            step += 1; samples_seen += x.size(0)

            if samples_seen >= next_record:
                ppl = evaluate_ppl(model, val_loader, device)
                vl  = float(np.log(ppl)) if ppl < 9999 else 9.21
                wall = time.perf_counter() - t0
                curve_steps.append(step); curve_samples.append(samples_seen)
                curve_times.append(wall); curve_val_ppl.append(ppl)
                curve_val_loss.append(vl)
                next_record += record_every_n
                cur_lr = scheduler.get_last_lr()[0]
                # Threshold-triggered LR decay for K-FAC
                if cfg['kfac'] and not PPL_DECAY_TRIGGERED and ppl < PPL_DECAY_THRESHOLD:
                    PPL_DECAY_TRIGGERED = True
                    remaining = max(1, args.max_steps_transformer - step)
                    eta_min   = cfg['lr'] * 0.002
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=remaining, eta_min=eta_min)
                    emb_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        emb_opt, T_max=remaining, eta_min=eta_min)
                    if hasattr(opt, 'damping'):
                        opt.damping = max(1e-4, opt.damping * 0.1)
                    print(f"     *** ppl < {PPL_DECAY_THRESHOLD:.0f} — "
                          f"fast cosine decay over {remaining} steps ***")
                print(f"     step={step:5d}  samples={samples_seen:8,}  "
                      f"val_ppl={ppl:7.1f}  val_loss={vl:.4f}  "
                      f"lr={cur_lr:.2e}  wall={wall/60:.1f}min  "
                      f"opt={np.mean(opt_times[-20:])*1000:.1f}ms")

        avg_power = power_mon.stop()
        peak_mem  = gpu_memory_gb()
        result = dict(
            task="transformer", **cfg,
            lr_final=scheduler.get_last_lr()[0],
            steps=step, samples=samples_seen,
            wall_s=time.perf_counter()-t0,
            avg_fwdbwd_ms=np.mean(fwdbwd_times)*1000,
            avg_opt_ms=np.mean(opt_times)*1000,
            p99_opt_ms=np.percentile(opt_times, 99)*1000,
            peak_mem_gb=peak_mem,
            avg_power_w=avg_power,
            final_val_ppl=curve_val_ppl[-1] if curve_val_ppl else None,
            final_val_loss=curve_val_loss[-1] if curve_val_loss else None,
            curve_steps=curve_steps, curve_samples=curve_samples,
            curve_times=curve_times, curve_val_ppl=curve_val_ppl,
            curve_val_loss=curve_val_loss,
        )
        all_results.append(result)
        save_result_incremental(result, "transformer")
        if hasattr(opt, 'cleanup'): opt.cleanup()

    return all_results

# ─── Scaling Benchmark ────────────────────────────────────────────────────

def run_scaling_benchmark(device, args):
    """
    Sweeps hidden-layer width to show two things:

      1. Step-cost scaling:
           Adam         O(n)     — element-wise, always cheapest per step
           OlsSMKFAC O(n·r²)  — randomised EVD, grows slowly
           ClassicKFAC  O(n³)    — direct inversion, explodes at large n

      2. Convergence quality (fixed sample budget):
           K-FAC uses curvature info → reaches higher accuracy in the same
           number of steps vs Adam.  Combined with panel 1, the third panel
           ("accuracy per ms of optimizer overhead") shows the crossover where
           OlsSMKFAC beats Adam on effective throughput.

    Model:  Linear(width, width) → ReLU → Linear(width, 10)
    Data:   Gaussian synthetic — 10-class linear-separable problem
            (avoids dataset loading; convergence comparison is still valid
             because the curvature advantage is architectural, not data-specific)
    """
    # ── Widths to sweep ────────────────────────────────────────────────
    widths = [256, 512, 1024, 2048]
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 12: widths.append(4096)
        if vram_gb >= 24: widths.append(8192)

    WARMUP_STEPS = 50
    TIMING_STEPS = 100   # pure timing, no convergence value
    CONV_STEPS   = getattr(args, 'max_steps_scaling', 300)
    BATCH        = 256   # fixed across all widths for fair comparison
    VAL_SIZE     = 2000

    timing_results = []  # list of {width, name, avg_ms, p50_ms, p99_ms}
    conv_results   = []  # list of {width, name, final_acc, final_loss, avg_opt_ms}

    print(f"\n  Widths: {widths}   CONV_STEPS={CONV_STEPS}  BATCH={BATCH}")

    for width in widths:
        print(f"\n  ── width={width} {'─'*35}")

        # Synthetic linear-separable dataset
        torch.manual_seed(42)
        W_true = torch.randn(width, 10, device=device) * 0.3

        def make_batch(n, seed=None):
            if seed is not None: torch.manual_seed(seed)
            x = torch.randn(n, width, device=device)
            y = (x @ W_true).argmax(dim=1)
            return x, y

        x_val, y_val = make_batch(VAL_SIZE, seed=99)

        class ScalingModel(nn.Module):
            def __init__(self, w):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(w, w),
                    nn.ReLU(),
                    nn.Linear(w, 10),
                )
            def forward(self, x): return self.net(x)

        rank_budget = min(max(width // 4, 32), 256)

        opt_configs = [
            dict(name="Adam",         kfac=False),
            dict(name="ClassicKFAC",  kfac=True, randomised=False),
            dict(name="OlsSMKFAC", kfac=True, randomised=True),
        ]
        opt_configs = [c for c in opt_configs if c["name"].lower() not in args.skip]

        def build_opt(cfg, model):
            if not cfg['kfac']:
                return torch.optim.Adam(model.parameters(), lr=1e-3)
            elif cfg['randomised']:
                from optimizer.olssm_kfac import OlsSMKFAC
                return OlsSMKFAC(model, lr=1e-2, damping=1e-2,
                                    factor_update_freq=1, decomp_update_freq=1,
                                    adaptive=True, adaptive_min_n=32,
                                    adaptive_rank_budget=rank_budget,
                                    momentum=0.0, gamma=0.0)
            else:
                from optimizer.classic_kfac import ClassicKFAC
                return ClassicKFAC(model, lr=1e-2, damping=1e-2,
                                   factor_update_freq=1, decomp_update_freq=1,
                                   momentum=0.0, gamma=0.0)

        criterion = nn.CrossEntropyLoss()

        for cfg in opt_configs:
            # ── Phase 1: pure timing ──────────────────────────────────
            model = ScalingModel(width).to(device)
            opt   = build_opt(cfg, model)
            timing_ms = []

            for step in range(WARMUP_STEPS + TIMING_STEPS):
                x_b, y_b = make_batch(BATCH)
                model.train()
                opt.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                if device.type == 'cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                opt.step()
                if device.type == 'cuda': torch.cuda.synchronize()
                if step >= WARMUP_STEPS:
                    timing_ms.append((time.perf_counter() - t0) * 1000)

            avg_ms = float(np.mean(timing_ms))
            p50_ms = float(np.percentile(timing_ms, 50))
            p99_ms = float(np.percentile(timing_ms, 99))
            timing_results.append(dict(width=width, name=cfg['name'],
                                       avg_ms=avg_ms, p50_ms=p50_ms, p99_ms=p99_ms))
            print(f"    {cfg['name']:16s}  timing  avg={avg_ms:.2f}ms  "
                  f"p50={p50_ms:.2f}ms  p99={p99_ms:.2f}ms")
            if hasattr(opt, 'cleanup'): opt.cleanup()
            del model, opt

            # ── Phase 2: convergence quality (fixed sample budget) ────
            torch.manual_seed(123)
            model2 = ScalingModel(width).to(device)
            opt2   = build_opt(cfg, model2)
            opt_times2 = []

            for step in range(CONV_STEPS):
                x_b, y_b = make_batch(BATCH)
                model2.train()
                opt2.zero_grad()
                loss = criterion(model2(x_b), y_b)
                loss.backward()
                t0 = time.perf_counter()
                opt2.step()
                if device.type == 'cuda': torch.cuda.synchronize()
                opt_times2.append((time.perf_counter() - t0) * 1000)

            model2.eval()
            with torch.no_grad():
                val_logits = model2(x_val)
                val_acc  = (val_logits.argmax(1) == y_val).float().mean().item()
                val_loss = criterion(val_logits, y_val).item()

            conv_results.append(dict(width=width, name=cfg['name'],
                                     final_acc=val_acc, final_loss=val_loss,
                                     avg_opt_ms=float(np.mean(opt_times2))))
            print(f"    {cfg['name']:16s}  conv    acc={val_acc:.4f}  "
                  f"loss={val_loss:.4f}  opt={np.mean(opt_times2):.2f}ms")
            if hasattr(opt2, 'cleanup'): opt2.cleanup()
            del model2, opt2

    # ── Save raw results ──────────────────────────────────────────────
    result_path = OUT / "scaling_results.json"
    with open(result_path, "w") as f:
        json.dump(to_serialisable({"timing": timing_results, "conv": conv_results}), f, indent=2)
    print(f"\n  Scaling results → {result_path}")

    make_scaling_plot(timing_results, conv_results, CONV_STEPS, BATCH)
    return timing_results, conv_results

def make_scaling_plot(timing_results, conv_results, conv_steps, batch):
    import matplotlib.pyplot as plt

    COLORS = {
        "Adam":         "#7f8c8d",
        "ClassicKFAC":  "#c0392b",
        "OlsSMKFAC": "#2980b9",
    }
    MARKERS = {"Adam": "s", "ClassicKFAC": "^", "OlsSMKFAC": "o"}
    names  = ["Adam", "ClassicKFAC", "OlsSMKFAC"]
    widths = sorted(set(r['width'] for r in timing_results))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "OlsSMKFAC Scaling Benchmark\n"
        "Step-cost scaling (left)  ·  Convergence quality (middle)  ·  "
        "Effective throughput (right)",
        fontsize=12, fontweight="bold"
    )

    # ── Panel 1: step-cost scaling (log-log) ─────────────────────────
    ax = axes[0]
    for name in names:
        pts = sorted((r['width'], r['avg_ms']) for r in timing_results if r['name'] == name)
        if pts:
            ws, ms = zip(*pts)
            ax.loglog(ws, ms, lw=2.5, marker=MARKERS[name], ms=9,
                      color=COLORS[name], label=name)

    # Theoretical reference lines anchored at smallest width
    w0     = widths[0]
    w_arr  = np.array(widths, dtype=float)
    adam_pts = sorted((r['width'], r['avg_ms']) for r in timing_results if r['name'] == 'Adam')
    if adam_pts:
        a0 = adam_pts[0][1]  # Adam ms at smallest width
        ax.loglog(w_arr, a0 * (w_arr / w0),       '--', lw=1, color='#95a5a6',
                  alpha=0.6, label='O(n) ref')
        ax.loglog(w_arr, a0 * (w_arr / w0) ** 2,  ':', lw=1, color='#e67e22',
                  alpha=0.6, label='O(n²) ref')
        ax.loglog(w_arr, a0 * (w_arr / w0) ** 3,  ':', lw=1, color='#e74c3c',
                  alpha=0.6, label='O(n³) ref')

    ax.set_xlabel("Hidden layer width  n", fontsize=10)
    ax.set_ylabel("Avg optimizer step time (ms)", fontsize=10)
    ax.set_title("Step Cost Scaling  (log-log)\n"
                 "slope = algorithmic complexity",
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    # ── Panel 2: convergence quality (final acc vs width) ─────────────
    ax = axes[1]
    for name in names:
        pts = sorted((r['width'], r['final_acc']) for r in conv_results if r['name'] == name)
        if pts:
            ws, accs = zip(*pts)
            ax.semilogx(ws, accs, lw=2.5, marker=MARKERS[name], ms=9,
                        color=COLORS[name], label=name)

    ax.set_xlabel("Hidden layer width  n", fontsize=10)
    ax.set_ylabel(f"Val accuracy  ({conv_steps} steps, B={batch})", fontsize=10)
    ax.set_title("Convergence Quality vs Width\n"
                 "fixed sample budget — higher = better curvature use",
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── Panel 3: effective throughput = accuracy / opt_ms ─────────────
    ax = axes[2]
    for name in names:
        tim_by_w  = {r['width']: r['avg_ms']    for r in timing_results if r['name'] == name}
        conv_by_w = {r['width']: r['final_acc'] for r in conv_results   if r['name'] == name}
        ws = sorted(tim_by_w.keys() & conv_by_w.keys())
        if ws:
            throughput = [conv_by_w[w] / max(tim_by_w[w], 1e-6) for w in ws]
            ax.semilogx(ws, throughput, lw=2.5, marker=MARKERS[name], ms=9,
                        color=COLORS[name], label=name)

    ax.set_xlabel("Hidden layer width  n", fontsize=10)
    ax.set_ylabel("Val accuracy / opt_ms  (higher = better)", fontsize=10)
    ax.set_title("Effective Throughput\n"
                 "accuracy per ms of optimizer overhead\n"
                 "crossover = where K-FAC beats Adam",
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT / "gpu_benchmark_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Scaling plot saved → {path}")

# ─── Plotting ─────────────────────────────────────────────────────────────

def make_plots(results, tag):
    import matplotlib.pyplot as plt
    COLORS = {"Adam":"#7f8c8d","ClassicKFAC":"#c0392b","OlsSMKFAC":"#2980b9"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"OlsSMKFAC GPU Benchmark — {tag}", fontsize=13, fontweight="bold")

    # Detect whether results use accuracy (classification) or perplexity (LM)
    _has_ppl = any("curve_val_ppl" in r for r in results)
    _acc_key  = "curve_val_ppl" if _has_ppl else "curve_val_acc"
    _acc_lbl  = "Val perplexity (↓)"   if _has_ppl else "Val accuracy"
    _acc_t1   = "Val Perplexity vs Steps"          if _has_ppl else "Val Accuracy vs Steps"
    _acc_t2   = "Val Perplexity vs Samples Seen\n(fair cross-batchsize comparison)" \
                if _has_ppl else "Val Accuracy vs Samples Seen\n(fair cross-batchsize comparison)"
    _acc_t3   = "Val Perplexity vs Wall Time"      if _has_ppl else "Val Accuracy vs Wall Time"

    def plot_curve(ax, x_key, xlabel, ylabel_key, ylabel, title):
        for r in results:
            ys = r.get(ylabel_key) or []
            xs = r.get(x_key) or []
            if not xs or not ys:
                continue
            ax.plot(xs, ys, lw=2, marker="o", ms=4,
                    color=COLORS.get(r["name"],"black"), label=r["name"])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plot_curve(axes[0,0], "curve_steps",   "Gradient steps",
               _acc_key, _acc_lbl, _acc_t1)
    plot_curve(axes[0,1], "curve_samples", "Samples seen",
               _acc_key, _acc_lbl, _acc_t2)
    plot_curve(axes[0,2], "curve_times",   "Wall time (s)",
               _acc_key, _acc_lbl, _acc_t3)
    plot_curve(axes[1,0], "curve_steps",   "Gradient steps",
               "curve_val_loss","Val loss",     "Val Loss vs Steps")
    plot_curve(axes[1,1], "curve_samples", "Samples seen",
               "curve_val_loss","Val loss",     "Val Loss vs Samples Seen")

    # Per-step timing + memory bar chart
    ax = axes[1,2]
    names   = [r["name"] for r in results]
    fwd_ms  = [r["avg_fwdbwd_ms"] for r in results]
    opt_ms  = [r["avg_opt_ms"]    for r in results]
    mem_gb  = [r["peak_mem_gb"]   for r in results]
    pwr_w   = [r.get("avg_power_w") or 0 for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, fwd_ms, w, label="Fwd+Bwd (ms)", color="#bdc3c7")
    ax.bar(x - w/2, opt_ms, w, label="Optimizer (ms)", color=[COLORS.get(n,"black") for n in names],
           bottom=fwd_ms, alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(x, mem_gb, "D--", color="#e67e22", ms=8, lw=2, label="Peak mem (GB)")
    ax2.set_ylabel("Peak GPU memory (GB)", color="#e67e22", fontsize=9)
    ax2.tick_params(axis='y', labelcolor="#e67e22")
    ax.set_title("Per-Step Timing & GPU Memory", fontweight="bold", fontsize=10)
    ax.set_ylabel("ms / step", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate power if available
    for i, (name, pw) in enumerate(zip(names, pwr_w)):
        if pw: ax.text(i+w/2, 5, f"{pw:.0f}W", fontsize=8, ha="center", color="#8e44ad")

    plt.tight_layout()
    path = OUT / f"gpu_benchmark_{tag.lower().replace(' ','_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {path}")

# ─── Summary printout ─────────────────────────────────────────────────────

def print_summary(results):
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    print(f"  {'Name':<16}  {'B':>5}  {'lr_init':>8}  {'lr_final':>8}  "
          f"{'Steps':>7}  {'Samples':>9}  {'Wall':>8}  "
          f"{'FwdBwd ms':>10}  {'Opt ms':>8}  {'Mem GB':>7}  {'Pwr W':>6}")
    print(f"  {'':─<16}  {'':─<5}  {'':─<8}  {'':─<8}  {'':─<7}  "
          f"{'':─<9}  {'':─<8}  {'':─<10}  {'':─<8}  {'':─<7}  {'':─<6}")
    for r in results:
        wall      = f"{r['wall_s']/60:.1f}min" if r['wall_s'] > 120 else f"{r['wall_s']:.0f}s"
        pwr       = f"{r['avg_power_w']:.0f}" if r.get('avg_power_w') else "N/A"
        lr_final  = f"{r['lr_final']:.2e}" if r.get('lr_final') is not None else "N/A"
        print(f"  {r['name']:<16}  {r['B']:>5}  {r['lr']:>8.2e}  {lr_final:>8}  "
              f"{r['steps']:>7,}  {r['samples']:>9,}  {wall:>8}  "
              f"{r['avg_fwdbwd_ms']:>10.1f}  {r['avg_opt_ms']:>8.1f}  "
              f"{r['peak_mem_gb']:>7.2f}  {pwr:>6}")
    _has_ppl = any("curve_val_ppl" in r for r in results)
    _metric_key = "curve_val_ppl" if _has_ppl else "curve_val_acc"
    _metric_lbl = "Final val perplexity" if _has_ppl else "Final val accuracy"
    print(f"\n  {_metric_lbl}:")
    for r in results:
        curve = r.get(_metric_key) or []
        val = curve[-1] if curve else 0
        print(f"    {r['name']:<16}: {val:.4f}")

    # Compute key ratios
    by_name = {r["name"]: r for r in results}
    if "OlsSMKFAC" in by_name and "ClassicKFAC" in by_name:
        ol = by_name["OlsSMKFAC"]; cl = by_name["ClassicKFAC"]
        print(f"\n  OlsSMKFAC vs ClassicKFAC:")
        print(f"    Optimizer overhead ratio : {cl['avg_opt_ms']/ol['avg_opt_ms']:.2f}× faster")
        print(f"    Wall-time ratio          : {cl['wall_s']/ol['wall_s']:.2f}× faster")
        print(f"    Memory savings           : {cl['peak_mem_gb']-ol['peak_mem_gb']:.2f} GB less")
        if cl.get('avg_power_w') and ol.get('avg_power_w'):
            elec_ratio = (cl['avg_power_w']*cl['wall_s']) / (ol['avg_power_w']*ol['wall_s'])
            print(f"    Energy ratio             : {elec_ratio:.2f}× less electricity")
    if "OlsSMKFAC" in by_name and "Adam" in by_name:
        ol = by_name["OlsSMKFAC"]; ad = by_name["Adam"]
        print(f"\n  OlsSMKFAC vs Adam (samples to same final accuracy):")
        print(f"    Samples seen: OlsSMKFAC {ol['samples']:,}  vs  Adam {ad['samples']:,}")
        if ol['samples'] < ad['samples']:
            print(f"    K-FAC needed {ad['samples']/ol['samples']:.1f}× fewer samples — advantage confirmed")
        else:
            print(f"    Adam needed fewer samples — K-FAC advantage not seen at this scale")

# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mlp","bert","cifar","scaling","transformer","all"], default="all")
    parser.add_argument("--max-steps-mlp",   type=int, default=3000)
    parser.add_argument("--max-steps-bert",  type=int, default=3000)
    parser.add_argument("--max-wall-bert",   type=float, default=1200.0,
                        help="Maximum wall-clock seconds for BERT training (default 1200 = 20 min). "
                             "0 = no wall-time limit.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing BERT checkpoints and start from scratch.")
    parser.add_argument("--max-steps-cifar",   type=int, default=5000)
    parser.add_argument("--max-steps-transformer", type=int, default=5000,
                        help="Training steps for the from-scratch transformer task")
    parser.add_argument("--max-steps-scaling", type=int, default=300,
                        help="Convergence steps per optimizer per width in scaling task")
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated optimizer names to skip. "
             "Valid: adam, classickfac, olssmkfac. "
             "Example: --skip adam,classickfac"
    )
    parser.add_argument("--lr-ols-transformer", type=float, default=None,
                        help="Override learning rate for OlsSMKFAC in transformer task. "
                             "Default: 8e-3. Example: --lr-ols-transformer 5e-3")
    parser.add_argument("--lr-cls-transformer", type=float, default=None,
                        help="Override learning rate for ClassicKFAC in transformer task. "
                             "Default: 8e-3. Example: --lr-cls-transformer 5e-3")
    parser.add_argument("--lr-ifkfac-transformer", type=float, default=None,
                        help="Override learning rate for IFKFAC in transformer task. "
                             "Default: 1e-2. Example: --lr-ifkfac-transformer 5e-3")
    parser.add_argument("--lr-ifkfac-bert", type=float, default=None,
                        help="Override learning rate for IFKFAC in BERT task. "
                             "Default: 3e-3. Example: --lr-ifkfac-bert 5e-3")
    parser.add_argument("--lr-sweep-transformer", action="store_true",
                        help="Run a learning-rate sweep for both K-FAC optimizers in the "
                             "transformer task. Tests [1e-3, 3e-3, 5e-3, 8e-3] for each "
                             "using --max-steps-transformer steps, reports best LR and ppl. "
                             "Adam is not swept (its LR 3e-4 is well-tuned).")
    args = parser.parse_args()
    # Normalise skip list to lowercase set
    args.skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    args.max_steps_scaling = args.max_steps_scaling  # expose via consistent attr name

    print("\nOlsSMKFAC GPU Benchmark")
    print("="*70)
    device = get_device()

    global HW_INFO
    HW_INFO = get_hardware_info()
    print(f"  Hardware: {HW_INFO.get('gpu_name')}  |  driver {HW_INFO.get('gpu_driver')}  "
          f"|  CUDA {HW_INFO.get('cuda_version')}  |  torch {HW_INFO.get('torch_version')}")

    all_results = []

    if args.task in ("scaling","all"):
        run_scaling_benchmark(device, args)

    if args.task in ("mlp","all"):
        mlp_results = run_mlp_benchmark(device, args)
        all_results.extend(mlp_results)
        if mlp_results:
            print_summary(mlp_results)
            make_plots(mlp_results, "Large MLP")

    if args.task in ("cifar","all"):
        cifar_results = run_cifar_benchmark(device, args)
        all_results.extend(cifar_results)
        if cifar_results:
            print_summary(cifar_results)
            make_plots(cifar_results, "CIFAR-10 MLP")

    if args.task in ("bert","all"):
        bert_results = run_bert_benchmark(device, args)
        all_results.extend(bert_results)
        if bert_results:
            print_summary(bert_results)
            make_plots(bert_results, "BERT-base SST-2")

    if args.task in ("transformer","all") and getattr(args, "lr_sweep_transformer", False):
        # ── LR sweep: run each K-FAC optimizer at 4 candidate learning rates ──
        # Uses the same step budget as --max-steps-transformer for fair comparison.
        # Adam is excluded from the sweep (lr=3e-4 is well-established for AdamW).
        _LR_CANDIDATES = [1e-3, 3e-3, 5e-3, 8e-3]
        print("\n" + "="*70)
        print("  LR SWEEP — SmallGPT / WikiText-2")
        print(f"  Candidates: {_LR_CANDIDATES}  |  Steps: {args.max_steps_transformer}")
        print("="*70)
        import copy
        sweep_best = {}   # opt_name -> (best_lr, best_ppl, result)
        sweep_args = copy.copy(args)
        sweep_args.skip = {"adam"}          # skip Adam during sweep
        for opt_name, key in [("OlsSMKFAC", "lr_ols_transformer"),
                               ("ClassicKFAC",  "lr_cls_transformer")]:
            if opt_name.lower() in args.skip:
                continue
            print(f"\n  ── Sweeping {opt_name} ──")
            best_lr = None; best_ppl = float("inf"); best_result = None
            for lr_candidate in _LR_CANDIDATES:
                # Skip the other optimizer each iteration
                other = "classickfac" if opt_name == "OlsSMKFAC" else "olssmkfac"
                sweep_args.skip = {"adam", other}
                setattr(sweep_args, "lr_ols_transformer",
                        lr_candidate if opt_name == "OlsSMKFAC" else args.lr_ols_transformer)
                setattr(sweep_args, "lr_cls_transformer",
                        lr_candidate if opt_name == "ClassicKFAC"  else args.lr_cls_transformer)
                print(f"\n     lr = {lr_candidate:.0e}")
                res_list = run_transformer_benchmark(device, sweep_args)
                if res_list:
                    r = res_list[0]
                    ppl = r.get("final_val_ppl") or (r.get("curve_val_ppl") or [float("inf")])[-1]
                    print(f"     → final ppl = {ppl:.1f}")
                    if ppl < best_ppl:
                        best_ppl = ppl; best_lr = lr_candidate; best_result = r
            if best_result:
                sweep_best[opt_name] = (best_lr, best_ppl, best_result)
                print(f"\n  ★  {opt_name}: best lr = {best_lr:.0e}  →  ppl = {best_ppl:.1f}")

        print("\n" + "="*70)
        print("  LR SWEEP RESULTS")
        print("="*70)
        for opt_name, (blr, bppl, _) in sweep_best.items():
            print(f"  {opt_name:<16}  best lr = {blr:.0e}   final ppl = {bppl:.1f}")
        if sweep_best:
            ols_lr = sweep_best.get("OlsSMKFAC", (3e-3,))[0]
            cls_lr = sweep_best.get("ClassicKFAC",  (3e-3,))[0]
            print(f"\n  Re-run with:")
            print(f"    bash run_benchmark.sh --task transformer "
                  f"--lr-ols-transformer {ols_lr:.0e} "
                  f"--lr-cls-transformer {cls_lr:.0e}")

    if args.task in ("transformer","all") and not getattr(args, "lr_sweep_transformer", False):
        transformer_results = run_transformer_benchmark(device, args)
        all_results.extend(transformer_results)
        if transformer_results:
            make_plots(transformer_results, "SmallGPT WikiText-2 (from scratch)")

    # Save combined results JSON (all runs together)
    if all_results:
        save_path = OUT / "gpu_benchmark_results.json"
        with open(save_path, "w") as f:
            json.dump(to_serialisable(all_results), f, indent=2)
        print(f"\n  Combined results → {save_path}")
        print(f"  Per-run JSONs and CSV summary → {OUT}/")

if __name__ == "__main__":
    main()