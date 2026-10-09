"""
benchmark/laplace_eig_audit.py  --  §5.9 factor-level audit, saved to JSON

Backs the §5.9 factor-level numbers with result files:
  * homebrew Classic K-FAC (optimizer/hooks.py KFACHooks): negative eigenvalues
    of the Kronecker factors A, G at fp32 and after bf16 storage of the Gram,
    plus d_A range and median per-layer kappa, exactly as
    laplace_cifar10._evd_from_gram / _report_eig_stats compute them;
  * ASDL reference Classic K-FAC (probe_asdl_kappa.py logic): the same counts
    for ASDL's captured Gram factors A, B.

One fp32 capture per (implementation, seed) serves both precisions: the bf16
regime is bf16 STORAGE of the fp32 Gram followed by fp32 eigh (no bf16 eigh
kernel exists), which is what the Laplace pipeline does.  IFKFAC is not
audited here: its eigenvalues are squared singular values of R and are
non-negative by construction.

Capture protocol = laplace_cifar10 (cached AdamW MAP checkpoint, one pass over
the 45k training split, batch 128, model-sampled labels, mean-reduced CE,
torch.manual_seed(seed) before the loader).  Label sampling makes the counts
vary slightly across runs.

Run:
    python -m benchmark.laplace_eig_audit                 # seeds 42 43 44, ~15 min (RTX 3080 Laptop)
    python -m benchmark.laplace_eig_audit --seeds 42 --no-asdl
Resumable: a seed whose file already has both captures is skipped; a file with
only the homebrew capture gets the ASDL capture added.
Output: benchmark/results/laplace_eig_audit_seed{seed}.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmark import laplace_cifar10 as L

OUT = ROOT / "benchmark" / "results"


def eig_row(name, tag, M):
    """fp32 and bf16-storage eigen statistics of one Gram factor."""
    out = {"layer": name, "factor": tag, "dim": int(M.shape[0])}
    for prec in ("fp32", "bf16"):
        Mp = M.float()
        if prec == "bf16":
            Mp = Mp.to(torch.bfloat16).to(torch.float32)
        Mp = 0.5 * (Mp + Mp.t())
        d = torch.linalg.eigvalsh(Mp)
        dc = d.clamp(min=0.0)
        out[f"n_neg_{prec}"] = int((d < 0).sum())
        out[f"d_min_{prec}"] = float(d.min())
        out[f"d_max_{prec}"] = float(d.max())
        # _report_eig_stats convention: max / max(min_clamped, 1e-30)
        out[f"kappa_{prec}"] = float(dc.max() / dc.min().clamp(min=1e-30))
    return out


def summarize(rows, a_tag):
    s = {}
    for prec in ("fp32", "bf16"):
        a = [r for r in rows if r["factor"] == a_tag]
        g = [r for r in rows if r["factor"] != a_tag]
        s[prec] = {
            "n_neg_A": sum(r[f"n_neg_{prec}"] for r in a),
            "n_neg_G": sum(r[f"n_neg_{prec}"] for r in g),
            "d_A_min": min(max(r[f"d_min_{prec}"], 0.0) for r in a),
            "d_A_max": max(r[f"d_max_{prec}"] for r in a),
            "median_kappa_A": st.median(r[f"kappa_{prec}"] for r in a),
            "median_kappa_G": st.median(r[f"kappa_{prec}"] for r in g),
        }
    s["n_layers"] = len({r["layer"] for r in rows})
    return s


def train_loader(train, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return torch.utils.data.DataLoader(train, batch_size=L.BATCH_SIZE, shuffle=True,
                                       num_workers=0, pin_memory=True)


def capture_homebrew(model, loader, device):
    """laplace_cifar10.capture_fisher's loop, returning the raw fp32 Grams."""
    opt = L.build_kfac("classic", model, precision="fp32")
    model.eval()
    opt.hooks.clear()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(logits.float(), -1), 1).squeeze(-1)
        loss = F.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
    raw = opt.hooks.get_factors()
    names = {m: n for n, m in model.named_modules()}
    rows = []
    for mod, (A, G) in raw.items():
        rows.append(eig_row(names[mod], "A", A))
        rows.append(eig_row(names[mod], "G", G))
    opt.hooks.remove()
    model.zero_grad(set_to_none=True)
    del opt
    return rows


def capture_asdl(model, loader, device):
    """probe_asdl_kappa.py's capture: ASDL accumulate_curvature, read kron.A / kron.B.

    `model` must be a freshly loaded network: PyTorch keeps a per-module flag
    once a full backward hook has been registered (even after the handle is
    removed), and then refuses ASDL's regular backward hooks on that module."""
    from optimizer.asdl_classic_kfac import AsdlClassicKFAC
    opt = AsdlClassicKFAC(
        model, lr=L.KFAC_LR, damping=L.KFAC_DAMPING_TRAIN,
        factor_update_freq=1, momentum=L.KFAC_MOMENTUM,
        grad_clip=L.GRAD_CLIP, gamma=L.KFAC_GAMMA, use_bf16_factors=False,
    )
    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(model(x).float(), -1), 1).squeeze(-1)
        opt.accumulate_curvature(x, y, loss_fn=F.cross_entropy)
    rows = []
    for name, module in model.named_modules():
        kron = getattr(getattr(module, "fisher", None), "kron", None)
        A, B = getattr(kron, "A", None), getattr(kron, "B", None)
        if A is None or B is None:
            continue
        rows.append(eig_row(name, "A", A.detach()))
        rows.append(eig_row(name, "B", B.detach()))
    for module in model.modules():          # drop ASDL state before the next capture
        if hasattr(module, "fisher"):
            try:
                delattr(module, "fisher")
            except Exception:
                pass
    del opt
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--no-asdl", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)
    train, _, _ = L.get_cifar10(device)
    OUT.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        p = OUT / f"laplace_eig_audit_seed{seed}.json"
        rec = json.loads(p.read_text()) if p.exists() else {}
        have_home = "summary" in rec.get("homebrew_classic", {})
        have_asdl = "summary" in rec.get("asdl_classic", {})
        if have_home and (have_asdl or args.no_asdl):
            print(f"  already done ({p.name}), skipping", flush=True)
            continue
        rec.update({"seed": seed, "device": str(device), "torch": torch.__version__,
                    "protocol": "fp32 capture; bf16 = bf16 storage of the fp32 Gram, fp32 eigh"})

        if have_home:
            print("  homebrew Classic: already in the result file", flush=True)
        else:
            model, _ = L.train_or_load_map(seed, train, device)
            t0 = time.perf_counter()
            rows = capture_homebrew(model, train_loader(train, seed), device)
            rec["homebrew_classic"] = {"summary": summarize(rows, "A"), "per_layer": rows,
                                       "wall_s": time.perf_counter() - t0}
            s = rec["homebrew_classic"]["summary"]
            print(f"  homebrew Classic ({s['n_layers']} layers): neg A/G fp32 = "
                  f"{s['fp32']['n_neg_A']}/{s['fp32']['n_neg_G']}   bf16 = "
                  f"{s['bf16']['n_neg_A']}/{s['bf16']['n_neg_G']}   "
                  f"d_A in [{s['fp32']['d_A_min']:.3g}, {s['fp32']['d_A_max']:.3g}]   "
                  f"median kappa(A) fp32 = {s['fp32']['median_kappa_A']:.3g}", flush=True)
            p.write_text(json.dumps(rec, indent=2))
            model = None

        if not args.no_asdl:
            t0 = time.perf_counter()
            try:
                model, _ = L.train_or_load_map(seed, train, device)   # fresh copy, no hook history
                rows = capture_asdl(model, train_loader(train, seed), device)
                model = None
                rec["asdl_classic"] = {"summary": summarize(rows, "A"), "per_layer": rows,
                                       "wall_s": time.perf_counter() - t0}
                s = rec["asdl_classic"]["summary"]
                print(f"  ASDL Classic ({s['n_layers']} layers): neg A/B fp32 = "
                      f"{s['fp32']['n_neg_A']}/{s['fp32']['n_neg_G']}   bf16 = "
                      f"{s['bf16']['n_neg_A']}/{s['bf16']['n_neg_G']}   "
                      f"(total bf16 {s['bf16']['n_neg_A'] + s['bf16']['n_neg_G']}, "
                      f"fp32 {s['fp32']['n_neg_A'] + s['fp32']['n_neg_G']})", flush=True)
            except Exception as e:
                rec["asdl_classic"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  ASDL skipped: {rec['asdl_classic']['error']}", flush=True)

        p.write_text(json.dumps(rec, indent=2))
        print(f"  saved {p.name}", flush=True)
        model = None
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
