"""
§5.8 disambiguation probe — does ASDL's captured Gram actually collapse at bf16?

ASDL_HANDOFF.md poses two hypotheses for why ASDL Classic §5.8 shows no
accuracy collapse at bf16 (fp32 88.14% ≈ bf16 88.18%, both λ=10000):

  A — ASDL silently upcasts factors to fp32, so the "bf16" run is not a fair
      κ² test.
  B — the high chosen damping (λ=10000) dominates the curvature, degenerating
      K-FAC to scaled SGD, so there is no κ left to collapse.

Reading the code already refutes a naive version of A: the §5.8 path
(`laplace_cifar10._capture_fisher_asdl`) reads ASDL's Gram factors as fp32 and
then OUR `_evd_from_gram` quantises A/B to bf16 itself before `eigh`.  So the
bf16 corruption is induced by our code regardless of ASDL internals.

This probe measures the κ² signature DIRECTLY at the factor level, independent
of any damping grid-search.  For each supported (Linear/Conv2d) layer it:

  1. Captures the fp32 Gram factors A, B via ASDL's accumulate_curvature.
  2. Runs `eigh` on the fp32 Gram   → (d_min, d_max, κ, n_neg).
  3. Runs `eigh` on the bf16-round-tripped Gram (exactly what _evd_from_gram
     does for the bf16 cell) → (d_min, d_max, κ, n_neg).
  4. Reports per-layer and aggregate deltas.

Decision rule:
  - If bf16 produces MANY more negative eigenvalues / much larger κ than fp32
    → the κ² collapse IS present at the factor level; §5.8's flat accuracy is
    the Laplace damping+clamp masking it (Hypothesis B family).  ASDL is a
    genuine bf16 K-FAC and matches homebrew Classic.
  - If bf16 ≈ fp32 (no extra negatives, similar κ) → ResNet-18's K-FAC factors
    simply aren't ill-conditioned enough for §5.8 to demonstrate the collapse
    at all; ASDL == homebrew Classic, both fine.  The collapse story lives in
    §5.2/§5.6 (training), not §5.8.

Either outcome refutes Hypothesis A (upcasting) as the explanation, because we
quantise the captured fp32 Gram ourselves.

Run:
    python benchmark/probe_asdl_kappa.py --seed 42
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import laplace_cifar10 as lap


def _eigh_stats(M: torch.Tensor):
    """Symmetrise, eigendecompose, return (d_min, d_max, kappa, n_neg)."""
    M = 0.5 * (M + M.t())
    d = torch.linalg.eigvalsh(M)
    d_min = float(d.min())
    d_max = float(d.max())
    n_neg = int((d < 0).sum())
    # κ on the positive part — what the Cholesky/solve actually sees after the
    # clamp(min=0) in _evd_from_gram.  Guard the denominator.
    d_pos_min = float(d[d > 0].min()) if (d > 0).any() else 0.0
    kappa = d_max / d_pos_min if d_pos_min > 0 else float("inf")
    return d_min, d_max, kappa, n_neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, default=1,
                        help="Fisher-capture passes over the train loader.")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}", flush=True)

    # CIFAR-10 + cached MAP model (seed 42 checkpoint already on disk).
    train, val, test = lap.get_cifar10(device)
    model, _ = lap.train_or_load_map(args.seed, train, device)
    model.eval()

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=lap.BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=True)

    # ASDL Classic K-FAC — same construction as build_kfac("asdl_classic", ...).
    from optimizer.asdl_classic_kfac import AsdlClassicKFAC
    opt = AsdlClassicKFAC(
        model, lr=lap.KFAC_LR, damping=lap.KFAC_DAMPING_TRAIN,
        factor_update_freq=1, momentum=lap.KFAC_MOMENTUM,
        grad_clip=lap.GRAD_CLIP, gamma=lap.KFAC_GAMMA,
        use_bf16_factors=False,
    )
    print(f"ignore_modules ({len(opt.ignore_modules)}): {opt.ignore_modules}",
          flush=True)

    # Capture fp32 Gram factors via ASDL (model stays at MAP — no weight step).
    import torch.nn.functional as F
    print(f"\nCapturing Fisher ({args.passes} pass(es))...", flush=True)
    for p in range(args.passes):
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                probs = torch.softmax(model(x).float(), dim=-1)
                y_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            opt.accumulate_curvature(x, y_sampled, loss_fn=F.cross_entropy)
        print(f"  pass {p + 1}/{args.passes} done", flush=True)

    # Read the explicit fp32 Gram factors ASDL stored per supported module.
    rows = []
    for name, module in model.named_modules():
        fisher = getattr(module, "fisher", None)
        if fisher is None:
            continue
        kron = getattr(fisher, "kron", None)
        if kron is None:
            continue
        A = getattr(kron, "A", None)
        B = getattr(kron, "B", None)
        if A is None or B is None:
            continue
        A = A.detach().float()
        B = B.detach().float()
        for tag, M in (("A", A), ("B", B)):
            d_min_f, d_max_f, kap_f, neg_f = _eigh_stats(M)
            Mb = M.to(torch.bfloat16).to(torch.float32)
            d_min_b, d_max_b, kap_b, neg_b = _eigh_stats(Mb)
            rows.append({
                "layer": name, "factor": tag, "dim": M.shape[0],
                "kap_fp32": kap_f, "kap_bf16": kap_b,
                "neg_fp32": neg_f, "neg_bf16": neg_b,
                "dmin_fp32": d_min_f, "dmin_bf16": d_min_b,
            })

    if not rows:
        print("\n!! No Gram factors captured — check ignore_modules / supported "
              "layers.", flush=True)
        return

    # ---- Per-layer table ----
    print(f"\n{'layer':<28} {'fac':>3} {'dim':>5} "
          f"{'κ(fp32)':>11} {'κ(bf16)':>11} {'neg fp32':>9} {'neg bf16':>9}",
          flush=True)
    print("-" * 92, flush=True)
    for r in rows:
        print(f"{r['layer']:<28} {r['factor']:>3} {r['dim']:>5} "
              f"{r['kap_fp32']:>11.2e} {r['kap_bf16']:>11.2e} "
              f"{r['neg_fp32']:>9d} {r['neg_bf16']:>9d}", flush=True)

    # ---- Aggregate verdict ----
    tot_neg_f = sum(r["neg_fp32"] for r in rows)
    tot_neg_b = sum(r["neg_bf16"] for r in rows)
    import statistics as st
    med_kap_f = st.median([r["kap_fp32"] for r in rows if r["kap_fp32"] != float("inf")] or [float("inf")])
    med_kap_b = st.median([r["kap_bf16"] for r in rows if r["kap_bf16"] != float("inf")] or [float("inf")])
    # Worst-case κ blow-up factor introduced by bf16 quantisation.
    blowups = [r["kap_bf16"] / r["kap_fp32"]
               for r in rows
               if r["kap_fp32"] not in (0.0, float("inf"))
               and r["kap_bf16"] != float("inf")]
    max_blowup = max(blowups) if blowups else float("inf")

    print("-" * 92, flush=True)
    print(f"\nAGGREGATE ({len(rows)} factor-matrices across "
          f"{len({r['layer'] for r in rows})} layers):", flush=True)
    print(f"  total negative eigenvalues:  fp32={tot_neg_f}   bf16={tot_neg_b}",
          flush=True)
    print(f"  median condition number:     fp32={med_kap_f:.2e}   "
          f"bf16={med_kap_b:.2e}", flush=True)
    print(f"  max κ blow-up (bf16/fp32):   {max_blowup:.2f}×", flush=True)

    print("\nVERDICT:", flush=True)
    if tot_neg_b > tot_neg_f + 5 or max_blowup > 100:
        print("  → bf16 quantisation DOES corrupt the Gram (κ² signature "
              "present at the factor level).", flush=True)
        print("    §5.8's flat accuracy is the Laplace damping + clamp(min=0) "
              "masking it.", flush=True)
        print("    Hypothesis B family: ASDL is a genuine bf16 K-FAC; the "
              "default λ grid dodges the collapse.", flush=True)
    else:
        print("  → bf16 ≈ fp32 at the factor level: ResNet-18's K-FAC factors "
              "are not ill-conditioned", flush=True)
        print("    enough for §5.8 to demonstrate the κ² collapse for ANYONE "
              "(homebrew Classic included).", flush=True)
        print("    The collapse story lives in §5.2/§5.6 (training), not §5.8 "
              "(BDL).", flush=True)
    print("  Either way: Hypothesis A (ASDL upcasting) is refuted — we quantise "
          "the captured fp32 Gram ourselves.", flush=True)


if __name__ == "__main__":
    main()
