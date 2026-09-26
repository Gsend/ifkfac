"""
benchmark/fisher_approx_small.py  --  EKFAC_LAPLACE_PLAN arm 4

Direct Fisher-approximation comparison on a model small enough to form the
exact Fisher densely.  Single-hidden-layer MLP (d_in -> H -> 10, ReLU, no
bias), trained to a MAP point; exact type-2 Fisher (= GGN for softmax-CE,
exact expectation over classes, no MC sampling) per layer, in fp64.

Approximations, each in fp32 and bf16 regimes:
    kfac_classic    Classic K-FAC   G (x) A, eigh of the (bf16-stored) Grams
    kfac_ifkfac      IFKFAC          R^T R = A via streaming TSQR, SVD of R
    ekfac_classic   EK-FAC          diag(Q^T F Q) in the Classic eigenbasis
    ekfac_ifkfac     EK-FAC          diag(Q^T F Q) in the IFKFAC eigenbasis
    *_ifkfac_Rbf16   (bf16 only) stress variant: running R stored in bf16 after
                    every TSQR update (fully-bf16 streaming storage)

Decomposition this gives (per layer):
    eigenvalue effect = kfac_X  vs ekfac_X   (same basis X)
    basis effect      = ekfac_classic vs ekfac_ifkfac
                        (EK-FAC is the Frobenius-optimal diagonal in its
                         basis - George et al. 2018 - so its error is a pure
                         basis-quality measure)
Note: in exact arithmetic kfac_classic == kfac_ifkfac and
ekfac_classic == ekfac_ifkfac.  Differences are numerical by construction.

Metrics per (layer, method, precision):
    rel_fro     ||F - F_hat||_F / ||F||_F
    align@k     ||Q_k(F)^T Q_k(F_hat)||_F^2 / k   (top-k subspace overlap, 1 = perfect)
plus kappa(A), kappa(G), #negative eigenvalues (Classic eigh), and whether
torch.linalg.eigh accepts a bf16 tensor at all (recorded, not hidden).

bf16 regime (identical to the ResNet-18 Laplace pipeline): factors are
STORED in bf16 (round-trip), arithmetic in fp32 because no bf16 eigh/SVD
kernel exists.  Classic: fp32 Gram of fp32 data, quantized once, then eigh.
IFKFAC: each TSQR input chunk quantized to bf16, running R in fp32
(= benchmark/kfac_bf16_compare.enable_bf16), then SVD.  EK-FAC: S computed from
fp32 activations/grads in the (bf16-stored) basis, then stored in bf16.

Sanity checks (plan step 1):
    * EK-FAC code path with S forced to d_G (x) d_A reproduces K-FAC exactly.
    * EK-FAC S == diag(Q^T F Q) computed from the dense exact Fisher.
    * Optional reference: curvlinops KFAC / EKFAC (type-2) dense blocks.

Run:
    python -m benchmark.fisher_approx_small --data digits          # sklearn, no download, CPU ok
    python -m benchmark.fisher_approx_small --data mnist --hidden 64
Output: benchmark/results/fisher_approx_small_{data}_h{H}_seed{s}.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

F64 = torch.float64
TOPK = (8, 32, 128)
TSQR_CHUNK = 256


def q(t):
    """bf16 storage round-trip, returns the input dtype."""
    return t.to(torch.bfloat16).to(t.dtype)


# ----------------------------------------------------------------------------- data / model
def load_data(name, seed, data_root):
    if name == "digits":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
    elif name == "mnist":
        from torchvision import datasets
        ds = datasets.MNIST(str(data_root), train=True, download=True)
        X = ds.data.reshape(-1, 784).float() / 255.0
        y = ds.targets.clone()
        Xc = X - X.mean(0)
        _, _, V = torch.pca_lowrank(Xc, q=64, center=False)
        X = Xc @ V[:, :64]          # PCA projection, NOT whitened
    else:
        raise ValueError(name)
    # Global scaling only: per-feature standardisation would whiten the inputs
    # and push kappa(A) -> 1, hiding exactly the conditioning effect under test.
    X = X[:, X.std(0) > 1e-8]
    X = (X - X.mean(0)) / X.std()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    n_tr = int(0.8 * X.shape[0])
    return X[perm[:n_tr]], y[perm[:n_tr]], X[perm[n_tr:]], y[perm[n_tr:]]


class MLP(nn.Module):
    def __init__(self, d_in, h, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h, bias=False)
        self.fc2 = nn.Linear(h, d_out, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train_map(model, X, y, Xv, yv, device, epochs, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = X.shape[0]
    bs = min(256, n)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(X[idx].to(device)), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (model(Xv.to(device)).argmax(-1).cpu() == yv).float().mean().item()
    return acc


# ----------------------------------------------------------------------------- per-sample quantities
@torch.no_grad()
def layer_stats(model, X, device):
    """Exact-expectation per-sample (a, g) per layer, fp64.
    g_{n,c} = sqrt(p_nc) * d loss(c) / d s_l  -> sum_c g g^T = B^T H B."""
    W1 = model.fc1.weight.detach().to(device, F64)
    W2 = model.fc2.weight.detach().to(device, F64)
    a1 = X.to(device, F64)
    s1 = a1 @ W1.t()
    a2 = F.relu(s1)
    z = a2 @ W2.t()
    p = torch.softmax(z, -1)                                   # (N, C)
    C = p.shape[1]
    eye = torch.eye(C, device=device, dtype=F64)
    dz = p[:, None, :] - eye[None, :, :]                       # (N, C_label, C)
    g2 = p.sqrt()[:, :, None] * dz                             # (N, C, 10)
    g1 = (g2 @ W2) * (s1 > 0).to(F64)[:, None, :]              # (N, C, H)
    return {"fc1": (a1, g1), "fc2": (a2, g2)}


def exact_fisher(a, g, chunk=64):
    """F = (1/N) sum_{n,c} (g (x) a)(g (x) a)^T, vec(W) row-major (out, in)."""
    N, n_in = a.shape
    n_out = g.shape[-1]
    D = n_out * n_in
    Fm = torch.zeros(D, D, device=a.device, dtype=F64)
    for i in range(0, N, chunk):
        v = (g[i:i + chunk, :, :, None] * a[i:i + chunk, None, None, :]).reshape(-1, D)
        Fm.addmm_(v.t(), v)
    return Fm / N


# ----------------------------------------------------------------------------- factor paths
def classic_factors(a, g, prec, log):
    N = a.shape[0]
    A = (a.t() @ a / N).float()
    Gr = g.reshape(-1, g.shape[-1])
    G = (Gr.t() @ Gr / N).float()
    if prec == "bf16":
        try:
            torch.linalg.eigh(A.to(torch.bfloat16))
            log["eigh_bf16_native"] = "ok"
        except Exception as e:
            log["eigh_bf16_native"] = f"unsupported: {type(e).__name__}: {str(e)[:80]}"
        A, G = q(A), q(G)
    A = 0.5 * (A + A.t()); G = 0.5 * (G + G.t())
    dA, UA = torch.linalg.eigh(A)
    dG, UG = torch.linalg.eigh(G)
    log["n_neg_A"] = int((dA < 0).sum()); log["n_neg_G"] = int((dG < 0).sum())
    dA, dG = dA.clamp(min=0), dG.clamp(min=0)
    if prec == "bf16":
        UA, UG, dA, dG = q(UA), q(UG), q(dA), q(dG)
    return UA.to(F64), dA.to(F64), UG.to(F64), dG.to(F64)


def tsqr(rows, prec, requant_R=False):
    """Streaming TSQR, fp32 arithmetic.
    bf16 regime (default, matches the pipeline's enable_bf16 chunk transform):
    quantize each INPUT chunk to bf16; the running R stays fp32.
    requant_R=True (stress variant 'ifkfac_Rbf16'): additionally store the
    running R in bf16 after every update - what a fully-bf16 streaming
    implementation would do; rounding error then accumulates with #updates."""
    d = rows.shape[1]
    R = torch.zeros(0, d, device=rows.device, dtype=torch.float32)
    for i in range(0, rows.shape[0], TSQR_CHUNK):
        ch = rows[i:i + TSQR_CHUNK].float()
        if prec == "bf16":
            ch = q(ch)
        R = torch.linalg.qr(torch.cat([R, ch], 0), mode="r").R
        if prec == "bf16" and requant_R:
            R = q(R)
    return R


def ifkfac_factors(a, g, prec, log, requant_R=False):
    N = a.shape[0]
    RA = tsqr(a / math.sqrt(N), prec, requant_R)
    RG = tsqr(g.reshape(-1, g.shape[-1]) / math.sqrt(N), prec, requant_R)
    log["tsqr_updates_A"] = math.ceil(a.shape[0] / TSQR_CHUNK)
    log["tsqr_updates_G"] = math.ceil(g.reshape(-1, g.shape[-1]).shape[0] / TSQR_CHUNK)
    _, SA, VhA = torch.linalg.svd(RA, full_matrices=False)
    _, SG, VhG = torch.linalg.svd(RG, full_matrices=False)
    UA, dA = VhA.t().contiguous(), SA ** 2
    UG, dG = VhG.t().contiguous(), SG ** 2
    # pad if R is short (fewer rows than dims) - not expected here
    assert UA.shape[0] == UA.shape[1] and UG.shape[0] == UG.shape[1]
    log["n_neg_A"] = 0; log["n_neg_G"] = 0
    if prec == "bf16":
        UA, UG, dA, dG = q(UA), q(UG), q(dA), q(dG)
    return UA.to(F64), dA.to(F64), UG.to(F64), dG.to(F64)


def ekfac_S(a, g, UA, UG, prec, chunk=256):
    """S_ij = (1/N) sum_{n,c} (U_G^T g)_i^2 (U_A^T a)_j^2, fp32-equivalent math in fp64."""
    N = a.shape[0]
    S = torch.zeros(UG.shape[0], UA.shape[0], device=a.device, dtype=F64)
    for i in range(0, N, chunk):
        ap = (a[i:i + chunk] @ UA) ** 2                        # (b, n_in)
        gp = (g[i:i + chunk] @ UG) ** 2                        # (b, C, n_out)
        S += torch.einsum("bco,bi->oi", gp, ap)
    S = S / N
    if prec == "bf16":
        S = q(S.float()).to(F64)
    return S


# ----------------------------------------------------------------------------- reconstruction / metrics
def kron_basis(UA, UG):
    return torch.kron(UG, UA)                                  # columns u_G,i (x) u_A,j


def recon(UA, UG, S):
    Q = kron_basis(UA, UG)
    return (Q * S.reshape(-1)[None, :]) @ Q.t(), Q


def topk_basis_hat(Q, S, k):
    idx = torch.argsort(S.reshape(-1), descending=True)[:k]
    return Q[:, idx]


def align(QF, Qh):
    k = QF.shape[1]
    return float((QF.t() @ Qh).pow(2).sum() / k)


LAMBDAS = (1.0, 10.0, 100.0)


def laplace_kl(Fm, Fh, N, lam):
    """KL( N(0, P^-1) || N(0, Phat^-1) ),  P = N F + lam I,  Phat = N Fhat + lam I.
    = 0.5 [ tr(Phat P^-1) - D + logdet P - logdet Phat ].  Sensitive to the
    SMALL end of the spectrum (where posterior variance lives), unlike rel_fro."""
    D = Fm.shape[0]
    I = torch.eye(D, device=Fm.device, dtype=F64)
    P = N * Fm + lam * I
    Ph = N * 0.5 * (Fh + Fh.t()) + lam * I
    Lc = torch.linalg.cholesky(P)
    tr = torch.cholesky_solve(Ph, Lc).diagonal().sum()
    ld_P = 2 * torch.log(Lc.diagonal()).sum()
    sign, ld_Ph = torch.linalg.slogdet(Ph)
    if sign <= 0:
        return float("inf")
    return float(0.5 * (tr - D + ld_P - ld_Ph)) / D     # per-dimension KL


def evaluate(Fm, evF, UF, UA, UG, S, N):
    Fh, Q = recon(UA, UG, S)
    out = {"rel_fro": float(torch.linalg.norm(Fm - Fh) / torch.linalg.norm(Fm))}
    for lam in LAMBDAS:
        out[f"kl@{lam:g}"] = laplace_kl(Fm, Fh, N, lam)
    for k in TOPK:
        if k <= Fm.shape[0]:
            out[f"align@{k}"] = align(UF[:, :k], topk_basis_hat(Q, S, k))
    return out


def kappa_eff(ev):
    """Condition number over the numerically non-zero spectrum (> 1e-10 * max)."""
    nz = ev[ev > 1e-10 * ev.max()]
    return float(nz.max() / nz.min())


# ----------------------------------------------------------------------------- curvlinops reference
def curvlinops_check(model, X, y, device, blocks):
    try:
        from curvlinops import KFACLinearOperator, EKFACLinearOperator
    except Exception as e:
        return {"status": f"skipped: {e}"}
    res = {}
    try:
        params = [model.fc1.weight, model.fc2.weight]
        data = [(X.to(device), y.to(device))]
        loss = nn.CrossEntropyLoss(reduction="mean")
        sizes = [p.numel() for p in params]
        offs = [0, sizes[0], sizes[0] + sizes[1]]
        for name, cls in (("kfac", KFACLinearOperator), ("ekfac", EKFACLinearOperator)):
            op = cls(model, loss, params, data, fisher_type="type-2", check_deterministic=False)
            dense = (op @ torch.eye(sum(sizes), device=device)).to(F64)
            for li, lname in enumerate(("fc1", "fc2")):
                ref = dense[offs[li]:offs[li + 1], offs[li]:offs[li + 1]]
                mine = blocks[lname][name]
                res[f"{name}_{lname}_rel_diff"] = float(torch.linalg.norm(ref - mine)
                                                        / torch.linalg.norm(ref))
        res["status"] = "ok"
    except Exception as e:
        res["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="digits", choices=["digits", "mnist"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-curvlinops", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or (300 if args.data == "digits" else 15)
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        t0 = time.perf_counter()
        X, y, Xv, yv = load_data(args.data, seed, args.data_root)
        torch.manual_seed(seed)
        model = MLP(X.shape[1], args.hidden).to(device)
        acc = train_map(model, X, y, Xv, yv, device, epochs, seed)
        print(f"\n=== {args.data}  d_in={X.shape[1]}  H={args.hidden}  seed={seed}  "
              f"N={X.shape[0]}  MAP val acc={acc*100:.1f}% ===", flush=True)
        stats = layer_stats(model, X, device)
        result = {"data": args.data, "hidden": args.hidden, "seed": seed,
                  "n_train": X.shape[0], "map_val_acc": acc, "layers": {}}
        blocks_fp32 = {}
        for lname, (a, g) in stats.items():
            Fm = exact_fisher(a, g)
            evF, UF = torch.linalg.eigh(Fm)
            evF, UF = evF.flip(0), UF.flip(1)
            N = a.shape[0]
            A64 = a.t() @ a / N
            Gr = g.reshape(-1, g.shape[-1])
            G64 = Gr.t() @ Gr / N
            eA, eG = torch.linalg.eigvalsh(A64), torch.linalg.eigvalsh(G64)
            L = {"dim": Fm.shape[0],
                 "kappa_A": kappa_eff(eA), "kappa_G": kappa_eff(eG),
                 "rank_A": int((eA > 1e-10 * eA.max()).sum()), "n_in": int(A64.shape[0]),
                 "rank_G": int((eG > 1e-10 * eG.max()).sum()), "n_out": int(G64.shape[0]),
                 "results": {}}
            # sanity 1: K-FAC == G (x) A; EK-FAC path with S forced to d_G (x) d_A reproduces it
            dA64, UA64 = torch.linalg.eigh(A64); dG64, UG64 = torch.linalg.eigh(G64)
            Fk_direct = torch.kron(G64, A64)
            Fk_kfe, Q64 = recon(UA64, UG64, dG64[:, None] * dA64[None, :])
            L["sanity_kfe_equals_kfac"] = float(torch.linalg.norm(Fk_direct - Fk_kfe)
                                                / torch.linalg.norm(Fk_direct))
            # sanity 2: EK-FAC S == diag(Q^T F Q)
            S_direct = torch.einsum("dk,de,ek->k", Q64, Fm, Q64).reshape(UG64.shape[0], -1)
            S_ek = ekfac_S(a, g, UA64, UG64, "fp32")
            L["sanity_ekfac_equals_diagQtFQ"] = float(torch.linalg.norm(S_direct - S_ek)
                                                      / torch.linalg.norm(S_direct))
            blocks_fp32[lname] = {"kfac": Fk_direct,
                                  "ekfac": recon(UA64, UG64, S_ek)[0]}
            variants = (("classic", classic_factors),
                        ("ifkfac", ifkfac_factors),
                        ("ifkfac_Rbf16", lambda a_, g_, p_, l_: ifkfac_factors(a_, g_, p_, l_, True)))
            for prec in ("fp32", "bf16"):
                for basis, fn in variants:
                    if basis == "ifkfac_Rbf16" and prec == "fp32":
                        continue                     # identical to ifkfac at fp32
                    log = {}
                    UA, dA, UG, dG = fn(a, g, prec, log)
                    r_k = evaluate(Fm, evF, UF, UA, UG, dG[:, None] * dA[None, :], N)
                    r_e = evaluate(Fm, evF, UF, UA, UG, ekfac_S(a, g, UA, UG, prec), N)
                    L["results"][f"kfac_{basis}/{prec}"] = {**r_k, **log}
                    L["results"][f"ekfac_{basis}/{prec}"] = r_e
            result["layers"][lname] = L
            print(f"  [{lname}] dim={L['dim']}  kappa_eff(A)={L['kappa_A']:.2e} "
                  f"(rank {L['rank_A']}/{L['n_in']})  kappa_eff(G)={L['kappa_G']:.2e} "
                  f"(rank {L['rank_G']}/{L['n_out']})  sanity: kfe==kfac "
                  f"{L['sanity_kfe_equals_kfac']:.1e}, ekfac==diag(QtFQ) "
                  f"{L['sanity_ekfac_equals_diagQtFQ']:.1e}", flush=True)
            kls = "  ".join(f"KL@{lam:<5g}" for lam in LAMBDAS)
            print(f"      {'method/prec':<20} rel_fro  align@32  {kls}  n_neg(A,G)", flush=True)
            for key, r in L["results"].items():
                kv = "  ".join(f"{r[f'kl@{lam:g}']:.3e}" for lam in LAMBDAS)
                nn_ = f"({r['n_neg_A']},{r['n_neg_G']})" if "n_neg_A" in r else ""
                print(f"      {key:<20} {r['rel_fro']:.4f}   {r.get('align@32', float('nan')):.4f}"
                      f"    {kv}  {nn_}", flush=True)
            if "eigh_bf16_native" in L["results"].get("kfac_classic/bf16", {}):
                print(f"      native bf16 eigh: {L['results']['kfac_classic/bf16']['eigh_bf16_native']}",
                      flush=True)
        if not args.no_curvlinops:
            result["curvlinops_reference"] = curvlinops_check(model, X, y, device, blocks_fp32)
            print(f"  curvlinops reference: {result['curvlinops_reference']}", flush=True)
        result["wall_s"] = time.perf_counter() - t0
        p = out_dir / f"fisher_approx_small_{args.data}_h{args.hidden}_seed{seed}.json"
        p.write_text(json.dumps(result, indent=2))
        print(f"  saved {p.name}  ({result['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
