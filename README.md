# IFKFAC — Inverse-Free K-FAC

A numerically stable, inverse-free K-FAC optimizer for PyTorch. Replaces the
Gram-matrix inversion at the heart of Classic K-FAC with a streaming TSQR
factor and four triangular solves — the textbook QR-stable OLS recipe
(Householder 1958; Golub 1965).

The result: **one order of κ better stability** than Classic K-FAC.
Concretely, at bf16 with the typical neural-network conditioning κ(X) ~ 10²,
Classic's error term is O(κ² · ε_bf16) ≈ 40 (saturated, preconditioner
destroyed), while IFKFAC's is O(κ · ε_bf16) ≈ 0.4 (preconditioner intact).

This is what makes the difference between K-FAC working at bf16 and not.

## Why use IFKFAC

K-FAC is the standard Kronecker-factored Fisher approximation underlying
several modern applications: Bayesian deep learning via Laplace approximation
(Daxberger et al. 2021), influence-function analysis at LLM scale (Grosse et
al. 2023), Elastic Weight Consolidation for continual learning (Kirkpatrick
et al. 2017), and several optimization domains where Adam-family methods
struggle (PINNs, deep autoencoders, variational quantum chemistry, RL).

The standard K-FAC implementation collapses under bf16 storage. IFKFAC
preserves K-FAC's algorithmic structure in modern mixed-precision pipelines.

## Installation

```bash
git clone <repo-url>
cd IFKFAC
pip install -e .
# or, with test deps:
pip install -e ".[test,demo]"
```

Requires PyTorch ≥ 2.0.

> **Not yet on PyPI.** `pip install ifkfac` does not work today — install from
> source as above. See [Publishing to PyPI](#publishing-to-pypi) for what it
> would take to make `pip install ifkfac` available.

## Quick start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ifkfac import IFKFAC

model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).cuda()
optimizer = IFKFAC(model, lr=1e-3, damping=1e-2)

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The same three hyperparameters as Classic K-FAC: `lr`, `damping`, and
`factor_update_freq`. No new tuning burden.

## Training in fp32 or bf16

Precision is controlled by a single constructor flag, `use_true_bf16`. In both
modes the model's **master weights stay in fp32** — the flag only changes the
dtype in which IFKFAC stores its `R` factors and runs the triangular solves.
The natural gradient is always cast back to fp32 before the weight update, so
this is the standard mixed-precision pattern (bf16 curvature, fp32 weights).

### fp32 (default)

```python
from ifkfac import IFKFAC

model = build_model().cuda()                 # fp32 parameters
optimizer = IFKFAC(model, lr=1e-3, damping=1e-2)   # use_true_bf16=False (default)

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### bf16

Keep the model in fp32 and set `use_true_bf16=True`. IFKFAC then stores `R_X`
and `R_G` as bfloat16 and routes QR / triangular solves through the hand-rolled
bf16 primitives in `ifkfac.bf16_linalg` (cuSOLVER lacks bf16 `geqrf` /
`triangular_solve` / `cholesky`). This is the regime where IFKFAC's `O(κ · ε)`
stability matters — Classic K-FAC's `O(κ² · ε)` error saturates here.

```python
from ifkfac import IFKFAC

model = build_model().cuda()                 # master weights remain fp32
optimizer = IFKFAC(
    model,
    lr=1e-3,
    damping=1e-2,
    use_true_bf16=True,                      # store R factors as bf16 — saves memory
)

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    # Optional: run the forward/backward under bf16 autocast as usual.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The two modes take identical hyperparameters — switching precision is just the
one flag, no re-tuning of `lr` / `damping` / `factor_update_freq`.

To see both side by side on a small problem:

```bash
python demo/demo_train_mlp.py --precision all     # fp32 and bf16 cells
```

## Usage patterns

### 1. As a drop-in optimizer (bf16-stable)

```python
optimizer = IFKFAC(
    model,
    lr=1e-3,
    damping=1e-2,
    factor_update_freq=20,   # refresh R factors every 20 steps
    momentum=0.9,
    use_true_bf16=True,      # store R factors as bf16 — saves memory
)
```

### 2. For Bayesian deep learning (K-FAC Laplace, §5.8 of the paper)

```python
from ifkfac import IFKFAC, LaplacePosterior

# Phase 1: train to MAP with your favorite optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
# ... standard training loop ...

# Phase 2: one extra pass over train data to capture K-FAC Fisher
kfac = IFKFAC(model, lr=0, damping=1e-3)
model.eval()
for x, y in train_loader:
    logits = model(x)
    # True Fisher: MC-sampled labels (Daxberger 2021 §3.1)
    probs = torch.softmax(logits.float(), dim=-1)
    y_sampled = torch.multinomial(probs, 1).squeeze(-1)
    loss = F.cross_entropy(logits, y_sampled)
    model.zero_grad(set_to_none=True)
    loss.backward()
factors = kfac.hooks.get_factors()
kfac.hooks.remove()

# Phase 3: form posterior, draw predictive samples
posterior = LaplacePosterior(
    model, factors_to_evd(factors),
    prior_precision=10.0, dataset_size=len(train_set),
)
probs, labels = posterior.predictive_full_loader(test_loader, device, n_samples=30)
```

### 3. For research comparisons against Classic K-FAC

```python
from ifkfac import IFKFAC, ClassicKFAC

# Same hyperparameters, drop in either to compare:
opt_ifkfac  = IFKFAC(model, lr=1e-3, damping=1e-2)
opt_classic = ClassicKFAC(model, lr=1e-3, damping=1e-2)
```

## How it works

For a layer with input `X ∈ ℝ^(p × n_in)` and back-propagated output gradient
`δ ∈ ℝ^(p × n_out)`, K-FAC approximates the per-layer Fisher as `A ⊗ G` where
`A = XᵀX / p` and `G = δᵀδ / p`. The natural-gradient update is

    ΔW = G⁻¹ · ∇W ℓ · A⁻¹

Classic K-FAC forms `A` and `G` explicitly and inverts them — that doubles
the condition number (`κ(XᵀX) = κ(X)²`) and the subsequent inversion adds
another factor of `κ(A)`. Total error: `O(κ(X)² · ε)`.

IFKFAC uses streaming TSQR to compute `R` factors such that `RᵀR = A`,
*without ever forming* `A`. The natural gradient is then applied via four
triangular solves with `R`. Total error: `O(κ(X) · ε)` — one order of κ
better, structurally.

See `paper.md` (or the paper) for the full theoretical analysis (Higham 2002,
Theorems 19.10 and 20.3).

## Tests

```bash
pytest tests/
```

Three test groups:
- `test_basic.py` — IFKFAC reduces loss on a small MLP (smoke test)
- `test_equivalence.py` — IFKFAC at fp32 reproduces Classic K-FAC at fp32 to
  numerical precision (the algorithms compute the same gradient when noise is
  negligible — this verifies IFKFAC isn't doing something subtly different)
- `test_bf16_stability.py` — the κ²/κ contrast: synthetic κ-sweep showing
  Classic's error saturates at high κ while IFKFAC's stays bounded

## Demo

```bash
python demo/demo_train_mlp.py --precision bf16
```

Trains a 4-layer MLP on synthetic data with both IFKFAC and Classic K-FAC at
bf16. Reproduces the headline κ²/κ contrast on a small problem in ~30 seconds.

## Publishing to PyPI

To make the library installable with `pip install ifkfac`, the following needs
to happen. The packaging is already 90% there — `pyproject.toml` uses a
standard PEP 621 layout with the setuptools backend — so most of this is
process, not code.

**1. Confirm the distribution name is available.** Check
[pypi.org/project/ifkfac](https://pypi.org/project/ifkfac/). If `ifkfac` is
taken, pick another `name` in `pyproject.toml` (the import package stays
`ifkfac` regardless).

**2. Tighten `pyproject.toml` metadata** so the PyPI project page renders well
and installs resolve correctly:
- Fill in real `authors = [{ name = "...", email = "..." }]` (currently the
  citation and author fields are placeholders).
- Add `[project.urls]` (Homepage / Repository / Issues) — these become the
  sidebar links on the PyPI page.
- Set `readme = "README.md"` (already present) so this file becomes the long
  description.
- Consider adding `"Development Status"`, `"Operating System"`, and Python
  minor-version classifiers.

**3. Decide the Torch dependency story.** `torch>=2.0` is fine as a floor, but
PyPI wheels of torch are CPU/CUDA-variant specific. Leave the dependency loose
and document that GPU users should install the matching torch build from the
official index first. Don't pin a `+cuXXX` build in `dependencies`.

**4. Build the distributions:**

```bash
python -m pip install --upgrade build twine
python -m build            # produces dist/ifkfac-0.1.0-py3-none-any.whl + .tar.gz
twine check dist/*         # validates metadata / long-description rendering
```

**5. Upload — test first, then real:**

```bash
twine upload --repository testpypi dist/*    # dry run on test.pypi.org
pip install -i https://test.pypi.org/simple/ ifkfac   # verify the install
twine upload dist/*                          # publish to the real PyPI
```

Use a PyPI **API token** (recommended) or, better, configure
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) so a GitHub
Actions workflow can publish on tagged releases without storing secrets.

**6. (Recommended) Automate releases.** Add a GitHub Actions workflow that, on
a version tag (e.g. `v0.1.0`), runs `python -m build` and `pypi-publish` via
OIDC Trusted Publishing. Bump `version` in `pyproject.toml` for every release —
PyPI refuses to overwrite an already-published version.

## Citation

If you use IFKFAC in research, please cite:

```
@article{ifkfac2026,
  title  = {Inversion-Free K-FAC: Stability and Robustness via QR-Stable OLS},
  author = {...},
  year   = {2026},
}
```

## License

MIT (see [LICENSE](LICENSE)).
