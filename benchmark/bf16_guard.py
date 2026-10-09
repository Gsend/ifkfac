"""
benchmark/bf16_guard.py  --  prove that a code region never materializes an fp32 tensor.

    from benchmark.bf16_guard import Bf16Guard
    with Bf16Guard() as g:
        loss = loss_fn(model(x), y); loss.backward(); opt.step()
    g.report()          # lists every op that produced a float tensor that is not bf16
    assert g.clean

The guard is a TorchDispatchMode: it sees every ATen operation executed while
it is active - the forward pass, the autograd backward pass, hook code and the
optimizer step, including ops issued from inside library code (SINGD, ASDL,
torch.optim) - and records any op whose output contains a floating-point
tensor whose dtype is not bfloat16 (fp16, fp32, fp64).  Integer and bool
tensors (token ids, indices, masks) and Python scalars (`.item()`) are
allowed.  For each violating (op, dtype) pair it keeps the Python call site
of its first occurrence, so the upcast can be found in the source.

What "bf16 all the way" means here: every tensor that exists between two
operations is bf16.  Inside a single GPU kernel, bf16 matmuls and reductions
still accumulate in fp32 and round the result to bf16 - that is how every
bf16 kernel works on current hardware (tensor cores, cuBLAS, PyTorch's
reduction kernels); it is not visible as an fp32 tensor and is not flagged.

Mixed precision (fp32 master weights, bf16 autocast forward/backward) uses
the guard per region instead of per step: only the K-FAC code runs under it
(the capture hooks and the optimizer step), and two options describe the
boundary with the fp32 model:
  ignore_views=True   view / alias ops (detach, reshape, transpose, ...) of an
                      fp32 input produce no new data; they are counted in
                      .views, not flagged
  allow=[(func, op)]  a non-bf16 output of op `op` ("*" = any) issued directly
                      by the Python function `func` (innermost frame outside
                      torch) is counted in .allowed, not flagged - used for
                      the update of the fp32 master weights
"""
from __future__ import annotations

import traceback
import sys
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

_IGNORE_FRAMES = ("bf16_guard.py", "torch/_dynamo", "torch/_compile.py", "torch/utils/_python_dispatch.py", "torch/_ops.py",
                  "torch/_tensor.py", "torch/autograd/", "torch/nn/modules/module.py")


def _call_site(limit: int = 6) -> str:
    frames = [f for f in traceback.extract_stack()[:-2]
              if not any(s in f.filename.replace("\\", "/") for s in _IGNORE_FRAMES)]
    return " <- ".join(f"{f.filename.replace(chr(92), '/').split('/')[-1]}:{f.lineno} {f.name}"
                       for f in reversed(frames[-limit:]))


def _innermost_user_frame() -> str:
    """Name of the innermost Python function outside torch and this file."""
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename.replace("\\", "/")
        if "/torch/" not in fn and not fn.endswith("bf16_guard.py"):
            return f.f_code.co_name
        f = f.f_back
    return ""


class Bf16Guard(TorchDispatchMode):
    def __init__(self, name: str = "", allow=(), ignore_views: bool = False):
        super().__init__()
        self.name = name
        self.allow = {(fn, op) for fn, op in allow}
        self.ignore_views = ignore_views
        self.violations: Counter = Counter()
        self.allowed: Counter = Counter()
        self.views: Counter = Counter()
        self.sites: dict = {}
        self.n_ops = 0

    def _permitted(self, op: str):
        if not self.allow:
            return None
        fn = _innermost_user_frame()
        if (fn, op) in self.allow or (fn, "*") in self.allow:
            return fn
        return None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.n_ops += 1
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != torch.bfloat16:
                op = str(func.overloadpacket.__name__)
                dt = str(t.dtype).replace("torch.", "")
                if self.ignore_views and getattr(func, "is_view", False):
                    self.views[(op, dt)] += 1
                    continue
                fn = self._permitted(op)
                if fn is not None:
                    self.allowed[(fn, op, dt)] += 1
                    continue
                key = (op, dt)
                self.violations[key] += 1
                if key not in self.sites:
                    self.sites[key] = _call_site()
        return out

    @property
    def clean(self) -> bool:
        return not self.violations

    def report(self, max_rows: int = 40) -> str:
        head = f"[bf16 guard{' ' + self.name if self.name else ''}] {self.n_ops} ops, "
        if self.clean:
            s = head + "no non-bf16 floating tensor produced"
        else:
            rows = [f"  {n:>6}x  {op:<32} -> {dt:<8} first at: {self.sites[(op, dt)]}"
                    for (op, dt), n in self.violations.most_common(max_rows)]
            s = head + f"{sum(self.violations.values())} non-bf16 outputs:\n" + "\n".join(rows)
        if self.allowed:
            s += "\n  permitted (" + ", ".join(f"{fn}: {op}->{dt} x{n}"
                                              for (fn, op, dt), n in sorted(self.allowed.items())) + ")"
        if self.views:
            s += f"\n  {sum(self.views.values())} view/alias ops of non-bf16 inputs (no data produced)"
        print(s, flush=True)
        return s
