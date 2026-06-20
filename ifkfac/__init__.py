"""
IFKFAC — Inverse-Free K-FAC

A numerically stable, inverse-free K-FAC optimizer for PyTorch.

Public API
----------
- IFKFAC                    : the optimizer (drop-in replacement for ClassicKFAC)
- ClassicKFAC               : the textbook normal-equations K-FAC, for comparison
- LaplacePosterior          : Daxberger 2021 K-FAC Laplace approximation
- IFKFACRankError           : raised when p < n on a layer (rank-deficient X)

Usage
-----

    from ifkfac import IFKFAC
    optimizer = IFKFAC(model, lr=1e-3, damping=1e-2)
    for x, y in loader:
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
"""
from .ifkfac import IFKFAC
from .classic_kfac import ClassicKFAC
from .laplace import LaplacePosterior
from .activation_hooks import RawActivationHooks, IFKFACRankError

__version__ = "0.1.0"
__all__ = [
    "IFKFAC",
    "ClassicKFAC",
    "LaplacePosterior",
    "RawActivationHooks",
    "IFKFACRankError",
]
