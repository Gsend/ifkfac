"""pytest sys.path setup: put the repository root on sys.path so tests can
import the ``benchmark`` and ``optimizer`` packages used by the paper scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
