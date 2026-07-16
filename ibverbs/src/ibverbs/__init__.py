"""ibverbs: low-level Pythonic bindings for libibverbs (RDMA).

This package exposes RDMA verbs as RAII Python objects. See the README for a
quickstart. The compiled fast paths live in :mod:`ibverbs._ibverbs`; Pythonic
enums live in :mod:`ibverbs.enums` and thin RC helpers in
:mod:`ibverbs.helpers`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import _ibverbs  # noqa: F401  (re-exported for `ibverbs._ibverbs`)

__all__ = ["__version__"]
