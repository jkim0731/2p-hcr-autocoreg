"""Import hub that pulls in every candidate module so their
`@register_candidate(...)` decorators execute.

Each candidate lives in ``bench/candidate_impls/<candidate_id>.py`` and
registers itself with :func:`harness.register_candidate`.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

_IMPL_PKG = "bench.candidate_impls"


def _load_all():
    try:
        pkg = importlib.import_module(_IMPL_PKG)
    except ModuleNotFoundError:
        return
    impl_dir = os.path.dirname(pkg.__file__)
    for _f, mod, _ in pkgutil.iter_modules([impl_dir]):
        importlib.import_module(f"{_IMPL_PKG}.{mod}")


_load_all()
