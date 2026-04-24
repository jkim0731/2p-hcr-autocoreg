"""M3 — Mask-NCC warm-start + P1 centroid refinement (hybrid).

Run M1 (density-NCC coarse) to get a `(R, S, t)`, apply it to CZ
centroids, and pass those *as* `cz_init` into P1.  This is where the
"mask coarse gives scale; centroid refinement fixes local residual"
hybrid actually pays off.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult
from bench.candidate_impls._m1_mask_ncc import run_m1
from bench.candidate_impls._p1_teaser import run_p1
from lib.centroid_helpers import centroids_um


@register_candidate("M3")
def run_m3(s) -> CoregResult:
    r_m1 = run_m1(s)
    cz_init = None
    if r_m1.transform is not None:
        cz_um, _ = centroids_um(s, "cz")
        tr = r_m1.transform
        R = np.asarray(tr.R); S = np.asarray(tr.scales); t = np.asarray(tr.translation)
        src_mean = np.asarray(tr.src_mean)
        # M1's convention: hcr_pos = ((cz - src_mean) * S) @ R.T + t
        cz_init = ((cz_um - src_mean) * S) @ R.T + t
    r_p1 = run_p1(s, cz_init=cz_init)
    diag = dict(m1=r_m1.diagnostics, p1=r_p1.diagnostics,
                used_m1_warmstart=cz_init is not None)
    return CoregResult(
        pairs_df=r_p1.pairs_df, confidence=r_p1.confidence,
        transform=r_p1.transform, diagnostics=diag,
    )
