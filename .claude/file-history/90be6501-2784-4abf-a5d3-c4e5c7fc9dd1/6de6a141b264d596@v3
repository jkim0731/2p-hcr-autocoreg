"""C1 — Image-coarse (I2) → centroid-refine (P1) hybrid.

Run I2 to get an MI-affine; apply its inverse (CZ→HCR direction) to the CZ
centroids; pass the warped CZ points as P1's `cz_init` warm-start.  If I2
fails, fall back to plain P1 (which uses its own default warm-start).
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
from lib.centroid_helpers import centroids_um


def _apply_sitk_inverse(pts_zyx, A, t, c):
    """SITK inverse (moving→fixed, CZ→HCR)."""
    A_inv = np.linalg.inv(A)
    return ((pts_zyx - c - t) @ A_inv.T) + c


@register_candidate("C1")
def run_c1(s, *, K: int = 30, c_bar: float = 40.0) -> CoregResult:
    from bench.candidate_impls._i2_sitk_affine import run_i2
    from bench.candidate_impls._p1_teaser import run_p1

    r_i2 = run_i2(s)
    cz_init = None
    i2_scales = None
    if r_i2.transform is not None:
        try:
            A = np.asarray(r_i2.diagnostics["sitk_matrix_zyx"])
            t_v = np.asarray(r_i2.diagnostics["sitk_translation_zyx"])
            c_v = np.asarray(r_i2.diagnostics["sitk_center_zyx"])
            cz_um, _ = centroids_um(s, "cz")
            cz_init = _apply_sitk_inverse(cz_um, A, t_v, c_v)
            i2_scales = r_i2.transform.scales.tolist()
        except Exception as e:
            cz_init = None
            i2_scales = f"warmstart_failed: {e}"

    # Widen K and inlier threshold because I2 residual (~100–500 µm) is larger
    # than the default P1 NN radius of ~20 µm; otherwise GT cells fall outside
    # the putative-K list and TEASER refit drifts further.
    r_p1 = run_p1(s, cz_init=cz_init, K=K, c_bar=c_bar)
    return CoregResult(
        pairs_df=r_p1.pairs_df,
        confidence=r_p1.confidence,
        transform=r_p1.transform,
        diagnostics={**r_p1.diagnostics,
                     "i2_transform_scales": i2_scales,
                     "i2_metric": r_i2.diagnostics.get("metric"),
                     "c1_K": K,
                     "c1_c_bar": c_bar,
                     "used_i2_warmstart": cz_init is not None},
    )
