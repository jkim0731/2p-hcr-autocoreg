"""Inspect ICP fit R, scales, translation vs landmark fit on 788406.

The ICP translation is seeded with HCR-centroid, but the CZ sub-ROI does
not necessarily land at the HCR centroid — it's a cortical slab sub-ROI,
and the HCR centroid is the mean of *all* GFP+ cells across the whole
cortex.  This is likely the source of the ~150 um offset vs the LM fit.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject, landmark_pairs_um
from benchmark_analysis import fit_anisotropic_similarity
from lib.centroid_helpers import centroids_um
from anisotropic_icp import estimate_scales_icp_multi_start


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


def main(sid="788406"):
    s = load_subject(sid)
    cz_zyx, _ = centroids_um(s, "cz")
    hcr_zyx, _ = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_zyx[:, [2, 1, 0]]
    hcr_xyz = hcr_zyx[:, [2, 1, 0]]

    R0 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)  # 180° Z (xyz)
    fit0 = _Fit(R=R0, src_mean=cz_xyz.mean(0), translation=hcr_xyz.mean(0),
                scales=np.ones(3))
    res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
    fit = res.fit
    print("ICP fit:")
    print(f"  R = \n{fit.R}")
    print(f"  scales (xyz) = {fit.scales}")
    print(f"  translation (xyz) = {fit.translation}")
    print(f"  rms_um = {fit.rms_um:.2f}")
    print(f"  iters = {res.iterations}, matched = {res.n_matched}")

    # Landmark fit for comparison (zyx)
    cz_lm_xyz, hcr_lm_xyz = landmark_pairs_um(s)
    cz_lm_zyx = cz_lm_xyz[:, [2, 1, 0]]
    hcr_lm_zyx = hcr_lm_xyz[:, [2, 1, 0]]
    fit_lm = fit_anisotropic_similarity(cz_lm_zyx, hcr_lm_zyx)
    print()
    print("LM fit (zyx convention):")
    print(f"  R = \n{fit_lm.R}")
    print(f"  scales (zyx) = {fit_lm.scales}")
    print(f"  translation (zyx) = {fit_lm.translation}")
    print(f"  rms_um = {fit_lm.rms_um:.2f}")

    # CZ cloud mean vs HCR cloud mean vs landmarked-CZ dst mean
    print()
    print(f"CZ cloud centroid (xyz):  {cz_xyz.mean(0)}")
    print(f"HCR cloud centroid (xyz): {hcr_xyz.mean(0)}")
    print(f"LM hcr centroid (xyz):    {hcr_lm_xyz.mean(0)}")
    print(f"Offset HCR-all vs LM HCR (xyz, um): {hcr_xyz.mean(0) - hcr_lm_xyz.mean(0)}")


if __name__ == "__main__":
    main()
