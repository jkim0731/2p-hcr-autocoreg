"""Analyse why R1 translation errs: compare against landmark centroids + GT center."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, depth_from_surface
from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject, landmark_pairs_um
from r1_coarse_align import (
    PRIOR_XY_EXPANSION,
    PRIOR_Z_EXPANSION,
    _rotation_about_z,
    coarse_align,
)


def main(subjects):
    print(f"{'sid':>7}  {'gt_trg_xy':>18}  {'hcr_gfp_c_xy':>18}  {'hcr_lm_c_xy':>18}  "
          f"{'r1_tx_xy':>18}  {'centroid_err':>12}  {'landmark_err':>12}")
    for sid in subjects:
        s = load_subject(sid)
        info = analyze_subject(s)
        cz_xyz = info["cz_xyz"]
        gfp_xyz = info["gfp_xyz"]
        cz_surface = info["cz_surface"]
        hcr_surface = info["hcr_surface"]

        cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
        gt = info["procrustes"]
        cz_center = cz_xyz.mean(axis=0)
        src_mean = cz_lm.mean(axis=0)
        dst_mean = hcr_lm.mean(axis=0)
        gt_target = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean

        fit = coarse_align(cz_xyz, gfp_xyz, cz_surface, hcr_surface)
        r1_t = fit.translation

        hcr_gfp_c = gfp_xyz.mean(axis=0)

        d_centroid = float(np.linalg.norm(hcr_gfp_c[:2] - gt_target[:2]))
        d_lm = float(np.linalg.norm(hcr_lm.mean(axis=0)[:2] - gt_target[:2]))

        print(f"{sid:>7}  "
              f"({gt_target[0]:6.0f},{gt_target[1]:6.0f})  "
              f"({hcr_gfp_c[0]:6.0f},{hcr_gfp_c[1]:6.0f})  "
              f"({hcr_lm.mean(axis=0)[0]:6.0f},{hcr_lm.mean(axis=0)[1]:6.0f})  "
              f"({r1_t[0]:6.0f},{r1_t[1]:6.0f})  "
              f"{d_centroid:12.0f}  {d_lm:12.0f}")

        # Diagnostics on HCR GFP+ extent to understand asymmetry
        print(f"   HCR GFP+ extent x=[{gfp_xyz[:,0].min():.0f},{gfp_xyz[:,0].max():.0f}] "
              f"y=[{gfp_xyz[:,1].min():.0f},{gfp_xyz[:,1].max():.0f}]   "
              f"GT_tgt_x offset from x-center: {gt_target[0] - 0.5*(gfp_xyz[:,0].min()+gfp_xyz[:,0].max()):.0f}")

        # What's the CZ footprint extent after prior transform?
        R_prior = _rotation_about_z(180.0)
        scales_prior = np.array([PRIOR_XY_EXPANSION, PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION])
        cz_prior = (cz_xyz - cz_center) @ R_prior * scales_prior
        cz_ext_x = cz_prior[:,0].max() - cz_prior[:,0].min()
        cz_ext_y = cz_prior[:,1].max() - cz_prior[:,1].min()
        hcr_ext_x = gfp_xyz[:,0].max() - gfp_xyz[:,0].min()
        hcr_ext_y = gfp_xyz[:,1].max() - gfp_xyz[:,1].min()
        print(f"   CZ prior extent: ({cz_ext_x:.0f},{cz_ext_y:.0f})   "
              f"HCR GFP+ extent: ({hcr_ext_x:.0f},{hcr_ext_y:.0f})")


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["788406", "790322", "767018", "782149", "755252", "767022"]
    main(subjects)
