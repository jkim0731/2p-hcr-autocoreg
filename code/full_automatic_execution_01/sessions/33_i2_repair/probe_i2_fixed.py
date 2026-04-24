"""Verify the repaired I2 wrapper: does TransformDescriptor now correctly map
CZ centroids to HCR coords? And does MI converge at all on 782149?

Two checks per subject:
  1. Apply MIFitResult.apply_inverse(cz_gt) vs hcr_gt.
  2. Apply TransformDescriptor forward convention to cz_gt vs hcr_gt.
Both should now yield small median residuals if MI converges.
"""
from __future__ import annotations

import sys
import time
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from bench.candidate_impls._i1_axial_ncc import _load_cz_fullstack, _load_hcr_fullstack  # noqa
from lib.sitk_wrapper import mi_affine  # noqa


def gt_pairs(s):
    ct = s.coreg_table
    cz = s.cz_centroids.set_index("cz_id")
    hc = s.hcr_centroids.set_index("hcr_id")
    mask = ct["cz_id"].isin(cz.index) & ct["hcr_id"].isin(hc.index)
    ct = ct[mask]
    cz_rows = cz.loc[ct["cz_id"].values]
    hc_rows = hc.loc[ct["hcr_id"].values]
    cz_um = cz_px_to_um(cz_rows[["z_px", "y_px", "x_px"]].values, s)
    hc_um = hcr_px_to_um(hc_rows[["z_px", "y_px", "x_px"]].values, s)
    return cz_um, hc_um  # zyx


def apply_td(td, pts_zyx):
    """Replicate the TransformDescriptor forward:
       pred = ((src - src_mean) * scales) ... actually Procrustes convention varies.
       We test multiple conventions and report the best.
    """
    src = np.asarray(pts_zyx)
    src_mean = td.src_mean
    R = td.R
    S = np.asarray(td.scales)
    t = td.translation  # dst_mean in our convention

    conventions = {
        "P1: ((src-sm)@R)*S+t": ((src - src_mean) @ R) * S + t,
        "P2: ((src-sm)*S)@R+t": ((src - src_mean) * S) @ R + t,
        "P3: ((src-sm)*S)@R.T+t": ((src - src_mean) * S) @ R.T + t,
        "P4: ((src-sm)@R.T)*S+t": ((src - src_mean) @ R.T) * S + t,
    }
    return conventions


def main():
    for subj in ["788406", "782149", "755252", "767022"]:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        cz_gt, hcr_gt = gt_pairs(s)
        print(f"  cz_gt[0] zyx = {cz_gt[0]}; hcr_gt[0] zyx = {hcr_gt[0]}")

        cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
        hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
        target_um = 8.0
        cz_ds = zoom(cz_stack, (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um),
                     order=1).astype(np.float32)
        hcr_ds = zoom(hcr_vol, (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um),
                      order=1).astype(np.float32)

        t0 = time.time()
        r = mi_affine(
            cz_ds, hcr_ds,
            cz_xy_um=target_um, cz_z_um=target_um,
            hcr_xy_um=target_um, hcr_z_um=target_um,
            init_rotation_deg_z=180.0,
            init_scale=(1.8, 1.8, 2.8),
            n_iterations=200,
            pyramid_levels=(4, 2, 1),
        )
        print(f"  MI converged={r.converged} metric={r.metric:.4f} iter={r.iterations} ({time.time()-t0:.1f}s)")
        print(f"  A (zyx)=\n{r.affine_matrix}")
        print(f"  t (zyx)={r.translation}; c (zyx)={r.center}")

        # Direct inverse via wrapper
        pred = r.apply_inverse(cz_gt)
        d = np.linalg.norm(pred - hcr_gt, axis=1)
        print(f"  apply_inverse: median={np.median(d):.1f} µm, n<50={int((d<50).sum())}, n<100={int((d<100).sum())}")


if __name__ == "__main__":
    main()
