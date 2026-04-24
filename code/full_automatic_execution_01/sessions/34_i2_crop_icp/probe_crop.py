"""Probe: does restricting HCR GFP+ to an I2-centered XY window unlock 782149?

Rationale from S32/S33:
  - 782149 HCR GFP+ has 3831 cells; only ~608 (16 %) are within ±50 µm of
    the GT region. The other 84 % are scattered and form wrong basins.
  - I2 (S33) places CZ at ~340 µm median residual but close in XY for most
    subjects. Cropping HCR to an XY window around I2's estimate prunes the
    wrong basins.
  - If after crop the correct basin dominates, reciprocal-NN ICP should
    find it.

Plan for each subject:
  1. Run I2 → seed translation (xyz).
  2. Crop HCR GFP+ to cells within ±W µm of I2's XY (and within ±H µm Z).
     Sweep W ∈ {400, 600, 800, ∞} and report n_lt50 oracle.
  3. Feed (R=180°, seed_t = crop_center) into multi-start ICP; rank by
     self-supervised recip×unique; report oracle.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from lib.centroid_helpers import centroids_um  # noqa
from lib.sitk_wrapper import mi_affine  # noqa
from bench.candidate_impls._i1_axial_ncc import _load_cz_fullstack, _load_hcr_fullstack  # noqa
from anisotropic_icp import estimate_scales_icp_multi_start  # noqa


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


R_XYZ = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)


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
    return cz_um[:, [2, 1, 0]], hc_um[:, [2, 1, 0]]


def run_i2(s, target_um=8.0):
    cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
    hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
    cz_ds = zoom(cz_stack, (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um),
                 order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol, (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um),
                  order=1).astype(np.float32)
    return mi_affine(cz_ds, hcr_ds,
                     cz_xy_um=target_um, cz_z_um=target_um,
                     hcr_xy_um=target_um, hcr_z_um=target_um,
                     init_rotation_deg_z=180.0,
                     init_scale=(1.8, 1.8, 2.8),
                     n_iterations=300, pyramid_levels=(4, 2, 1))


def score_recip_unique(pred_xyz, hcr_xyz):
    th = cKDTree(hcr_xyz); tc = cKDTree(pred_xyz)
    d_c2h, idx_c2h = th.query(pred_xyz, k=1)
    _, idx_h2c = tc.query(hcr_xyz, k=1)
    recip = int(((idx_h2c[idx_c2h] == np.arange(len(pred_xyz))) & (d_c2h < 30)).sum())
    uniq = len(np.unique(idx_c2h)) / len(pred_xyz)
    return recip * uniq, recip, uniq


def oracle_lt50(fit, cz_gt_xyz, hcr_gt_xyz):
    pred = (cz_gt_xyz * fit.scales) @ fit.R.T + fit.translation
    d = np.linalg.norm(pred - hcr_gt_xyz, axis=1)
    return int((d < 50).sum()), float(np.median(d))


def main():
    rows = []
    for subj in ["788406", "755252", "767022", "782149"]:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        cz_um, _ = centroids_um(s, "cz")
        hcr_um, _ = centroids_um(s, "hcr_gfp")
        cz_xyz = cz_um[:, [2, 1, 0]]
        hcr_xyz = hcr_um[:, [2, 1, 0]]
        cz_gt, hcr_gt = gt_pairs(s)
        print(f"  n_cz={len(cz_xyz)}, n_hcr_gfp={len(hcr_xyz)}, n_gt={len(cz_gt)}")

        # Run I2
        t0 = time.time()
        r_i2 = run_i2(s)
        pred_zyx = r_i2.apply_inverse(cz_um)
        pred_xyz_i2 = pred_zyx[:, [2, 1, 0]]
        i2_center_xyz = pred_xyz_i2.mean(0)
        i2_median = float(np.median(np.linalg.norm(pred_xyz_i2[:len(cz_gt)] - hcr_gt, axis=1))) if len(cz_gt) else float('nan')
        print(f"  I2 ({time.time()-t0:.1f}s) center_xyz = {i2_center_xyz}; seed median to GT = {i2_median:.0f} µm")

        # Per-W crop sweep
        for W in [400, 600, 800, 1200, 9999]:
            mask = (np.abs(hcr_xyz[:, 0] - i2_center_xyz[0]) <= W) & \
                   (np.abs(hcr_xyz[:, 1] - i2_center_xyz[1]) <= W)
            hcr_cropped = hcr_xyz[mask]
            if len(hcr_cropped) < 100:
                print(f"  W={W}: only {len(hcr_cropped)} HCR cells — skip")
                continue
            # Seed with crop center (xy) + cz mean z after scale-guess
            # Use hcr_cropped XY centroid as translation seed
            seed_t = hcr_cropped.mean(0)
            f0 = _Fit(R=R_XYZ, src_mean=cz_xyz.mean(0),
                      translation=seed_t, scales=np.ones(3))
            best = dict(n=0, trim=None, rank=-1, sxy=float('nan'), sz=float('nan'), med=float('nan'))
            ss_best = dict(n=0, trim=None, rank=-1, sxy=float('nan'), sz=float('nan'), med=float('nan'))
            for trim in [0.4, 0.6, 0.8, 0.9]:
                try:
                    r = estimate_scales_icp_multi_start(cz_xyz, hcr_cropped, f0,
                                                        inlier_residual_quantile=trim)
                except Exception:
                    continue
                if r.fit is None:
                    continue
                pred = (cz_xyz * r.fit.scales) @ r.fit.R.T + r.fit.translation
                rank, recip, uniq = score_recip_unique(pred, hcr_cropped)
                n50, med = oracle_lt50(r.fit, cz_gt, hcr_gt)
                if n50 > best["n"]:
                    best = dict(n=n50, trim=trim, rank=rank, sxy=r.fit.scales[0], sz=r.fit.scales[2], med=med)
                if rank > ss_best["rank"]:
                    ss_best = dict(n=n50, trim=trim, rank=rank, sxy=r.fit.scales[0], sz=r.fit.scales[2], med=med)
            print(f"  W={W} (n_hcr={len(hcr_cropped)}):"
                  f" OR n<50={best['n']} trim={best['trim']} sxy={best['sxy']:.2f} sz={best['sz']:.2f} med={best['med']:.0f};"
                  f" SS n<50={ss_best['n']} trim={ss_best['trim']} rank={ss_best['rank']:.1f}")
            rows.append(dict(subject=subj, W=W, n_hcr=len(hcr_cropped),
                             or_n=best['n'], or_trim=best['trim'], or_sxy=best['sxy'], or_sz=best['sz'], or_median=best['med'],
                             ss_n=ss_best['n'], ss_trim=ss_best['trim'], ss_rank=ss_best['rank']))

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("/root/capsule/code/full_automatic_execution_01/sessions/34_i2_crop_icp/crop_sweep.csv", index=False)


if __name__ == "__main__":
    main()
