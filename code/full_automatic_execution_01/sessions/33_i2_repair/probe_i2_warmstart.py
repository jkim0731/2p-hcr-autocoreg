"""Test whether I2 provides a usable warm-start for ICP on the hard subjects.

Approach:
  1. Run I2 to get MIFitResult (A, t, c).
  2. Apply I2's inverse to CZ centroids → gives I2's estimate of where CZ
     lands in HCR µm.
  3. Compute the effective ``translation_xyz = pred_centroid - R*cz_centroid``
     in a Procrustes ``(R, scales, t)`` form that ICP accepts.
  4. Feed this as a single seed to ``estimate_scales_icp_multi_start`` and
     measure n_gt_lt50 vs default (hcr_gfp_centroid) seed.
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


def run_i2_fit(s, target_um=8.0):
    cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
    hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
    cz_ds = zoom(cz_stack, (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um),
                 order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol, (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um),
                  order=1).astype(np.float32)
    r = mi_affine(cz_ds, hcr_ds,
                  cz_xy_um=target_um, cz_z_um=target_um,
                  hcr_xy_um=target_um, hcr_z_um=target_um,
                  init_rotation_deg_z=180.0,
                  init_scale=(1.8, 1.8, 2.8),
                  n_iterations=300,
                  pyramid_levels=(4, 2, 1))
    return r


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

        # Run I2
        t0 = time.time()
        r = run_i2_fit(s)
        print(f"  I2 run ({time.time()-t0:.1f}s), converged={r.converged}, iter={r.iterations}")
        # Apply inverse (CZ zyx → HCR zyx)
        pred_zyx_i2 = r.apply_inverse(cz_gt[:, [2, 1, 0]])
        pred_xyz_i2 = pred_zyx_i2[:, [2, 1, 0]]
        d_i2 = np.linalg.norm(pred_xyz_i2 - hcr_gt, axis=1)
        print(f"  I2 raw: median={np.median(d_i2):.1f} µm, n<50={int((d_i2<50).sum())}, n<100={int((d_i2<100).sum())}")

        # Build I2 warm-start fit for ICP. Apply I2's inverse to all CZ to get
        # the predicted HCR positions; then compute a translation seed.
        pred_all_zyx = r.apply_inverse(cz_um)
        pred_all_xyz = pred_all_zyx[:, [2, 1, 0]]
        # With R_xyz and scales from I2:
        A_zyx_inv = np.linalg.inv(r.affine_matrix)
        U, S_vals, Vt = np.linalg.svd(A_zyx_inv)
        R_zyx_ortho = U @ Vt
        if np.linalg.det(R_zyx_ortho) < 0:
            # SVD gave improper rotation; flip last col of U.
            U[:, -1] = -U[:, -1]
            R_zyx_ortho = U @ Vt
        # Convert R_zyx → R_xyz
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
        R_xyz_i2 = P @ R_zyx_ortho @ P.T
        scales_xyz_i2 = np.array([S_vals[2], S_vals[1], S_vals[0]])  # zyx→xyz

        # Translation seed: midpoint of pred_all_xyz
        seed_t_i2 = pred_all_xyz.mean(0)
        print(f"  I2 seed t_xyz = {seed_t_i2}")
        print(f"  Default hcr_gfp t_xyz = {hcr_xyz.mean(0)}")
        print(f"  I2 R_xyz ≈ {np.round(R_xyz_i2, 3)}")
        print(f"  I2 scales_xyz ≈ {np.round(scales_xyz_i2, 3)}")

        # A) Baseline: multi-start ICP with default seeds
        f_default = _Fit(R=R_XYZ, src_mean=cz_xyz.mean(0),
                         translation=hcr_xyz.mean(0), scales=np.ones(3))
        res_default = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f_default)
        n50_d, med_d = oracle_lt50(res_default.fit, cz_gt, hcr_gt) if res_default.fit else (0, float('nan'))
        print(f"  Default ICP: n<50={n50_d}, median={med_d:.1f}, sxy={res_default.sxy:.2f}, sz={res_default.sz:.2f}")

        # B) I2-seeded single-start
        f_i2 = _Fit(R=R_xyz_i2, src_mean=cz_xyz.mean(0),
                    translation=seed_t_i2, scales=np.ones(3))
        # Multi-start with I2-seed means forcing the initial t; ICP will keep
        # R fixed and refine scales+t via the Procrustes inner loop.
        res_i2 = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f_i2)
        n50_i, med_i = oracle_lt50(res_i2.fit, cz_gt, hcr_gt) if res_i2.fit else (0, float('nan'))
        print(f"  I2-seed ICP: n<50={n50_i}, median={med_i:.1f}, sxy={res_i2.sxy:.2f}, sz={res_i2.sz:.2f}")

        # C) I2-seed with multiple trim levels
        best_i2_trim = dict(n=0, trim=None)
        for trim in [0.4, 0.6, 0.8, 0.9]:
            try:
                res_it = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f_i2,
                                                          inlier_residual_quantile=trim)
                if res_it.fit is None:
                    continue
                n, med = oracle_lt50(res_it.fit, cz_gt, hcr_gt)
                if n > best_i2_trim["n"]:
                    best_i2_trim = dict(n=n, trim=trim, med=med)
            except Exception as e:
                pass
        print(f"  I2-seed+trim best: n<50={best_i2_trim.get('n')}, trim={best_i2_trim.get('trim')}")

        rows.append(dict(subject=subj,
                         i2_median=float(np.median(d_i2)), i2_n_lt50=int((d_i2<50).sum()),
                         default_n50=n50_d, default_median=med_d,
                         i2_seed_n50=n50_i, i2_seed_median=med_i,
                         i2_trim_n50=best_i2_trim.get("n"), i2_trim_level=best_i2_trim.get("trim")))

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("/root/capsule/code/full_automatic_execution_01/sessions/33_i2_repair/warmstart.csv", index=False)


if __name__ == "__main__":
    main()
