"""Fair comparison: multi-start ICP with 6 default seeds vs 7 seeds (6 + I2).

For each subject, also try the I2-only seed. Rank every seed/trim combination
by recip×unique and report which won.
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
    return mi_affine(cz_ds, hcr_ds,
                     cz_xy_um=target_um, cz_z_um=target_um,
                     hcr_xy_um=target_um, hcr_z_um=target_um,
                     init_rotation_deg_z=180.0,
                     init_scale=(1.8, 1.8, 2.8),
                     n_iterations=300,
                     pyramid_levels=(4, 2, 1))


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


def i2_seed_t_xyz(s, cz_xyz):
    """Run I2 and return (seed_translation_xyz, i2_median_um)."""
    r = run_i2_fit(s)
    cz_um = cz_xyz[:, [2, 1, 0]]  # zyx
    pred_zyx = r.apply_inverse(cz_um)
    t_xyz = pred_zyx[:, [2, 1, 0]].mean(0)
    return t_xyz, r


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

        # Get I2 seed translation
        t_i2, r_i2 = i2_seed_t_xyz(s, cz_xyz)
        print(f"  I2 seed t_xyz = {t_i2}")

        # The 6 default translation seeds
        gfp_c = hcr_xyz.mean(0)
        seeds = {
            "hcr_gfp":    gfp_c,
            "gfp_dz+100": gfp_c + [0, 0, 100],
            "gfp_dz-100": gfp_c + [0, 0, -100],
            "gfp_dz+200": gfp_c + [0, 0, 200],
            "gfp_q25":    np.quantile(hcr_xyz, 0.25, axis=0),
            "gfp_q75":    np.quantile(hcr_xyz, 0.75, axis=0),
            "i2":         t_i2,
        }

        # Try each seed with trim ∈ {0.4, 0.6, 0.8, 0.9}
        best_ss = dict(rank=-1, n=0, seed=None, trim=None)
        best_or = dict(n=-1, seed=None, trim=None, median=float('nan'))
        for name, t_seed in seeds.items():
            for trim in [0.4, 0.6, 0.8, 0.9]:
                f0 = _Fit(R=R_XYZ, src_mean=cz_xyz.mean(0),
                          translation=np.asarray(t_seed, float),
                          scales=np.ones(3))
                try:
                    r = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f0,
                                                        inlier_residual_quantile=trim)
                except Exception as e:
                    continue
                if r.fit is None:
                    continue
                pred = (cz_xyz * r.fit.scales) @ r.fit.R.T + r.fit.translation
                rank, recip, uniq = score_recip_unique(pred, hcr_xyz)
                n50, med = oracle_lt50(r.fit, cz_gt, hcr_gt)
                if rank > best_ss["rank"]:
                    best_ss = dict(rank=rank, n=n50, seed=name, trim=trim, median=med)
                if n50 > best_or["n"]:
                    best_or = dict(n=n50, seed=name, trim=trim, median=med)
                if name == "i2":
                    print(f"  i2 trim={trim}: n<50={n50}, rank={rank:.1f}, med={med:.0f}, sxy={r.fit.scales[0]:.2f}, sz={r.fit.scales[2]:.2f}")
        print(f"  SS best: seed={best_ss['seed']}, trim={best_ss['trim']}, n<50={best_ss['n']}, median={best_ss.get('median', float('nan')):.1f}")
        print(f"  OR best: seed={best_or['seed']}, trim={best_or['trim']}, n<50={best_or['n']}, median={best_or['median']:.1f}")

        rows.append(dict(subject=subj,
                         ss_seed=best_ss["seed"], ss_trim=best_ss["trim"], ss_n50=best_ss["n"],
                         or_seed=best_or["seed"], or_trim=best_or["trim"], or_n50=best_or["n"],
                         or_median=best_or["median"]))

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("/root/capsule/code/full_automatic_execution_01/sessions/33_i2_repair/seven_seeds.csv", index=False)


if __name__ == "__main__":
    main()
