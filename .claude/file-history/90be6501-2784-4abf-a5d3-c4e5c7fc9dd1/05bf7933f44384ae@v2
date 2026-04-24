"""S40 — P1 multi-start with I2-crop + default warmstart seeds.

S39 showed P1 at default converges to wrong basin on 755252, 767022, 782149.
Default P1 uses default_warmstart_zyx (6-translation grid multi-start) and
still fails on stress subjects. S34 showed that I2-crop W=600 unlocks 755252
on ICP (+53 n_lt50 over uncropped). Hypothesis: if P1 is fed multiple seed
cz_init arrays and ranked by a no-peek SS score, the I2-crop seed may rescue
stress subjects without regressing 788406.

Seeds tried per subject:
  (A) default     — P1's default_warmstart_zyx (6-translation grid).
  (B) i2_direct   — apply I2 MI-affine directly to CZ centroids (zyx).
  (C) i2_crop600  — I2-predicted CZ center → crop HCR to ±600 µm XY → cz_init
                    = 180°-about-Z rotated CZ + cropped-HCR XY centroid.

SS score: number of pairs with residual_um < 30 from run_p1's pairs_df. Winner
= max SS. GT metrics (r@5, r@10, r@20, recall_id) reported for validation only.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401  (registers all)
from benchmark_data_loader import load_subject  # noqa
from lib.centroid_helpers import centroids_um  # noqa
from lib.sitk_wrapper import mi_affine  # noqa
from bench.candidate_impls._p1_teaser import run_p1  # noqa
from bench.candidate_impls._i1_axial_ncc import _load_cz_fullstack, _load_hcr_fullstack  # noqa
from bench.harness import compare_to_gt  # noqa


def i2_estimate(s, target_um=8.0):
    cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
    hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
    cz_ds = zoom(cz_stack,
                 (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um),
                 order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol,
                  (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um),
                  order=1).astype(np.float32)
    return mi_affine(cz_ds, hcr_ds,
                     cz_xy_um=target_um, cz_z_um=target_um,
                     hcr_xy_um=target_um, hcr_z_um=target_um,
                     init_rotation_deg_z=180.0,
                     init_scale=(1.8, 1.8, 2.8),
                     n_iterations=200, pyramid_levels=(4, 2, 1))


def build_i2_direct_seed(cz_um_zyx, r_i2):
    return r_i2.apply_inverse(cz_um_zyx)


def build_i2_crop_seed(cz_um_zyx, hcr_um_zyx, r_i2, W=600.0):
    pred_zyx = r_i2.apply_inverse(cz_um_zyx)
    pred_xyz = pred_zyx[:, [2, 1, 0]]
    i2_center_xy = pred_xyz.mean(0)[:2]
    hcr_xyz = hcr_um_zyx[:, [2, 1, 0]]
    mask = (np.abs(hcr_xyz[:, 0] - i2_center_xy[0]) <= W) & \
           (np.abs(hcr_xyz[:, 1] - i2_center_xy[1]) <= W)
    hcr_cropped_xyz = hcr_xyz[mask]
    if len(hcr_cropped_xyz) < 20:
        return None, len(hcr_cropped_xyz)
    cz_xyz = cz_um_zyx[:, [2, 1, 0]]
    R_xyz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    cz_rot_xyz = (cz_xyz - cz_xyz.mean(0)) @ R_xyz.T
    t_xyz = hcr_cropped_xyz.mean(0)
    cz_init_xyz = cz_rot_xyz + t_xyz
    return cz_init_xyz[:, [2, 1, 0]], len(hcr_cropped_xyz)


def ss_score(pairs_df, threshold_um=30.0):
    if pairs_df is None or pairs_df.empty:
        return 0
    return int((pairs_df["residual_um"] < threshold_um).sum())


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    all_rows = []
    for subj in subjects:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        cz_um, _ = centroids_um(s, "cz")
        hcr_um, _ = centroids_um(s, "hcr_gfp")
        print(f"  n_cz={len(cz_um)} n_hcr_gfp={len(hcr_um)} n_gt={len(s.coreg_table)}", flush=True)

        seeds = {"default": None}
        t_i2 = time.time()
        try:
            r_i2 = i2_estimate(s)
            i2_t = time.time() - t_i2
            print(f"  I2 done in {i2_t:.1f}s metric={r_i2.metric:.3f}", flush=True)
            seeds["i2_direct"] = build_i2_direct_seed(cz_um, r_i2)
            crop_seed, n_crop = build_i2_crop_seed(cz_um, hcr_um, r_i2, W=600.0)
            seeds["i2_crop600"] = crop_seed
            print(f"  i2_crop600: n_hcr_in_window={n_crop}", flush=True)
        except Exception as e:
            print(f"  I2 failed: {e}", flush=True)
            seeds["i2_direct"] = None
            seeds["i2_crop600"] = None

        for name, cz_init in seeds.items():
            if name != "default" and cz_init is None:
                print(f"  [{name:12s}] skipped", flush=True)
                continue
            t1 = time.time()
            try:
                if cz_init is None:
                    r = run_p1(s)
                else:
                    r = run_p1(s, cz_init=cz_init)
            except Exception as e:
                print(f"  [{name:12s}] FAILED: {e}", flush=True)
                continue
            dt = time.time() - t1
            ss = ss_score(r.pairs_df)
            scores = compare_to_gt(r.pairs_df, s)
            print(f"  [{name:12s}] SS={ss:3d}  r@20={scores['recall_at_20um']:.3f} "
                  f"r@10={scores['recall_at_10um']:.3f} "
                  f"r@5={scores['recall_at_5um']:.3f} "
                  f"rec_id={scores['recall']:.3f} "
                  f"med={scores['median_error_um']:.0f}µm "
                  f"wall={dt:.1f}s", flush=True)
            all_rows.append(dict(
                subject=subj, seed=name, ss=ss,
                r5=scores['recall_at_5um'], r10=scores['recall_at_10um'],
                r20=scores['recall_at_20um'], recall_id=scores['recall'],
                median=scores['median_error_um'], wall=round(dt, 1),
            ))

    df = pd.DataFrame(all_rows)
    df.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/40_p1_multistart/multistart.csv",
        index=False,
    )
    print("\n=== FULL TABLE ===")
    print(df.to_string(index=False))

    print("\n=== BEST PER SUBJECT ===")
    for subj in subjects:
        sub = df[df.subject == subj]
        if len(sub) == 0:
            print(f"  {subj}: no results")
            continue
        ss_best = sub.sort_values("ss", ascending=False).iloc[0]
        r20_best = sub.sort_values("r20", ascending=False).iloc[0]
        default_row = sub[sub.seed == "default"]
        default_r20 = float(default_row.iloc[0].r20) if len(default_row) else float("nan")
        pick_flag = "✓" if ss_best.seed == r20_best.seed else "✗(SS≠ORACLE)"
        print(f"  {subj}: SS-pick={ss_best.seed:12s} (ss={ss_best.ss:3d}, "
              f"r@20={ss_best.r20:.3f}) | ORACLE={r20_best.seed:12s} "
              f"r@20={r20_best.r20:.3f} | default r@20={default_r20:.3f} | {pick_flag}")


if __name__ == "__main__":
    main()
