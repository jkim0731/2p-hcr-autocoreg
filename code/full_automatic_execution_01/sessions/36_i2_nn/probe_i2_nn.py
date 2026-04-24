"""Probe: does I2's 340 µm median alignment already contain nearest-neighbor
correspondence signal on 782149 (before ICP)?

S30-S35 established centroid ICP always slides to the wrong basin on 782149.
The culprit is XY density (3831 HCR cells, GT region 335 µm off-centroid).
Hypothesis: I2's global MI-affine doesn't use HCR centroid density — it uses
tissue image structure — so it may land near truth even when centroid ICP runs
away. If so, NN matching on the I2-warped CZ cells (no ICP at all) could
already produce non-zero n_lt50 on 782149.

Plan (per subject):
  1. Run I2 → CoregResult with affine warp.
  2. Map each CZ cell via I2: pred_hcr = apply_inverse(cz_zyx) → xyz.
  3. For each pred, find nearest HCR GFP+ cell (cKDTree, k=1).
  4. Form (cz_id, hcr_id) pairs; compute n_lt50 vs coreg_table GT.
  5. Also compute 'NN with reciprocity' (keeps only cz↔hcr that agree both ways).
  6. Print I2-affine diagnostics (scales, residual median on GT, n_lt50 at NN).

Outcome to decide next step:
  - If 782149 n_lt50 > 0 with I2+NN, the signal is there: try I3 B-spline refine.
  - If 782149 n_lt50 == 0, I2 alone is insufficient and we need either:
      * I3 with proper affine composition (sitk_wrapper rewrite), or
      * feature-based matching (G1 GNN / F6 descriptors).
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
    return (
        cz_rows.index.values,
        hc_rows.index.values,
        cz_um[:, [2, 1, 0]],
        hc_um[:, [2, 1, 0]],
    )


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


def main():
    rows = []
    for subj in ["788406", "755252", "767022", "782149"]:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        cz_um, cz_ids = centroids_um(s, "cz")
        hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
        cz_xyz = cz_um[:, [2, 1, 0]]
        hcr_xyz = hcr_um[:, [2, 1, 0]]
        gt_cz_ids, gt_hcr_ids, cz_gt_xyz, hcr_gt_xyz = gt_pairs(s)
        n_gt = len(gt_cz_ids)
        print(f"  n_cz={len(cz_xyz)}, n_hcr_gfp={len(hcr_xyz)}, n_gt={n_gt}")

        # Run I2
        t0 = time.time()
        r_i2 = run_i2(s)
        pred_zyx = r_i2.apply_inverse(cz_um)
        pred_xyz = pred_zyx[:, [2, 1, 0]]
        dt = time.time() - t0

        # I2 quality on GT pairs
        cz_id_to_idx = {cid: i for i, cid in enumerate(cz_ids)}
        gt_idx_in_cz = np.array([cz_id_to_idx[c] for c in gt_cz_ids if c in cz_id_to_idx])
        gt_keep = np.array([c in cz_id_to_idx for c in gt_cz_ids])
        if gt_keep.sum() == 0:
            print(f"  I2 ({dt:.1f}s) — no GT CZ cells in centroid table; skip")
            continue
        pred_gt_xyz = pred_xyz[gt_idx_in_cz]
        hcr_gt_keep = hcr_gt_xyz[gt_keep]
        residuals = np.linalg.norm(pred_gt_xyz - hcr_gt_keep, axis=1)
        i2_median = float(np.median(residuals))
        i2_rms = float(np.sqrt((residuals ** 2).mean()))
        i2_n_lt50_direct = int((residuals < 50).sum())
        print(f"  I2 ({dt:.1f}s) GT median={i2_median:.0f} µm rms={i2_rms:.0f} µm "
              f"n_lt50_direct={i2_n_lt50_direct}/{int(gt_keep.sum())}")

        # NN matching: for each pred, find nearest HCR GFP+ cell
        th = cKDTree(hcr_xyz)
        d_c2h, idx_c2h = th.query(pred_xyz, k=1)

        # Build pair (cz_id -> hcr_id) via NN
        nn_cz_to_hcr = {cz_ids[i]: hcr_ids[idx_c2h[i]] for i in range(len(cz_ids))}

        # Check how many GT pairs are recovered, with distance to GT HCR
        n_nn_correct = 0
        n_gt_eligible = 0
        nn_dists = []
        for cid, hid_gt, hcr_gt_pos in zip(gt_cz_ids, gt_hcr_ids, hcr_gt_xyz):
            if cid not in cz_id_to_idx:
                continue
            n_gt_eligible += 1
            hid_pred = nn_cz_to_hcr[cid]
            if hid_pred == hid_gt:
                n_nn_correct += 1
            # Distance between predicted NN HCR cell and GT HCR cell
            j = np.where(hcr_ids == hid_pred)[0][0]
            d = float(np.linalg.norm(hcr_xyz[j] - hcr_gt_pos))
            nn_dists.append(d)
        nn_dists = np.array(nn_dists)
        nn_n_lt50 = int((nn_dists < 50).sum()) if len(nn_dists) else 0

        # Reciprocal NN
        tc = cKDTree(pred_xyz)
        _, idx_h2c = tc.query(hcr_xyz, k=1)
        recip_mask = idx_h2c[idx_c2h] == np.arange(len(pred_xyz))
        n_recip = int(recip_mask.sum())
        # Reciprocal subset on GT
        recip_correct = 0
        recip_dists = []
        for k, cid in enumerate(gt_cz_ids):
            if cid not in cz_id_to_idx:
                continue
            i = cz_id_to_idx[cid]
            if not recip_mask[i]:
                continue
            hid_pred = hcr_ids[idx_c2h[i]]
            if hid_pred == gt_hcr_ids[k]:
                recip_correct += 1
            j = np.where(hcr_ids == hid_pred)[0][0]
            d = float(np.linalg.norm(hcr_xyz[j] - hcr_gt_xyz[k]))
            recip_dists.append(d)
        recip_dists = np.array(recip_dists)
        recip_n_lt50 = int((recip_dists < 50).sum()) if len(recip_dists) else 0

        print(f"  NN: {n_nn_correct}/{n_gt_eligible} IDs exact, n_lt50_pos={nn_n_lt50} "
              f"(median NN-to-GT dist={np.median(nn_dists):.0f} µm)" if len(nn_dists) else "  NN: no GT")
        print(f"  Recip NN: n_recip_total={n_recip}, GT-subset IDs exact={recip_correct}, "
              f"n_lt50_pos={recip_n_lt50} / {len(recip_dists)}")

        rows.append(dict(
            subject=subj, n_gt=n_gt_eligible,
            i2_dt_s=round(dt, 1),
            i2_median_um=round(i2_median, 0),
            i2_rms_um=round(i2_rms, 0),
            i2_direct_n_lt50=i2_n_lt50_direct,
            nn_id_exact=n_nn_correct,
            nn_pos_n_lt50=nn_n_lt50,
            recip_gt_id_exact=recip_correct,
            recip_pos_n_lt50=recip_n_lt50,
            recip_total=n_recip,
        ))

    print("\n=== SUMMARY ===")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv("/root/capsule/code/full_automatic_execution_01/sessions/36_i2_nn/i2_nn.csv",
              index=False)


if __name__ == "__main__":
    main()
