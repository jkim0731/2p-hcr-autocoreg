"""Sweep widened M1 (S31) across all 6 subjects and report diagnostics vs ground truth.

Compares:
- Recovered (sxy, sz) vs landmark-fit GT scales
- Recovered translation vs landmark-fit GT translation
- NCC peak + robust-z
- ICP-seeded-by-M1 GT recall at 5/50 µm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from benchmark_analysis import fit_anisotropic_similarity  # noqa
from bench.candidate_impls import _m1_mask_ncc as _m1mod  # noqa
from bench.harness import CANDIDATES  # noqa
from anisotropic_icp import estimate_scales_icp_multi_start  # noqa
from lib.centroid_helpers import centroids_um  # noqa

from dataclasses import dataclass


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


SUBJECTS = ["788406", "755252", "767018", "767022", "782149", "790322"]


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


def run_one(subj):
    import time
    s = load_subject(subj)
    # ground truth
    cz_gt_xyz, hcr_gt_xyz = gt_pairs(s)
    gt_fit = fit_anisotropic_similarity(cz_gt_xyz, hcr_gt_xyz)
    gt_sxy = float(np.mean([gt_fit.scales[0], gt_fit.scales[1]]))
    gt_sz = float(gt_fit.scales[2])

    t0 = time.time()
    res = CANDIDATES["M1"](s)
    t_m1 = time.time() - t0

    if res.transform is None:
        return dict(subject=subj, error="no transform", t_m1_s=t_m1)
    d = res.diagnostics

    # Compute M1's predicted (R, S, t) in (x, y, z) convention
    # M1 returns T such that: hcr_pos_zyx = (cz - cz_c) @ R.T * S + t
    #   where R = diag(1, -1, -1) (180° Z) and S = (sz, sxy, sxy) in zyx
    # For ICP seed conversion:
    sxy = float(d["sxy"]); sz = float(d["sz"])
    t_zyx = np.asarray(d["t_um"])
    # t_zyx is where CZ's centroid (cz_c) lands in HCR
    # (because (cz_c - cz_c) = 0, so t = template center in HCR)
    # Convert to (x, y, z): swap
    t_xyz = t_zyx[[2, 1, 0]]

    # Seed ICP from M1's (S, t) with R = 180° about Z
    cz_um, _ = centroids_um(s, "cz")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_um[:, [2, 1, 0]]
    hcr_xyz = hcr_um[:, [2, 1, 0]]
    # ProcrustesFit convention: dst = ((src - src_mean) @ R) * scales + t_dst_mean
    # We want: (cz - cz_c) * S @ R.T = target in HCR µm relative to t
    # So seed fit has R = 180°Z, scales = (sxy, sxy, sz) in xyz, translation = t_xyz
    # src_mean = cz_xyz.mean(0), and the effective translation passed to
    # estimate_scales_icp_multi_start is hcr_mean used as dst center.
    # Simpler: just set translation directly.
    R_xyz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    seed_fit = _Fit(R=R_xyz,
                    src_mean=cz_xyz.mean(0),
                    translation=t_xyz,
                    scales=np.array([sxy, sxy, sz]))

    # Also evaluate M1's raw (pre-ICP) predictions on GT
    cz_gt_um = cz_gt_xyz  # already in xyz
    pred_m1 = ((cz_gt_um - cz_xyz.mean(0)) @ R_xyz) * np.array([sxy, sxy, sz]) + t_xyz
    d_m1 = np.linalg.norm(pred_m1 - hcr_gt_xyz, axis=1)
    m1_n_lt5 = int((d_m1 < 5).sum())
    m1_n_lt50 = int((d_m1 < 50).sum())
    m1_n_lt100 = int((d_m1 < 100).sum())
    m1_median = float(np.median(d_m1))

    # Run ICP from M1 seed
    t1 = time.time()
    try:
        icp_res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, seed_fit)
        if icp_res.fit is not None:
            fit = icp_res.fit
            pred_icp = ((cz_gt_um * fit.scales) @ fit.R.T) + fit.translation
            d_icp = np.linalg.norm(pred_icp - hcr_gt_xyz, axis=1)
            icp_n_lt5 = int((d_icp < 5).sum())
            icp_n_lt50 = int((d_icp < 50).sum())
            icp_median = float(np.median(d_icp))
            icp_sxy = float(np.mean([fit.scales[0], fit.scales[1]]))
            icp_sz = float(fit.scales[2])
        else:
            icp_n_lt5 = icp_n_lt50 = 0; icp_median = float("nan")
            icp_sxy = icp_sz = float("nan")
    except Exception as e:
        icp_n_lt5 = icp_n_lt50 = 0; icp_median = float("nan")
        icp_sxy = icp_sz = float("nan")
    t_icp = time.time() - t1

    return dict(
        subject=subj,
        # M1 outputs
        m1_ncc=float(d["best_ncc"]), m1_robust_z=float(d["robust_z"]),
        m1_sxy=sxy, m1_sz=sz, m1_n_peaks=int(d["n_peaks"]),
        m1_t_xyz=t_xyz.tolist(),
        # GT reference
        gt_sxy=gt_sxy, gt_sz=gt_sz, n_gt=len(cz_gt_xyz),
        # M1 raw GT evaluation
        m1_n_lt5=m1_n_lt5, m1_n_lt50=m1_n_lt50, m1_n_lt100=m1_n_lt100,
        m1_median_um=m1_median,
        # M1 → ICP GT evaluation
        icp_n_lt5=icp_n_lt5, icp_n_lt50=icp_n_lt50,
        icp_median_um=icp_median, icp_sxy=icp_sxy, icp_sz=icp_sz,
        t_m1_s=t_m1, t_icp_s=t_icp,
    )


def main():
    rows = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===")
        r = run_one(subj)
        print(json.dumps(r, indent=2, default=str))
        rows.append(r)
    df = pd.DataFrame(rows)
    out = Path("/root/capsule/code/full_automatic_execution_01/sessions/31_M1_widened")
    df.to_csv(out / "sweep.csv", index=False)
    print("\nSaved → sweep.csv")
    print(df[["subject", "m1_ncc", "m1_robust_z", "m1_sxy", "m1_sz",
              "gt_sxy", "gt_sz", "m1_n_lt50", "m1_n_lt100", "m1_median_um",
              "icp_n_lt50", "icp_median_um", "t_m1_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
