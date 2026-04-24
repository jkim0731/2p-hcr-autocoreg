"""Probe rotation-augmented ICP seeds on 782149 (and the 5 other subjects).

For each subject, run ICP from a grid of (translation_seed × rotation_perturbation)
initial poses, and record per-seed recip×unique_frac score and GT recall at 5µm.

This is a diagnostic — it peeks at GT only to label which seeds would have been
best, so we can validate whether the recip×unique_frac self-supervised ranker
still picks a good seed when rotation variation is added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01/lib")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from anisotropic_icp import estimate_scales_icp_multi_start  # noqa
from centroid_helpers import centroids_um  # noqa

from dataclasses import dataclass


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


SUBJECTS = ["788406", "755252", "767018", "767022", "782149", "790322"]


def gt_pairs_um(s):
    """Return (cz_gt_um_xyz, hcr_gt_um_xyz) aligned from coreg_table."""
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


def score_seed(pred_xyz, hcr_xyz, cz_gt_xyz, hcr_gt_xyz, fit):
    """Return (score, recip, uniq, n_gt_lt5)."""
    th = cKDTree(hcr_xyz); tc = cKDTree(pred_xyz)
    d_c2h, idx_c2h = th.query(pred_xyz, k=1)
    _, idx_h2c = tc.query(hcr_xyz, k=1)
    recip = int(((idx_h2c[idx_c2h] == np.arange(len(pred_xyz))) & (d_c2h < 30)).sum())
    uniq_frac = len(np.unique(idx_c2h)) / len(pred_xyz)
    # GT recall at 5µm (validation only, not used for ranking in production)
    cz_gt_pred = ((cz_gt_xyz * fit.scales) @ fit.R.T) + fit.translation
    d_gt = np.linalg.norm(cz_gt_pred - hcr_gt_xyz, axis=1)
    n_gt_lt5 = int((d_gt < 5).sum())
    n_gt_lt50 = int((d_gt < 50).sum())
    med_gt = float(np.median(d_gt))
    return recip * uniq_frac, recip, uniq_frac, n_gt_lt5, n_gt_lt50, med_gt


def probe_subject(subj):
    s = load_subject(subj)
    cz_pts, _ = centroids_um(s, "cz")
    hcr_pts, _ = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_pts[:, [2, 1, 0]]
    hcr_xyz = hcr_pts[:, [2, 1, 0]]
    cz_gt, hcr_gt = gt_pairs_um(s)

    R180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)

    # Translation seeds (same 6 as S29)
    gfp_c = hcr_xyz.mean(0)
    t_seeds = {
        "hcr_gfp":    gfp_c,
        "gfp_dz+100": gfp_c + [0, 0, 100],
        "gfp_dz-100": gfp_c + [0, 0, -100],
        "gfp_dz+200": gfp_c + [0, 0, 200],
        "gfp_q25":    np.quantile(hcr_xyz, 0.25, axis=0),
        "gfp_q75":    np.quantile(hcr_xyz, 0.75, axis=0),
    }
    # Rotation seed perturbations: (axis, angle_deg)
    r_seeds = [
        ("none",    None),
        ("rz+5",    ("z",  5)),
        ("rz-5",    ("z", -5)),
        ("rz+10",   ("z", 10)),
        ("rz-10",   ("z",-10)),
        ("rx+5",    ("x",  5)),
        ("rx-5",    ("x", -5)),
        ("rx+10",   ("x", 10)),
        ("rx-10",   ("x",-10)),
    ]

    results = []
    for tname, seed_t in t_seeds.items():
        for rname, rpert in r_seeds:
            if rpert is None:
                R_init = R180
            else:
                axis, ang = rpert
                Rp = Rot.from_rotvec(np.radians(ang) * np.array(
                    {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
                )).as_matrix()
                R_init = Rp @ R180
            f0 = _Fit(R=R_init, src_mean=cz_xyz.mean(0),
                      translation=np.asarray(seed_t, float), scales=np.ones(3))
            try:
                r = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f0)
                if r.fit is None:
                    continue
            except Exception:
                continue
            pred = (cz_xyz * r.fit.scales) @ r.fit.R.T + r.fit.translation
            score, recip, uniq, ngt5, ngt50, medgt = score_seed(
                pred, hcr_xyz, cz_gt, hcr_gt, r.fit)
            results.append(dict(
                subject=subj, t_seed=tname, r_seed=rname,
                score=score, recip=recip, uniq=uniq,
                n_gt_lt5=ngt5, n_gt_lt50=ngt50, med_gt_um=medgt,
                sxy=r.sxy, sz=r.sz,
            ))
    return pd.DataFrame(results)


def main():
    out_dir = Path("/root/capsule/code/full_automatic_execution_01/sessions/30_rotation_multistart")
    all_df = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===")
        df = probe_subject(subj)
        df.to_csv(out_dir / f"probe_{subj}.csv", index=False)
        # top-5 by self-supervised score
        topk = df.sort_values("score", ascending=False).head(5)
        print("\nTop-5 by recip×uniq score:")
        print(topk[["t_seed", "r_seed", "score", "recip", "uniq",
                    "n_gt_lt5", "n_gt_lt50", "med_gt_um"]].to_string(index=False))
        # top-5 by GT (oracle)
        topk_gt = df.sort_values("n_gt_lt5", ascending=False).head(5)
        print("\nTop-5 by GT rec@5 (oracle):")
        print(topk_gt[["t_seed", "r_seed", "score", "recip", "uniq",
                       "n_gt_lt5", "n_gt_lt50", "med_gt_um"]].to_string(index=False))
        all_df.append(df)
    big = pd.concat(all_df, ignore_index=True)
    big.to_csv(out_dir / "probe_all.csv", index=False)
    print(f"\nSaved {len(big)} rows → probe_all.csv")


if __name__ == "__main__":
    main()
