"""Probe trim-sweep ICP: run multi-start ICP at 6 trim levels per subject.

Each (seed × trim) combination is scored by:
- Self-supervised: recip × unique_frac (existing S29 ranker)
- Oracle (GT-based, diagnostic only): n_gt_lt5, n_gt_lt50, median residual

Writes per-subject CSV of all (seed, trim) scores, plus an overall
summary showing which (seed, trim) the self-supervised ranker picks vs
the oracle-best.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
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
TRIMS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # inlier_residual_quantile sweep
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


def seeds_xyz(hcr_xyz):
    gfp_c = hcr_xyz.mean(0)
    return {
        "hcr_gfp":    gfp_c,
        "gfp_dz+100": gfp_c + [0, 0, 100],
        "gfp_dz-100": gfp_c + [0, 0, -100],
        "gfp_dz+200": gfp_c + [0, 0, 200],
        "gfp_q25":    np.quantile(hcr_xyz, 0.25, axis=0),
        "gfp_q75":    np.quantile(hcr_xyz, 0.75, axis=0),
    }


def score_recip_unique(pred_xyz, hcr_xyz):
    th = cKDTree(hcr_xyz); tc = cKDTree(pred_xyz)
    d_c2h, idx_c2h = th.query(pred_xyz, k=1)
    _, idx_h2c = tc.query(hcr_xyz, k=1)
    recip = int(((idx_h2c[idx_c2h] == np.arange(len(pred_xyz))) & (d_c2h < 30)).sum())
    uniq_frac = len(np.unique(idx_c2h)) / len(pred_xyz)
    return recip * uniq_frac, recip, uniq_frac


def oracle_scores(pred_xyz_full, cz_xyz, cz_gt, hcr_gt):
    """Evaluate on GT pairs."""
    # pred_xyz_full is the warped *full* cz_xyz; need subset for GT pairs
    # Better: re-apply the fit to cz_gt.
    # But we only have pred_xyz (already warped). Use nearest-index mapping.
    # Simpler: pass fit explicitly.
    raise NotImplementedError("use oracle_scores_from_fit instead")


def oracle_scores_from_fit(fit, cz_gt, hcr_gt, post_offset_xyz=None):
    pred_gt = ((cz_gt * fit.scales) @ fit.R.T) + fit.translation
    if post_offset_xyz is not None:
        pred_gt = pred_gt + post_offset_xyz
    d = np.linalg.norm(pred_gt - hcr_gt, axis=1)
    return dict(
        n_gt_lt5=int((d < 5).sum()),
        n_gt_lt50=int((d < 50).sum()),
        n_gt_lt100=int((d < 100).sum()),
        median_um=float(np.median(d)),
    )


def run_one(subj):
    s = load_subject(subj)
    cz_um, _ = centroids_um(s, "cz")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_um[:, [2, 1, 0]]
    hcr_xyz = hcr_um[:, [2, 1, 0]]
    cz_gt, hcr_gt = gt_pairs(s)
    ncz = len(cz_xyz)

    rows = []
    for seed_name, seed_t in seeds_xyz(hcr_xyz).items():
        for trim in TRIMS:
            f0 = _Fit(R=R_XYZ, src_mean=cz_xyz.mean(0),
                      translation=np.asarray(seed_t, float),
                      scales=np.ones(3))
            t0 = time.time()
            try:
                res = estimate_scales_icp_multi_start(
                    cz_xyz, hcr_xyz, f0,
                    inlier_residual_quantile=trim,
                )
            except Exception as e:
                rows.append(dict(subject=subj, seed=seed_name, trim=trim,
                                 status=f"err:{type(e).__name__}"))
                continue
            dt = time.time() - t0
            if res.fit is None:
                rows.append(dict(subject=subj, seed=seed_name, trim=trim,
                                 status="no_fit", t_s=dt))
                continue
            fit = res.fit
            pred = (cz_xyz * fit.scales) @ fit.R.T + fit.translation
            rank, recip, uniq = score_recip_unique(pred, hcr_xyz)
            oracle = oracle_scores_from_fit(fit, cz_gt, hcr_gt)
            rows.append(dict(
                subject=subj, seed=seed_name, trim=trim, status="ok",
                t_s=dt,
                rank_score=rank, recip30=recip, uniq_frac=uniq,
                sxy=float(np.mean(fit.scales[:2])),
                sz=float(fit.scales[2]),
                **oracle,
            ))
    return rows


def main():
    all_rows = []
    summary = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===", flush=True)
        rows = run_one(subj)
        all_rows.extend(rows)
        ok = [r for r in rows if r.get("status") == "ok"]
        if not ok:
            print("  no valid fits")
            continue
        # Self-supervised best
        ss_best = max(ok, key=lambda r: r["rank_score"])
        # Oracle best (diagnostic only)
        or_best = max(ok, key=lambda r: r["n_gt_lt50"])
        print(f"  SS pick: seed={ss_best['seed']:12s} trim={ss_best['trim']:.1f}  "
              f"rank={ss_best['rank_score']:.2f} n_gt_lt50={ss_best['n_gt_lt50']:3d} "
              f"median={ss_best['median_um']:6.1f} µm")
        print(f"  OR best: seed={or_best['seed']:12s} trim={or_best['trim']:.1f}  "
              f"rank={or_best['rank_score']:.2f} n_gt_lt50={or_best['n_gt_lt50']:3d} "
              f"median={or_best['median_um']:6.1f} µm")
        summary.append(dict(
            subject=subj,
            ss_seed=ss_best["seed"], ss_trim=ss_best["trim"],
            ss_rank=ss_best["rank_score"], ss_n_lt50=ss_best["n_gt_lt50"],
            ss_median=ss_best["median_um"],
            or_seed=or_best["seed"], or_trim=or_best["trim"],
            or_rank=or_best["rank_score"], or_n_lt50=or_best["n_gt_lt50"],
            or_median=or_best["median_um"],
        ))

    out = Path("/root/capsule/code/full_automatic_execution_01/sessions/32_trimmed_icp")
    pd.DataFrame(all_rows).to_csv(out / "probe_all.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
