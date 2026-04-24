"""Evaluate richer self-supervised ranking signals for multi-start ICP seeds.

For each (subject, seed) we've already computed:
  - inl30 (count of CZ points w/ HCR 1-NN within 30µm)
  - rms (ICP converged residual)
  - scales (anisotropic)
  - GT median distance (for label only; not used in ranking)

We want a ranking function r(converged_pred, hcr_points) that is high for the
"correct" basin and low for "wrong" basins, using no GT.

Candidate signals:
  (a) inl30 alone (baseline)
  (b) reciprocal-NN count: number of (i,j) where j=argmin(|pred_i - hcr_*|)
      AND i=argmin(|hcr_j - pred_*|). A correct fit has many reciprocal
      pairs; a wrong-but-dense basin has few (because many CZ map to the
      same HCR).
  (c) unique-HCR fraction: |unique HCR 1-NNs| / n_cz. Higher = less
      collapse onto a few HCR points.
  (d) combined: inl30 × reciprocal × unique_frac
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from lib.centroid_helpers import centroids_um
from anisotropic_icp import estimate_scales_icp_multi_start


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


def run_icp_seed(cz_xyz, hcr_xyz, seed_t_xyz):
    R0 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    fit0 = _Fit(R=R0, src_mean=cz_xyz.mean(0), translation=seed_t_xyz, scales=np.ones(3))
    try:
        res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
        if res.fit is None:
            return None
        fit = res.fit
        pred = (cz_xyz * fit.scales) @ fit.R.T + fit.translation
        return dict(pred_xyz=pred, fit=fit, rms=float(fit.rms_um))
    except Exception:
        return None


def score_signals(pred_xyz, hcr_xyz, gt_pred, gt_hcr):
    """Return dict of ranking signals + GT quality."""
    tree_h = cKDTree(hcr_xyz)
    d_cz2h, idx_cz2h = tree_h.query(pred_xyz, k=1)
    inl30 = int((d_cz2h < 30).sum())
    inl50 = int((d_cz2h < 50).sum())
    tree_c = cKDTree(pred_xyz)
    d_h2c, idx_h2c = tree_c.query(hcr_xyz, k=1)
    # Reciprocal within 30µm
    recip_mask = (idx_h2c[idx_cz2h] == np.arange(len(pred_xyz))) & (d_cz2h < 30)
    recip = int(recip_mask.sum())
    # Unique HCR targets among CZ's 1-NNs
    unique_h = int(len(np.unique(idx_cz2h)))
    unique_frac = unique_h / len(pred_xyz)
    # Mean distance to 5-NN in HCR — should be smallish if fit is tight
    d5, _ = tree_h.query(pred_xyz, k=5)
    mean5 = float(d5.mean())
    # GT
    gt_d = np.linalg.norm(pred_xyz[gt_pred] - hcr_xyz[gt_hcr], axis=1)
    gt_med = float(np.median(gt_d))
    gt_lt50 = int((gt_d < 50).sum())
    return dict(inl30=inl30, inl50=inl50, recip=recip,
                unique_frac=unique_frac, mean5=mean5,
                gt_med=gt_med, gt_lt50=gt_lt50, n_gt=len(gt_d))


def main():
    seeds_for_subject = [
        ("hcr_all",   lambda hcr_xyz, hcr_all_xyz: hcr_all_xyz.mean(0)),
        ("hcr_gfp",   lambda hcr_xyz, hcr_all_xyz: hcr_xyz.mean(0)),
        ("gfp_dz+100", lambda hcr_xyz, hcr_all_xyz: hcr_xyz.mean(0) + [0, 0, 100]),
        ("gfp_dz-100", lambda hcr_xyz, hcr_all_xyz: hcr_xyz.mean(0) + [0, 0, -100]),
        ("gfp_dz+200", lambda hcr_xyz, hcr_all_xyz: hcr_xyz.mean(0) + [0, 0, 200]),
        ("gfp_q25",   lambda hcr_xyz, hcr_all_xyz: np.quantile(hcr_xyz, 0.25, axis=0)),
        ("gfp_q75",   lambda hcr_xyz, hcr_all_xyz: np.quantile(hcr_xyz, 0.75, axis=0)),
    ]

    all_records = []
    for sid in ["767022", "782149", "755252", "788406", "767018", "790322"]:
        s = load_subject(sid)
        cz_zyx, cz_ids = centroids_um(s, "cz")
        hcr_zyx, hcr_ids = centroids_um(s, "hcr_gfp")
        hcr_all_zyx, _ = centroids_um(s, "hcr_all")
        cz_xyz = cz_zyx[:, [2, 1, 0]]
        hcr_xyz = hcr_zyx[:, [2, 1, 0]]
        hcr_all_xyz = hcr_all_zyx[:, [2, 1, 0]]

        id_to_cz = {int(i): k for k, i in enumerate(cz_ids)}
        id_to_hcr = {int(i): k for k, i in enumerate(hcr_ids)}
        rows = [(id_to_cz[int(r.cz_id)], id_to_hcr[int(r.hcr_id)])
                for _, r in s.coreg_table.iterrows()
                if int(r.cz_id) in id_to_cz and int(r.hcr_id) in id_to_hcr]
        idx_cz = np.array([r[0] for r in rows])
        idx_hcr = np.array([r[1] for r in rows])

        print(f"\n=== {sid} (n_cz={len(cz_xyz)}, n_gt={len(idx_cz)}) ===")
        print(f"{'seed':12s}{'inl30':>7s}{'inl50':>7s}{'recip':>7s}{'uniq_f':>8s}{'mean5':>8s}  GT:  {'med':>6s}{'<50':>5s}")
        for name, seed_fn in seeds_for_subject:
            seed_t = np.asarray(seed_fn(hcr_xyz, hcr_all_xyz), float)
            r = run_icp_seed(cz_xyz, hcr_xyz, seed_t)
            if r is None:
                continue
            sig = score_signals(r["pred_xyz"], hcr_xyz, idx_cz, idx_hcr)
            print(f"{name:12s}{sig['inl30']:7d}{sig['inl50']:7d}{sig['recip']:7d}"
                  f"{sig['unique_frac']:8.3f}{sig['mean5']:8.1f}       {sig['gt_med']:6.1f}{sig['gt_lt50']:5d}")
            all_records.append(dict(subject=sid, seed=name, **sig, rms=r["rms"]))

    # Summary — for each subject, find seed with lowest gt_med, and see how ranked by each signal
    import pandas as pd
    df = pd.DataFrame(all_records)
    print("\n=== Which signal best predicts the true-best seed? ===")
    for sid in df["subject"].unique():
        sub = df[df.subject == sid].reset_index(drop=True)
        true_best = sub["gt_lt50"].idxmax()
        rank_inl30 = sub.sort_values("inl30", ascending=False).reset_index()["index"].tolist().index(true_best) + 1
        rank_recip = sub.sort_values("recip", ascending=False).reset_index()["index"].tolist().index(true_best) + 1
        rank_uniq  = sub.sort_values("unique_frac", ascending=False).reset_index()["index"].tolist().index(true_best) + 1
        rank_comb  = sub.assign(c=sub["recip"] * sub["unique_frac"]).sort_values("c", ascending=False).reset_index()["index"].tolist().index(true_best) + 1
        print(f"  {sid}: true best ({sub.seed[true_best]:12s} gt<50={sub.gt_lt50[true_best]:3d}) — "
              f"rank by inl30={rank_inl30}, recip={rank_recip}, uniq={rank_uniq}, recip×uniq={rank_comb}")


if __name__ == "__main__":
    main()
