"""S45 pivot — HCR-only feature re-ranker diagnostic.

The original M4 per-cell Dice plan is STRUCTURALLY BLOCKED:
  * CZ `*_seg-mask-outline.tif` is a single-label binary mask on all 4 test
    subjects — not per-cell labels. No per-cell CZ volumes exist.
  * Only 788406 has HCR `metrics.pickle` (so only 788406 has per-cell HCR
    `volume`, `counts`, `density` in `hcr_gfp_df`).

Without CZ-side per-cell morphology there is no pairing feature that mixes
both sides. BUT: an HCR-only re-ranker is still testable —
  *Among P1's K=5 putative HCR partners for a given CZ cell, does HCR's
   per-cell volume/density/counts distribution of correct partners differ
   from wrong partners?*

If yes, we can re-rank P1's top-K by a putative's HCR-side deviation from
the empirical correct-partner distribution.

Strategy (788406 only — other subjects lack metrics.pickle):
  1. Load s + run P1's putative builder (not just the final pairs_df — we
     need the top-K list per CZ cell).
  2. Among CZ cells where the GT partner IS in the top-K, tag which
     putative is correct and which are wrong.
  3. Compare HCR-side feature distributions (volume, density, counts,
     log(counts), log(volume), log(density)) on correct-putative HCRs vs
     wrong-putative HCRs.
  4. Fit a simple logistic regression on those features → is identity
     discriminable at all?

This is a DIAGNOSTIC only — it answers "is there signal available?".
If not, attacking the ranker via HCR-only features is hopeless and the
session should pivot to image-based CZ per-cell features.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import (centroids_um, default_warmstart_zyx,
                                   apply_aniso_fit)  # noqa: E402
from lib.cell_features import (extract_cell_features,
                                invariant_feature_mask)  # noqa: E402
from benchmark_analysis import fit_anisotropic_similarity  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402


def p1_putatives(s, K: int = 5):
    """Rebuild P1's top-K putative list per CZ cell (same logic as _p1_teaser).

    Returns:
      order: (N_cz, K) int array of HCR indices per CZ cell.
      cz_ids, hcr_ids: 1-D arrays aligning `order` to native IDs.
      cz_um, hcr_um: centroid arrays.
      D: (N_cz, N_hcr) Euclidean distance (post-warmstart).
      cosS: (N_cz, N_hcr) feature cosine.
    """
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    Fc, names, _ = extract_cell_features(s, "cz")
    Fg, _, _ = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
    Fcn = (Fc[:, keep] - mu) / sd
    Fgn = (Fg[:, keep] - mu) / sd
    Fcn = Fcn / (np.linalg.norm(Fcn, axis=1, keepdims=True) + 1e-9)
    Fgn = Fgn / (np.linalg.norm(Fgn, axis=1, keepdims=True) + 1e-9)

    D = cdist(cz_init, hcr_um)
    cosS = Fcn @ Fgn.T
    score = D - 25.0 * cosS
    order = np.argsort(score, axis=1)[:, :K]
    return order, cz_ids, hcr_ids, cz_um, hcr_um, D, cosS


def main():
    subj = "788406"
    print(f"=== {subj} — HCR-feature re-ranker diagnostic ===", flush=True)
    s = load_subject(subj)
    if "volume" not in s.hcr_gfp_df.columns:
        print("  SKIP — no HCR volume metadata.")
        return

    t0 = time.time()
    order, cz_ids, hcr_ids, cz_um, hcr_um, D, cosS = p1_putatives(s, K=5)
    print(f"  putatives built in {time.time()-t0:.1f}s  "
          f"n_cz={len(cz_ids)}  n_hcr_gfp={len(hcr_ids)}  K=5", flush=True)

    # GT map: cz_id -> hcr_id
    gt_map = dict(zip(s.coreg_table["cz_id"].astype(int),
                      s.coreg_table["hcr_id"].astype(int)))
    hcr_pos_of_id = dict(zip(hcr_ids, range(len(hcr_ids))))

    # Per-HCR features from hcr_gfp_df. Some HCR cells may be missing from gfp_df
    # if they were filtered — build a dict keyed by hcr_id.
    feat = s.hcr_gfp_df.set_index("hcr_id")
    feat_cols = [c for c in ["counts", "volume", "density"] if c in feat.columns]
    print(f"  HCR feature cols: {feat_cols}", flush=True)

    correct_rows = []
    wrong_rows = []
    cz_in_topk = 0
    cz_not_in_topk = 0
    for i, cid in enumerate(cz_ids):
        gt_h = gt_map.get(int(cid))
        if gt_h is None:
            continue
        gt_pos = hcr_pos_of_id.get(int(gt_h))
        if gt_pos is None:
            continue
        topk_hcr_positions = order[i]
        topk_hcr_native_ids = [int(hcr_ids[j]) for j in topk_hcr_positions]
        if gt_h not in topk_hcr_native_ids:
            cz_not_in_topk += 1
            continue
        cz_in_topk += 1
        for rank, (jpos, jnat) in enumerate(zip(topk_hcr_positions, topk_hcr_native_ids)):
            if jnat not in feat.index:
                continue
            row = dict(
                cz_id=int(cid), hcr_id=int(jnat),
                rank=int(rank),
                dist_um=float(D[i, jpos]),
                cos=float(cosS[i, jpos]),
            )
            for c in feat_cols:
                row[c] = float(feat.loc[jnat, c])
            if int(jnat) == int(gt_h):
                correct_rows.append(row)
            else:
                wrong_rows.append(row)

    correct = pd.DataFrame(correct_rows)
    wrong = pd.DataFrame(wrong_rows)
    print(f"  GT in top-5: {cz_in_topk}/{cz_in_topk + cz_not_in_topk} "
          f"({cz_in_topk/max(1,cz_in_topk+cz_not_in_topk):.1%})", flush=True)
    print(f"  n_correct_putatives={len(correct)} n_wrong_putatives={len(wrong)}", flush=True)

    # Compare per-feature distributions.
    print(f"\n  feature     | correct_med (IQR) | wrong_med (IQR) | AUC (correct>wrong?)")
    print(f"  ------------|-------------------|-----------------|---------------------")
    for c in ["counts", "volume", "density", "dist_um", "cos"]:
        if c not in correct.columns and c not in ["dist_um", "cos"]:
            continue
        cv = correct[c].values.astype(float); wv = wrong[c].values.astype(float)
        cv = cv[np.isfinite(cv)]; wv = wv[np.isfinite(wv)]
        if len(cv) == 0 or len(wv) == 0:
            continue
        def med_iqr(a):
            return np.median(a), np.percentile(a, 75) - np.percentile(a, 25)
        cm, ci = med_iqr(cv)
        wm, wi = med_iqr(wv)
        # AUC of "correct has larger c than wrong putative"
        # Use pairwise: for each cz where correct exists in same cz-group as wrong
        auc_num = 0
        auc_den = 0
        # Build per-cz groupings for within-CZ ranking
        by_cz_correct = correct.groupby("cz_id")
        by_cz_wrong = wrong.groupby("cz_id")
        for cid_, cg in by_cz_correct:
            if cid_ not in by_cz_wrong.groups:
                continue
            wg = by_cz_wrong.get_group(cid_)
            if c not in cg.columns or c not in wg.columns:
                continue
            cvals = cg[c].values.astype(float)
            wvals = wg[c].values.astype(float)
            for x in cvals:
                for y in wvals:
                    if np.isfinite(x) and np.isfinite(y):
                        auc_num += (x > y) + 0.5 * (x == y)
                        auc_den += 1
        auc = auc_num / max(1, auc_den)
        print(f"  {c:11s} | {cm:10.3g} ({ci:7.3g}) | {wm:10.3g} ({wi:7.3g}) | {auc:.3f}")

    # Fit a simple logistic on [dist, cos, log(counts), log(volume)] to predict correct.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Xc = []; yc = []
    if len(correct) and len(wrong):
        cols_used = [c for c in ["dist_um", "cos"] if c in correct.columns]
        for c in ["counts", "volume", "density"]:
            if c in correct.columns:
                cols_used.append(c)
        X = pd.concat([correct.assign(y=1), wrong.assign(y=0)], ignore_index=True)
        for c in ["counts", "volume", "density"]:
            if c in X.columns:
                X[f"log_{c}"] = np.log(np.clip(X[c].astype(float), 1e-9, None))
        # Within-CZ z-score for each feature (so the model sees *relative* deviation)
        for c in list(cols_used) + [f"log_{k}" for k in ["counts", "volume", "density"]
                                     if f"log_{k}" in X.columns]:
            g = X.groupby("cz_id")[c]
            mu = g.transform("mean"); sd = g.transform("std").replace(0, 1e-6)
            X[f"z_{c}"] = (X[c] - mu) / sd
        use = [f"z_{c}" for c in cols_used] + \
              [f"z_log_{c}" for c in ["counts", "volume", "density"]
               if f"z_log_{c}" in X.columns]
        Xarr = X[use].values.astype(float)
        yarr = X["y"].values.astype(int)
        mask = np.isfinite(Xarr).all(1)
        Xarr = Xarr[mask]; yarr = yarr[mask]
        if len(np.unique(yarr)) == 2 and len(Xarr) >= 20:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(Xarr, yarr)
            p = lr.predict_proba(Xarr)[:, 1]
            auc_all = roc_auc_score(yarr, p)
            print(f"\n  LR on within-CZ z-features: AUC={auc_all:.3f} (in-sample)")
            print(f"  coef mapping:")
            for n, c in zip(use, lr.coef_[0]):
                print(f"    {n:20s} {c:+.3f}")
            # Compare to baseline: dist + cos only
            base_cols = [u for u in use if u in ("z_dist_um", "z_cos")]
            if len(base_cols) == 2:
                Xb = X[base_cols].values.astype(float)
                mask2 = np.isfinite(Xb).all(1)
                lr_b = LogisticRegression(max_iter=1000)
                lr_b.fit(Xb[mask2], X["y"].values[mask2])
                p_b = lr_b.predict_proba(Xb[mask2])[:, 1]
                auc_b = roc_auc_score(X["y"].values[mask2], p_b)
                print(f"  Baseline (z_dist + z_cos only) AUC={auc_b:.3f}")
                print(f"  Lift from HCR features: {auc_all - auc_b:+.3f}")


if __name__ == "__main__":
    main()
