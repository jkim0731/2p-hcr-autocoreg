"""S45 follow-up — GT-in-top-K ceiling for P1's putative generator.

The HCR-feat-rerank probe showed only 48.9% of CZ cells have GT in top-5.
That caps any top-K re-ranker. Measure the curve K ↦ GT-in-top-K across
{788406, 755252, 767022, 782149} to decide whether widening K is useful.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa: E402
from lib.cell_features import extract_cell_features, invariant_feature_mask  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402


def p1_order(s):
    """Full argsort of HCR candidates per CZ cell — same scoring as P1."""
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
    order = np.argsort(score, axis=1)
    return order, cz_ids, hcr_ids


def main():
    Ks = [1, 3, 5, 10, 20, 50, 100, 500]
    subjects = ["788406", "755252", "767022", "782149"]
    rows = []
    for subj in subjects:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        t0 = time.time()
        order, cz_ids, hcr_ids = p1_order(s)
        gt_map = dict(zip(s.coreg_table["cz_id"].astype(int),
                          s.coreg_table["hcr_id"].astype(int)))
        hcr_id_pos = dict(zip(hcr_ids, range(len(hcr_ids))))
        cz_gt = []  # per-CZ: rank of GT in order, or -1
        for i, cid in enumerate(cz_ids):
            gt_h = gt_map.get(int(cid))
            if gt_h is None:
                continue
            gt_pos = hcr_id_pos.get(int(gt_h))
            if gt_pos is None:
                cz_gt.append(-1)
                continue
            # rank of GT in order
            m = np.where(order[i] == gt_pos)[0]
            cz_gt.append(int(m[0]) if len(m) else -1)
        cz_gt = np.array(cz_gt)
        n_gt = len(cz_gt)
        print(f"  n_gt_with_cz_centroid={n_gt}  order build={time.time()-t0:.1f}s",
              flush=True)
        for K in Ks:
            frac = float((cz_gt >= 0) & (cz_gt < K)).mean() \
                if False else float(((cz_gt >= 0) & (cz_gt < K)).mean())
            print(f"    K={K:4d} → GT-in-top-K = {frac:.3f} ({int(frac*n_gt)}/{n_gt})")
            rows.append(dict(subject=subj, K=K, gt_in_topk=frac,
                             n_gt=n_gt, n_hit=int(frac*n_gt)))

        # Also: what rank does GT land at when present?
        present = cz_gt[cz_gt >= 0]
        if len(present):
            print(f"  GT rank quantiles (when present, n={len(present)}): "
                  f"p25={int(np.percentile(present,25))} "
                  f"p50={int(np.percentile(present,50))} "
                  f"p75={int(np.percentile(present,75))} "
                  f"p95={int(np.percentile(present,95))} "
                  f"max={int(present.max())}")
        missing = int((cz_gt == -1).sum())
        print(f"  GT missing from HCR GFP+ list entirely: {missing}/{n_gt}")

    df = pd.DataFrame(rows)
    df.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/45_iou_augmented_p1/topk_ceiling.csv",
        index=False,
    )
    print("\n\n=== SUMMARY — GT-in-top-K (rows=subject, cols=K) ===")
    pivot = df.pivot(index="subject", columns="K", values="gt_in_topk")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
