"""S46-b — top-K ceiling shift with HCR-image quality bonus.

The within-putative AUC probe established that:
  * CZ-side image features are constant per CZ group (AUC = 0.5
    everywhere — useless for within-CZ re-ranking).
  * HCR-side image features discriminate: AUC 0.58 on 788406, 0.72–0.75
    on stress subjects (755252, 767022).
  * Cross-modal abs-difference is ANTI-correlated (AUC 0.25–0.42), so a
    cross-modal cosine term would add noise.

Therefore the augmented score is the baseline P1 score MINUS a global
HCR-quality bonus:
    score(i, j) = D(i, j) - 25 * cos_F6(i, j) - beta * hcr_quality(j)
where `hcr_quality(j)` is a within-HCR z-scored sum of (mean, std, p90,
|lap|) — a positive scalar per HCR cell reflecting how image-prominent
it is. Higher-quality HCR cells become preferred globally; this
re-ranks noisy/spurious HCR candidates downward per-CZ.

The test: does this lift the GT-in-top-K ceiling on stress subjects?
S45 baseline: 788406 K=20 → 0.615; 755252 K=20 → 0.280; 767022 K=20 →
0.480; 782149 K=20 → 0.000.

Run: `python probe_topk_shift.py [subject_id ...]`
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa: E402
from lib.cell_features import extract_cell_features, invariant_feature_mask  # noqa: E402
from image_features import hcr_image_features  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def augmented_order(s, *, betas=(0.0,), img_bbox_um=(2.0, 3.0, 3.0)):
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    Fc, names, _ = extract_cell_features(s, "cz")
    Fg, _, _ = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
    Fcn = (Fc[:, keep] - mu) / sd; Fgn = (Fg[:, keep] - mu) / sd
    Fcn = Fcn / (np.linalg.norm(Fcn, axis=1, keepdims=True) + 1e-9)
    Fgn = Fgn / (np.linalg.norm(Fgn, axis=1, keepdims=True) + 1e-9)

    D = cdist(cz_init, hcr_um)
    cosS = Fcn @ Fgn.T

    # HCR image features for the GFP+ subset only.
    t0 = time.time()
    hcr_px_all = s.hcr_centroids[["z_px", "y_px", "x_px"]].values
    hcr_ids_all = s.hcr_centroids["hcr_id"].astype(int).values
    gfp_ids = set(s.hcr_gfp_df["hcr_id"].astype(int).tolist())
    keep_hcr = np.array([int(i) in gfp_ids for i in hcr_ids_all])
    hcr_px_gfp = hcr_px_all[keep_hcr]
    hcr_img_gfp = hcr_image_features(s, hcr_px_gfp, channel="488",
                                     level=2, bbox_um=img_bbox_um)
    hcr_img_ids = hcr_ids_all[keep_hcr]
    print(f"  hcr_img in {time.time()-t0:.1f}s  n={len(hcr_img_gfp)}",
          flush=True)

    # Align rows to the centroid helper's hcr_ids order.
    hid_to_row = {int(h): i for i, h in enumerate(hcr_img_ids)}
    hcr_img = np.array([hcr_img_gfp[hid_to_row[int(h)]] for h in hcr_ids])

    # Within-HCR z-score each column; sum as scalar quality.
    mu = np.nanmean(hcr_img, axis=0)
    sd = np.nanstd(hcr_img, axis=0) + 1e-6
    zI = (hcr_img - mu) / sd
    zI = np.nan_to_num(zI, nan=0.0)
    hcr_quality = zI.sum(axis=1)  # (N_hcr,)
    print(f"  hcr_quality range: [{hcr_quality.min():.2f}, "
          f"{hcr_quality.max():.2f}]  mean={hcr_quality.mean():.2f}",
          flush=True)

    orders = {}
    for b in betas:
        score = D - 25.0 * cosS - float(b) * hcr_quality[None, :]
        orders[b] = np.argsort(score, axis=1)
    return orders, cz_ids, hcr_ids


def topk_curve(order, cz_ids, hcr_ids, coreg_table, Ks):
    gt_map = dict(zip(coreg_table["cz_id"].astype(int),
                      coreg_table["hcr_id"].astype(int)))
    hcr_pos = {int(h): i for i, h in enumerate(hcr_ids)}
    ranks = []
    for i_cz, cid in enumerate(cz_ids):
        gt_h = gt_map.get(int(cid))
        if gt_h is None:
            continue
        gt_pos = hcr_pos.get(int(gt_h))
        if gt_pos is None:
            ranks.append(-1); continue
        m = np.where(order[i_cz] == gt_pos)[0]
        ranks.append(int(m[0]) if len(m) else -1)
    ranks = np.array(ranks)
    out = {"n": len(ranks)}
    for K in Ks:
        out[K] = float(((ranks >= 0) & (ranks < K)).mean())
    present = ranks[ranks >= 0]
    if len(present):
        out["p50"] = int(np.percentile(present, 50))
        out["p95"] = int(np.percentile(present, 95))
        out["max"] = int(present.max())
    return out


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    betas = [0.0, 5.0, 15.0, 30.0, 60.0]
    Ks = [5, 20, 50, 100, 500]

    all_rows = []
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        orders, cz_ids, hcr_ids = augmented_order(s, betas=betas)
        for b in betas:
            curve = topk_curve(orders[b], cz_ids, hcr_ids, s.coreg_table, Ks)
            row = {"subject": sid, "beta": b,
                   **{f"K={k}": curve.get(k, np.nan) for k in Ks},
                   "p50": curve.get("p50", -1), "p95": curve.get("p95", -1),
                   "max": curve.get("max", -1), "n": curve["n"]}
            msg = "  ".join([f"K={k}={curve.get(k, np.nan):.3f}" for k in Ks])
            print(f"  beta={b:5.1f}  {msg}  p50={curve.get('p50',-1)}  "
                  f"p95={curve.get('p95',-1)}", flush=True)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_csv = "/root/capsule/code/full_automatic_execution_01/sessions/46_cz_image_features/topk_shift.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    print("\n=== SUMMARY — GT-in-top-20 as a function of beta (HCR-quality weight) ===")
    pivot = df.pivot(index="beta", columns="subject", values="K=20")
    print(pivot.to_string())
    print("\n=== SUMMARY — GT-in-top-5 (baseline) as function of beta ===")
    pivot5 = df.pivot(index="beta", columns="subject", values="K=5")
    print(pivot5.to_string())


if __name__ == "__main__":
    main()
