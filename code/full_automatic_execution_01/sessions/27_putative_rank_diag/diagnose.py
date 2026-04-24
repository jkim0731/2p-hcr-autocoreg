"""Diagnostic: after M1 warm-start, where does the GT HCR partner rank in P1's
top-K putative set for each CZ cell with known ground truth?

This separates two failure modes:
  (a) Coarse is still off — GT partner is far away in µm, any top-K based on
      distance+features won't include it.
  (b) Coarse is fine but feature+distance scoring doesn't rank GT highly — GT
      is geometrically close but semantically not top-K.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from bench.candidate_impls._m1_mask_ncc import run_m1
from lib.centroid_helpers import centroids_um
from lib.cell_features import extract_cell_features, invariant_feature_mask


def diagnose(subject_id="788406", K_sweep=(5, 10, 20, 50, 100)):
    s = load_subject(subject_id)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    print(f"Loaded subject {subject_id}: N_cz={len(cz_um)}, N_hcr_gfp={len(hcr_um)}")

    # Ground truth
    gt = s.coreg_table.copy()
    cz_id_to_idx = {int(cid): i for i, cid in enumerate(cz_ids)}
    hcr_id_to_idx = {int(hid): i for i, hid in enumerate(hcr_ids)}
    gt_pairs = []
    for _, row in gt.iterrows():
        ci = cz_id_to_idx.get(int(row["cz_id"]))
        hi = hcr_id_to_idx.get(int(row["hcr_id"]))
        if ci is not None and hi is not None:
            gt_pairs.append((ci, hi))
    print(f"GT pairs where both CZ and HCR are present: {len(gt_pairs)}")

    # Run M1
    r_m1 = run_m1(s)
    tr = r_m1.transform
    R = np.asarray(tr.R); S = np.asarray(tr.scales); t = np.asarray(tr.translation)
    src_mean = np.asarray(tr.src_mean)
    cz_init = ((cz_um - src_mean) * S) @ R.T + t
    print(f"M1 transform: sxy={S[1]:.2f}, sz={S[0]:.2f}, t={t}, src_mean={src_mean}")
    print(f"M1 diagnostics: {r_m1.diagnostics}")

    # Distance from warped CZ to its GT partner
    pred_gt_dist = []
    for ci, hi in gt_pairs:
        d = np.linalg.norm(cz_init[ci] - hcr_um[hi])
        pred_gt_dist.append(d)
    pred_gt_dist = np.array(pred_gt_dist)
    print(f"\nCZ-warp → GT-HCR distance distribution (µm):")
    print(f"  median={np.median(pred_gt_dist):.1f}, p95={np.percentile(pred_gt_dist, 95):.1f}, "
          f"max={pred_gt_dist.max():.1f}, min={pred_gt_dist.min():.1f}")
    print(f"  <5µm: {(pred_gt_dist < 5).sum()}, <10µm: {(pred_gt_dist < 10).sum()}, "
          f"<20µm: {(pred_gt_dist < 20).sum()}, <50µm: {(pred_gt_dist < 50).sum()}, "
          f"<100µm: {(pred_gt_dist < 100).sum()}")

    # Rebuild the full P1 putative ranker for every CZ cell
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

    # For each GT pair, the rank of the GT HCR within this CZ's sorted list.
    ranks = []
    for ci, hi in gt_pairs:
        order = np.argsort(score[ci])
        rank = int(np.where(order == hi)[0][0])
        ranks.append(rank)
    ranks = np.array(ranks)
    print(f"\nRank of GT HCR partner within P1 putative ranking (score = D - 25·cos):")
    print(f"  median rank = {int(np.median(ranks))}")
    for K in K_sweep:
        inside = int((ranks < K).sum())
        print(f"  <rank {K}: {inside}/{len(ranks)} = {100*inside/len(ranks):.1f}%")

    # Also rank by pure distance
    ranks_d = []
    for ci, hi in gt_pairs:
        order = np.argsort(D[ci])
        rank = int(np.where(order == hi)[0][0])
        ranks_d.append(rank)
    ranks_d = np.array(ranks_d)
    print(f"\nSame, ranked by pure distance only:")
    print(f"  median rank = {int(np.median(ranks_d))}")
    for K in K_sweep:
        inside = int((ranks_d < K).sum())
        print(f"  <rank {K}: {inside}/{len(ranks_d)} = {100*inside/len(ranks_d):.1f}%")

    # And pure cosine feature
    ranks_c = []
    for ci, hi in gt_pairs:
        order = np.argsort(-cosS[ci])
        rank = int(np.where(order == hi)[0][0])
        ranks_c.append(rank)
    ranks_c = np.array(ranks_c)
    print(f"\nSame, ranked by pure cosine (features) only:")
    print(f"  median rank = {int(np.median(ranks_c))}")
    for K in K_sweep:
        inside = int((ranks_c < K).sum())
        print(f"  <rank {K}: {inside}/{len(ranks_c)} = {100*inside/len(ranks_c):.1f}%")

    # Save to JSON for notebook.
    out = _ROOT / "sessions/27_putative_rank_diag"
    out.mkdir(parents=True, exist_ok=True)
    import json
    json.dump({
        "subject_id": subject_id,
        "n_gt_pairs": len(gt_pairs),
        "m1_diag": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in r_m1.diagnostics.items()},
        "pred_gt_dist_quantiles": {
            "p50": float(np.median(pred_gt_dist)),
            "p95": float(np.percentile(pred_gt_dist, 95)),
            "p99": float(np.percentile(pred_gt_dist, 99)),
            "max": float(pred_gt_dist.max()),
            "min": float(pred_gt_dist.min()),
        },
        "dist_thresholds_counts": {
            "lt5": int((pred_gt_dist < 5).sum()),
            "lt10": int((pred_gt_dist < 10).sum()),
            "lt20": int((pred_gt_dist < 20).sum()),
            "lt50": int((pred_gt_dist < 50).sum()),
            "lt100": int((pred_gt_dist < 100).sum()),
            "lt200": int((pred_gt_dist < 200).sum()),
        },
        "rank_distribution_score": {
            "p50": int(np.median(ranks)),
            "p95": int(np.percentile(ranks, 95)),
            **{f"lt_{K}": int((ranks < K).sum()) for K in K_sweep},
        },
        "rank_distribution_distance_only": {
            "p50": int(np.median(ranks_d)),
            **{f"lt_{K}": int((ranks_d < K).sum()) for K in K_sweep},
        },
        "rank_distribution_cosine_only": {
            "p50": int(np.median(ranks_c)),
            **{f"lt_{K}": int((ranks_c < K).sum()) for K in K_sweep},
        },
        "N_gt_with_both_ids": len(gt_pairs),
    }, open(out / "diagnose.json", "w"), indent=2)

    # Save per-pair arrays for notebook plot
    np.savez(out / "diagnose_arrays.npz",
             pred_gt_dist=pred_gt_dist,
             ranks_score=ranks,
             ranks_dist=ranks_d,
             ranks_cos=ranks_c)
    return {
        "pred_gt_dist": pred_gt_dist,
        "ranks": ranks,
        "ranks_d": ranks_d,
        "ranks_c": ranks_c,
        "gt_pairs": gt_pairs,
        "cz_init": cz_init,
        "hcr_um": hcr_um,
    }


if __name__ == "__main__":
    diagnose()
