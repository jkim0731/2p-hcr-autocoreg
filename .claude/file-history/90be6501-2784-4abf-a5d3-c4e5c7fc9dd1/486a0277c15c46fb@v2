"""Cheat with CORRECT coordinate order.

fit_anisotropic_similarity expects (x, y, z). centroids_um returns (z, y, x).
The previous cheat mixed them — this test corrects it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject, landmark_pairs_um
from benchmark_analysis import fit_anisotropic_similarity
from lib.centroid_helpers import centroids_um


def main(sid="788406"):
    s = load_subject(sid)
    cz_um_zyx, cz_ids = centroids_um(s, "cz")      # (z, y, x)
    hcr_um_zyx, hcr_ids = centroids_um(s, "hcr_gfp") # (z, y, x)

    # Convert centroids to (x, y, z) to match landmark_pairs_um convention
    cz_um_xyz = cz_um_zyx[:, [2, 1, 0]]
    hcr_um_xyz = hcr_um_zyx[:, [2, 1, 0]]

    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)  # (x, y, z)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)  # fit in (x,y,z)
    print(f"Landmark fit (xyz): scales={fit.scales}, rms={fit.rms_um:.2f}µm")

    # Apply fit to centroids in (x, y, z) order using the FITTER's convention
    src_mean_xyz = cz_lm.mean(0)
    dst_mean_xyz = hcr_lm.mean(0)
    # Forward: pred = ((src - src_mean) @ R) * scales + dst_mean
    cz_init_xyz = ((cz_um_xyz - src_mean_xyz) @ fit.R) * fit.scales + dst_mean_xyz
    # Convert back to (z, y, x)
    cz_init_zyx = cz_init_xyz[:, [2, 1, 0]]

    gt = s.coreg_table.copy()
    cz_idx_map = {int(c): i for i, c in enumerate(cz_ids)}
    hcr_idx_map = {int(h): i for i, h in enumerate(hcr_ids)}
    gt_pairs = [(cz_idx_map[int(r["cz_id"])], hcr_idx_map[int(r["hcr_id"])])
                for _, r in gt.iterrows()
                if cz_idx_map.get(int(r["cz_id"])) is not None
                and hcr_idx_map.get(int(r["hcr_id"])) is not None]

    # Distance to GT partners in (z, y, x)
    d_gt = np.array([np.linalg.norm(cz_init_zyx[ci] - hcr_um_zyx[hi])
                     for ci, hi in gt_pairs])
    print(f"CZ-warp → GT-HCR distance (corrected ordering):")
    print(f"  median={np.median(d_gt):.1f}, p95={np.percentile(d_gt, 95):.1f}, "
          f"min={d_gt.min():.1f}, max={d_gt.max():.1f}")
    print(f"  <5µm: {(d_gt<5).sum()}, <10µm: {(d_gt<10).sum()}, "
          f"<20µm: {(d_gt<20).sum()}, <50µm: {(d_gt<50).sum()}, "
          f"<100µm: {(d_gt<100).sum()}")

    # Rank
    D = cdist(cz_init_zyx, hcr_um_zyx)
    ranks = []
    for ci, hi in gt_pairs:
        order = np.argsort(D[ci])
        ranks.append(int(np.where(order == hi)[0][0]))
    ranks = np.array(ranks)
    print(f"\nGT-partner rank (pure distance): "
          f"p50={int(np.median(ranks))}, lt1={int((ranks<1).sum())}, "
          f"lt5={int((ranks<5).sum())}, lt10={int((ranks<10).sum())}, "
          f"lt20={int((ranks<20).sum())}, lt50={int((ranks<50).sum())}")

    # Greedy nearest-neighbour
    greedy = np.argmin(D, axis=1)
    correct_greedy = sum(1 for ci, hi in gt_pairs if greedy[ci] == hi)
    print(f"\nGreedy NN correct: {correct_greedy}/{len(gt_pairs)}")

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(D)
    hung_map = {int(row_ind[k]): int(col_ind[k]) for k in range(len(row_ind))}
    correct_hung = sum(1 for ci, hi in gt_pairs if hung_map.get(ci) == hi)
    print(f"Hungarian correct: {correct_hung}/{len(gt_pairs)}")


if __name__ == "__main__":
    main()
