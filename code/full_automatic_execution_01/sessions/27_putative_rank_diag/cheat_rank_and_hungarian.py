"""With the landmark-based cheat warm-start, check:
 1. GT HCR partner's rank in proximity from cz_init.
 2. Pure Hungarian match accuracy at cz_init.
 3. Pure greedy nearest-neighbour match accuracy.

If Hungarian on the landmark-cheat cz_init still gives 0, the issue isn't P1 —
it's how we're mapping hcr IDs back.
"""
from __future__ import annotations
import sys, json
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
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    cz_init = (cz_um * fit.scales) @ fit.R.T + fit.translation

    gt = s.coreg_table.copy()
    cz_idx_map = {int(c): i for i, c in enumerate(cz_ids)}
    hcr_idx_map = {int(h): i for i, h in enumerate(hcr_ids)}
    gt_pairs = []
    for _, row in gt.iterrows():
        ci = cz_idx_map.get(int(row["cz_id"]))
        hi = hcr_idx_map.get(int(row["hcr_id"]))
        if ci is not None and hi is not None:
            gt_pairs.append((ci, hi))
    print(f"GT pairs reachable: {len(gt_pairs)}")

    # Rank of GT
    D = cdist(cz_init, hcr_um)
    ranks = []
    dists_gt = []
    for ci, hi in gt_pairs:
        order = np.argsort(D[ci])
        ranks.append(int(np.where(order == hi)[0][0]))
        dists_gt.append(D[ci, hi])
    ranks = np.array(ranks)
    dists_gt = np.array(dists_gt)
    print(f"GT-partner distance at cz_init (µm): median={np.median(dists_gt):.1f}, "
          f"p50={np.percentile(dists_gt, 50):.1f}, p95={np.percentile(dists_gt, 95):.1f}")
    print(f"GT-partner rank (pure distance): "
          f"p50={int(np.median(ranks))}, "
          f"lt1={int((ranks < 1).sum())}, "
          f"lt5={int((ranks < 5).sum())}, "
          f"lt20={int((ranks < 20).sum())}")

    # Greedy NN match
    greedy = np.argmin(D, axis=1)
    correct_greedy = sum(
        1 for (ci, hi), g in zip(gt_pairs, [greedy[ci] for ci, _ in gt_pairs]) if g == hi
    )
    print(f"Greedy NN match correct: {correct_greedy}/{len(gt_pairs)}")

    # Hungarian on all-vs-all
    # If |HCR| >> |CZ|, this is a partial assignment. Use rectangular LAP.
    print(f"Running Hungarian on D.shape = {D.shape}...")
    import time
    t0 = time.time()
    row_ind, col_ind = linear_sum_assignment(D)
    t1 = time.time() - t0
    print(f"Hungarian runtime: {t1:.1f}s")
    hung_assign = {int(row_ind[k]): int(col_ind[k]) for k in range(len(row_ind))}
    correct_hung = sum(1 for (ci, hi) in gt_pairs if hung_assign.get(ci) == hi)
    print(f"Hungarian match correct: {correct_hung}/{len(gt_pairs)}")

    # Save
    out = _ROOT / "sessions/27_putative_rank_diag"
    json.dump({
        "landmark_fit_rms_um": float(fit.rms_um),
        "gt_partner_dist_at_cz_init": {
            "p50": float(np.median(dists_gt)),
            "p95": float(np.percentile(dists_gt, 95)),
            "p99": float(np.percentile(dists_gt, 99)),
            "max": float(dists_gt.max()),
        },
        "rank_pure_distance": {
            "p50": int(np.median(ranks)),
            **{f"lt_{K}": int((ranks < K).sum()) for K in (1, 5, 10, 20, 50, 100)},
        },
        "greedy_correct": correct_greedy,
        "hungarian_correct": correct_hung,
        "n_gt": len(gt_pairs),
        "hungarian_runtime_s": t1,
    }, open(out / "cheat_rank_and_hungarian.json", "w"), indent=2)


if __name__ == "__main__":
    main()
