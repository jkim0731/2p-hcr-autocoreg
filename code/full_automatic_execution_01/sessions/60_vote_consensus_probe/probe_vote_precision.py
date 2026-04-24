"""S60 — method-vote consensus probe.

Hypothesis: pairs (cz_id, hcr_id) voted by multiple methods of
{P1, P4, P6} are substantially more precise than single-vote pairs.
If so, a learned QF1 gate (or even a simple 2-vote threshold) can
lift C5's r@20 on non-saturated subjects by dropping false positives.

Scoring: per-tier (1-vote, 2-vote, 3-vote) count pairs whose cz_id
is in coreg_table, count hits at 20 µm.

Fail-fast criterion: if r@20 in the 2-vote tier is not >= 1.5x the
1-vote tier, consensus filtering won't lift C5 meaningfully and we
accept the plateau.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._p1_teaser import run_p1  # noqa: E402
from bench.candidate_impls._p4_spectral import run_p4  # noqa: E402
from bench.candidate_impls._p6_bcpd import run_p6  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402


def is_hit_20um(pred_hcr_pos, gt_hcr_id, hcr_by_id):
    if gt_hcr_id not in hcr_by_id:
        return False
    return np.linalg.norm(pred_hcr_pos - hcr_by_id[gt_hcr_id]) <= 20.0


def tier_stats(pairs_dict, s):
    """pairs_dict: dict of {(cz_id, hcr_id): set_of_methods}.

    Returns per-vote-count stats: {n_votes: (n, in_gt, hits@20)}.
    """
    _, hcr_ids = centroids_um(s, "hcr_gfp")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    hcr_by_id = dict(zip(hcr_ids, hcr_um))
    coreg = s.coreg_table
    gt = {int(r['cz_id']): int(r['hcr_id']) for _, r in coreg.iterrows()}

    tiers = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
    for (cz_id, hcr_id), methods in pairs_dict.items():
        n_votes = len(methods)
        tiers[n_votes][0] += 1  # total emitted
        if cz_id not in gt:
            continue
        tiers[n_votes][1] += 1  # in gt
        # need the predicted hcr position — find it from any method's df
        # since methods agree on hcr_id, any entry works
        j_by_id = {int(hid): i for i, hid in enumerate(hcr_ids)}
        if hcr_id not in j_by_id:
            continue
        pred_pos = hcr_um[j_by_id[hcr_id]]
        gh = gt[cz_id]
        if is_hit_20um(pred_pos, gh, hcr_by_id):
            tiers[n_votes][2] += 1
    return tiers


def main():
    subjects = ["788406", "790322", "767018"]
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        p1 = run_p1(s)
        p4 = run_p4(s)
        p6 = run_p6(s)
        print(f"  P1={len(p1.pairs_df)} P4={len(p4.pairs_df)} P6={len(p6.pairs_df)}",
              flush=True)

        pairs = {}
        for label, df in [("P1", p1.pairs_df), ("P4", p4.pairs_df), ("P6", p6.pairs_df)]:
            for _, r in df.iterrows():
                k = (int(r['cz_id']), int(r['hcr_id']))
                pairs.setdefault(k, set()).add(label)

        tiers = tier_stats(pairs, s)
        print(f"  votes=1: n={tiers[1][0]:4d}  in_gt={tiers[1][1]:4d}  "
              f"hits@20={tiers[1][2]:3d}  "
              f"p@gt={tiers[1][2]/max(tiers[1][1],1):.3f}",
              flush=True)
        print(f"  votes=2: n={tiers[2][0]:4d}  in_gt={tiers[2][1]:4d}  "
              f"hits@20={tiers[2][2]:3d}  "
              f"p@gt={tiers[2][2]/max(tiers[2][1],1):.3f}",
              flush=True)
        print(f"  votes=3: n={tiers[3][0]:4d}  in_gt={tiers[3][1]:4d}  "
              f"hits@20={tiers[3][2]:3d}  "
              f"p@gt={tiers[3][2]/max(tiers[3][1],1):.3f}",
              flush=True)
        # Ratio check: 2-vote vs 1-vote precision
        p1v = tiers[1][2] / max(tiers[1][1], 1)
        p2v = tiers[2][2] / max(tiers[2][1], 1)
        p3v = tiers[3][2] / max(tiers[3][1], 1)
        ratio = p2v / max(p1v, 0.01)
        print(f"  2-vote/1-vote precision ratio = {ratio:.2f}x", flush=True)


if __name__ == "__main__":
    main()
