"""S60 — benchmark C6 (consensus-vote) vs C5 (priority) on 6 subjects.

Headline metric: sum r@20. C5 baseline 1.080 (near single-method-oracle
1.081); C6 success = sum r@20 > 1.080 means consensus-vote is a
Pareto improvement over priority dispatch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

import bench.candidates  # noqa: F401  (registers all)
from bench.harness import CANDIDATES  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402


def score(df, s):
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    hcr_by_id = dict(zip(hcr_ids, hcr_um))
    coreg = s.coreg_table
    gt = {int(r['cz_id']): int(r['hcr_id']) for _, r in coreg.iterrows()}
    hits = 0; total = 0; errs = []
    for _, r in df.iterrows():
        c = int(r['cz_id'])
        if c not in gt:
            continue
        total += 1
        gh = gt[c]
        if gh not in hcr_by_id:
            continue
        gt_pos = hcr_by_id[gh]
        pr_pos = np.array([r['hcr_z_um'], r['hcr_y_um'], r['hcr_x_um']])
        d = np.linalg.norm(gt_pos - pr_pos)
        errs.append(d)
        if d <= 20.0:
            hits += 1
    # Use |coreg_table| as denominator to match S56's reporting convention
    denom = len(gt)
    r20 = hits / max(denom, 1)
    med = float(np.median(errs)) if errs else float("inf")
    return r20, med, hits, total, denom


def main():
    subjects = ["788406", "790322", "755252", "767022", "767018", "782149"]
    rows = []
    sum_c5 = 0.0
    sum_c6 = 0.0
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)

        t0 = time.time()
        r5 = CANDIDATES["C5"](s)
        w5 = time.time() - t0
        r20_5, med_5, hits_5, tot_5, denom_5 = score(r5.pairs_df, s)
        print(f"  C5 r@20={r20_5:.3f} med={med_5:.1f}µm hits={hits_5}/{tot_5} "
              f"emitted={len(r5.pairs_df)} wall={w5:.1f}s", flush=True)

        t0 = time.time()
        r6 = CANDIDATES["C6"](s)
        w6 = time.time() - t0
        r20_6, med_6, hits_6, tot_6, denom_6 = score(r6.pairs_df, s)
        print(f"  C6 r@20={r20_6:.3f} med={med_6:.1f}µm hits={hits_6}/{tot_6} "
              f"emitted={len(r6.pairs_df)} wall={w6:.1f}s", flush=True)

        delta = r20_6 - r20_5
        print(f"  Δr@20 = {delta:+.3f}", flush=True)
        rows.append(dict(subject=sid,
                         c5_r20=r20_5, c5_hits=hits_5, c5_n=len(r5.pairs_df),
                         c6_r20=r20_6, c6_hits=hits_6, c6_n=len(r6.pairs_df),
                         delta=delta))
        sum_c5 += r20_5; sum_c6 += r20_6

    out = pd.DataFrame(rows)
    out.to_csv("sessions/60_vote_consensus_probe/bench_c6.csv", index=False)

    print(f"\n=== SUMS ===", flush=True)
    print(f"  Sum C5 r@20 = {sum_c5:.3f}", flush=True)
    print(f"  Sum C6 r@20 = {sum_c6:.3f}", flush=True)
    print(f"  Δ           = {sum_c6 - sum_c5:+.3f}", flush=True)


if __name__ == "__main__":
    main()
