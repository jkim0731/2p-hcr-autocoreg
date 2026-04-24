"""S60 — benchmark C6b (consensus + C5 fallback) vs C5 on 6 subjects.

C6b matches C5's density-dispatch rule for the fallback (union_conf on
sparse, priority on mid/dense) but overrides with consensus pick when
≥ 2 methods agree. Isolates consensus-first effect from sparse-fallback
divergence that hurt C6 v1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

import bench.candidates  # noqa: F401
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
    denom = len(gt)
    r20 = hits / max(denom, 1)
    med = float(np.median(errs)) if errs else float("inf")
    return r20, med, hits, total, denom


def main():
    subjects = ["788406", "790322", "755252", "767022", "767018", "782149"]
    # C5 baselines from sessions/60_vote_consensus_probe/bench_c6.csv
    c5_r20 = {
        "788406": 0.2617534942820839,
        "790322": 0.2892030848329049,
        "755252": 0.09546165884194054,
        "767022": 0.11727616645649433,
        "767018": 0.315018315018315,
        "782149": 0.0,
    }
    rows = []
    sum_c5 = sum(c5_r20.values())
    sum_c6b = 0.0
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)

        t0 = time.time()
        r = CANDIDATES["C6b"](s)
        w = time.time() - t0
        r20, med, hits, tot, denom = score(r.pairs_df, s)
        c5_val = c5_r20[sid]
        delta = r20 - c5_val
        print(f"  C5  r@20={c5_val:.3f} (cached)", flush=True)
        print(f"  C6b r@20={r20:.3f} med={med:.1f}µm hits={hits}/{tot} "
              f"emitted={len(r.pairs_df)} wall={w:.1f}s", flush=True)
        print(f"  Δr@20 (C6b - C5) = {delta:+.3f}", flush=True)
        rows.append(dict(subject=sid,
                         c5_r20=c5_val,
                         c6b_r20=r20, c6b_hits=hits, c6b_n=len(r.pairs_df),
                         delta=delta))
        sum_c6b += r20

    out = pd.DataFrame(rows)
    out.to_csv("sessions/60_vote_consensus_probe/bench_c6b.csv", index=False)

    print(f"\n=== SUMS ===", flush=True)
    print(f"  Sum C5  r@20 = {sum_c5:.3f}", flush=True)
    print(f"  Sum C6b r@20 = {sum_c6b:.3f}", flush=True)
    print(f"  Δ (C6b - C5) = {sum_c6b - sum_c5:+.3f}", flush=True)


if __name__ == "__main__":
    main()
