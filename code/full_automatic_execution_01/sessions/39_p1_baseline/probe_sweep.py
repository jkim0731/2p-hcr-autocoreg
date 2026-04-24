"""S39 sweep — P1 K × c_bar on 4 benchmark subjects.

Default P1 uses K=5 (top-5 F6+distance putatives per CZ) and c_bar=15 µm
(TLS inlier threshold). Baseline rec@5: 788406=0.20, 755252=0.03, 767022=0.08,
782149=0.00. Sweep K ∈ {5, 10, 20, 40} × c_bar ∈ {15, 30, 60} to see if
widening the putative pool or the inlier threshold lifts stress subjects.

Reports the best configuration per subject plus the run matrix.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from bench.harness import run_candidate  # noqa
import bench.candidates  # noqa: F401
import pandas as pd


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    configs = []
    for K in [5, 10, 20, 40]:
        for cbar in [15.0, 30.0, 60.0]:
            configs.append((K, cbar))
    rows = []
    for subj in subjects:
        for (K, cbar) in configs:
            t0 = time.time()
            row = run_candidate("P1", subj, write_csv=False,
                                 extra_kwargs=dict(K=K, c_bar=cbar))
            wall = round(time.time() - t0, 1)
            r5 = row.get("recall_at_5um", 0)
            r10 = row.get("recall_at_10um", 0)
            r20 = row.get("recall_at_20um", 0)
            recall = row.get("recall", 0)
            n_gt = row.get("n_gt", 0)
            n_pred = row.get("n_pred", 0)
            med = row.get("median_error_um", float('nan'))
            print(f"  {subj} K={K:2d} c_bar={cbar:4.0f}  rec_id={recall:.3f} "
                  f"r@5={r5:.3f} r@10={r10:.3f} r@20={r20:.3f} "
                  f"n_pred={n_pred}/{n_gt} med={med:.0f} wall={wall}s",
                  flush=True)
            rows.append(dict(subject=subj, K=K, c_bar=cbar,
                              recall_id=recall, r5=r5, r10=r10, r20=r20,
                              n_pred=n_pred, n_gt=n_gt, median_err=med,
                              wall=wall))

    df = pd.DataFrame(rows)
    df.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/39_p1_baseline/sweep.csv",
        index=False,
    )

    print("\n=== BEST PER SUBJECT (by r@20) ===")
    for subj in subjects:
        sub = df[df.subject == subj]
        best = sub.sort_values("r20", ascending=False).iloc[0]
        print(f"  {subj}: K={int(best.K)} c_bar={best.c_bar:.0f} "
              f"r@20={best.r20:.3f} r@5={best.r5:.3f} "
              f"recall_id={best.recall_id:.3f}")


if __name__ == "__main__":
    main()
