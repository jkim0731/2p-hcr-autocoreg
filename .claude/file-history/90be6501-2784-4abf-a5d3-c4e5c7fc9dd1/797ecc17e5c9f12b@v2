"""S39 probe — fresh P1 TEASER baseline on the 4 benchmark subjects.

S38 ended with a recommendation to pivot from G1 GNN to P1 TEASER. Before
tuning K / c_bar / TPS params, get the current P1-at-default recall on all 4
subjects so we know the starting point and can compare against S28/S29's
P1/P14/C1 numbers.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from bench.harness import run_candidate  # noqa
import bench.candidates  # noqa: F401  (registers all)


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    rows = []
    for subj in subjects:
        print(f"\n=== P1 on {subj} ===", flush=True)
        t0 = time.time()
        row = run_candidate("P1", subj, write_csv=True)
        row["wall_s"] = round(time.time() - t0, 1)
        print(f"  rec@5={row.get('rec@5', 0):.3f} "
              f"rec@10={row.get('rec@10', 0):.3f} "
              f"rec@20={row.get('rec@20', 0):.3f} "
              f"n_pred={row.get('n_pred', 0)} n_gt={row.get('n_gt_covered', 0)} "
              f"wall={row['wall_s']}s", flush=True)
        rows.append(row)

    print("\n=== SUMMARY ===")
    import pandas as pd
    cols = ["subject_id", "n_pred", "n_gt_covered", "rec@5", "rec@10", "rec@20",
            "median_err_um", "wall_s"]
    df = pd.DataFrame(rows)
    keep = [c for c in cols if c in df.columns]
    print(df[keep].to_string(index=False))


if __name__ == "__main__":
    main()
