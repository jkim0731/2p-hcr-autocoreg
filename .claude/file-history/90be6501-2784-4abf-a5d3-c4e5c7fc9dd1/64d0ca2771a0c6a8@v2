"""S56 — Validate C5 candidate registration via F9 harness on 6 subjects."""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import run_candidate  # noqa: E402
import bench.candidates  # noqa: F401, E402


SUBJECTS = ["788406", "790322", "755252", "767022", "767018", "782149"]


def main():
    rows = []
    for subj in SUBJECTS:
        t0 = time.time()
        try:
            r = run_candidate("C5", subj, write_csv=True,
                              extra_kwargs=dict(method="auto",
                                                 sparse_threshold=10_000,
                                                 dense_threshold=20_000,
                                                 verbose=True))
            dt = time.time() - t0
            print(f"{subj} C5-auto r@5={r.get('recall_at_5um',0):.3f} "
                  f"r@10={r.get('recall_at_10um',0):.3f} "
                  f"r@20={r.get('recall_at_20um',0):.3f} "
                  f"n_pred={r.get('n_pred',0)} "
                  f"med={r.get('median_error_um', float('nan')):.1f}µm "
                  f"wall={dt:.1f}s", flush=True)
            rows.append((subj, r.get("recall_at_20um", 0),
                         r.get("median_error_um", float("nan")), dt))
        except Exception as e:
            print(f"{subj} FAILED: {e}", flush=True)
            rows.append((subj, 0.0, float("nan"), 0.0))

    print()
    print("Summary (C5 auto):")
    total_r20 = 0
    for subj, r20, med, dt in rows:
        print(f"  {subj}: r@20={r20:.3f} med={med:.1f}µm wall={dt:.1f}s")
        total_r20 += r20
    print(f"  SUM r@20 = {total_r20:.3f}")


if __name__ == "__main__":
    main()
