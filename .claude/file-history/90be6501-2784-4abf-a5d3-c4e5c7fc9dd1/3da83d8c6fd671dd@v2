"""S53 — P6 CPD nonrigid benchmark across 4 primary + 2 stress subjects."""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import run_candidate  # noqa: E402
import bench.candidates  # noqa: F401, E402


PRIMARY = ["788406", "790322", "755252", "767022"]
STRESS = ["767018", "782149"]


def main():
    rows = []
    for subj in PRIMARY + STRESS:
        t0 = time.time()
        try:
            r = run_candidate("P6", subj, write_csv=False,
                              extra_kwargs=dict(method="cpd_nonrigid",
                                                 maxiter=30, w_outlier=0.3,
                                                 crop_pad_um=150.0,
                                                 match_radius_um=60.0,
                                                 hcr_quality_beta=0.0))
            dt = time.time() - t0
            print(f'{subj} r@5={r.get("recall_at_5um", 0):.3f} '
                  f'r@10={r.get("recall_at_10um", 0):.3f} '
                  f'r@20={r.get("recall_at_20um", 0):.3f} '
                  f'med={r.get("median_error_um", float("nan")):.1f}µm '
                  f'wall={dt:.1f}s', flush=True)
            rows.append((subj, r.get("recall_at_20um", 0),
                         r.get("median_error_um", float("nan")), dt))
        except Exception as e:
            print(f'{subj} FAILED: {e}', flush=True)
            rows.append((subj, 0.0, float("nan"), 0.0))

    print()
    print("Summary:")
    for subj, r20, med, dt in rows:
        print(f"  {subj}: r@20={r20:.3f} med={med:.1f}µm wall={dt:.1f}s")


if __name__ == "__main__":
    main()
