"""S46-d — benchmark C1 (I2 warm-start → P1) after I2 init-direction fix."""
from __future__ import annotations

import sys

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401
from bench.harness import run_candidate  # noqa: E402


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    for subj in subjects:
        # P1 (baseline, ship defaults)
        r_p1 = run_candidate("P1", subj, write_csv=False)
        # C1 (I2-warmstart → P1)
        r_c1 = run_candidate("C1", subj, write_csv=False)
        def fmt(r):
            return (f"r@5={r.get('recall_at_5um',0):.3f} "
                    f"r@20={r.get('recall_at_20um',0):.3f} "
                    f"rid={r.get('recall',0):.3f} "
                    f"med={r.get('median_error_um') or 0:.1f}µm "
                    f"n={r.get('n_pred',0)} "
                    f"rt={r.get('runtime_s',0):.1f}s")
        print(f"\n  {subj}:")
        print(f"    P1: {fmt(r_p1)}")
        print(f"    C1: {fmt(r_c1)}")


if __name__ == "__main__":
    main()
