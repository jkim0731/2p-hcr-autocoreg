"""S53 — P6 BCPD single-subject probe."""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import run_candidate  # noqa: E402
import bench.candidates  # noqa: F401, E402


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "788406"
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    print(f"P6 BCPD on {subj}, maxiter={maxiter}, w={w}", flush=True)
    t0 = time.time()
    r = run_candidate("P6", subj, write_csv=False,
                      extra_kwargs=dict(maxiter=maxiter, w_outlier=w))
    dt = time.time() - t0
    print(f"subj={subj} maxiter={maxiter} w={w} "
          f"r@5={r.get('recall_at_5um', 0.0):.3f} "
          f"r@10={r.get('recall_at_10um', 0.0):.3f} "
          f"r@20={r.get('recall_at_20um', 0.0):.3f} "
          f"med={r.get('median_error_um', float('nan')):.1f} "
          f"wall={dt:.1f}s")


if __name__ == "__main__":
    main()
