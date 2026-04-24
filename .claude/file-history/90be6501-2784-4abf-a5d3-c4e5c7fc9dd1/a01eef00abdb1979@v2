"""S52 — Quick single-subject probe of asymmetric G1."""
from __future__ import annotations

import sys
import time
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import run_candidate  # noqa: E402
import bench.candidates  # noqa: F401, E402


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "788406"
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    use_f6 = (sys.argv[3].lower() in ("1", "true", "yes")) if len(sys.argv) > 3 else False
    asym = (sys.argv[4].lower() in ("1", "true", "yes")) if len(sys.argv) > 4 else True
    print(f"G1 asymmetric={asym} use_f6={use_f6} on {subj}, n_train_iter={n_iter}", flush=True)
    t0 = time.time()
    r = run_candidate("G1", subj, write_csv=False,
                      extra_kwargs=dict(n_train_iter=n_iter, asymmetric=asym, use_f6=use_f6))
    dt = time.time() - t0
    print(f"subj={subj} n_iter={n_iter} asym={asym} use_f6={use_f6} "
          f"rec_id={r.get('recall', 0.0):.3f} "
          f"r@5={r.get('recall_at_5um', 0.0):.3f} "
          f"r@10={r.get('recall_at_10um', 0.0):.3f} "
          f"r@20={r.get('recall_at_20um', 0.0):.3f} "
          f"med={r.get('median_error_um', float('nan')):.1f} "
          f"wall={dt:.1f}s")


if __name__ == "__main__":
    main()
