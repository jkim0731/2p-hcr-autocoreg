"""S57 — I3 probe on 782149 (the centroid-unreachable subject).

C5 (P1⊕P4⊕P6) scores r@20=0 on 782149; this subject has 12° pia tilt + thin
Z (878 µm) + 34% match rate. Test whether I3's image-level affine (skip_bspline
= I2 + NN pair emission) recovers any signal.

If I3_skip_bspline gives r@20 > 0 here, that's a breakthrough. If not, I3 via
direct pair emission is structurally wrong and we should document + defer.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import CANDIDATES, compare_to_gt  # noqa: E402
import bench.candidates  # noqa: F401, E402
from benchmark_data_loader import load_subject  # noqa: E402


def main():
    for sid in ["782149", "755252", "767018"]:
        print(f"\n=========== {sid} ===========", flush=True)
        s = load_subject(sid)
        print(f"n_cz={len(s.cz_centroids)} n_hcr_gfp={len(s.hcr_gfp_df)}",
              flush=True)

        # Only test skip_bspline — the B-spline path regressed on 788406, and
        # the I2-affine-alone baseline is the cheap-and-meaningful first test.
        for name, kwargs in [
            ("I3_skip_bspline", dict(skip_bspline=True)),
        ]:
            print(f"\n--- {name} ---", flush=True)
            t0 = time.time()
            try:
                r = CANDIDATES["I3"](s, **kwargs)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            wall = time.time() - t0
            d = r.diagnostics
            pairs = r.pairs_df
            print(f"  wall={wall:.1f}s n_pairs={len(pairs)} "
                  f"nn_p50={d.get('nn_dists_p50', 0):.1f} µm "
                  f"nn_p90={d.get('nn_dists_p90', 0):.1f} µm",
                  flush=True)
            if len(pairs):
                sc = compare_to_gt(pairs, s)
                print(f"  r@5={sc['recall_at_5um']:.3f} "
                      f"r@10={sc['recall_at_10um']:.3f} "
                      f"r@20={sc['recall_at_20um']:.3f} "
                      f"med_err={sc['median_error_um']:.1f} "
                      f"p95={sc['p95_error_um']:.1f} µm "
                      f"n_pred={sc['n_pred']} n_gt={sc['n_gt']}", flush=True)
            else:
                print("  no pairs emitted", flush=True)


if __name__ == "__main__":
    main()
