"""v2-S03 sub-stages (b) + (c) — P3 RANSAC + P5 FGW with LP warm-start.

Same monkey-patch trick as bench_a: `default_warmstart_zyx` returns
locked-prior-warped CZ centroids, so P3 (RANSAC over putatives) and P5
(FGW with sxy-rescaled CZ centroids implicit in the LP warp) both
consume the Stage A frame.

This is the "soft" version of the v2 plan §5 (b)/(c); the v2 plan also
calls for sxy + t_xy *locked* inside P3's RANSAC fit.  We first
benchmark the naive LP-warmstart version; if it doesn't beat v1, the
next iter will add a constrained-aniso-affine fit to P3.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_02/lib")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_02/sessions/v2_S03_revival_pass")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/sessions/03c_onset_features/iterations")

import pandas as pd

from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject  # noqa: E402
from bench_a_p1_p4_p6_c5 import patch_warmstart_to_locked_prior  # noqa: E402

OUT = Path("/root/capsule/code/full_automatic_execution_02/sessions/v2_S03_revival_pass")
METHODS = ["P3", "P5"]


def main():
    from bench.harness import run_candidate
    import bench.candidates  # noqa
    import lib.centroid_helpers as ch

    rows = []
    for sid in BENCHMARK_SUBJECTS:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        original, lp = patch_warmstart_to_locked_prior(s)
        print(f"  LP sxy={lp.sxy_value:.3f} sz_init={lp.scales[0]:.3f} "
              f"theta_z={lp.rotation_deg_z:.2f}° pwr_ncc={lp.pwr_ncc:.3f}",
              flush=True)
        try:
            for mid in METHODS:
                t0 = time.time()
                try:
                    r = run_candidate(mid, sid, write_csv=False)
                    dt = time.time() - t0
                    row = dict(
                        subject_id=sid, method=mid,
                        n_pred=int(r.get("n_pred", 0)),
                        recall_at_5um=float(r.get("recall_at_5um", 0)),
                        recall_at_10um=float(r.get("recall_at_10um", 0)),
                        recall_at_20um=float(r.get("recall_at_20um", 0)),
                        recall_at_30um=float(r.get("recall_at_30um", 0)),
                        median_error_um=float(r.get("median_error_um",
                                                     float("nan"))),
                        runtime_s=round(dt, 1),
                    )
                    rows.append(row)
                    print(f"  {mid}: r@10={row['recall_at_10um']:.3f} "
                          f"r@20={row['recall_at_20um']:.3f} "
                          f"r@30={row['recall_at_30um']:.3f} "
                          f"n={row['n_pred']:4d} "
                          f"med={row['median_error_um']:.1f}µm "
                          f"({dt:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  {mid} FAILED: {type(e).__name__}: {e}",
                          flush=True)
                    rows.append(dict(
                        subject_id=sid, method=mid, n_pred=0,
                        recall_at_5um=0.0, recall_at_10um=0.0,
                        recall_at_20um=0.0, recall_at_30um=0.0,
                        median_error_um=float("nan"),
                        runtime_s=round(time.time() - t0, 1),
                        error=f"{type(e).__name__}: {e}",
                    ))
        finally:
            ch.default_warmstart_zyx = original

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bench_bc_results.csv", index=False)
    print("\n=== summary ===")
    pivot = df.pivot(index="subject_id", columns="method",
                      values="recall_at_20um")
    print("r@20:")
    print(pivot.round(3).to_string())
    print(f"\nsum r@20:")
    print(pivot.sum().round(3).to_string())


if __name__ == "__main__":
    main()
