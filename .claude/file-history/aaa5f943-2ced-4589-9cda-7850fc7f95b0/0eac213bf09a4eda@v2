"""v2-S03 sub-stage (a) — re-bench P1/P4/P6/C5 with Stage A locked prior.

The v1 candidates source their warm-start from
`lib.centroid_helpers.default_warmstart_zyx`, which uses the session-07
anisotropic ICP (REVOKED — GT-tuned scales).  We monkey-patch that
helper to return our locked-prior-warped CZ centroids, so all four
candidates (P1, P4, P6, C5) consume the Stage A frame uniformly.

Outputs:
* `bench_a_results.csv`  per-method × per-subject r@5/10/20/30, n_pred,
  median residual, runtime
* `bench_a_<sid>_<MID>.json`  raw diagnostics
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_02/lib")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/sessions/03c_onset_features/iterations")

import numpy as np
import pandas as pd

from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject  # noqa: E402
from locked_prior_warm import (  # noqa: E402
    apply_to_cz_um,
    compute_locked_prior_warm_start,
)

OUT = Path("/root/capsule/code/full_automatic_execution_02/sessions/v2_S03_revival_pass")
METHODS = ["P1", "P4", "P6", "C5"]


def patch_warmstart_to_locked_prior(s):
    """Monkey-patch `lib.centroid_helpers.default_warmstart_zyx` so that
    every candidate consumes the locked-prior warm-start.

    Returns the original function (caller restores it after the bench).
    """
    import lib.centroid_helpers as ch

    lp = compute_locked_prior_warm_start(s)
    original = ch.default_warmstart_zyx

    def _locked(cz_um_zyx, hcr_um_zyx, **_kw):
        cz_init = apply_to_cz_um(lp, cz_um_zyx)
        info = dict(
            warmstart_kind="locked_prior",
            sxy_value=float(lp.sxy_value),
            sxy_source=str(lp.sxy_source),
            sz_value=float(lp.scales[0]),
            translation_zyx=lp.translation.tolist(),
            rotation_deg_z=float(lp.rotation_deg_z),
            pwr_ncc=float(lp.pwr_ncc),
            pwr_method=str(lp.pwr_method),
        )
        return cz_init, info

    ch.default_warmstart_zyx = _locked
    return original, lp


def main():
    from bench.harness import run_candidate  # noqa: E402
    import bench.candidates  # noqa: F401, E402
    import lib.centroid_helpers as ch

    rows = []
    for sid in BENCHMARK_SUBJECTS:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        original, lp = patch_warmstart_to_locked_prior(s)
        print(
            f"  LP sxy={lp.sxy_value:.3f} sz_init={lp.scales[0]:.3f} "
            f"theta_z={lp.rotation_deg_z:.2f}° "
            f"t=(z={lp.translation[0]:+.0f}, y={lp.translation[1]:+.0f}, "
            f"x={lp.translation[2]:+.0f}) "
            f"pwr_ncc={lp.pwr_ncc:.3f} [{lp.pwr_method}]",
            flush=True,
        )
        try:
            for mid in METHODS:
                t0 = time.time()
                try:
                    extra = (
                        dict(method="auto", sparse_threshold=10_000,
                             dense_threshold=20_000, verbose=False)
                        if mid == "C5"
                        else dict()
                    )
                    r = run_candidate(mid, sid, write_csv=False,
                                      extra_kwargs=extra)
                    dt = time.time() - t0
                    row = dict(
                        subject_id=sid,
                        method=mid,
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
                    print(
                        f"  {mid:3s}: r@10={row['recall_at_10um']:.3f} "
                        f"r@20={row['recall_at_20um']:.3f} "
                        f"r@30={row['recall_at_30um']:.3f} "
                        f"n={row['n_pred']:4d} "
                        f"med={row['median_error_um']:.1f}µm "
                        f"({dt:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {mid:3s} FAILED: {type(e).__name__}: {e}",
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
    df.to_csv(OUT / "bench_a_results.csv", index=False)
    print("\n=== summary ===")
    pivot_r20 = df.pivot(index="subject_id", columns="method",
                          values="recall_at_20um")
    print("r@20:")
    print(pivot_r20.round(3).to_string())
    print(f"\nsum r@20 (P1, P4, P6, C5):")
    print((pivot_r20.sum().round(3)).to_string())


if __name__ == "__main__":
    main()
