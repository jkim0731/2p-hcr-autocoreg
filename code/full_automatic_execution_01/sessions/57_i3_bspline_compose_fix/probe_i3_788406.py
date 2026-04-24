"""S57 — Single-subject probe for I3 (I2 affine + B-spline) on 788406.

Compares three pipelines:
  1. I2 alone (affine only)
  2. I3 with skip_bspline=True (= I2 affine + NN pair emission)
  3. I3 with bspline (the fix)

Measures:
  - I2's affine residual (dist between I3's predicted HCR and actual HCR
    centroids for putative NN pairs)
  - pairs_df recall at 20/50 µm against coreg_table.csv
  - wall times
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import CANDIDATES, compare_to_gt  # noqa: E402
import bench.candidates  # noqa: F401, E402
from benchmark_data_loader import load_subject  # noqa: E402


def main():
    s = load_subject("788406")
    print(f"n_cz={len(s.cz_centroids)} n_hcr_gfp={len(s.hcr_gfp_df)}", flush=True)

    for name, kwargs in [
        ("I3_skip_bspline", dict(skip_bspline=True)),
        ("I3_bspline_20_g60",   dict(skip_bspline=False, bspline_iterations=20,
                                      grid_spacing_um=(60.0, 60.0, 120.0),
                                      bspline_sampling_fraction=0.02)),
        ("I3_bspline_30_g60",   dict(skip_bspline=False, bspline_iterations=30,
                                      grid_spacing_um=(60.0, 60.0, 120.0),
                                      bspline_sampling_fraction=0.05)),
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
              f"nn_p90={d.get('nn_dists_p90', 0):.1f} µm", flush=True)
        if "bspline_diag" in d:
            bd = d["bspline_diag"]
            print(f"  bspline: {bd}", flush=True)
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
