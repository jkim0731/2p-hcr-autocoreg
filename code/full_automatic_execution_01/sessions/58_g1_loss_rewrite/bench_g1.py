"""S58 — benchmark G1 with the new per-side-averaged loss on all 6 subjects.

Runs `run_g1` with asymmetric=True, hidden=96, n_layers=4, cross_layers=3,
n_train_iter=300, use_f6=True (the S52 best config).  Compares r@20 vs C5
(the shipping ensemble baseline at sum 1.080).

Also emits diagnostic: final train loss (new loss should descend well
below the S52 plateau of ~4.95).
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import CANDIDATES, compare_to_gt  # noqa: E402
import bench.candidates  # noqa: F401, E402
from benchmark_data_loader import load_subject  # noqa: E402


def main():
    sids = ["788406", "790322", "755252", "767022", "767018", "782149"]
    rows = []
    for sid in sids:
        print(f"\n=========== {sid} ===========", flush=True)
        s = load_subject(sid)
        print(f"n_cz={len(s.cz_centroids)} n_hcr_gfp={len(s.hcr_gfp_df)}",
              flush=True)

        t0 = time.time()
        try:
            r = CANDIDATES["G1"](s, n_train_iter=1000, use_f6=False,
                                  asymmetric=True,
                                  hidden=96, n_layers=4, cross_layers=3)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        wall = time.time() - t0
        d = r.diagnostics
        pairs = r.pairs_df
        print(f"  wall={wall:.1f}s n_pairs={len(pairs)} "
              f"train_loss_final={d.get('train_loss_final')!s}",
              flush=True)
        if len(pairs):
            sc = compare_to_gt(pairs, s)
            print(f"  r@5={sc['recall_at_5um']:.3f} "
                  f"r@10={sc['recall_at_10um']:.3f} "
                  f"r@20={sc['recall_at_20um']:.3f} "
                  f"med_err={sc['median_error_um']:.1f} "
                  f"p95={sc['p95_error_um']:.1f} µm "
                  f"n_pred={sc['n_pred']} n_gt={sc['n_gt']}", flush=True)
            rows.append(dict(subject=sid, r_at_20=sc["recall_at_20um"],
                              median_err_um=sc["median_error_um"],
                              n_pairs=len(pairs), wall_s=wall,
                              train_loss_final=d.get("train_loss_final")))
        else:
            print("  no pairs emitted", flush=True)
            rows.append(dict(subject=sid, r_at_20=0.0, median_err_um=float("nan"),
                              n_pairs=0, wall_s=wall,
                              train_loss_final=d.get("train_loss_final")))

    if rows:
        print("\n\n=== SUMMARY ===", flush=True)
        print(f"{'sid':8s} {'r@20':>6s} {'med':>6s} {'n':>6s} {'wall':>6s} "
              f"{'loss':>6s}", flush=True)
        tot = 0.0
        for r in rows:
            tot += r["r_at_20"]
            loss = r["train_loss_final"]
            loss_s = f"{loss:.3f}" if isinstance(loss, (int, float)) else "n/a"
            print(f"{r['subject']:8s} {r['r_at_20']:6.3f} "
                  f"{r['median_err_um']:6.1f} {r['n_pairs']:6d} "
                  f"{r['wall_s']:6.1f} {loss_s:>6s}", flush=True)
        print(f"sum r@20 = {tot:.3f}  (C5=1.080)", flush=True)


if __name__ == "__main__":
    main()
