"""Probe: does the existing G1 GNN matcher solve 782149?

F6 (lib/cell_features.py) and G1 (bench/candidate_impls/_g1_gnn_matcher.py)
are already implemented. G1 trains self-supervised on F8 synthetic warps of
the HCR GFP+ cloud, then runs cross-graph attention + Sinkhorn on the real
CZ <-> HCR pair using F6 features.

S36 established that centroid-ICP and I2-affine paths both fail on 782149
because of XY density mismatch (wrong-basin). G1 is feature-based rather
than density-based, so the expected behavior is that per-cell features
(k-NN angles, depth rank, inter-ROI distance ranks) should match the GT
region even when the spatial centroid is 335 µm off.

Plan:
  1. Run G1 on 788406, 755252, 767022, 782149 via the bench harness.
  2. Report n_pred, recall, recall@5/10/20, median error.
  3. If 782149 > 0, document as the first method to crack that subject.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
from bench import candidates as _cand  # noqa: F401  (registration)
from bench.harness import run_candidate


def main():
    out_dir = Path("/root/capsule/code/full_automatic_execution_01/sessions/37_g1_stress")
    out_dir.mkdir(exist_ok=True)
    rows = []
    for subj in ["788406", "755252", "767022", "782149"]:
        print(f"\n=== {subj} ===", flush=True)
        t0 = time.time()
        row = run_candidate("G1", subj, write_csv=False)
        dt = time.time() - t0
        row["wall_s"] = round(dt, 1)
        # Remove stderr blob if present
        err = row.pop("error", None)
        if err:
            print(f"  ERROR: {err[:200]}")
        print(f"  n_pred={row['n_pred']} n_gt={row['n_gt']} "
              f"rec={row['recall']:.3f} prec={row['precision']:.3f} "
              f"rec@5={row['recall_at_5um']:.3f} rec@10={row['recall_at_10um']:.3f} "
              f"rec@20={row['recall_at_20um']:.3f} med={row['median_error_um']} "
              f"({dt:.1f}s)", flush=True)
        rows.append(row)

    with open(out_dir / "g1_stress.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(f"{'subject':>8} {'n_pred':>7} {'n_gt':>5} {'rec':>5} {'prec':>5} {'r@5':>5} {'r@10':>5} {'r@20':>5} {'med':>5} {'wall':>6}")
    for r in rows:
        print(f"{r['subject_id']:>8} {r['n_pred']:>7} {r['n_gt']:>5} "
              f"{r['recall']:>5.3f} {r['precision']:>5.3f} "
              f"{r['recall_at_5um']:>5.3f} {r['recall_at_10um']:>5.3f} {r['recall_at_20um']:>5.3f} "
              f"{str(r['median_error_um'])[:5]:>5} {r['wall_s']:>6.1f}")


if __name__ == "__main__":
    main()
