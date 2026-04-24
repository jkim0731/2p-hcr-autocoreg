"""S44 — Cross-modality bakeoff on 4 subjects.

Centroid-only ensembles plateau at P1-alone r@20 = 0.362 (S41 oracle
0.377 is unreachable; S42 SS-ensemble regresses to 0.306; S43 consensus
intersect precision median 0.088).

Reach for the next modality. Run candidates that consume masks or
images alongside P1 as baseline:

  P1  — TEASER baseline (centroid only) — reference.
  M1  — 3D mask-NCC on GFP+ segmentation masks via F1/F2.
  M3  — M1 warm-start → P1 hybrid (mask coarse + centroid refine).
  C1  — I2 MI-affine warm-start → P1 hybrid (image coarse + centroid).
  I2  — SimpleITK MI-affine on CZ z-stack × HCR 488 (after S33 repair).

Primary question: does any method reach 782149 r@20 > 0 OR lift
755252/767022 above the centroid plateau (P1 = 0.044, 0.108)?
Success threshold: r@20 > 0.02 on 782149, OR ≥0.03 lift on
755252/767022 without regressing 788406.
"""
from __future__ import annotations

import sys
import time
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401 (registers all)
from bench.harness import run_candidate  # noqa: E402


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    candidates = ["P1", "M1", "M3", "C1", "I2"]
    rows = []
    for subj in subjects:
        for cand in candidates:
            t0 = time.time()
            try:
                row = run_candidate(cand, subj, write_csv=False)
            except Exception as e:
                print(f"  {subj} {cand}: FAILED {e}", flush=True)
                continue
            wall = round(time.time() - t0, 1)
            r5 = row.get("recall_at_5um", 0)
            r10 = row.get("recall_at_10um", 0)
            r20 = row.get("recall_at_20um", 0)
            rec = row.get("recall", 0)
            med = row.get("median_error_um", float("nan"))
            n_pred = row.get("n_pred", 0)
            n_gt = row.get("n_gt", 0)
            print(f"  {subj} {cand:3s}  rec_id={rec:.3f} r@5={r5:.3f} "
                  f"r@10={r10:.3f} r@20={r20:.3f} "
                  f"n_pred={n_pred}/{n_gt} med={med:.0f} wall={wall}s",
                  flush=True)
            rows.append(dict(subject=subj, candidate=cand,
                             recall_id=rec, r5=r5, r10=r10, r20=r20,
                             median_err=med, n_pred=n_pred, n_gt=n_gt,
                             wall=wall))

    df = pd.DataFrame(rows)
    df.to_csv("/root/capsule/code/full_automatic_execution_01/sessions/44_mi_bakeoff/mi_bakeoff.csv",
              index=False)

    print("\n=== SUBJECT × CANDIDATE r@20 MATRIX ===")
    pivot = df.pivot(index="subject", columns="candidate", values="r20")
    print(pivot.to_string())

    print("\n=== SUBJECT × CANDIDATE recall_id MATRIX ===")
    pivot_id = df.pivot(index="subject", columns="candidate", values="recall_id")
    print(pivot_id.to_string())

    print("\n=== SUBJECT × CANDIDATE median_err MATRIX ===")
    pivot_med = df.pivot(index="subject", columns="candidate", values="median_err")
    print(pivot_med.to_string())

    print("\n=== PER SUBJECT BEST CANDIDATE (r@20) ===")
    for subj in subjects:
        sub = df[df.subject == subj]
        best = sub.sort_values("r20", ascending=False).iloc[0]
        print(f"  {subj}: {best.candidate} r@20={best.r20:.3f} "
              f"rec_id={best.recall_id:.3f} med={best.median_err:.0f}")


if __name__ == "__main__":
    main()
