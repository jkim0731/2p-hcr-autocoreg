"""S41 — P-series tier-1 bakeoff on 4 subjects.

S40 established that per-pair ranking (not warm-start) is the stress-subject
bottleneck on P1. Before investing in IoU-augmented P1, check whether any
alternative point-cloud method has a different failure mode:

  P1  — TEASER++ (GNC-TLS outlier rejection)
  P3  — RANSAC + anisotropic affine
  P4  — Spectral graph matching (pairwise consistency)
  P5  — Fused / partial GW
  P14 — Hungarian on feature+distance affinity (baseline)

All 4 subjects × 5 candidates = 20 runs via F9. Report recall_id, r@5, r@10,
r@20, median error. A method with different failure mode would likely show a
stress-subject lift (767022 > 0.131 r@20, 755252 > 0.08, or 782149 > 0) or a
qualitatively different median-error pattern.
"""
from __future__ import annotations

import sys
import time
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401 (registers all)
from bench.harness import run_candidate  # noqa


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    candidates = ["P1", "P3", "P4", "P5", "P14"]
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
    df.to_csv("/root/capsule/code/full_automatic_execution_01/sessions/41_p_bakeoff/bakeoff.csv",
              index=False)

    print("\n=== SUBJECT × CANDIDATE r@20 MATRIX ===")
    pivot = df.pivot(index="subject", columns="candidate", values="r20")
    print(pivot.to_string())

    print("\n=== SUBJECT × CANDIDATE recall_id MATRIX ===")
    pivot_id = df.pivot(index="subject", columns="candidate", values="recall_id")
    print(pivot_id.to_string())

    print("\n=== PER SUBJECT, BEST P-CANDIDATE BY r@20 ===")
    for subj in subjects:
        sub = df[df.subject == subj]
        best = sub.sort_values("r20", ascending=False).iloc[0]
        print(f"  {subj}: {best.candidate} r@20={best.r20:.3f} "
              f"r@10={best.r10:.3f} rec_id={best.recall_id:.3f} "
              f"med={best.median_err:.0f}")


if __name__ == "__main__":
    main()
