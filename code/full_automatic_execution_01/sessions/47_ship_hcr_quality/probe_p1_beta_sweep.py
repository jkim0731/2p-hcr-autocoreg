"""S47 — ship HCR-quality bonus into P1 (beta sweep).

S46-b showed on raw argsort that `score = D - 25*cos - beta*hcr_quality(j)`
lifts GT-in-top-20 on 755252 by +14.9 pp at beta=5. This probe runs the
full P1 pipeline end-to-end (TEASER/GNC-TLS + TPS + one-to-one) via F9
at beta ∈ {0, 3, 5, 8}, measures per-subject recall_id / r@5/10/20 and
median error, and picks a production default.

Primary question: does the +15 pp top-K lift survive the downstream
TLS + TPS + one-to-one selection?

Run: `python probe_p1_beta_sweep.py`
"""
from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401  (registers candidates)
from bench.harness import run_candidate  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    betas = [0.0, 3.0, 5.0, 8.0]
    rows = []
    for subj in subjects:
        for beta in betas:
            t0 = time.time()
            try:
                row = run_candidate(
                    "P1", subj, write_csv=False,
                    extra_kwargs=dict(hcr_quality_beta=beta),
                )
            except Exception as e:
                print(f"  {subj} beta={beta}: FAILED {e}", flush=True)
                continue
            wall = round(time.time() - t0, 1)
            r5 = row.get("recall_at_5um", 0)
            r10 = row.get("recall_at_10um", 0)
            r20 = row.get("recall_at_20um", 0)
            rec = row.get("recall", 0)
            med = row.get("median_error_um", float("nan"))
            n_pred = row.get("n_pred", 0)
            n_gt = row.get("n_gt", 0)
            print(f"  {subj} beta={beta:4.1f}  rec_id={rec:.3f} "
                  f"r@5={r5:.3f} r@10={r10:.3f} r@20={r20:.3f} "
                  f"n={n_pred}/{n_gt} med={med:.0f} wall={wall}s",
                  flush=True)
            rows.append(dict(subject=subj, beta=beta,
                             recall_id=rec, r5=r5, r10=r10, r20=r20,
                             median_err=med, n_pred=n_pred, n_gt=n_gt,
                             wall=wall))

    df = pd.DataFrame(rows)
    out = "/root/capsule/code/full_automatic_execution_01/sessions/47_ship_hcr_quality/p1_beta_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    for metric in ("r20", "recall_id", "median_err"):
        print(f"\n=== SUBJECT × BETA {metric} ===")
        pivot = df.pivot(index="subject", columns="beta", values=metric)
        print(pivot.to_string())


if __name__ == "__main__":
    main()
