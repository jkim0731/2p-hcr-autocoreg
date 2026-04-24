"""S47 — K=20 × beta sweep to see if widening K unlocks stress-subject lift.

S47 first pass (K=5 × beta sweep): beta=5 gives 788406 +2.4 pp r@20 but
only +1.2 pp on 755252 (vs +14.9 pp at raw argsort K=20). Hypothesis:
K=5 truncation cuts the signal. Widen K to 20 and re-sweep beta.
"""
from __future__ import annotations

import sys
import time
import warnings
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401
from bench.harness import run_candidate  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    configs = [(5, 0.0), (5, 5.0), (20, 0.0), (20, 3.0), (20, 5.0)]
    rows = []
    for subj in subjects:
        for K, beta in configs:
            t0 = time.time()
            try:
                row = run_candidate("P1", subj, write_csv=False,
                                     extra_kwargs=dict(K=K, hcr_quality_beta=beta))
            except Exception as e:
                print(f"  {subj} K={K} beta={beta}: FAILED {e}", flush=True)
                continue
            wall = round(time.time() - t0, 1)
            r20 = row.get("recall_at_20um", 0)
            rec = row.get("recall", 0)
            med = row.get("median_error_um", float("nan"))
            n_pred = row.get("n_pred", 0)
            n_gt = row.get("n_gt", 0)
            print(f"  {subj} K={K:2d} beta={beta:4.1f}  rec_id={rec:.3f} "
                  f"r@20={r20:.3f} n={n_pred}/{n_gt} med={med:.0f} wall={wall}s",
                  flush=True)
            rows.append(dict(subject=subj, K=K, beta=beta, recall_id=rec,
                              r20=r20, median_err=med, n_pred=n_pred,
                              n_gt=n_gt, wall=wall))

    df = pd.DataFrame(rows)
    out = "/root/capsule/code/full_automatic_execution_01/sessions/47_ship_hcr_quality/k20_beta_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\n=== r@20 by (K, beta) ===")
    piv = df.pivot_table(index="subject", columns=["K", "beta"], values="r20")
    print(piv.to_string())
    print("\n=== rec_id by (K, beta) ===")
    piv = df.pivot_table(index="subject", columns=["K", "beta"], values="recall_id")
    print(piv.to_string())


if __name__ == "__main__":
    main()
