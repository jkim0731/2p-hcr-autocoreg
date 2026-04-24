"""S51 — Sweep hcr_quality_beta on P4 spectral GM.

Mirrors S47's P1 hcr_quality bonus: subtract β·hq[j] from the putative score.
Goal: quantify orthogonal lift on P4 and see if β>0 unlocks regressions on
any of the four benchmark subjects (esp. 755252 where P4 previously tied P1).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import run_candidate  # noqa: E402
import bench.candidates  # noqa: F401, E402


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    betas = [0.0, 3.0, 5.0, 8.0]
    print(f"{'subj':<8}{'beta':>5}  {'rec_id':>7}  {'r@5':>6}  {'r@10':>6}  {'r@20':>6}  {'med':>7}  {'wall':>6}")
    for subj in subjects:
        for beta in betas:
            t0 = time.time()
            try:
                r = run_candidate("P4", subj, write_csv=False,
                                  extra_kwargs=dict(hcr_quality_beta=beta))
                rid = r.get("recall", 0.0)
                r5 = r.get("recall_at_5um", 0.0)
                r10 = r.get("recall_at_10um", 0.0)
                r20 = r.get("recall_at_20um", 0.0)
                med = r.get("median_error_um", float("nan"))
                dt = time.time() - t0
                if med is None:
                    med = float("nan")
                print(f"{subj:<8}{beta:>5.1f}  {rid:>7.3f}  {r5:>6.3f}  {r10:>6.3f}  {r20:>6.3f}  {med:>7.1f}  {dt:>6.1f}s",
                      flush=True)
            except Exception as e:
                print(f"{subj:<8}{beta:>5.1f}  FAIL: {e}", flush=True)


if __name__ == "__main__":
    main()
