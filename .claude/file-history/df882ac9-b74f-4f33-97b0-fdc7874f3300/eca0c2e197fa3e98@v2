"""F9 re-bench for v2-S01 locked-prior candidates.

Runs P1_LP / P4_LP / P6_LP / C5_LP on all 6 benchmark subjects via the
v1 harness; writes per-subject CSV plus a summary CSV next to this file.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent.parent
sys.path.insert(0, str(_CODE / "full_automatic_execution_01"))
sys.path.insert(0, str(_CODE / "full_automatic_execution_02" / "lib"))
sys.path.insert(0, str(_CODE / "dev_code"))

from bench import candidates as _v1_candidates  # noqa: F401, E402
import locked_prior_candidates  # noqa: F401, E402  (registers *_LP)
from bench.harness import run_candidate  # noqa: E402
from benchmark_data_loader import BENCHMARK_SUBJECTS  # noqa: E402

CANDIDATES = ["P1_LP", "P4_LP", "P6_LP", "C5_LP"]
COLS = [
    "candidate_id", "subject_id",
    "n_pred", "recall", "recall_at_5um", "recall_at_10um", "recall_at_20um",
    "median_error_um", "p95_error_um", "runtime_s", "error_first_line",
]


def main():
    out_csv = _HERE / "lp_bench_results.csv"
    rows = []
    for cid in CANDIDATES:
        for sid in BENCHMARK_SUBJECTS:
            t0 = time.time()
            try:
                row = run_candidate(cid, sid, write_csv=True)
                err = (row.get("error") or "").splitlines()[0] if row.get("error") else ""
            except Exception as exc:  # noqa: BLE001
                row = {
                    "candidate_id": cid, "subject_id": sid,
                    "n_pred": 0, "recall": 0.0,
                    "recall_at_5um": 0.0, "recall_at_10um": 0.0, "recall_at_20um": 0.0,
                    "median_error_um": float("nan"), "p95_error_um": float("nan"),
                    "runtime_s": time.time() - t0,
                }
                err = f"{type(exc).__name__}: {exc}"
            r = {k: row.get(k) for k in COLS if k != "error_first_line"}
            r["error_first_line"] = err
            rows.append(r)
            print(
                f"[{cid}:{sid}] r@10={row.get('recall_at_10um', 0):.2f} "
                f"r@20={row.get('recall_at_20um', 0):.2f} "
                f"med={row.get('median_error_um', float('nan')):.1f} "
                f"n_pred={row.get('n_pred', 0)} "
                f"err={err[:40]}",
                flush=True,
            )

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
