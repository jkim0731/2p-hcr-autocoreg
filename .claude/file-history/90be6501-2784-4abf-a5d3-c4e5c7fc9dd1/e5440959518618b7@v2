"""S62 — benchmark B3 (C5-seeded TPS expansion) vs C5.

Validation roster narrowed to three subjects (788406, 790322, 767018).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

import bench.candidates  # noqa: F401
from bench.harness import CANDIDATES  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402


def score(df, s):
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    hcr_by_id = dict(zip(hcr_ids, hcr_um))
    coreg = s.coreg_table
    gt = {int(r["cz_id"]): int(r["hcr_id"]) for _, r in coreg.iterrows()}
    hits = 0
    total = 0
    errs = []
    for _, r in df.iterrows():
        c = int(r["cz_id"])
        if c not in gt:
            continue
        total += 1
        gh = gt[c]
        if gh not in hcr_by_id:
            continue
        gt_pos = hcr_by_id[gh]
        pr_pos = np.array([r["hcr_z_um"], r["hcr_y_um"], r["hcr_x_um"]])
        d = np.linalg.norm(gt_pos - pr_pos)
        errs.append(d)
        if d <= 20.0:
            hits += 1
    denom = len(gt)
    r20 = hits / max(denom, 1)
    med = float(np.median(errs)) if errs else float("inf")
    return r20, med, hits, total, denom


def main():
    subjects = ["788406", "790322", "767018"]
    c5_r20 = {
        "788406": 0.2617534942820839,
        "790322": 0.2892030848329049,
        "767018": 0.315018315018315,
    }

    rows = []
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)

        t0 = time.time()
        r = CANDIDATES["B3"](s)
        w = time.time() - t0
        r20, med, hits, tot, denom = score(r.pairs_df, s)
        c5_val = c5_r20[sid]
        delta = r20 - c5_val
        print(
            f"  B3 r@20={r20:.3f} med={med:.1f}µm hits={hits}/{tot} "
            f"emitted={len(r.pairs_df)} wall={w:.1f}s",
            flush=True,
        )
        print(f"  C5 r@20={c5_val:.3f} (baseline)", flush=True)
        print(f"  Δr@20 (B3 - C5) = {delta:+.3f}", flush=True)

        diag = r.diagnostics or {}
        rows.append(
            dict(
                subject=sid,
                c5_r20=c5_val,
                b3_r20=r20,
                b3_hits=hits,
                b3_n=len(r.pairs_df),
                b3_med=med,
                n_seed_spatial=diag.get("n_seed_spatial", 0),
                n_seed_kept=diag.get("n_seed_kept", 0),
                delta=delta,
                wall_s=w,
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(
        "sessions/62_b_retrial_smart_seed/bench_b3.csv",
        index=False,
    )

    sum_c5 = sum(c5_r20.values())
    sum_b3 = out["b3_r20"].sum()
    print("\n=== SUMS (3-subject roster) ===", flush=True)
    print(f"  Sum C5 r@20 = {sum_c5:.3f}", flush=True)
    print(f"  Sum B3 r@20 = {sum_b3:.3f}", flush=True)
    print(f"  Δ (B3 - C5) = {sum_b3 - sum_c5:+.3f}", flush=True)


if __name__ == "__main__":
    main()
