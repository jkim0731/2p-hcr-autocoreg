"""S64 subgoal 6 — C2-LOSO 3-subject bench.

Validation roster: 788406 / 790322 / 767018 (matching S63 G1-LOSO and C5
baselines). Training pool: the other 5 subjects per held-out run.

Uses one shared preload across subjects — each subject's CZ + HCR patches +
centroid features are extracted ONCE up-front (~90 s/subject), then the
resulting dicts are reused across the 3 held-out runs. This saves 2× the
HCR-L2 volume loads vs calling `run_c2_loso` per held-out.

Compares r@20 to:
  - C5 baseline (centroid-only hand-feature GNN):  0.263 / 0.289 / 0.315
  - G1-LOSO (single-run centroid-only):            0.013 / 0.113 / 0.055

Decision rule (grand plan §9.6): sum Δr@20 across 3 held-out vs. C5 > 0
→ promote C2; ≤ 0 → document the negative and close G-series on
centroid+patch data.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

import torch  # noqa: E402

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from bench.candidate_impls._c2_image_gnn import (  # noqa: E402
    C2Matcher, SIX_SUBJECTS, _preload_c2_subject, _train_c2_stage2, _infer_c2,
)


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
    g1loso_r20 = {
        "788406": 0.013,
        "790322": 0.113,
        "767018": 0.055,
    }

    print(f"=== PRELOAD all {len(SIX_SUBJECTS)} subjects (one HCR L2 load each) ===",
          flush=True)
    t0 = time.time()
    preloaded: dict[str, dict] = {}
    for sid in SIX_SUBJECTS:
        preloaded[sid] = _preload_c2_subject(sid, verbose=True)
    print(f"  preload done in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for sid in subjects:
        print(f"\n=== {sid} (held-out) ===", flush=True)
        held_data = preloaded[sid]
        train_data = [preloaded[tid] for tid in SIX_SUBJECTS if tid != sid]

        hand_dim = train_data[0]["f_cz"].shape[1]
        torch.manual_seed(0)
        model = C2Matcher(
            hand_dim=hand_dim, patch_dim=64,
            hidden=96, n_layers=4, cross_layers=3,
        )
        print(
            f"  C2 model hand_dim={hand_dim} patch_dim=64 "
            f"total_params={model.param_count:,}",
            flush=True,
        )

        t0 = time.time()
        losses = _train_c2_stage2(
            model, train_data, n_iter=2000, lr=1e-4, rng_seed=0,
        )
        train_t = time.time() - t0

        t0 = time.time()
        df = _infer_c2(model, held_data)
        infer_t = time.time() - t0

        s = load_subject(sid)
        r20, med, hits, tot, denom = score(df, s)
        c5_val = c5_r20[sid]
        g1_val = g1loso_r20[sid]
        delta_c5 = r20 - c5_val
        delta_g1 = r20 - g1_val
        train_loss_mean50 = (
            float(np.mean(losses[-50:])) if len(losses) >= 50 else None
        )
        print(
            f"  C2-LOSO r@20={r20:.3f} med={med:.1f}µm "
            f"hits={hits}/{tot} emitted={len(df)} "
            f"train_t={train_t:.1f}s infer_t={infer_t:.1f}s "
            f"train_loss_mean50={train_loss_mean50}",
            flush=True,
        )
        print(f"  C5        r@20={c5_val:.3f} (baseline)", flush=True)
        print(f"  G1-LOSO   r@20={g1_val:.3f} (baseline)", flush=True)
        print(f"  Δr@20 (C2 - C5)      = {delta_c5:+.3f}", flush=True)
        print(f"  Δr@20 (C2 - G1-LOSO) = {delta_g1:+.3f}", flush=True)
        rows.append(dict(
            subject=sid,
            c5_r20=c5_val,
            g1loso_r20=g1_val,
            c2_r20=r20,
            c2_hits=hits,
            c2_n=len(df),
            c2_med=med,
            train_loss_final=(float(losses[-1]) if losses else None),
            train_loss_mean50=train_loss_mean50,
            delta_c5=delta_c5,
            delta_g1=delta_g1,
            train_s=train_t,
            infer_s=infer_t,
        ))

        out = pd.DataFrame(rows)
        out.to_csv(
            "sessions/64_c2_image_conditioned_gnn/bench_c2_loso.csv",
            index=False,
        )

    sum_c5 = sum(c5_r20.values())
    sum_g1 = sum(g1loso_r20.values())
    sum_c2 = out["c2_r20"].sum()
    print("\n=== SUMS (3-subject roster) ===", flush=True)
    print(f"  Sum C5       r@20 = {sum_c5:.3f}", flush=True)
    print(f"  Sum G1-LOSO  r@20 = {sum_g1:.3f}", flush=True)
    print(f"  Sum C2-LOSO  r@20 = {sum_c2:.3f}", flush=True)
    print(f"  Δ (C2 - C5)      = {sum_c2 - sum_c5:+.3f}", flush=True)
    print(f"  Δ (C2 - G1-LOSO) = {sum_c2 - sum_g1:+.3f}", flush=True)
    decision = "PROMOTE" if sum_c2 > sum_c5 else "CLOSE G-series on centroid+patch"
    print(f"  Decision (§9.6): {decision}", flush=True)


if __name__ == "__main__":
    main()
