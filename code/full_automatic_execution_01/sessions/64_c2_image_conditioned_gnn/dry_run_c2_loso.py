"""S64 subgoal 5 dry run — C2-LOSO with 1 training subject, 100 iterations.

Held-out: 788406. Training pool: 790322 only. Verifies:
  - Patch preprocessing (CZ + HCR) completes on both subjects
  - Stage 2 supervised loss descends over 100 iterations
  - Inference emits a non-empty pairs DataFrame
  - Ground-truth recall@20µm and position error printed for calibration

Budget (approximate on CPU):
  - Preload each subject: ~60 s HCR L2 load + ~5 s patch extraction
  - Stage 2 iter: ~6–10 s (dominated by CNN forward+backward on
    (~1000, ~2000) tensors).
  - Total: ~12–15 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._c2_image_gnn import run_c2_loso  # noqa: E402


def _hits_at_d(pairs_df, subject, d_um=20.0):
    """For each emitted pair, check if ground-truth distance in HCR space ≤ d_um."""
    if len(pairs_df) == 0:
        return 0, 0, 0.0

    gt_pairs = {(int(r.cz_id), int(r.hcr_id)) for _, r in subject.coreg_table.iterrows()}
    if not gt_pairs:
        return 0, 0, 0.0

    # We compute position error using the GT's HCR centroid vs emitted HCR centroid.
    gt_by_cz = {int(r.cz_id): int(r.hcr_id) for _, r in subject.coreg_table.iterrows()}

    from lib.centroid_helpers import centroids_um
    hcr_um, hcr_ids = centroids_um(subject, "hcr_gfp")
    hcr_pos = {int(v): i for i, v in enumerate(hcr_ids)}

    hits = 0
    labeled = 0
    errors = []
    for _, r in pairs_df.iterrows():
        cz_id = int(r["cz_id"])
        if cz_id not in gt_by_cz:
            continue
        labeled += 1
        emitted_hcr = int(r["hcr_id"])
        gt_hcr = gt_by_cz[cz_id]
        # emitted HCR centroid
        if emitted_hcr not in hcr_pos or gt_hcr not in hcr_pos:
            continue
        em = hcr_um[hcr_pos[emitted_hcr]]
        gt = hcr_um[hcr_pos[gt_hcr]]
        err = float(np.linalg.norm(em - gt))
        errors.append(err)
        if err <= d_um:
            hits += 1
    med_err = float(np.median(errors)) if errors else float("nan")
    return hits, labeled, med_err


def main() -> int:
    held_sid = "788406"
    train_sids = ["790322"]

    t_start = time.time()
    print(
        f"=== dry run C2-LOSO: held={held_sid} train={train_sids} "
        f"n_iter=100 ===", flush=True,
    )

    result = run_c2_loso(
        load_subject(held_sid),
        n_train_iter=100,
        train_subjects=train_sids,
        rng_seed=0,
        verbose=True,
    )

    df = result.pairs_df
    print(f"\n=== inference: {len(df)} pairs emitted ===", flush=True)
    print(f"  confidence median={result.confidence:.3f}", flush=True)

    print("\n=== GT recall @ 20 µm (labeled CZs only) ===", flush=True)
    s_held = load_subject(held_sid)
    hits, labeled, med_err = _hits_at_d(df, s_held, d_um=20.0)
    r20 = hits / labeled if labeled else 0.0
    print(
        f"  hits@20/labeled = {hits}/{labeled} = r@20={r20:.3f} "
        f"median_err={med_err:.1f} µm (labeled pairs)",
        flush=True,
    )
    print(f"\n=== diagnostics ===", flush=True)
    for k, v in result.diagnostics.items():
        print(f"  {k}: {v}", flush=True)

    dt = time.time() - t_start
    print(f"\n=== total time: {dt:.1f}s ({dt/60:.1f} min) ===", flush=True)

    # Validation: loss descended + emitted pairs nonzero.
    assert result.diagnostics.get("n_emitted", 0) > 0, "inference emitted 0 pairs"
    assert (result.diagnostics.get("train_loss_final") or 1e9) < 10.0, \
        "training loss did not descend below 10"

    print("\n=== PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
