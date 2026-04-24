"""S42 — P1 + P4 + P14 ensemble with no-peek SS selection.

S41 bakeoff showed per-subject oracle r@20 = 0.377 vs P1-only 0.362
(+0.015). P14 ties P1 on 788406/755252; P4 ties P14 on 755252; P1 wins
767022. All fail 782149.

This probe wraps P1, P4, P14 via F9, scores each candidate's emitted
`pairs_df` by a no-peek SS metric — residuals from a same-sample
anisotropic-affine fit on its own pairs — and picks the per-subject
winner. If the SS ranker matches the oracle on ≥3/4 subjects without
regressing 788406 or 767022 vs P1, the ensemble ships as S42.

SS metric (universal, no GT):
  1. Fit `fit_anisotropic_similarity(CZ_xyz, HCR_xyz)` on the candidate's
     own pairs_df.
  2. Per-pair residual norm from the fit.
  3. SS = count of pairs with residual < 30 µm.

Rationale: SS counts pairs consistent with a single global affine;
wrong-basin outputs have high within-basin residuals OR thin pair sets,
both of which push SS down. Matches the S40 SS definition generalised
to every P candidate (P1 already emits `residual_um` = TPS residual,
which is a stricter version of the same idea).
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401 (registers all)
from bench.harness import run_candidate, CANDIDATES  # noqa: E402
from bench.harness import compare_to_gt, _cz_centroids_um, _hcr_centroids_um  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from benchmark_analysis import fit_anisotropic_similarity  # noqa: E402


def ss_score(pairs_df: pd.DataFrame, threshold_um: float = 30.0) -> tuple[int, float, int]:
    """No-peek SS score: (count residual<thr, median residual, n_pairs).

    Fits an anisotropic-similarity on the candidate's own pairs_df and
    returns the number of pairs whose same-sample residual is below
    `threshold_um`. Does NOT touch the ground-truth table.
    """
    if pairs_df is None or len(pairs_df) < 4:
        return 0, float("nan"), int(0 if pairs_df is None else len(pairs_df))
    cz = pairs_df[["cz_x_um", "cz_y_um", "cz_z_um"]].values.astype(float)
    hcr = pairs_df[["hcr_x_um", "hcr_y_um", "hcr_z_um"]].values.astype(float)
    try:
        fit = fit_anisotropic_similarity(cz, hcr)
    except Exception:
        return 0, float("nan"), int(len(pairs_df))
    r = np.linalg.norm(fit.residuals_um, axis=1)
    return int((r < threshold_um).sum()), float(np.median(r)), int(len(pairs_df))


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    candidates = ["P1", "P4", "P14"]

    # Step 1: run each (cand, subj); cache pairs_df + GT metrics.
    cache = {}  # (subj, cand) -> (row_gt_metrics, pairs_df, wall)
    for subj in subjects:
        s = load_subject(subj)
        for cand in candidates:
            t0 = time.time()
            fn = CANDIDATES[cand]
            try:
                result = fn(s)
                pairs_df = result.pairs_df
            except Exception as e:
                print(f"  {subj} {cand}: FAILED {e}", flush=True)
                continue
            gt = compare_to_gt(pairs_df, s)
            wall = round(time.time() - t0, 1)
            ss_n, ss_med, ss_total = ss_score(pairs_df)
            rec_id = gt["recall"]
            r5 = gt["recall_at_5um"]
            r10 = gt["recall_at_10um"]
            r20 = gt["recall_at_20um"]
            med = gt["median_error_um"]
            cache[(subj, cand)] = dict(
                subject=subj, candidate=cand, wall=wall,
                ss_n=ss_n, ss_med=ss_med, ss_total=ss_total,
                recall_id=rec_id, r5=r5, r10=r10, r20=r20,
                median_err=med, n_pred=gt["n_pred"], n_gt=gt["n_gt"],
            )
            print(f"  {subj} {cand:3s}  ss_n={ss_n:4d}/{ss_total:<4d} "
                  f"ss_med={ss_med:6.1f} "
                  f"rec_id={rec_id:.3f} r@20={r20:.3f} med={med:6.1f} "
                  f"wall={wall}s", flush=True)

    # Step 2: build per-subject ensemble decision table.
    print("\n=== SS-PICKER vs ORACLE (r@20) ===")
    header = f"{'subj':<8s} {'ss-pick':>8s} {'ss@20':>6s} {'oracle':>7s} {'oracle@20':>9s} " \
             f"{'P1@20':>6s} {'P4@20':>6s} {'P14@20':>7s}  match?"
    print(header)
    rows = []
    ss_pick_r20_total = 0.0
    oracle_r20_total = 0.0
    p1_r20_total = 0.0
    for subj in subjects:
        subs = {c: cache[(subj, c)] for c in candidates if (subj, c) in cache}
        if not subs:
            continue
        # SS pick
        best_ss = max(subs.values(), key=lambda r: r["ss_n"])
        # Oracle pick
        best_oracle = max(subs.values(), key=lambda r: r["r20"])
        agree = best_ss["candidate"] == best_oracle["candidate"]
        mark = "*" if agree else " "
        p1 = subs["P1"]["r20"]
        p4 = subs["P4"]["r20"]
        p14 = subs["P14"]["r20"]
        print(f"{subj:<8s} {best_ss['candidate']:>8s} {best_ss['r20']:>6.3f} "
              f"{best_oracle['candidate']:>7s} {best_oracle['r20']:>9.3f} "
              f"{p1:>6.3f} {p4:>6.3f} {p14:>7.3f}  {mark}")
        ss_pick_r20_total += best_ss["r20"]
        oracle_r20_total += best_oracle["r20"]
        p1_r20_total += p1
        rows.append(dict(
            subject=subj,
            ss_pick=best_ss["candidate"], ss_r20=best_ss["r20"],
            oracle_pick=best_oracle["candidate"], oracle_r20=best_oracle["r20"],
            p1_r20=p1, p4_r20=p4, p14_r20=p14,
            match=int(agree),
        ))

    print(f"\nTotals (sum across 4 subjects):")
    print(f"  SS-pick r@20 total: {ss_pick_r20_total:.3f}")
    print(f"  Oracle r@20 total:  {oracle_r20_total:.3f}")
    print(f"  P1-only r@20 total: {p1_r20_total:.3f}")
    print(f"  SS lift over P1:    {ss_pick_r20_total - p1_r20_total:+.3f}")
    print(f"  Oracle lift / P1:   {oracle_r20_total - p1_r20_total:+.3f}")

    # Step 3: save matrix + decision table.
    df_all = pd.DataFrame(list(cache.values()))
    df_all.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/42_p_ensemble/ensemble_raw.csv",
        index=False,
    )
    df_dec = pd.DataFrame(rows)
    df_dec.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/42_p_ensemble/ensemble_decision.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
