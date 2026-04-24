"""S43 — P1 ∩ P14 cross-candidate consensus.

S42 showed an SS-selected ensemble regresses P1 by −0.055 r@20 — the
no-peek residual-fit score cannot separate "self-consistent + correct"
from "self-consistent + wrong-pair" across different matchers.

This probe bypasses scoring entirely. Strategy:

  intersect(cz_id):  P1(cz_id).hcr_id == P14(cz_id).hcr_id → high-conf
  complement:         P14 disagrees → inherit P1's assignment (baseline)

Outputs two variants:

  - ``intersect``   — keep only agreed pairs (high precision, low recall).
  - ``fallback_p1`` — agreed pairs have confidence=1, disagreed cells keep P1.

No ranker. Consensus is a pure identity-based filter.

If fallback_p1 r@20 ≥ P1-only on all 4 subjects and intersect precision
≥ 0.95, ship fallback_p1 as S43 with intersect-subset used as the
human-confirmable seed.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401 (registers all)
from bench.harness import CANDIDATES, compare_to_gt  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402


def build_consensus(p1_pairs: pd.DataFrame, p14_pairs: pd.DataFrame):
    """Return (intersect_df, fallback_df).

    intersect_df: rows where P1(cz_id).hcr_id == P14(cz_id).hcr_id.
    fallback_df: P1's full output; intersect rows get confidence=1.
    """
    # Reduce both to one row per cz_id (first wins — pairs_df already
    # drops duplicates upstream in P1/P14 so this is a no-op defensively).
    p1 = p1_pairs.drop_duplicates("cz_id").set_index("cz_id")
    p14 = p14_pairs.drop_duplicates("cz_id").set_index("cz_id")

    common_cz = p1.index.intersection(p14.index)
    agree_mask = p1.loc[common_cz, "hcr_id"].values == p14.loc[common_cz, "hcr_id"].values
    agree_cz = common_cz[agree_mask]

    intersect_df = p1.loc[agree_cz].reset_index()
    # Mark consensus confidence as 1.0 (both matchers agree).
    if len(intersect_df):
        intersect_df["confidence"] = 1.0

    fallback_df = p1_pairs.copy()
    if len(agree_cz):
        fallback_df.loc[fallback_df["cz_id"].isin(agree_cz), "confidence"] = 1.0

    return intersect_df, fallback_df, len(agree_cz), len(common_cz)


def main():
    subjects = ["788406", "755252", "767022", "782149"]
    rows = []
    summary_rows = []

    for subj in subjects:
        s = load_subject(subj)

        t0 = time.time()
        p1_result = CANDIDATES["P1"](s)
        p1_wall = time.time() - t0
        p1_pairs = p1_result.pairs_df
        p1_gt = compare_to_gt(p1_pairs, s)

        t0 = time.time()
        p14_result = CANDIDATES["P14"](s)
        p14_wall = time.time() - t0
        p14_pairs = p14_result.pairs_df
        p14_gt = compare_to_gt(p14_pairs, s)

        intersect_df, fallback_df, n_agree, n_common = build_consensus(
            p1_pairs, p14_pairs
        )
        intersect_gt = compare_to_gt(intersect_df, s)
        fallback_gt = compare_to_gt(fallback_df, s)

        n_gt = p1_gt["n_gt"]
        agree_frac = n_agree / max(1, n_common)

        print(f"\n=== {subj} ===", flush=True)
        print(f"  P1       rec_id={p1_gt['recall']:.3f} r@20={p1_gt['recall_at_20um']:.3f} "
              f"med={p1_gt['median_error_um']:.1f} n_pred={p1_gt['n_pred']}/{n_gt} "
              f"wall={p1_wall:.1f}s", flush=True)
        print(f"  P14      rec_id={p14_gt['recall']:.3f} r@20={p14_gt['recall_at_20um']:.3f} "
              f"med={p14_gt['median_error_um']:.1f} n_pred={p14_gt['n_pred']}/{n_gt} "
              f"wall={p14_wall:.1f}s", flush=True)
        print(f"  common_cz={n_common}  agree={n_agree}  agree_frac={agree_frac:.3f}",
              flush=True)
        print(f"  INTERSECT rec_id={intersect_gt['recall']:.3f} "
              f"r@20={intersect_gt['recall_at_20um']:.3f} "
              f"precision={intersect_gt['precision']:.3f} "
              f"n_pred={intersect_gt['n_pred']}", flush=True)
        print(f"  FALLBACK  rec_id={fallback_gt['recall']:.3f} "
              f"r@20={fallback_gt['recall_at_20um']:.3f} "
              f"precision={fallback_gt['precision']:.3f} "
              f"n_pred={fallback_gt['n_pred']}", flush=True)

        rows.append(dict(
            subject=subj, variant="P1",
            recall_id=p1_gt["recall"], r5=p1_gt["recall_at_5um"],
            r10=p1_gt["recall_at_10um"], r20=p1_gt["recall_at_20um"],
            precision=p1_gt["precision"],
            median_err=p1_gt["median_error_um"],
            n_pred=p1_gt["n_pred"], n_gt=n_gt, wall=round(p1_wall, 1),
        ))
        rows.append(dict(
            subject=subj, variant="P14",
            recall_id=p14_gt["recall"], r5=p14_gt["recall_at_5um"],
            r10=p14_gt["recall_at_10um"], r20=p14_gt["recall_at_20um"],
            precision=p14_gt["precision"],
            median_err=p14_gt["median_error_um"],
            n_pred=p14_gt["n_pred"], n_gt=n_gt, wall=round(p14_wall, 1),
        ))
        rows.append(dict(
            subject=subj, variant="intersect",
            recall_id=intersect_gt["recall"], r5=intersect_gt["recall_at_5um"],
            r10=intersect_gt["recall_at_10um"], r20=intersect_gt["recall_at_20um"],
            precision=intersect_gt["precision"],
            median_err=intersect_gt["median_error_um"],
            n_pred=intersect_gt["n_pred"], n_gt=n_gt, wall=0.0,
        ))
        rows.append(dict(
            subject=subj, variant="fallback_p1",
            recall_id=fallback_gt["recall"], r5=fallback_gt["recall_at_5um"],
            r10=fallback_gt["recall_at_10um"], r20=fallback_gt["recall_at_20um"],
            precision=fallback_gt["precision"],
            median_err=fallback_gt["median_error_um"],
            n_pred=fallback_gt["n_pred"], n_gt=n_gt, wall=0.0,
        ))
        summary_rows.append(dict(
            subject=subj, n_gt=n_gt, n_common=n_common, n_agree=n_agree,
            agree_frac=round(agree_frac, 3),
            p1_r20=p1_gt["recall_at_20um"], p14_r20=p14_gt["recall_at_20um"],
            intersect_r20=intersect_gt["recall_at_20um"],
            intersect_prec=intersect_gt["precision"],
            fallback_r20=fallback_gt["recall_at_20um"],
            fallback_prec=fallback_gt["precision"],
        ))

    df = pd.DataFrame(rows)
    df.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/43_p_consensus/consensus_raw.csv",
        index=False,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/43_p_consensus/consensus_summary.csv",
        index=False,
    )

    print("\n\n=== SUMMARY (r@20) ===")
    print(summary.to_string(index=False))

    p1_total = summary["p1_r20"].sum()
    p14_total = summary["p14_r20"].sum()
    intersect_total = summary["intersect_r20"].sum()
    fallback_total = summary["fallback_r20"].sum()

    print(f"\nTotals (sum across 4 subjects):")
    print(f"  P1 r@20:         {p1_total:.3f}")
    print(f"  P14 r@20:        {p14_total:.3f}")
    print(f"  intersect r@20:  {intersect_total:.3f}")
    print(f"  fallback r@20:   {fallback_total:.3f}")
    print(f"  fallback lift:   {fallback_total - p1_total:+.3f}")

    # Intersect precision (weighted by n_pred)
    w = summary["intersect_r20"] * summary["n_gt"]  # rough weighted r@20
    print(f"  intersect median-precision: {summary['intersect_prec'].median():.3f}")
    print(f"  intersect min-precision:    {summary['intersect_prec'].min():.3f}")


if __name__ == "__main__":
    main()
