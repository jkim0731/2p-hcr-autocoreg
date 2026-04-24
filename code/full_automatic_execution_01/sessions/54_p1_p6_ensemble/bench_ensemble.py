"""S54 — P1 ⊕ P6 ensemble benchmark across 6 subjects.

Strategies compared against P1-only and P6-only baselines:

  - union_conf: union of pairs_df, keep higher-confidence on cz_id collision.
  - union_p1_first: union, keep P1 on cz_id collision (P1 is dense-subject winner).
  - intersection: only pairs where both methods agree on (cz_id, hcr_id).
  - pick_p1_if_n_pred_hi: pick P1 when P1.n_pred > threshold, else P6.
  - oracle_best: upper bound — pick method with best r@20 per subject (peeks at GT; diagnostic only).

All scoring via harness.compare_to_gt so metrics are identical to F9.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from bench.harness import CANDIDATES, compare_to_gt  # noqa: E402
import bench.candidates  # noqa: F401, E402
from benchmark_data_loader import load_subject  # noqa: E402


SUBJECTS = ["788406", "790322", "755252", "767022", "767018", "782149"]

P1_KW = dict(hcr_quality_beta=5.0)
P6_KW = dict(method="cpd_nonrigid", maxiter=30, w_outlier=0.3,
             crop_pad_um=150.0, match_radius_um=60.0, hcr_quality_beta=0.0)


def _score(pairs_df: pd.DataFrame, s) -> dict:
    return compare_to_gt(pairs_df, s)


def _union_conf(p1: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    """Union; on cz_id collision, keep higher confidence."""
    cat = pd.concat([p1, p6], ignore_index=True)
    if len(cat) == 0:
        return cat
    cat = cat.sort_values("confidence", ascending=False)
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.sort_values("cz_id").reset_index(drop=True)
    return cat


def _union_p1_first(p1: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    """Union; on cz_id collision, keep P1."""
    cat = pd.concat([p1.assign(_src="p1"), p6.assign(_src="p6")], ignore_index=True)
    if len(cat) == 0:
        return cat
    cat["_p1_first"] = (cat["_src"] == "p1").astype(int)
    cat = cat.sort_values(["cz_id", "_p1_first"], ascending=[True, False])
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.drop(columns=["_src", "_p1_first"]).reset_index(drop=True)
    return cat


def _intersection(p1: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    """Only pairs where both methods predict same (cz_id, hcr_id)."""
    if len(p1) == 0 or len(p6) == 0:
        return pd.DataFrame(columns=p1.columns if len(p1) else p6.columns)
    p1_one = p1.drop_duplicates(subset=["cz_id"], keep="first").set_index("cz_id")
    p6_one = p6.drop_duplicates(subset=["cz_id"], keep="first").set_index("cz_id")
    common = p1_one.index.intersection(p6_one.index)
    hits = [cid for cid in common
            if int(p1_one.loc[cid, "hcr_id"]) == int(p6_one.loc[cid, "hcr_id"])]
    return p1_one.loc[hits].reset_index()


def main():
    rows = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===", flush=True)
        t_load = time.time()
        s = load_subject(subj)
        print(f"  load {time.time()-t_load:.1f}s  n_cz={len(s.cz_centroids)} "
              f"n_hcr_gfp={len(s.hcr_gfp_df)}", flush=True)

        t_p1 = time.time()
        try:
            res_p1 = CANDIDATES["P1"](s, **P1_KW)
            wall_p1 = time.time() - t_p1
            p1_df = res_p1.pairs_df
            sc_p1 = _score(p1_df, s)
            print(f"  P1 r@20={sc_p1['recall_at_20um']:.3f} n={sc_p1['n_pred']} "
                  f"med={sc_p1['median_error_um']:.1f}µm wall={wall_p1:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"  P1 FAILED: {e}", flush=True)
            p1_df = pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                          "cz_x_um","cz_y_um","cz_z_um",
                                          "hcr_x_um","hcr_y_um","hcr_z_um"])
            sc_p1 = _score(p1_df, s); wall_p1 = 0.0

        t_p6 = time.time()
        try:
            res_p6 = CANDIDATES["P6"](s, **P6_KW)
            wall_p6 = time.time() - t_p6
            p6_df = res_p6.pairs_df
            sc_p6 = _score(p6_df, s)
            print(f"  P6 r@20={sc_p6['recall_at_20um']:.3f} n={sc_p6['n_pred']} "
                  f"med={sc_p6['median_error_um']:.1f}µm wall={wall_p6:.1f}s",
                  flush=True)
        except Exception as e:
            print(f"  P6 FAILED: {e}", flush=True)
            p6_df = pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                          "cz_x_um","cz_y_um","cz_z_um",
                                          "hcr_x_um","hcr_y_um","hcr_z_um"])
            sc_p6 = _score(p6_df, s); wall_p6 = 0.0

        union_conf = _union_conf(p1_df, p6_df)
        union_p1_first = _union_p1_first(p1_df, p6_df)
        inter = _intersection(p1_df, p6_df)

        sc_uc = _score(union_conf, s)
        sc_up1 = _score(union_p1_first, s)
        sc_in = _score(inter, s)

        oracle_r20 = max(sc_p1["recall_at_20um"], sc_p6["recall_at_20um"])

        print(f"  union_conf    r@20={sc_uc['recall_at_20um']:.3f} n={sc_uc['n_pred']} "
              f"med={sc_uc['median_error_um']:.1f}µm", flush=True)
        print(f"  union_p1_first r@20={sc_up1['recall_at_20um']:.3f} n={sc_up1['n_pred']} "
              f"med={sc_up1['median_error_um']:.1f}µm", flush=True)
        print(f"  intersect     r@20={sc_in['recall_at_20um']:.3f} n={sc_in['n_pred']} "
              f"prec={sc_in['precision']:.3f}", flush=True)
        print(f"  oracle_best   r@20={oracle_r20:.3f}", flush=True)

        rows.append(dict(
            subject=subj,
            p1_r20=sc_p1["recall_at_20um"], p1_n=sc_p1["n_pred"],
            p1_med=sc_p1["median_error_um"], p1_wall=wall_p1,
            p6_r20=sc_p6["recall_at_20um"], p6_n=sc_p6["n_pred"],
            p6_med=sc_p6["median_error_um"], p6_wall=wall_p6,
            uc_r20=sc_uc["recall_at_20um"], uc_n=sc_uc["n_pred"],
            uc_med=sc_uc["median_error_um"],
            up1_r20=sc_up1["recall_at_20um"], up1_n=sc_up1["n_pred"],
            up1_med=sc_up1["median_error_um"],
            in_r20=sc_in["recall_at_20um"], in_n=sc_in["n_pred"],
            in_prec=sc_in["precision"],
            oracle_r20=oracle_r20,
        ))

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "ensemble_results.csv"
    df.to_csv(out, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"SUM r@20: P1={df.p1_r20.sum():.3f} P6={df.p6_r20.sum():.3f} "
          f"union_conf={df.uc_r20.sum():.3f} union_p1_first={df.up1_r20.sum():.3f} "
          f"intersect={df.in_r20.sum():.3f} oracle={df.oracle_r20.sum():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
