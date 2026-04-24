"""S56 — P1 ⊕ P4 ⊕ P6 three-way ensemble benchmark.

S51 showed P4+β=5 lifts 755252 to r@20=0.091 (best of any single method).
S54 showed P1⊕P6 ensemble at 0.971 sum r@20 but misses the P4 sparse-tile lift.
This session extends to three-way.

Strategies:
  - p1_p6_auto: S54 baseline (union_p1_first dense, union_conf sparse).
  - union_conf3: all three; higher-confidence wins cz_id collision.
  - union_p1_first3: P1 wins collisions; then P4; then P6.
  - union_p4_for_sparse: on sparse (n_hcr_gfp < 10k) use union_conf over {P1,P4,P6};
    else union_p1_first over {P1,P4,P6}. Asymmetric: P4 should mainly help
    mid-density subjects (755252 has 30k cells total but sparse GFP+ per µm³).
  - oracle3: max(P1,P4,P6) per subject.
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
P4_KW = dict(hcr_quality_beta=5.0)
P6_KW = dict(method="cpd_nonrigid", maxiter=30, w_outlier=0.3,
             crop_pad_um=150.0, match_radius_um=60.0, hcr_quality_beta=0.0)


def _score(pairs_df: pd.DataFrame, s) -> dict:
    return compare_to_gt(pairs_df, s)


def _union_conf_n(dfs: "list[pd.DataFrame]") -> pd.DataFrame:
    cat = pd.concat(dfs, ignore_index=True)
    if len(cat) == 0:
        return cat
    cat = cat.sort_values("confidence", ascending=False)
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.sort_values("cz_id").reset_index(drop=True)
    return cat


def _union_priority(dfs: "list[pd.DataFrame]") -> pd.DataFrame:
    """Priority by list order — first df wins on cz_id collision."""
    pieces = []
    for i, d in enumerate(dfs):
        if len(d) == 0:
            continue
        pieces.append(d.assign(_pri=i))
    if not pieces:
        return pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                      "cz_x_um","cz_y_um","cz_z_um",
                                      "hcr_x_um","hcr_y_um","hcr_z_um"])
    cat = pd.concat(pieces, ignore_index=True)
    cat = cat.sort_values(["cz_id", "_pri"], ascending=[True, True])
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.drop(columns=["_pri"]).reset_index(drop=True)
    return cat


def main():
    rows = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===", flush=True)
        t_load = time.time()
        s = load_subject(subj)
        n_hcr = len(s.hcr_gfp_df)
        print(f"  load {time.time()-t_load:.1f}s  n_cz={len(s.cz_centroids)} "
              f"n_hcr_gfp={n_hcr}", flush=True)

        results = {}
        for cid, kw in [("P1", P1_KW), ("P4", P4_KW), ("P6", P6_KW)]:
            t0 = time.time()
            try:
                r = CANDIDATES[cid](s, **kw)
                wall = time.time() - t0
                sc = _score(r.pairs_df, s)
                print(f"  {cid} r@20={sc['recall_at_20um']:.3f} n={sc['n_pred']} "
                      f"med={sc['median_error_um']:.1f}µm wall={wall:.1f}s",
                      flush=True)
                results[cid] = (r.pairs_df, sc, wall)
            except Exception as e:
                print(f"  {cid} FAILED: {e}", flush=True)
                empty = pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                               "cz_x_um","cz_y_um","cz_z_um",
                                               "hcr_x_um","hcr_y_um","hcr_z_um"])
                results[cid] = (empty, _score(empty, s), 0.0)

        p1_df, sc_p1, _ = results["P1"]
        p4_df, sc_p4, _ = results["P4"]
        p6_df, sc_p6, _ = results["P6"]

        # S54 baseline (P1⊕P6 auto)
        if n_hcr < 10_000:
            p1p6_auto = _union_conf_n([p1_df, p6_df])
        else:
            p1p6_auto = _union_priority([p1_df, p6_df])
        sc_b = _score(p1p6_auto, s)

        # Three-way union_conf (highest-conf wins)
        uc3 = _union_conf_n([p1_df, p4_df, p6_df])
        sc_uc3 = _score(uc3, s)

        # Three-way priority (P1 > P4 > P6)
        up14p6 = _union_priority([p1_df, p4_df, p6_df])
        sc_up14p6 = _score(up14p6, s)

        # Three-way auto: sparse → uc3, dense → up14p6
        if n_hcr < 10_000:
            auto3 = uc3
        else:
            auto3 = up14p6
        sc_auto3 = _score(auto3, s)

        # Three-way with P4-favored for mid-density (10k ≤ n_hcr < 20k uses P1>P4>P6;
        # n_hcr >= 20k uses P4>P1>P6 because dense-tile subjects like 755252 benefit
        # from P4's pairwise-consistency putative generator over P1's F6-NN)
        if n_hcr < 10_000:
            auto3_p4 = uc3
        elif n_hcr >= 20_000:
            auto3_p4 = _union_priority([p4_df, p1_df, p6_df])
        else:
            auto3_p4 = up14p6
        sc_auto3p4 = _score(auto3_p4, s)

        oracle3 = max(sc_p1["recall_at_20um"], sc_p4["recall_at_20um"],
                      sc_p6["recall_at_20um"])

        print(f"  p1p6_auto      r@20={sc_b['recall_at_20um']:.3f} "
              f"n={sc_b['n_pred']}", flush=True)
        print(f"  uc3            r@20={sc_uc3['recall_at_20um']:.3f} "
              f"n={sc_uc3['n_pred']}", flush=True)
        print(f"  up3(P1>P4>P6)  r@20={sc_up14p6['recall_at_20um']:.3f} "
              f"n={sc_up14p6['n_pred']}", flush=True)
        print(f"  auto3          r@20={sc_auto3['recall_at_20um']:.3f} "
              f"n={sc_auto3['n_pred']}", flush=True)
        print(f"  auto3_p4       r@20={sc_auto3p4['recall_at_20um']:.3f} "
              f"n={sc_auto3p4['n_pred']}", flush=True)
        print(f"  oracle3        r@20={oracle3:.3f}", flush=True)

        rows.append(dict(
            subject=subj, n_hcr=n_hcr,
            p1_r20=sc_p1["recall_at_20um"], p1_n=sc_p1["n_pred"], p1_med=sc_p1["median_error_um"],
            p4_r20=sc_p4["recall_at_20um"], p4_n=sc_p4["n_pred"], p4_med=sc_p4["median_error_um"],
            p6_r20=sc_p6["recall_at_20um"], p6_n=sc_p6["n_pred"], p6_med=sc_p6["median_error_um"],
            p1p6_auto=sc_b["recall_at_20um"],
            uc3=sc_uc3["recall_at_20um"],
            up3=sc_up14p6["recall_at_20um"],
            auto3=sc_auto3["recall_at_20um"],
            auto3_p4=sc_auto3p4["recall_at_20um"],
            oracle3=oracle3,
        ))

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "three_way_results.csv"
    df.to_csv(out, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"SUM r@20: P1={df.p1_r20.sum():.3f} P4={df.p4_r20.sum():.3f} "
          f"P6={df.p6_r20.sum():.3f} p1p6_auto={df.p1p6_auto.sum():.3f} "
          f"uc3={df.uc3.sum():.3f} up3={df.up3.sum():.3f} "
          f"auto3={df.auto3.sum():.3f} auto3_p4={df.auto3_p4.sum():.3f} "
          f"oracle3={df.oracle3.sum():.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
