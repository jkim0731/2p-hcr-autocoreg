"""S55 — Test whether calibration beats C5's existing dispatch rule.

Re-runs P1/P4/P6 once per subject, then evaluates 8 merge strategies:
  - raw_up_P1_P4_P6    priority P1>P4>P6 on all subjects (uniform)
  - raw_up_P4_P1_P6    priority P4>P1>P6 on all subjects (uniform)
  - raw_uc3            raw-confidence-sort union on all subjects (uniform)
  - cal_uc3            calibrated-confidence-sort union on all subjects (uniform)
  - C5_auto            C5's current rule (sparse→raw uc3; mid→up_P1P4P6; dense→up_P4P1P6)
  - C5_auto_cal        C5 rule with calibrated uc3 on sparse branch
  - cal_uc3_mid_up     calibrated uc3 uniformly except mid (priority)
  - best_per_subject   oracle (pick best strategy per subject)
"""
from __future__ import annotations

import pickle
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
CALIB_DIR = Path("/root/capsule/code/full_automatic_execution_01/lib/calibrators")

P1_KW = dict(hcr_quality_beta=5.0)
P4_KW = dict(hcr_quality_beta=5.0)
P6_KW = dict(method="cpd_nonrigid", maxiter=30, w_outlier=0.3,
             crop_pad_um=150.0, match_radius_um=60.0, hcr_quality_beta=0.0)


def _up_priority(dfs):
    pieces = [d.assign(_pri=i) for i, d in enumerate(dfs) if len(d) > 0]
    if not pieces:
        return pd.DataFrame()
    cat = pd.concat(pieces, ignore_index=True)
    return (cat.sort_values(["cz_id", "_pri"], ascending=[True, True])
               .drop_duplicates("cz_id", keep="first")
               .drop(columns=["_pri"]).reset_index(drop=True))


def _uc(dfs):
    cat = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    if not len(cat):
        return pd.DataFrame()
    return (cat.sort_values("confidence", ascending=False)
               .drop_duplicates("cz_id", keep="first")
               .sort_values("cz_id").reset_index(drop=True))


def main():
    with open(CALIB_DIR / "p1.pkl", "rb") as f:
        cal_p1 = pickle.load(f)
    with open(CALIB_DIR / "p4.pkl", "rb") as f:
        cal_p4 = pickle.load(f)
    with open(CALIB_DIR / "p6.pkl", "rb") as f:
        cal_p6 = pickle.load(f)
    cals = dict(P1=cal_p1, P4=cal_p4, P6=cal_p6)

    rows = []
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        n_hcr = len(s.hcr_gfp_df)

        dfs = {}
        for mid, kw in [("P1", P1_KW), ("P4", P4_KW), ("P6", P6_KW)]:
            t0 = time.time()
            r = CANDIDATES[mid](s, **kw)
            print(f"  {mid} n={len(r.pairs_df)} wall={time.time()-t0:.1f}s",
                  flush=True)
            dfs[mid] = r.pairs_df

        # Calibrated copies
        dfs_cal = {}
        for mid in ["P1", "P4", "P6"]:
            d = dfs[mid].copy() if len(dfs[mid]) else pd.DataFrame()
            if len(d):
                d["confidence"] = cals[mid].predict(d["confidence"].values)
            dfs_cal[mid] = d

        out = {"subject": subj, "n_hcr": n_hcr}
        strategies = {
            "raw_up_P1_P4_P6":  _up_priority([dfs["P1"], dfs["P4"], dfs["P6"]]),
            "raw_up_P4_P1_P6":  _up_priority([dfs["P4"], dfs["P1"], dfs["P6"]]),
            "raw_up_P6_P1_P4":  _up_priority([dfs["P6"], dfs["P1"], dfs["P4"]]),
            "raw_uc3":          _uc([dfs["P1"], dfs["P4"], dfs["P6"]]),
            "cal_uc3":          _uc([dfs_cal["P1"], dfs_cal["P4"], dfs_cal["P6"]]),
        }
        # C5 auto (raw)
        if n_hcr < 10_000:
            strategies["C5_auto"] = strategies["raw_uc3"]
        elif n_hcr >= 20_000:
            strategies["C5_auto"] = strategies["raw_up_P4_P1_P6"]
        else:
            strategies["C5_auto"] = strategies["raw_up_P1_P4_P6"]
        # C5 auto with calibrated union_conf on sparse
        if n_hcr < 10_000:
            strategies["C5_auto_cal"] = strategies["cal_uc3"]
        else:
            strategies["C5_auto_cal"] = strategies["C5_auto"]
        # cal_uc3 uniformly except mid density: mid uses priority
        if 10_000 <= n_hcr < 20_000:
            strategies["cal_uc3_mid_up"] = strategies["raw_up_P1_P4_P6"]
        else:
            strategies["cal_uc3_mid_up"] = strategies["cal_uc3"]

        for name, df in strategies.items():
            sc = compare_to_gt(df, s) if len(df) else dict(recall_at_20um=0.0)
            out[name] = sc["recall_at_20um"]
            print(f"  {name:22s} r@20={out[name]:.3f}", flush=True)
        rows.append(out)

    df = pd.DataFrame(rows)
    out_csv = Path(__file__).parent / "strategy_sweep.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== SUMMARY ===")
    strategy_cols = [c for c in df.columns if c not in ("subject", "n_hcr")]
    for c in strategy_cols:
        print(f"  {c:22s} sum r@20 = {df[c].sum():.3f}")
    print()
    best_per_subj = df[strategy_cols].max(axis=1)
    print(f"  {'best_per_subject':22s} sum r@20 = {best_per_subj.sum():.3f}")
    for i, subj in enumerate(df["subject"]):
        winners = df.iloc[i][strategy_cols].idxmax()
        print(f"    {subj}: {winners} r@20={df.iloc[i][winners]:.3f}")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
