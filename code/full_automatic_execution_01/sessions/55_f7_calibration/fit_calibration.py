"""S55 — F7 per-method confidence calibration (LOSO + final fit).

For P1, P4, P6:
1. Run each method on all 6 subjects.
2. Per pair, label is_correct := distance(pred_hcr_centroid, GT_hcr_centroid) < 20 µm
   for CZ cells in GT; drop pairs whose CZ is not in GT (unlabeled).
3. LOSO: fit IsotonicRegression on 5 subjects, evaluate Brier on held-out subject.
4. Final fit: one calibrator per method using all 6 subjects.
5. Pickle calibrators to `lib/calibrators/{p1,p4,p6}.pkl`.

Also runs a downstream test: three-way union_conf with calibrated vs raw
confidences for each subject, measuring sum r@20 difference.
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
from lib.calibrate import fit_isotonic  # noqa: E402


SUBJECTS = ["788406", "790322", "755252", "767022", "767018", "782149"]
CALIB_DIR = Path("/root/capsule/code/full_automatic_execution_01/lib/calibrators")
CALIB_DIR.mkdir(exist_ok=True)

P1_KW = dict(hcr_quality_beta=5.0)
P4_KW = dict(hcr_quality_beta=5.0)
P6_KW = dict(method="cpd_nonrigid", maxiter=30, w_outlier=0.3,
             crop_pad_um=150.0, match_radius_um=60.0, hcr_quality_beta=0.0)


def _label_pairs(pairs_df: pd.DataFrame, s) -> pd.DataFrame:
    """Return pairs_df with `is_correct` column in {0, 1} for pairs whose
    CZ cell is in GT; drops pairs whose CZ is not in GT."""
    gt = s.coreg_table.set_index("cz_id")
    if pairs_df.empty or gt.empty:
        return pd.DataFrame(columns=list(pairs_df.columns) + ["is_correct"])
    from bench.harness import _hcr_centroids_um
    hcr_cent = _hcr_centroids_um(s).set_index("hcr_id")

    pred = pairs_df.drop_duplicates("cz_id", keep="first").set_index("cz_id")
    joined = pred.join(gt.rename(columns={"hcr_id": "gt_hcr_id"})[["gt_hcr_id"]],
                        how="inner")
    gt_xyz = hcr_cent.reindex(joined["gt_hcr_id"])[
        ["hcr_x_um", "hcr_y_um", "hcr_z_um"]].values.astype(float)
    pred_xyz = joined[["hcr_x_um", "hcr_y_um", "hcr_z_um"]].values.astype(float)
    d = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    joined = joined.assign(dist_um=d, is_correct=(d < 20.0).astype(int))
    return joined.reset_index()


def collect_data():
    """Collect (method, subject, conf, is_correct, cz_id, hcr_id) rows."""
    rows = []
    per_subject_pairs = {}  # {(subj, method): pairs_df with is_correct}
    for subj in SUBJECTS:
        print(f"\n=== {subj} ===", flush=True)
        t = time.time()
        s = load_subject(subj)
        print(f"  load {time.time()-t:.1f}s n_cz={len(s.cz_centroids)} "
              f"n_hcr_gfp={len(s.hcr_gfp_df)}", flush=True)

        for mid, kw in [("P1", P1_KW), ("P4", P4_KW), ("P6", P6_KW)]:
            t0 = time.time()
            try:
                r = CANDIDATES[mid](s, **kw)
                pairs = r.pairs_df
                labeled = _label_pairs(pairs, s)
                n_lab = len(labeled)
                n_pos = int(labeled["is_correct"].sum()) if n_lab else 0
                print(f"  {mid} n_pairs={len(pairs)} n_labeled={n_lab} "
                      f"n_correct={n_pos} "
                      f"conf_q=[{pairs['confidence'].quantile(0.1):.3f}, "
                      f"{pairs['confidence'].quantile(0.5):.3f}, "
                      f"{pairs['confidence'].quantile(0.9):.3f}] "
                      f"wall={time.time()-t0:.1f}s", flush=True)
                per_subject_pairs[(subj, mid)] = (pairs, labeled)
                for _, r_row in labeled.iterrows():
                    rows.append(dict(
                        method=mid, subject=subj,
                        conf=float(r_row["confidence"]),
                        is_correct=int(r_row["is_correct"]),
                    ))
            except Exception as e:
                print(f"  {mid} FAILED: {e}", flush=True)
                per_subject_pairs[(subj, mid)] = (pd.DataFrame(), pd.DataFrame())

    return pd.DataFrame(rows), per_subject_pairs


def loso_fit(all_rows: pd.DataFrame):
    """Per-method LOSO: Brier before/after per held-out subject."""
    print("\n=== LOSO per-method calibration ===", flush=True)
    loso_summary = []
    for mid in ["P1", "P4", "P6"]:
        mid_rows = all_rows[all_rows["method"] == mid]
        if mid_rows.empty:
            continue
        print(f"\n[{mid}] total n={len(mid_rows)} "
              f"n_correct={int(mid_rows['is_correct'].sum())} "
              f"base_rate={mid_rows['is_correct'].mean():.3f}")
        for held in SUBJECTS:
            train = mid_rows[mid_rows["subject"] != held]
            test = mid_rows[mid_rows["subject"] == held]
            if train.empty or test.empty or test["is_correct"].nunique() < 2:
                print(f"  hold={held} train_n={len(train)} test_n={len(test)} "
                      f"n_pos_test={int(test['is_correct'].sum())} SKIP")
                continue
            cal = fit_isotonic(train["conf"].values, train["is_correct"].values)
            test_probs_raw = test["conf"].values
            test_probs_cal = cal.predict(test["conf"].values)
            from sklearn.metrics import brier_score_loss
            b_raw = brier_score_loss(test["is_correct"].values,
                                     np.clip(test_probs_raw, 0, 1))
            b_cal = brier_score_loss(test["is_correct"].values, test_probs_cal)
            print(f"  hold={held} train_n={len(train)} test_n={len(test)} "
                  f"brier_raw={b_raw:.4f} brier_cal={b_cal:.4f} "
                  f"Δ={b_cal-b_raw:+.4f}")
            loso_summary.append(dict(method=mid, hold=held,
                                     n_train=len(train), n_test=len(test),
                                     brier_raw=b_raw, brier_cal=b_cal))
    return pd.DataFrame(loso_summary)


def final_fit(all_rows: pd.DataFrame):
    """Fit final calibrator per method on all 6 subjects."""
    print("\n=== Final calibration (all-subject fit) ===", flush=True)
    calibrators = {}
    for mid in ["P1", "P4", "P6"]:
        mid_rows = all_rows[all_rows["method"] == mid]
        if mid_rows.empty:
            continue
        cal = fit_isotonic(mid_rows["conf"].values, mid_rows["is_correct"].values)
        print(f"[{mid}] fit_n={len(mid_rows)} brier_train={cal.brier:.4f} "
              f"thr={cal.thresholds}")
        calibrators[mid] = cal
        out = CALIB_DIR / f"{mid.lower()}.pkl"
        with open(out, "wb") as f:
            pickle.dump(cal, f)
        print(f"  wrote {out}")
    return calibrators


def eval_union_conf(per_subject_pairs, calibrators):
    """Compare three-way union_conf r@20: raw vs calibrated confidences."""
    print("\n=== Union_conf r@20: raw vs calibrated ===", flush=True)
    rows = []
    for subj in SUBJECTS:
        s = load_subject(subj)
        n_hcr = len(s.hcr_gfp_df)
        dfs = {}
        for mid in ["P1", "P4", "P6"]:
            pairs, _ = per_subject_pairs.get((subj, mid), (pd.DataFrame(), None))
            dfs[mid] = pairs

        # Raw uc3
        raw_cat = pd.concat([d for d in dfs.values() if len(d)], ignore_index=True)
        if len(raw_cat) == 0:
            print(f"{subj} no pairs")
            continue
        raw_merged = (raw_cat.sort_values("confidence", ascending=False)
                             .drop_duplicates("cz_id", keep="first")
                             .sort_values("cz_id").reset_index(drop=True))
        sc_raw = compare_to_gt(raw_merged, s)

        # Calibrated uc3
        cal_pieces = []
        for mid in ["P1", "P4", "P6"]:
            d = dfs[mid]
            if len(d) == 0 or mid not in calibrators:
                cal_pieces.append(d)
                continue
            d = d.copy()
            d["confidence"] = calibrators[mid].predict(d["confidence"].values)
            cal_pieces.append(d)
        cal_cat = pd.concat([d for d in cal_pieces if len(d)], ignore_index=True)
        cal_merged = (cal_cat.sort_values("confidence", ascending=False)
                             .drop_duplicates("cz_id", keep="first")
                             .sort_values("cz_id").reset_index(drop=True))
        sc_cal = compare_to_gt(cal_merged, s)

        print(f"  {subj} n_hcr={n_hcr} uc3_raw r@20={sc_raw['recall_at_20um']:.3f} "
              f"uc3_cal r@20={sc_cal['recall_at_20um']:.3f} "
              f"Δ={sc_cal['recall_at_20um']-sc_raw['recall_at_20um']:+.3f}")
        rows.append(dict(subject=subj, n_hcr=n_hcr,
                         uc3_raw=sc_raw["recall_at_20um"],
                         uc3_cal=sc_cal["recall_at_20um"]))
    df = pd.DataFrame(rows)
    print(f"\n  SUM: uc3_raw={df.uc3_raw.sum():.3f} "
          f"uc3_cal={df.uc3_cal.sum():.3f} "
          f"Δ={df.uc3_cal.sum()-df.uc3_raw.sum():+.3f}")
    return df


def main():
    t_start = time.time()
    all_rows, per_subject_pairs = collect_data()
    out_csv = Path(__file__).parent / "labeled_pairs.csv"
    all_rows.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    loso_df = loso_fit(all_rows)
    loso_df.to_csv(Path(__file__).parent / "loso_summary.csv", index=False)

    calibrators = final_fit(all_rows)

    uc3_df = eval_union_conf(per_subject_pairs, calibrators)
    uc3_df.to_csv(Path(__file__).parent / "uc3_raw_vs_cal.csv", index=False)

    print(f"\nTotal wall: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
