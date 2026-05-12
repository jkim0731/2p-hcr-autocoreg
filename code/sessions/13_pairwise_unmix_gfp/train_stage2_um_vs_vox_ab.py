"""A/B retrain of v5d stage-2: µm-feature variant vs vox-feature variant.

Both variants:
  * use the same merged label log (1983 actions in
    code/sessions/v3_S11_roi_quality/outputs/roi_qc_actions.jsonl)
  * use the same DROP_DEAD_FEATURES + DROP_LOW_GAIN_V3_EXTRA lists
  * **do NOT add within-subject percentile-rank features**
    (`PCT_RANK_COLS` is empty) — rank features are sensitive to
    sample-prep distribution shifts (e.g. tissue thickness).

Mode `um`:
  * load v2/v3_extra/v4/v5 (no v6_vox)
  * keep all µm features (DROP_UM_FEATURES is dropped from the drop set)

Mode `vox`:
  * load v2/v3_extra/v4/v5/v6_vox
  * drop all µm features (keep current v5d behaviour)

Outputs go to code/sessions/13_pairwise_unmix_gfp/outputs/abtest_um_vs_vox/<mode>/
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ROOT = Path("/root/capsule")
CACHE = ROOT / "code/dev_code/cached_roi_quality"
SESS_S11 = ROOT / "code/sessions/v3_S11_roi_quality"
SESS_AB = ROOT / "code/sessions/13_pairwise_unmix_gfp/outputs/abtest_um_vs_vox"
LABEL_LOG = SESS_S11 / "outputs/roi_qc_actions.jsonl"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

DROP_FEATURES_BASE = {"hcr_id", "y", "label", "human_label", "sid"}

DROP_UM_FEATURES = {
    "axis3d_extent_um", "axis3d_lambda1_um2", "axis3d_lambda2_um2",
    "axis3d_lambda3_um2", "axis3d_peak_sep_um_intens",
    "axis3d_raw_extent_um", "bbox_x_extent_um", "bbox_y_extent_um",
    "bbox_z_extent_um", "c405_core4um_p50_opened",
    "c405_shell4um_p50_opened", "c405_shell_minus_core4um_p50",
    "c405_shell_minus_core4um_p90", "c405_shell_over_core4um_p50_ratio",
    "core4um_voxel_frac_opened", "equivalent_diameter_um_opened",
    "equivalent_diameter_um_raw", "n_neighbors_30um",
    "proj_xy_main_extent_um", "proj_xy_orth_fwhm_um",
    "proj_yz_main_extent_um", "proj_yz_orth_fwhm_um",
    "proj_zx_main_extent_um", "proj_zx_orth_fwhm_um",
    "sa_to_vol_um_inv_opened", "sa_to_vol_um_inv_raw",
    "surface_area_um2_opened", "surface_area_um2_raw",
    "volume_um3_opened", "volume_um3_raw", "volume_um3_raw_v4",
}
DROP_DEAD_FEATURES = {
    "boundary_touching", "n_components_after_opening",
    "tight_bbox_in_pickle_bbox", "volume_pickle_minus_zarr_l2_eq",
    "protrusion_touches_other",
}
DROP_LOW_GAIN_V3_EXTRA = {
    "axis3d_n_peaks_inner_area",
    "axis3d_peak_prom_2nd_intens",
    "proj_yz_main_n_peaks_inner",
    "proj_zx_main_n_peaks_inner",
    "proj_xy_main_n_peaks_inner",
    "axis3d_raw_n_peaks_inner_intens",
    "proj_zx_orth_n_peaks_at_main",
    "proj_xy_orth_n_peaks_at_main",
    "proj_yz_orth_n_peaks_at_main",
    "axis3d_n_peaks_inner_intens",
}

CLASS_NAMES = ["bad", "bad_ok", "good", "merged"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
BINARY_POS = {"good", "bad_ok"}
BINARY_NEG = {"bad", "merged"}

LGB_BINARY = dict(
    objective="binary", learning_rate=0.05, num_leaves=31,
    min_data_in_leaf=20, feature_fraction=0.85, bagging_fraction=0.85,
    bagging_freq=5, lambda_l2=1.0, is_unbalance=True,
    metric=["binary_logloss", "auc"], verbosity=-1, seed=20260430,
)
LGB_MULTI = dict(
    objective="multiclass", num_class=len(CLASS_NAMES),
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=15,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    lambda_l2=1.0, metric=["multi_logloss"], verbosity=-1, seed=20260430,
)
N_ESTIMATORS = 400
EARLY_STOP = 30


def _load_label_log() -> pd.DataFrame:
    return pd.read_json(LABEL_LOG, lines=True)


def _active_labels(log: pd.DataFrame, sid: str) -> pd.DataFrame:
    sub = log[log["sid"].astype(str) == sid].copy()
    if sub.empty:
        return pd.DataFrame(columns=["hcr_id", "label"])
    tomb_ids: set[int] = set()
    if (sub["label"] == "_undone_").any():
        for _, r in sub[sub["label"] == "_undone_"].iterrows():
            ub = r.get("undoes") or {}
            try:
                tomb_ids.add(int(ub.get("hcr_id", -1)))
            except (TypeError, ValueError):
                pass
    sub = sub[sub["label"].isin(["good", "bad", "bad_ok", "merged", "unsure"])]
    sub = sub[~sub["hcr_id"].astype(int).isin(tomb_ids)]
    if "ts" in sub.columns:
        sub = sub.sort_values("ts")
    sub = sub.drop_duplicates(subset=["hcr_id"], keep="last")
    return sub[["hcr_id", "label"]].reset_index(drop=True)


def _load_features(sid: str, mode: str) -> pd.DataFrame:
    f = pd.read_parquet(CACHE / f"{sid}_features_v2.parquet")
    g = pd.read_parquet(CACHE / f"{sid}_features_v3_extra.parquet")
    h = pd.read_parquet(CACHE / f"{sid}_features_v4.parquet")
    k = pd.read_parquet(CACHE / f"{sid}_features_v5.parquet")
    out = f.merge(g, on="hcr_id", how="left", suffixes=("", "_v3"))
    out = out.merge(h, on="hcr_id", how="left", suffixes=("", "_v4"))
    out = out.merge(k, on="hcr_id", how="left", suffixes=("", "_v5"))
    if mode == "vox":
        v = pd.read_parquet(CACHE / f"{sid}_features_v6_vox.parquet")
        out = out.merge(v, on="hcr_id", how="left", suffixes=("", "_v6"))
    out["sid"] = sid
    return out


def _build_drop_set(mode: str) -> set:
    drop = DROP_FEATURES_BASE | DROP_DEAD_FEATURES | DROP_LOW_GAIN_V3_EXTRA
    if mode == "vox":
        drop = drop | DROP_UM_FEATURES
    return drop


def _build_matrix(df: pd.DataFrame, feature_columns, drop: set):
    if feature_columns is None:
        cols = [
            c for c in df.columns
            if c not in drop
            and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
        ]
    else:
        cols = list(feature_columns)
    X = df[cols].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("float32")
    return X, cols


def _early_stop_split(n: int, frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, int(round(n * (1 - frac))))
    return idx[:cut], idx[cut:]


def _train(params, X_tr, y_tr) -> lgb.Booster:
    tr, va = _early_stop_split(len(X_tr), frac=0.15, seed=42)
    train_set = lgb.Dataset(X_tr.iloc[tr], y_tr[tr])
    valid_set = lgb.Dataset(X_tr.iloc[va], y_tr[va], reference=train_set)
    return lgb.train(
        params, train_set, num_boost_round=N_ESTIMATORS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)],
    )


def run(mode: str, write_production: bool = False) -> dict:
    print("=" * 70)
    print(f"A/B mode: {mode}  (no rank features, no within-subject pct_subj)"
          + ("  [WRITE-PRODUCTION]" if write_production else ""))
    print("=" * 70)
    OUT = SESS_AB / mode
    OUT.mkdir(parents=True, exist_ok=True)

    log = _load_label_log()
    print(f"label log: {len(log)} rows")
    feats = {sid: _load_features(sid, mode) for sid in SUBJECTS}
    labs = {sid: _active_labels(log, sid) for sid in SUBJECTS}
    drop = _build_drop_set(mode)

    X_probe, cols0 = _build_matrix(feats[SUBJECTS[0]], None, drop)
    print(f"feature matrix: {X_probe.shape[1]} columns "
          f"(input merged width {feats[SUBJECTS[0]].shape[1]}, drop set {len(drop)})")

    # ── BINARY ───────────────────────────────────────────────
    print("\nBINARY:")
    feature_columns = None
    bin_metrics, bin_imp = [], {}
    for held in SUBJECTS:
        tr_X, tr_y = [], []
        for sid in SUBJECTS:
            if sid == held:
                continue
            f = feats[sid]
            l = labs[sid]
            l = l[l["label"].isin(BINARY_POS | BINARY_NEG)].copy()
            l["y"] = l["label"].isin(BINARY_POS).astype("int8")
            merged = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            X, cols = _build_matrix(merged, feature_columns, drop)
            feature_columns = cols
            tr_X.append(X)
            tr_y.append(merged["y"].to_numpy("int8"))
        X_tr = pd.concat(tr_X, axis=0).reset_index(drop=True)
        y_tr = np.concatenate(tr_y)

        f_held = feats[held]
        X_held, _ = _build_matrix(f_held, feature_columns, drop)
        l_held = labs[held]
        l_held = l_held[l_held["label"].isin(BINARY_POS | BINARY_NEG)].copy()
        l_held["y"] = l_held["label"].isin(BINARY_POS).astype("int8")

        booster = _train(LGB_BINARY, X_tr, y_tr)
        p_full = booster.predict(X_held, num_iteration=booster.best_iteration)
        scored = pd.DataFrame({"hcr_id": f_held["hcr_id"].to_numpy(), "score": p_full})
        eval_df = l_held.merge(scored, on="hcr_id", how="left")
        y_eval = eval_df["y"].to_numpy("int8")
        p_eval = eval_df["score"].to_numpy("float64")
        auc = roc_auc_score(y_eval, p_eval) if len(np.unique(y_eval)) > 1 else float("nan")
        ap = average_precision_score(y_eval, p_eval) if len(np.unique(y_eval)) > 1 else float("nan")
        brier = brier_score_loss(y_eval, p_eval)
        acc = accuracy_score(y_eval, (p_eval >= 0.5).astype("int8"))
        bin_metrics.append({"held": held, "n_tr": len(X_tr), "n_ev": len(y_eval),
                            "auc": auc, "ap": ap, "brier": brier, "acc@0.5": acc,
                            "iter": booster.best_iteration})
        print(f"  [hold {held}] n_tr={len(X_tr)} n_ev={len(y_eval)}  "
              f"AUC={auc:.4f} AP={ap:.4f} Brier={brier:.4f} acc@0.5={acc:.3f}  iter={booster.best_iteration}")
        gain = booster.feature_importance(importance_type="gain")
        for c, g in zip(feature_columns, gain):
            bin_imp.setdefault(c, []).append(float(g))

        if write_production:
            oof_df = pd.DataFrame({
                "hcr_id": f_held["hcr_id"].to_numpy("int64"),
                "score": p_full.astype("float32"),
            }).merge(
                l_held.rename(columns={"label": "human_label"})[["hcr_id", "human_label"]],
                on="hcr_id", how="left",
            )
            oof_df.to_parquet(CACHE / f"{held}_stage2_binary_score_v5d_{mode}.parquet", index=False)

    bm = pd.DataFrame(bin_metrics).sort_values("held")
    bm.to_csv(OUT / f"binary_loso_metrics_{mode}.csv", index=False)
    fi_b = (pd.DataFrame({c: pd.Series(v) for c, v in bin_imp.items()})
            .mean().sort_values(ascending=False)
            .rename("mean_gain").reset_index().rename(columns={"index": "feature"}))
    fi_b.to_csv(OUT / f"binary_feature_importance_{mode}.csv", index=False)
    print(f"\n  binary mean AUC={bm['auc'].mean():.4f}  AP={bm['ap'].mean():.4f}  Brier={bm['brier'].mean():.4f}")

    # ── 4-CLASS ────────────────────────────────────────────
    print("\n4-CLASS:")
    multi_metrics, multi_imp = [], {}
    overall_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    feature_columns = None
    for held in SUBJECTS:
        tr_X, tr_y = [], []
        for sid in SUBJECTS:
            if sid == held:
                continue
            f = feats[sid]
            l = labs[sid]
            l = l[l["label"].isin(CLASS_NAMES)].copy()
            l["y"] = l["label"].map(CLASS_TO_IDX).astype("int8")
            merged = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            X, cols = _build_matrix(merged, feature_columns, drop)
            feature_columns = cols
            tr_X.append(X)
            tr_y.append(merged["y"].to_numpy("int8"))
        X_tr = pd.concat(tr_X, axis=0).reset_index(drop=True)
        y_tr = np.concatenate(tr_y)

        f_held = feats[held]
        X_held, _ = _build_matrix(f_held, feature_columns, drop)
        l_held = labs[held]
        l_held = l_held[l_held["label"].isin(CLASS_NAMES)].copy()
        l_held["y"] = l_held["label"].map(CLASS_TO_IDX).astype("int8")

        booster = _train(LGB_MULTI, X_tr, y_tr)
        proba_full = booster.predict(X_held, num_iteration=booster.best_iteration)
        scored = pd.DataFrame(proba_full, columns=[f"p_{c}" for c in CLASS_NAMES])
        scored["hcr_id"] = f_held["hcr_id"].to_numpy("int64")
        eval_df = l_held.merge(scored, on="hcr_id", how="left")
        y_eval = eval_df["y"].to_numpy("int8")
        proba_eval = eval_df[[f"p_{c}" for c in CLASS_NAMES]].to_numpy("float64")
        y_pred = proba_eval.argmax(axis=1)
        acc = accuracy_score(y_eval, y_pred)
        f1m = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        f1p = f1_score(y_eval, y_pred, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
        cm = confusion_matrix(y_eval, y_pred, labels=list(range(len(CLASS_NAMES))))
        overall_cm += cm
        row = {"held": held, "n_tr": len(X_tr), "n_ev": len(y_eval),
               "acc": acc, "f1_macro": f1m, "iter": booster.best_iteration}
        for c, fv in zip(CLASS_NAMES, f1p):
            row[f"f1_{c}"] = float(fv)
        multi_metrics.append(row)
        print(f"  [hold {held}] n_tr={len(X_tr)} n_ev={len(y_eval)}  "
              f"acc={acc:.3f} f1_macro={f1m:.3f}  "
              + " ".join(f"f1_{c}={f:.2f}" for c, f in zip(CLASS_NAMES, f1p))
              + f"  iter={booster.best_iteration}")
        gain = booster.feature_importance(importance_type="gain")
        for c, g in zip(feature_columns, gain):
            multi_imp.setdefault(c, []).append(float(g))

        if write_production:
            oof = scored[["hcr_id"] + [f"p_{c}" for c in CLASS_NAMES]].merge(
                l_held.rename(columns={"label": "human_label"})[["hcr_id", "human_label"]],
                on="hcr_id", how="left",
            )
            oof.to_parquet(CACHE / f"{held}_stage2_4class_proba_v5d_{mode}.parquet", index=False)

    mm = pd.DataFrame(multi_metrics).sort_values("held")
    mm.to_csv(OUT / f"4class_loso_metrics_{mode}.csv", index=False)
    fi_m = (pd.DataFrame({c: pd.Series(v) for c, v in multi_imp.items()})
            .mean().sort_values(ascending=False)
            .rename("mean_gain").reset_index().rename(columns={"index": "feature"}))
    fi_m.to_csv(OUT / f"4class_feature_importance_{mode}.csv", index=False)
    cm_df = pd.DataFrame(overall_cm,
                         index=[f"true_{c}" for c in CLASS_NAMES],
                         columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm_df.to_csv(OUT / f"4class_confusion_overall_{mode}.csv")
    print(f"\n  4cls mean acc={mm['acc'].mean():.4f}  mean f1_macro={mm['f1_macro'].mean():.4f}")

    if write_production:
        print("\n" + "-" * 70 + "\nproduction models on all labelled rows\n" + "-" * 70)
        Xb_parts, yb_parts = [], []
        for sid in SUBJECTS:
            f = feats[sid]; l = labs[sid]
            l = l[l["label"].isin(BINARY_POS | BINARY_NEG)].copy()
            l["y"] = l["label"].isin(BINARY_POS).astype("int8")
            m = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            X, _ = _build_matrix(m, feature_columns, drop)
            Xb_parts.append(X); yb_parts.append(m["y"].to_numpy("int8"))
        Xb = pd.concat(Xb_parts, axis=0).reset_index(drop=True)
        yb = np.concatenate(yb_parts)
        n_iter_b = max(int(np.median([m["iter"] for m in bin_metrics])), 80)
        bin_prod = lgb.train(LGB_BINARY, lgb.Dataset(Xb, yb),
                             num_boost_round=n_iter_b,
                             callbacks=[lgb.log_evaluation(0)])
        bin_prod.save_model(str(CACHE / f"roi_quality_stage2_binary_v5d_{mode}.txt"))
        print(f"  binary production ({mode}): {len(Xb)} rows ({yb.sum()} pos / {(yb==0).sum()} neg), {n_iter_b} iters")

        Xm_parts, ym_parts = [], []
        for sid in SUBJECTS:
            f = feats[sid]; l = labs[sid]
            l = l[l["label"].isin(CLASS_NAMES)].copy()
            l["y"] = l["label"].map(CLASS_TO_IDX).astype("int8")
            m = f.merge(l[["hcr_id", "y"]], on="hcr_id", how="inner")
            X, _ = _build_matrix(m, feature_columns, drop)
            Xm_parts.append(X); ym_parts.append(m["y"].to_numpy("int8"))
        Xm = pd.concat(Xm_parts, axis=0).reset_index(drop=True)
        ym = np.concatenate(ym_parts)
        n_iter_m = max(int(np.median([m["iter"] for m in multi_metrics])), 80)
        multi_prod = lgb.train(LGB_MULTI, lgb.Dataset(Xm, ym),
                               num_boost_round=n_iter_m,
                               callbacks=[lgb.log_evaluation(0)])
        multi_prod.save_model(str(CACHE / f"roi_quality_stage2_4class_v5d_{mode}.txt"))
        print(f"  4-class production ({mode}): {len(Xm)} rows "
              f"({', '.join(f'{c}={(ym==CLASS_TO_IDX[c]).sum()}' for c in CLASS_NAMES)}), {n_iter_m} iters")

        meta = {
            "version": f"v5d_{mode}",
            "mode": mode,
            "feature_columns": feature_columns,
            "subjects": SUBJECTS,
            "class_names": CLASS_NAMES,
            "binary_pos": sorted(BINARY_POS),
            "binary_neg": sorted(BINARY_NEG),
            "pct_rank_columns": [],  # rank features intentionally disabled
            "drop_um_features": (mode == "vox"),
            "uses_v6_vox": (mode == "vox"),
            "binary": {
                "n_train_total": int(len(Xb)),
                "n_iter_prod": int(n_iter_b),
                "loso_mean_auc": float(bm["auc"].mean()),
                "loso_mean_ap": float(bm["ap"].mean()),
                "loso_mean_brier": float(bm["brier"].mean()),
                "loso_mean_acc05": float(bm["acc@0.5"].mean()),
                "params": LGB_BINARY,
            },
            "four_class": {
                "n_train_total": int(len(Xm)),
                "n_iter_prod": int(n_iter_m),
                "loso_mean_acc": float(mm["acc"].mean()),
                "loso_mean_f1_macro": float(mm["f1_macro"].mean()),
                "params": LGB_MULTI,
            },
        }
        (CACHE / f"roi_quality_stage2_meta_v5d_{mode}.json").write_text(json.dumps(meta, indent=2))
        print(f"  meta -> {CACHE / f'roi_quality_stage2_meta_v5d_{mode}.json'}")
        print(f"  per-subject OOF -> {CACHE}/<sid>_stage2_*_v5d_{mode}.parquet")

    return {
        "mode": mode,
        "n_features": len(feature_columns),
        "binary_mean_auc": float(bm["auc"].mean()),
        "binary_mean_ap": float(bm["ap"].mean()),
        "binary_mean_brier": float(bm["brier"].mean()),
        "binary_mean_acc": float(bm["acc@0.5"].mean()),
        "fourcls_mean_acc": float(mm["acc"].mean()),
        "fourcls_mean_f1_macro": float(mm["f1_macro"].mean()),
        "fourcls_mean_f1_per_class": {c: float(mm[f"f1_{c}"].mean()) for c in CLASS_NAMES},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["um", "vox"], choices=["um", "vox"])
    ap.add_argument("--write-production", action="store_true",
                    help="Also write per-subject OOF parquets and production "
                         "boosters under cached_roi_quality/ with `_v5d_<mode>` suffix")
    args = ap.parse_args()
    SESS_AB.mkdir(parents=True, exist_ok=True)
    summary = []
    t0 = time.time()
    for mode in args.modes:
        summary.append(run(mode, write_production=args.write_production))
    summary_path = SESS_AB / "ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== AB DONE in {time.time() - t0:.0f}s — summary -> {summary_path} ===")
    for s in summary:
        print(f"  {s['mode']}: nF={s['n_features']:3d}  "
              f"binAUC={s['binary_mean_auc']:.4f}  "
              f"4cls_acc={s['fourcls_mean_acc']:.4f}  "
              f"f1m={s['fourcls_mean_f1_macro']:.4f}  "
              + " ".join(f"f1_{c}={s['fourcls_mean_f1_per_class'][c]:.3f}" for c in CLASS_NAMES))


if __name__ == "__main__":
    main()
