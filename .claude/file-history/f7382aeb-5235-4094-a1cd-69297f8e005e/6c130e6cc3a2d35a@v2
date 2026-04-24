"""Subgoal 04/R1/02 — GFP+ threshold for intensity-only subjects (755252, 767022).

Mirrors subgoal 01 but operates on per-cell 488 ``mean`` intensity (and
``mean - background``) rather than spot counts/density. The same
distribution-driven strategy families (Yen/Otsu/Isodata/Triangle on
``log(x)``, plus the v2.1–2.4 additions PeakGauss3, FiltGMM-n, MirrorGauss,
all at 0.1st / 1st percentile lower-tail of the fitted peak Gaussian) are
evaluated and scored against the canonical ``coreg_table.csv`` HCR IDs.

Data source: ``data/cell_data_mean_{subj}_R1.csv``, columns
``[channel, cell_id, sum, count, mean, background]``. Only channel 488 is
used. ``mean - background`` drops to non-positive for 13–19 % of cells, so
the ``_bg`` feature path is restricted to cells with ``mean > background``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse strategy implementations from subgoal 01 by loading it as a module.
import importlib.util as _iu

_SG1_PATH = _THIS_DIR / "04_r1_subgoal_01_gfp_threshold.py"
_spec = _iu.spec_from_file_location("_sg1", _SG1_PATH)
_sg1 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_sg1)

from benchmark_data_loader import DATA_DIR, load_subject  # noqa: E402


INTENSITY_SUBJECTS = ["755252", "767022"]
OUT_DIR = _THIS_DIR.parent / "sessions" / "04_R1_coarse_align"
FIG_DIR = OUT_DIR / "subgoal_02_figures"
RESULT_JSON = OUT_DIR / "subgoal_02_intensity_threshold_results.json"

COVERAGE_TARGET = 0.95


# =====================================================================
# Data loading
# =====================================================================
def load_raw_intensity_table(subject_id: str) -> pd.DataFrame:
    """Return per-cell channel-488 ``[hcr_id, mean, background]`` table."""
    path = DATA_DIR / f"cell_data_mean_{subject_id}_R1.csv"
    df = pd.read_csv(path)
    df = df[df["channel"] == 488].copy()
    rename = {"cell_id": "hcr_id"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["hcr_id"] = df["hcr_id"].astype(int)
    df["mean_minus_bg"] = df["mean"].astype(float) - df["background"].astype(float)
    return df[["hcr_id", "mean", "background", "mean_minus_bg"]].reset_index(drop=True)


# =====================================================================
# Score a threshold
# =====================================================================
def score_threshold(
    raw: pd.DataFrame,
    n_hcr_total: int,
    feature: str,
    threshold: float,
    coreg_hcr_ids: list[int],
) -> dict:
    vals = raw[feature].values.astype(float)
    keep = vals >= threshold
    n_gfp = int(keep.sum())
    gfp_frac = n_gfp / n_hcr_total if n_hcr_total else float("nan")
    out = {
        "feature": feature,
        "threshold": float(threshold),
        "n_gfp": n_gfp,
        "n_hcr_total": n_hcr_total,
        "gfp_frac": gfp_frac,
    }
    if coreg_hcr_ids:
        gfp_ids = set(raw.loc[keep, "hcr_id"].astype(int).tolist())
        in_gfp = sum(1 for x in coreg_hcr_ids if x in gfp_ids)
        out["coreg_coverage"] = in_gfp / len(coreg_hcr_ids)
        out["n_coreg"] = len(coreg_hcr_ids)
    return out


# =====================================================================
# Per-subject runner
# =====================================================================
def analyze_one(subject_id: str) -> dict:
    s = load_subject(subject_id)
    raw = load_raw_intensity_table(subject_id)
    n_hcr_total = len(s.hcr_centroids)
    coreg_hcr_ids = (
        s.coreg_table["hcr_id"].astype(int).tolist() if len(s.coreg_table) else []
    )

    mean_vals = raw["mean"].values.astype(float)
    mean_pos = mean_vals[mean_vals > 0]

    bg_vals = raw["mean_minus_bg"].values.astype(float)
    bg_pos = bg_vals[bg_vals > 0]  # 13–19 % of cells have mean<=bg

    strategies: dict[str, dict] = {}

    def add(name, feature, threshold, source):
        strategies[name] = {
            "feature": feature,
            "threshold": float(threshold),
            "source": source,
        }

    # --- Reference (no threshold = current behaviour) ---
    add("baseline_no_threshold", "mean", 0.0, "legacy — every HCR cell is GFP+")

    # --- Classic one-shot methods on log(mean) ---
    from skimage.filters import (
        threshold_yen,
        threshold_otsu,
        threshold_isodata,
        threshold_triangle,
    )

    def _log_filter(x: np.ndarray, method: str) -> float:
        if len(x) == 0:
            return float("inf")
        lx = np.log(x)
        funcs = {
            "yen": threshold_yen,
            "otsu": threshold_otsu,
            "isodata": threshold_isodata,
            "triangle": threshold_triangle,
        }
        t = float(funcs[method](lx, nbins=256))
        return float(np.exp(t))

    add("Yen_log_mean", "mean", _log_filter(mean_pos, "yen"), "Yen on log(mean>0)")
    add("Otsu_log_mean", "mean", _log_filter(mean_pos, "otsu"), "Otsu on log(mean>0)")
    add("Isodata_log_mean", "mean", _log_filter(mean_pos, "isodata"), "ISODATA on log(mean>0)")
    add("Triangle_log_mean", "mean", _log_filter(mean_pos, "triangle"), "Triangle on log(mean>0)")

    # --- Peak-family at 1 % and 0.1 % on log(mean) ---
    for p, tag in [(1.0, "p1"), (0.1, "p0.1")]:
        add(f"PeakGauss3_mean_{tag}",
            "mean",
            _sg1.strategy_peak_gauss_lower_pct(mean_pos, percentile=p, n_components=3),
            f"GMM-3 on log(mean>0), rightmost component, {p}% lower tail")
    add("MirrorGauss_mean_p0.1",
        "mean",
        _sg1.strategy_mirror_gauss(mean_pos, percentile=0.1),
        "Rightmost peak of log(mean) + right reflection, 0.1% lower tail")

    # --- Peak family on log(mean - background), restricted to mean>bg ---
    if len(bg_pos) >= 100:
        add("Yen_log_mean_bg", "mean_minus_bg", _log_filter(bg_pos, "yen"),
            "Yen on log(mean-bg) for cells with mean>bg")
        add("Otsu_log_mean_bg", "mean_minus_bg", _log_filter(bg_pos, "otsu"),
            "Otsu on log(mean-bg)")
        add("Isodata_log_mean_bg", "mean_minus_bg", _log_filter(bg_pos, "isodata"),
            "ISODATA on log(mean-bg)")
        add("Triangle_log_mean_bg", "mean_minus_bg", _log_filter(bg_pos, "triangle"),
            "Triangle on log(mean-bg)")
        for p, tag in [(1.0, "p1"), (0.1, "p0.1")]:
            add(f"PeakGauss3_mean_bg_{tag}",
                "mean_minus_bg",
                _sg1.strategy_peak_gauss_lower_pct(bg_pos, percentile=p, n_components=3),
                f"GMM-3 on log(mean-bg), rightmost component, {p}% lower tail")
        add("MirrorGauss_mean_bg_p0.1",
            "mean_minus_bg",
            _sg1.strategy_mirror_gauss(bg_pos, percentile=0.1),
            "Rightmost peak of log(mean-bg) + right reflection, 0.1%")

    # --- FiltGMM family on mean>=min_filt (use the raw-mean bulk p25 as filter) ---
    # Rationale: the raw histogram has a bulk autofluorescence mode in the
    # low end; filter out that floor before the GMM sees it. Use the p25
    # of mean as the filter (scales with the distribution).
    for mf_label, mf in [("p25", float(np.quantile(mean_pos, 0.25)))]:
        for nc in [3, 4, 5]:
            add(
                f"FiltGMM{nc}_mean_{mf_label}_p0.1",
                "mean",
                _sg1.strategy_peak_gauss_filtered_counts(
                    mean_pos, n_components=nc, percentile=0.1,
                    min_count=int(np.ceil(mf)), ceil_int=False,
                ),
                f"GMM-{nc} on log(mean>={mf:.0f}), rightmost component, 0.1%",
            )

    # --- Evaluate ---
    out_strategies = {}
    for name, cfg in strategies.items():
        out_strategies[name] = {
            **cfg,
            **score_threshold(raw, n_hcr_total, cfg["feature"], cfg["threshold"], coreg_hcr_ids),
        }

    return {
        "subject": subject_id,
        "n_hcr_total": n_hcr_total,
        "n_mean_pos": int(len(mean_pos)),
        "n_bg_pos": int(len(bg_pos)),
        "n_coreg_rows": len(coreg_hcr_ids),
        "mean_percentiles": {
            q: float(np.quantile(mean_pos, p))
            for q, p in [("p01", 0.01), ("p25", 0.25), ("p50", 0.5),
                         ("p75", 0.75), ("p95", 0.95), ("p99", 0.99)]
        },
        "bg_percentiles": {
            q: float(np.quantile(bg_pos, p))
            for q, p in [("p01", 0.01), ("p25", 0.25), ("p50", 0.5),
                         ("p75", 0.75), ("p95", 0.95), ("p99", 0.99)]
        } if len(bg_pos) else {},
        "strategies": out_strategies,
    }


# =====================================================================
# Plotting
# =====================================================================
def _save_histograms(per_subject: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for feature, log_label, fname, overlay_names in [
        ("mean", "log10(mean)", "log_mean_histograms.png",
         [("baseline_no_threshold", "C0", "--", "baseline"),
          ("PeakGauss3_mean_p0.1", "C8", "-", "PeakGauss3 p0.1"),
          ("MirrorGauss_mean_p0.1", "C2", "-", "MirrorGauss p0.1")]),
        ("mean_minus_bg", "log10(mean - bg)", "log_mean_bg_histograms.png",
         [("PeakGauss3_mean_bg_p0.1", "C8", "-", "PeakGauss3 p0.1"),
          ("MirrorGauss_mean_bg_p0.1", "C2", "-", "MirrorGauss p0.1")]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=False)
        for ax, sid in zip(axes, INTENSITY_SUBJECTS):
            raw = load_raw_intensity_table(sid)
            vals = raw[feature].values.astype(float)
            vals_pos = vals[vals > 0]
            x = np.log10(vals_pos)
            bins = np.linspace(x.min(), x.max() + 1e-6, 80)
            ax.hist(x, bins=bins, color="0.6", edgecolor="k", linewidth=0.3,
                    label=f"all HCR (n={len(vals_pos)})")

            s = load_subject(sid)
            cg_ids = set(s.coreg_table["hcr_id"].astype(int).tolist())
            cg_vals = raw.loc[raw["hcr_id"].isin(cg_ids), feature].values.astype(float)
            cg_vals = cg_vals[cg_vals > 0]
            ax2 = ax.twinx()
            if len(cg_vals):
                ax2.hist(np.log10(cg_vals), bins=bins, color="C0", alpha=0.7,
                         label=f"coreg cells ({len(cg_vals)})")
            ax2.set_ylabel("coreg cells (right)", color="C0")
            ax2.tick_params(axis="y", labelcolor="C0")

            # Peak Gauss PDF overlay
            peak_params = _sg1._fit_peak_gauss_component(vals_pos, n_components=3)
            _sg1._overlay_peak_gaussian(ax, bins, peak_params, label_prefix="PeakGauss3: ")
            mirror_params = _sg1._fit_mirror_gauss_params(vals_pos)
            _sg1._overlay_mirror_gaussian(ax, bins, mirror_params, label_prefix="MirrorGauss: ")

            strat = per_subject[sid]["strategies"]
            for name, col, ls, tag in overlay_names:
                if name not in strat:
                    continue
                t = strat[name]["threshold"]
                if t and np.isfinite(t) and t > 0:
                    ax.axvline(np.log10(t), color=col, ls=ls, lw=1.4,
                               label=f"{tag}={t:.1f}")
            ax.set_title(f"{sid}")
            ax.set_xlabel(log_label)
            ax.set_ylabel("all HCR (left)", color="0.3")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=130)
        plt.close(fig)


# =====================================================================
# CLI
# =====================================================================
def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_subject = {}
    for sid in INTENSITY_SUBJECTS:
        print(f"\n=== {sid} ===", flush=True)
        per_subject[sid] = analyze_one(sid)
        r = per_subject[sid]
        print(f"n_hcr_total={r['n_hcr_total']}, n_mean_pos={r['n_mean_pos']}, "
              f"n_bg_pos={r['n_bg_pos']}, n_coreg={r['n_coreg_rows']}")
        for name, d in r["strategies"].items():
            cov = d.get("coreg_coverage", float("nan"))
            print(f"  {name:38s} feat={d['feature']:14s} t={d['threshold']:9.2f}  "
                  f"gfp_frac={d['gfp_frac']:.3f}  coreg_cov={cov:.3f}  "
                  f"n_gfp={d['n_gfp']}")

    # Cross-subject summary
    strategy_names = list(next(iter(per_subject.values()))["strategies"].keys())
    summary = {}
    for name in strategy_names:
        covs = [per_subject[sid]["strategies"][name].get("coreg_coverage")
                for sid in INTENSITY_SUBJECTS]
        fracs = [per_subject[sid]["strategies"][name]["gfp_frac"]
                 for sid in INTENSITY_SUBJECTS]
        ts = [per_subject[sid]["strategies"][name]["threshold"]
              for sid in INTENSITY_SUBJECTS]
        covs_valid = [c for c in covs if c is not None]
        summary[name] = {
            "threshold_per_subject": dict(zip(INTENSITY_SUBJECTS, ts)),
            "gfp_frac_per_subject": dict(zip(INTENSITY_SUBJECTS, fracs)),
            "coreg_cov_per_subject": dict(zip(INTENSITY_SUBJECTS, covs)),
            "gfp_frac_max": float(np.max(fracs)),
            "gfp_frac_mean": float(np.mean(fracs)),
            "coreg_cov_min": float(np.nanmin(covs_valid)) if covs_valid else None,
            "coreg_cov_mean": float(np.nanmean(covs_valid)) if covs_valid else None,
        }

    # Winners — same rule as subgoal 01: cov_min >= 0.95, rank by lowest frac_max
    winners = []
    for name, row in summary.items():
        if name.startswith("baseline_"):
            continue
        if row["coreg_cov_min"] is None or row["coreg_cov_min"] < COVERAGE_TARGET:
            continue
        winners.append((name, row["gfp_frac_max"], row["gfp_frac_mean"], row["coreg_cov_min"]))
    winners.sort(key=lambda x: (x[1], x[2], -x[3]))

    out = {
        "coverage_target": COVERAGE_TARGET,
        "per_subject": per_subject,
        "cross_subject_summary": summary,
        "winners_ranked": [w[0] for w in winners],
        "winners_detail": [
            {"name": w[0], "gfp_frac_max": w[1], "gfp_frac_mean": w[2], "coreg_cov_min": w[3]}
            for w in winners
        ],
    }
    with open(RESULT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {RESULT_JSON}")

    try:
        _save_histograms(per_subject)
        print(f"Wrote figures to {FIG_DIR}")
    except Exception as exc:  # noqa: BLE001
        print(f"plot skipped: {exc!r}")

    return out


if __name__ == "__main__":
    run()
