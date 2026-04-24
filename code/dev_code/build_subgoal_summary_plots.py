"""Generate summary comparison plots for subgoal 01 & 02 notebooks.

Produces the following into the session figure dirs:

  subgoal_01_figures/
    gfp_frac_comparison.png      # per-subject GFP+ frac pre / v2.1 / v2.2 (spot)
    final_protocol_fit.png       # log(density) hist + PeakGauss3 fit + p0.1 thr
                                 # (the chosen v2.2 spot protocol)

  subgoal_02_figures/
    gfp_frac_comparison.png      # intensity subjects pre / v2.2 + cov overlay
    strategy_frontier.png        # frac_max vs cov_min over 17 strategies
    final_protocol_fit.png       # log(mean - bg) hist + PeakGauss3 fit + p1 thr
                                 # (the chosen v2.2 intensity protocol)

All numbers match the v2.2 addendum table in
``sessions/04_R1_coarse_align/log.md``.
"""
from __future__ import annotations

import importlib.util as _iu
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/capsule/code")
DEV = ROOT / "dev_code"
SESS = ROOT / "sessions" / "04_R1_coarse_align"
FIG_01 = SESS / "subgoal_01_figures"
FIG_02 = SESS / "subgoal_02_figures"
SUBGOAL_01_JSON = SESS / "subgoal_01_gfp_threshold_results.json"
SUBGOAL_02_JSON = SESS / "subgoal_02_intensity_threshold_results.json"

if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

# Subgoal-01 driver module loaded by path because the filename starts with a digit.
_SG1_PATH = DEV / "04_r1_subgoal_01_gfp_threshold.py"
_spec = _iu.spec_from_file_location("_sg1", _SG1_PATH)
_sg1 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_sg1)

_SG2_PATH = DEV / "04_r1_subgoal_02_intensity_threshold.py"
_spec2 = _iu.spec_from_file_location("_sg2", _SG2_PATH)
_sg2 = _iu.module_from_spec(_spec2)
_spec2.loader.exec_module(_sg2)

from benchmark_data_loader import load_subject  # noqa: E402


SPOT_SUBJECTS = ["788406", "790322", "767018", "782149"]
INTENSITY_SUBJECTS = ["755252", "767022"]

# Per-subject GFP+ counts across threshold versions (canonical record:
# log.md §"Benchmark results (final R1)" / §v2.1 addendum / §v2.2 addendum).
N_GFP = {
    "788406": {"pre": 69680, "v2.1": 20464, "v2.2": 17427, "n_hcr": 127275},
    "790322": {"pre": 72532, "v2.1": 14907, "v2.2": 10131, "n_hcr": 106379},
    "767018": {"pre": 58959, "v2.1": 10322, "v2.2":  9161, "n_hcr": 108506},
    "782149": {"pre": 25251, "v2.1":  4556, "v2.2":  3831, "n_hcr": 39291},
    "755252": {"pre": 77785, "v2.1": 77785, "v2.2": 30804, "n_hcr": 84233},
    "767022": {"pre": 72213, "v2.1": 72213, "v2.2": 14239, "n_hcr": 76336},
}


def _bar_group(ax, subjects, series, colors, labels, ylabel, title,
                   hline=None, hline_label=None):
    n_series = len(series)
    x = np.arange(len(subjects))
    width = 0.8 / n_series
    for i, (vals, col, lab) in enumerate(zip(series, colors, labels)):
        offs = (i - (n_series - 1) / 2) * width
        ax.bar(x + offs, vals, width=width, color=col, edgecolor="k",
               linewidth=0.3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, color="red", ls="--", lw=1.0,
                   label=hline_label or f"target = {hline}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_subgoal_01_gfp_frac() -> None:
    FIG_01.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    subjects = SPOT_SUBJECTS
    pre = [100 * N_GFP[s]["pre"] / N_GFP[s]["n_hcr"] for s in subjects]
    v21 = [100 * N_GFP[s]["v2.1"] / N_GFP[s]["n_hcr"] for s in subjects]
    v22 = [100 * N_GFP[s]["v2.2"] / N_GFP[s]["n_hcr"] for s in subjects]
    _bar_group(
        ax, subjects,
        series=[pre, v21, v22],
        colors=["0.7", "C0", "C2"],
        labels=["pre (counts≥5)", "v2.1 (yen_log_joint)",
                "v2.2 (peakgauss3_density_p0.1)"],
        ylabel="GFP+ % of HCR cells",
        title="Subgoal 01 — GFP+ fraction per spot subject across threshold versions",
    )
    fig.tight_layout()
    out = FIG_01 / "gfp_frac_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_subgoal_02_gfp_frac() -> None:
    FIG_02.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    subjects = INTENSITY_SUBJECTS
    pre = [100 * N_GFP[s]["pre"] / N_GFP[s]["n_hcr"] for s in subjects]
    v22 = [100 * N_GFP[s]["v2.2"] / N_GFP[s]["n_hcr"] for s in subjects]
    _bar_group(
        ax, subjects,
        series=[pre, v22],
        colors=["0.7", "C2"],
        labels=["pre (no threshold, 100 %)",
                "v2.2 (peakgauss3_mean_bg_p1)"],
        ylabel="GFP+ % of HCR cells",
        title="Subgoal 02 — GFP+ fraction per intensity subject",
    )
    cov_by_sid = {"755252": 0.933, "767022": 0.946}
    x = np.arange(len(subjects))
    width = 0.4
    for i, s in enumerate(subjects):
        ax.text(x[i] + width / 2, v22[i] + 1.0,
                f"cov={cov_by_sid[s]:.3f}",
                ha="center", va="bottom", fontsize=8, color="C2")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    out = FIG_02 / "gfp_frac_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_subgoal_02_frontier() -> None:
    """Scatter of strategies on (coreg_cov_min, gfp_frac_max); winner starred."""
    if not SUBGOAL_02_JSON.exists():
        raise FileNotFoundError(SUBGOAL_02_JSON)
    with open(SUBGOAL_02_JSON) as f:
        res = json.load(f)
    summary = res["cross_subject_summary"]

    winner = "PeakGauss3_mean_bg_p1"  # v2.2 chosen, not the ranking-rule winner
    rule_winner = "PeakGauss3_mean_bg_p0.1"  # subgoal-rule winner, for reference

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for name, row in summary.items():
        x = row["coreg_cov_min"]
        y = row["gfp_frac_max"]
        if x is None:
            continue
        is_winner = name == winner
        is_rule = name == rule_winner
        col = "C3" if is_winner else ("C1" if is_rule else "0.5")
        size = 110 if is_winner or is_rule else 40
        zorder = 5 if is_winner or is_rule else 2
        ax.scatter(x, y, s=size, color=col, edgecolor="k", linewidth=0.6,
                   zorder=zorder, label=None)
        if is_winner or is_rule or name == "baseline_no_threshold":
            ax.annotate(name, (x, y), xytext=(4, -4), textcoords="offset points",
                        fontsize=7)

    ax.axvline(0.95, color="red", ls="--", lw=1.0, label="cov_min target = 0.95")
    ax.scatter([], [], s=110, color="C3", edgecolor="k", label=f"v2.2 chosen: {winner}")
    ax.scatter([], [], s=110, color="C1", edgecolor="k",
               label=f"subgoal-rule winner: {rule_winner}")
    ax.scatter([], [], s=40, color="0.5", edgecolor="k", label="other strategies")
    ax.set_xlabel("coreg_cov_min (min over 755252, 767022)")
    ax.set_ylabel("gfp_frac_max (max over 755252, 767022)")
    ax.set_title("Subgoal 02 — intensity strategies on the (coverage, stringency) frontier")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_02 / "strategy_frontier.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Final-protocol fits: hist + rightmost-GMM Gaussian + threshold line
# ---------------------------------------------------------------------------

def _panel_fit(ax, x_all_log10, x_coreg_log10, peak_params, threshold_val,
               title, xlabel, thr_label):
    """Shared histogram + fit + threshold-line panel."""
    bins = np.linspace(x_all_log10.min(), x_all_log10.max() + 1e-6, 80)
    ax.hist(x_all_log10, bins=bins, color="0.6", edgecolor="k",
            linewidth=0.3, label=f"all HCR (n={len(x_all_log10)})")
    ax2 = ax.twinx()
    if len(x_coreg_log10):
        ax2.hist(x_coreg_log10, bins=bins, color="C0", alpha=0.7,
                 label=f"coreg cells ({len(x_coreg_log10)})")
    ax2.set_ylabel("coreg cells (right)", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    _sg1._overlay_peak_gaussian(ax, bins, peak_params, label_prefix="PeakGauss3: ")
    if threshold_val and np.isfinite(threshold_val) and threshold_val > 0:
        ax.axvline(np.log10(threshold_val), color="C3", ls="-", lw=1.6,
                   label=thr_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("all HCR (left)", color="0.3")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")


def plot_subgoal_01_final_fit() -> None:
    """log(density) histogram + PeakGauss3 fit + p0.1 threshold, 4 spot subjects."""
    FIG_01.mkdir(parents=True, exist_ok=True)
    if not SUBGOAL_01_JSON.exists():
        raise FileNotFoundError(SUBGOAL_01_JSON)
    with open(SUBGOAL_01_JSON) as f:
        res = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, sid in zip(axes.flatten(), SPOT_SUBJECTS):
        raw = _sg1.load_raw_gfp_table(sid)
        if "density" not in raw.columns:
            ax.set_title(f"{sid} — no density column")
            continue
        d = raw["density"].values.astype(float)
        d_pos = d[d > 0]
        x_all = np.log10(d_pos)
        s = load_subject(sid, gfp_threshold_method="counts_min", gfp_min_spots=1)
        cg_ids = set(s.coreg_table["hcr_id"].astype(int).tolist())
        cg_d = raw.loc[raw["hcr_id"].isin(cg_ids), "density"].values.astype(float)
        cg_d = cg_d[cg_d > 0]
        peak = _sg1._fit_peak_gauss_component(d, n_components=3)
        t_d = float(
            res["per_subject"][sid]["strategies"]["PeakGauss3_density_p0.1"][
                "threshold_density"
            ]
        )
        frac = res["per_subject"][sid]["strategies"][
            "PeakGauss3_density_p0.1"
        ]["gfp_frac"]
        _panel_fit(
            ax,
            x_all_log10=x_all,
            x_coreg_log10=np.log10(cg_d) if len(cg_d) else np.array([]),
            peak_params=peak,
            threshold_val=t_d,
            title=f"{sid}  (n={len(d_pos)}, gfp_frac={frac:.3f})",
            xlabel="log10(density)",
            thr_label=f"p0.1 thr = {t_d:.4f}",
        )
    fig.suptitle(
        "Subgoal 01 — chosen protocol `peakgauss3_density_p0.1`: "
        "log(density) histogram + rightmost-GMM fit + per-subject threshold",
        y=1.02,
    )
    fig.tight_layout()
    out = FIG_01 / "final_protocol_fit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_subgoal_02_final_fit() -> None:
    """log(mean - bg) histogram + PeakGauss3 fit + p1 threshold, 2 intensity subjects."""
    FIG_02.mkdir(parents=True, exist_ok=True)
    if not SUBGOAL_02_JSON.exists():
        raise FileNotFoundError(SUBGOAL_02_JSON)
    with open(SUBGOAL_02_JSON) as f:
        res = json.load(f)
    summary = res["cross_subject_summary"]["PeakGauss3_mean_bg_p1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=False)
    for ax, sid in zip(axes, INTENSITY_SUBJECTS):
        raw = _sg2.load_raw_intensity_table(sid)
        vals = raw["mean_minus_bg"].values.astype(float)
        vals_pos = vals[vals > 0]
        x_all = np.log10(vals_pos)
        s = load_subject(sid)
        cg_ids = set(s.coreg_table["hcr_id"].astype(int).tolist())
        cg_vals = raw.loc[raw["hcr_id"].isin(cg_ids),
                          "mean_minus_bg"].values.astype(float)
        cg_vals = cg_vals[cg_vals > 0]
        peak = _sg1._fit_peak_gauss_component(vals_pos, n_components=3)
        t = float(summary["threshold_per_subject"][sid])
        frac = float(summary["gfp_frac_per_subject"][sid])
        cov = float(summary["coreg_cov_per_subject"][sid])
        _panel_fit(
            ax,
            x_all_log10=x_all,
            x_coreg_log10=np.log10(cg_vals) if len(cg_vals) else np.array([]),
            peak_params=peak,
            threshold_val=t,
            title=f"{sid}  (n={len(vals_pos)}, frac={frac:.3f}, cov={cov:.3f})",
            xlabel="log10(mean - bg)",
            thr_label=f"p1 thr = {t:.2f}",
        )
    fig.suptitle(
        "Subgoal 02 — chosen protocol `peakgauss3_mean_bg_p1`: "
        "log(mean - bg) histogram + rightmost-GMM fit + per-subject threshold",
        y=1.04,
    )
    fig.tight_layout()
    out = FIG_02 / "final_protocol_fit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_subgoal_01_gfp_frac()
    plot_subgoal_02_gfp_frac()
    plot_subgoal_02_frontier()
    plot_subgoal_01_final_fit()
    plot_subgoal_02_final_fit()


if __name__ == "__main__":
    main()
