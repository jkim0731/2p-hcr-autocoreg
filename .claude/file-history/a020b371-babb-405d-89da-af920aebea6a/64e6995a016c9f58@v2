"""Session 07f — per-cell z-bbox ratio as sz estimator.

Uses `roi_area_sxy` per-cell tight-bbox tables (already cached for the
4 spot subjects) to compute:

    sz_bbox = median(bbox_dz_HCR) / median(bbox_dz_CZ)

and a depth-binned variant `sz_bin(d)`. GT is diagnostic only, never
enters the estimator.

The estimator is biased by 2P axial-PSF broadening (CZ bbox z is
inflated by the CZ z-PSF) — we measure that bias here, and its
cross-subject stability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from benchmark_analysis import analyze_subject, fit_anisotropic_similarity
from benchmark_data_loader import landmark_pairs_um, load_subject
import roi_area_sxy

SPOT_SUBJECTS = ["788406", "790322", "767018", "782149"]

D_SKIN_UM = 100.0
D_TOP_PAD_CZ_UM = 5.0     # exclude CZ cells within 5 µm of top/bottom CZ plane
D_TOP_PAD_HCR_UM = 2.0    # exclude HCR cells within 2 µm of top/bottom HCR plane
BIN_STRIDE_UM = 50.0
MIN_PER_BIN = 20
FIG_DIR = Path("/root/capsule/code/sessions/07f_sz_from_zbbox/figures")
OUT_JSON = Path("/root/capsule/code/sessions/07f_sz_from_zbbox/results.json")


def _drop_edge_cells(df, z_col: str, lo: float, hi: float, pad: float):
    """Drop cells whose centroid is within `pad` of the z-slab edges."""
    return df[(df[z_col] >= lo + pad) & (df[z_col] <= hi - pad)].copy()


def analyze_subject_zbbox(sid: str) -> dict:
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    # --- bootstrap strict-GFP+ set (same as sxy estimator) ---
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gfp_thr", THIS / "07b_gfp_intersection_threshold.py"
    )
    _gfp_thr = importlib.util.module_from_spec(spec)
    sys.modules["_gfp_thr"] = _gfp_thr
    spec.loader.exec_module(_gfp_thr)  # type: ignore
    gi = _gfp_thr.analyze_subject(sid)
    strict_cutoff = float(gi.cutoff_linear)
    strict_df = _gfp_thr.strict_gfp_df(sid, strict_cutoff)
    strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)
    fov_ids = roi_area_sxy._prefilter_center_fov(sid, s, strict_hcr_ids)

    # --- per-cell bbox tables ---
    cz_df = roi_area_sxy.cz_cell_tight_bboxes(sid, s, cz_surf)
    hcr_df = roi_area_sxy.hcr_cell_tight_bboxes(sid, s, hcr_surf, fov_ids)

    # --- z-slab edge exclusion ---
    cz_z_lo, cz_z_hi = float(cz_df["z_um"].min()), float(cz_df["z_um"].max())
    hcr_z_lo, hcr_z_hi = float(hcr_df["z_um"].min()), float(hcr_df["z_um"].max())
    cz_df_noedge = _drop_edge_cells(cz_df, "z_um", cz_z_lo, cz_z_hi, D_TOP_PAD_CZ_UM)
    hcr_df_noedge = _drop_edge_cells(hcr_df, "z_um", hcr_z_lo, hcr_z_hi, D_TOP_PAD_HCR_UM)

    # --- cortex-band filter ---
    cz_span = float(np.nanpercentile(cz_df_noedge["depth_um"], 99))
    hcr_span = float(np.nanpercentile(hcr_df_noedge["depth_um"], 99))
    cz_f = cz_df_noedge[
        (cz_df_noedge.depth_um >= D_SKIN_UM) & (cz_df_noedge.depth_um <= cz_span)
    ].copy()
    hcr_f = hcr_df_noedge[
        (hcr_df_noedge.depth_um >= D_SKIN_UM) & (hcr_df_noedge.depth_um <= hcr_span)
    ].copy()

    # --- estimator ---
    med_cz = float(cz_f["bbox_dz_um"].median())
    med_hcr = float(hcr_f["bbox_dz_um"].median())
    mean_cz = float(cz_f["bbox_dz_um"].mean())
    mean_hcr = float(hcr_f["bbox_dz_um"].mean())
    sz_bbox_med = med_hcr / med_cz
    sz_bbox_mean = mean_hcr / mean_cz

    # --- depth-binned ---
    d_top = min(cz_span, hcr_span)
    edges = np.arange(D_SKIN_UM, d_top, BIN_STRIDE_UM)
    sz_bin = {}
    for lo in edges:
        hi = lo + BIN_STRIDE_UM
        cz_band = cz_f[(cz_f.depth_um >= lo) & (cz_f.depth_um < hi)]
        hcr_band = hcr_f[(hcr_f.depth_um >= lo) & (hcr_f.depth_um < hi)]
        if len(cz_band) >= MIN_PER_BIN and len(hcr_band) >= MIN_PER_BIN:
            sz_bin[float(lo)] = {
                "sz_bin_med": float(hcr_band.bbox_dz_um.median() / cz_band.bbox_dz_um.median()),
                "cz_dz_med": float(cz_band.bbox_dz_um.median()),
                "hcr_dz_med": float(hcr_band.bbox_dz_um.median()),
                "n_cz": int(len(cz_band)),
                "n_hcr": int(len(hcr_band)),
            }

    # --- aspect ratio (bbox_dz / geomean(bbox_dx, bbox_dy)) ---
    cz_ar = cz_f["bbox_dz_um"] / np.sqrt(cz_f["bbox_dx_um"] * cz_f["bbox_dy_um"])
    hcr_ar = hcr_f["bbox_dz_um"] / np.sqrt(hcr_f["bbox_dx_um"] * hcr_f["bbox_dy_um"])

    # --- GT ---
    cz_lm, hc_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hc_lm)
    sz_gt = float(fit.scales[2])
    sxy_gt = float(np.sqrt(fit.scales[0] * fit.scales[1]))

    return {
        "sid": sid,
        "sz_bbox_med": sz_bbox_med,
        "sz_bbox_mean": sz_bbox_mean,
        "sz_gt": sz_gt,
        "sxy_gt": sxy_gt,
        "err_pct_med": 100.0 * (sz_bbox_med - sz_gt) / sz_gt,
        "err_pct_mean": 100.0 * (sz_bbox_mean - sz_gt) / sz_gt,
        "bbox_stats": {
            "cz_dz_median_um": med_cz,
            "hcr_dz_median_um": med_hcr,
            "cz_dz_mean_um": mean_cz,
            "hcr_dz_mean_um": mean_hcr,
            "cz_dz_p25_um": float(cz_f["bbox_dz_um"].quantile(0.25)),
            "cz_dz_p75_um": float(cz_f["bbox_dz_um"].quantile(0.75)),
            "hcr_dz_p25_um": float(hcr_f["bbox_dz_um"].quantile(0.25)),
            "hcr_dz_p75_um": float(hcr_f["bbox_dz_um"].quantile(0.75)),
            "cz_dz_n_unique": int(cz_f["bbox_dz_um"].nunique()),
        },
        "aspect_ratio": {
            "cz_median": float(cz_ar.median()),
            "cz_p25": float(cz_ar.quantile(0.25)),
            "cz_p75": float(cz_ar.quantile(0.75)),
            "hcr_median": float(hcr_ar.median()),
            "hcr_p25": float(hcr_ar.quantile(0.25)),
            "hcr_p75": float(hcr_ar.quantile(0.75)),
        },
        "n_cz": int(len(cz_f)),
        "n_hcr": int(len(hcr_f)),
        "cz_span_um": cz_span,
        "hcr_span_um": hcr_span,
        "cz_z_range_um": [cz_z_lo, cz_z_hi],
        "hcr_z_range_um": [hcr_z_lo, hcr_z_hi],
        "sz_bin": sz_bin,
        "cz_bbox_dz_um_all": cz_f["bbox_dz_um"].tolist(),
        "hcr_bbox_dz_um_all": hcr_f["bbox_dz_um"].tolist(),
        "cz_depth_um_all": cz_f["depth_um"].tolist(),
        "hcr_depth_um_all": hcr_f["depth_um"].tolist(),
    }


# -----------------------------------------------------------
# plotting
# -----------------------------------------------------------
def plot_hist_per_subject(rec: dict, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cz = np.array(rec["cz_bbox_dz_um_all"])
    hcr = np.array(rec["hcr_bbox_dz_um_all"])
    bins_log = np.logspace(np.log10(2), np.log10(max(hcr.max(), cz.max()) * 1.1), 40)
    ax.hist(cz, bins=bins_log, alpha=0.5, label=f"CZ (n={len(cz)})", color="#1f77b4", density=True)
    ax.hist(hcr, bins=bins_log, alpha=0.5, label=f"HCR (n={len(hcr)})", color="#d62728", density=True)
    ax.axvline(rec["bbox_stats"]["cz_dz_median_um"], color="#1f77b4", ls="--",
               label=f"CZ med {rec['bbox_stats']['cz_dz_median_um']:.1f}")
    ax.axvline(rec["bbox_stats"]["hcr_dz_median_um"], color="#d62728", ls="--",
               label=f"HCR med {rec['bbox_stats']['hcr_dz_median_um']:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("per-cell bbox_dz (µm)")
    ax.set_ylabel("density")
    ax.set_title(f"{rec['sid']}  sz_bbox={rec['sz_bbox_med']:.2f} "
                 f"GT={rec['sz_gt']:.2f}  err={rec['err_pct_med']:+.1f}%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_depth_profile(rec: dict, outpath: Path):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # top: per-modality median bbox_dz vs depth
    sz_bin = rec["sz_bin"]
    if not sz_bin:
        fig.text(0.5, 0.5, "no depth bins with ≥ 20 cells/side", ha="center")
        fig.savefig(outpath, dpi=140)
        plt.close(fig)
        return
    bins = sorted(float(b) for b in sz_bin.keys())
    cz_vals = [sz_bin[str(b) if str(b) in sz_bin else b]["cz_dz_med"] for b in bins]
    hcr_vals = [sz_bin[str(b) if str(b) in sz_bin else b]["hcr_dz_med"] for b in bins]
    sz_vals = [sz_bin[str(b) if str(b) in sz_bin else b]["sz_bin_med"] for b in bins]

    axes[0].plot(bins, cz_vals, "o-", color="#1f77b4", label="CZ med")
    axes[0].plot(bins, hcr_vals, "o-", color="#d62728", label="HCR med")
    axes[0].set_ylabel("median bbox_dz (µm)")
    axes[0].legend()

    axes[1].plot(bins, sz_vals, "o-", color="black", label=r"$sz_{bbox}(d)$")
    axes[1].axhline(rec["sz_gt"], ls="--", color="gray", label=f"sz_GT={rec['sz_gt']:.2f}")
    axes[1].axhline(rec["sz_bbox_med"], ls=":", color="orange",
                    label=f"sz_bbox={rec['sz_bbox_med']:.2f}")
    axes[1].set_xlabel("pia depth (µm)")
    axes[1].set_ylabel(r"$sz_{bbox}(d) = $ HCR/CZ")
    axes[1].legend()
    axes[1].set_title(f"{rec['sid']} depth profile")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_subject_summary(records: list, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sids = [r["sid"] for r in records]
    errs = [r["err_pct_med"] for r in records]
    sz_est = [r["sz_bbox_med"] for r in records]
    sz_gt = [r["sz_gt"] for r in records]

    ax = axes[0]
    bars = ax.bar(sids, errs, color=["#2ca02c" if abs(e) <= 10 else "#d62728" for e in errs])
    ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-10, 10, color="gray", alpha=0.15, label="±10%")
    ax.axhspan(-5, 5, color="green", alpha=0.10, label="±5%")
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width()/2, e + (1 if e >= 0 else -3),
                f"{e:+.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("err_pct (sz_bbox vs sz_GT)")
    ax.set_title("per-subject err; mean±std = "
                 f"{np.mean(errs):+.1f}% ± {np.std(errs):.1f}%")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    ax.plot([1.5, 4.0], [1.5, 4.0], "k--", lw=0.5, label="y=x")
    ax.scatter(sz_gt, sz_est, s=60, color="#d62728")
    for sid, g, e in zip(sids, sz_gt, sz_est):
        ax.annotate(sid, (g, e), fontsize=8, xytext=(5, 0), textcoords="offset points")
    ax.set_xlabel("sz_GT")
    ax.set_ylabel("sz_bbox (median)")
    ax.set_title("sz_bbox vs GT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_aspect_ratio(rec: dict, outpath: Path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    cz_ar = np.array(rec["cz_bbox_dz_um_all"]) / np.sqrt(
        np.array(rec.get("cz_bbox_dx_um_all", []) or 1.0)
    )
    # Simpler — just show the summaries
    ar = rec["aspect_ratio"]
    ax.errorbar([0.8], [ar["cz_median"]], yerr=[[ar["cz_median"] - ar["cz_p25"]],
                                                [ar["cz_p75"] - ar["cz_median"]]],
                fmt="o", color="#1f77b4", capsize=5, label=f"CZ  {ar['cz_median']:.2f}")
    ax.errorbar([1.2], [ar["hcr_median"]], yerr=[[ar["hcr_median"] - ar["hcr_p25"]],
                                                 [ar["hcr_p75"] - ar["hcr_median"]]],
                fmt="o", color="#d62728", capsize=5, label=f"HCR  {ar['hcr_median']:.2f}")
    ax.axhline(1.0, ls="--", color="gray", lw=0.5, label="isotropic = 1")
    ax.set_xlim(0.3, 1.7)
    ax.set_xticks([0.8, 1.2])
    ax.set_xticklabels(["CZ", "HCR"])
    ax.set_ylabel("bbox_dz / sqrt(bbox_dx·bbox_dy)")
    ax.set_title(f"{rec['sid']} cell aspect ratio (p25/med/p75)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in SPOT_SUBJECTS:
        print(f"\n== {sid} ==")
        try:
            rec = analyze_subject_zbbox(sid)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        records.append(rec)
        print(f"  sz_bbox={rec['sz_bbox_med']:.3f}  sz_GT={rec['sz_gt']:.3f}  "
              f"err={rec['err_pct_med']:+.1f}%  n_cz={rec['n_cz']} n_hcr={rec['n_hcr']}")
        plot_hist_per_subject(rec, FIG_DIR / f"zbbox_hist_{sid}.png")
        plot_depth_profile(rec, FIG_DIR / f"sz_depth_profile_{sid}.png")
        plot_aspect_ratio(rec, FIG_DIR / f"aspect_ratio_{sid}.png")

    if records:
        plot_subject_summary(records, FIG_DIR / "subject_summary.png")

    # trim heavy list columns for on-disk JSON
    json_records = []
    for r in records:
        rj = {k: v for k, v in r.items() if not k.endswith("_all")}
        json_records.append(rj)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(json_records, indent=2))

    if records:
        errs = [r["err_pct_med"] for r in records]
        print(f"\nSummary: mean err {np.mean(errs):+.2f}%  std {np.std(errs):.2f}%  "
              f"spread [{min(errs):+.1f}, {max(errs):+.1f}]")


if __name__ == "__main__":
    main()
