"""Stage A of the surface-detection sub-plan:
characterize HCR ROI segmentation errors using the current image-based pia
as a reference, then report whether simple per-ROI features can identify
out-of-tissue ROIs above the pia.

Stages B (filter-then-shallowest-z plane fit) and C (ROI-prior + image-based
refinement) are appended below.  Each stage writes a CSV under
code/sessions/01_analyze_benchmark_data/ and at most a few figures.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS,
    SubjectData,
    hcr_px_to_um,
    load_subject,
)
from benchmark_analysis import (
    _robust_plane_fit,
    analyze_subject,
    depth_from_surface,
    depth_profile,
    estimate_pia_surface_from_image,
    filter_in_tissue,
    load_hcr_combined,
)

OUT_DIR = Path("/root/capsule/code/sessions/01_analyze_benchmark_data")
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ==============================================================================
# Per-ROI feature loading
# ==============================================================================
def load_hcr_roi_features(s: SubjectData) -> pd.DataFrame:
    """Return a DataFrame with one row per HCR ROI.

    Columns always present:
      hcr_id, x_um, y_um, z_um, local_density_30um, k5_mean_um
    Optional (where data exists):
      volume_px, bbox_z_px, bbox_y_px, bbox_x_px, spot_488_counts, spot_488_density
    """
    cen_px = s.hcr_centroids[["z_px", "y_px", "x_px"]].values
    cen_um = hcr_px_to_um(cen_px, s)              # columns z, y, x in um
    cen_xyz = cen_um[:, [2, 1, 0]]                 # (x, y, z) for geometric ops

    df = pd.DataFrame({
        "hcr_id": s.hcr_centroids["hcr_id"].astype(int).values,
        "x_um": cen_xyz[:, 0],
        "y_um": cen_xyz[:, 1],
        "z_um": cen_xyz[:, 2],
    })

    # Local density: number of neighbors within 30 um
    tree = cKDTree(cen_xyz)
    df["local_density_30um"] = tree.query_ball_point(cen_xyz, r=30.0, return_length=True)

    # Mean distance to 5 nearest neighbors (excludes self)
    d, _ = tree.query(cen_xyz, k=6)
    df["k5_mean_um"] = d[:, 1:].mean(axis=1)

    # Volume / bbox (if metrics.pickle exists)
    metrics_path = s.hcr_dir / "cell_body_segmentation" / "metrics.pickle"
    if metrics_path.exists():
        with open(metrics_path, "rb") as f:
            m = pickle.load(f)
        rows = []
        for hcr_id, info in m.items():
            bb = info.get("global_bbox")
            if bb is None or len(bb) < 6:
                bb = [np.nan] * 6
            rows.append((int(hcr_id), info.get("volume", np.nan),
                         bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]))
        m_df = pd.DataFrame(rows, columns=["hcr_id", "volume_px",
                                            "bbox_z_px", "bbox_y_px", "bbox_x_px"])
        df = df.merge(m_df, on="hcr_id", how="left")

    # GFP+ spot features (all spot-based subjects)
    if not s.hcr_gfp_df.empty and "counts" in s.hcr_gfp_df.columns:
        gfp = s.hcr_gfp_df[["hcr_id", "counts"]].rename(
            columns={"counts": "spot_488_counts"}
        )
        if "density" in s.hcr_gfp_df.columns:
            gfp["spot_488_density"] = s.hcr_gfp_df["density"].values
        df = df.merge(gfp, on="hcr_id", how="left")

    return df


# ==============================================================================
# Stage A: separability of above-pia vs in-tissue by ROI feature
# ==============================================================================
def stage_A(subjects: dict[str, SubjectData], analyses: dict[str, dict]) -> pd.DataFrame:
    """For each subject compute the ROI features table, classify above-pia vs
    in-tissue using the current image-based surface, and report per-feature
    separability (group medians, fraction identifiable by a simple cut).
    """
    all_rows = []

    fig, axes = plt.subplots(len(subjects), 3, figsize=(13, 3 * len(subjects)),
                              sharex=False)
    if len(subjects) == 1:
        axes = axes[None, :]

    for row, sid in enumerate(subjects):
        s = subjects[sid]
        r = analyses[sid]
        surf = r["hcr_surface_image"]
        if surf is None:
            print(f"{sid}: no image-based surface — skipping")
            continue

        feats = load_hcr_roi_features(s)
        depths = depth_from_surface(feats[["x_um", "y_um", "z_um"]].values, surf)
        feats["depth_um"] = depths
        feats["above_pia"] = depths < -5.0

        n_total = len(feats)
        n_above = int(feats["above_pia"].sum())
        frac_above = n_above / max(n_total, 1)

        # Per-feature separability
        feature_stats = {}
        for feat in ("local_density_30um", "k5_mean_um", "volume_px",
                     "spot_488_counts"):
            if feat not in feats.columns:
                continue
            a = feats.loc[feats["above_pia"], feat].dropna()
            b = feats.loc[~feats["above_pia"], feat].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            feature_stats[f"{feat}_above_med"] = float(a.median())
            feature_stats[f"{feat}_intis_med"] = float(b.median())
            feature_stats[f"{feat}_above_p75"] = float(np.percentile(a, 75))
            feature_stats[f"{feat}_intis_p25"] = float(np.percentile(b, 25))
            # fraction of above-pia ROIs captured by a simple 1D cut
            # (cut at 25th percentile of in-tissue — below for density-like
            # feats, above for NN-distance feats)
            if feat == "k5_mean_um":
                cut = float(np.percentile(b, 75))
                captured = float((a >= cut).mean())
                feature_stats[f"{feat}_cut"] = cut
                feature_stats[f"{feat}_frac_above_captured"] = captured
            else:
                cut = float(np.percentile(b, 25))
                captured = float((a <= cut).mean())
                feature_stats[f"{feat}_cut"] = cut
                feature_stats[f"{feat}_frac_above_captured"] = captured

        all_rows.append({
            "subject": sid,
            "n_hcr_roi": n_total,
            "n_above_pia": n_above,
            "frac_above_pia": frac_above,
            **feature_stats,
        })

        # Plot: density-30um, k5_mean, and volume (if present) histograms split by class
        for ax, feat in zip(axes[row], ("local_density_30um", "k5_mean_um",
                                         "volume_px" if "volume_px" in feats.columns
                                         else "spot_488_counts")):
            if feat not in feats.columns:
                ax.set_title(f"{sid} — {feat} (n/a)")
                ax.axis("off")
                continue
            a = feats.loc[feats["above_pia"], feat].dropna()
            b = feats.loc[~feats["above_pia"], feat].dropna()
            if len(a) < 5 or len(b) < 5:
                ax.set_title(f"{sid} — {feat} (too few above)")
                ax.axis("off")
                continue
            bins = 60
            if feat == "volume_px":
                a_plot = np.log10(a.clip(lower=1))
                b_plot = np.log10(b.clip(lower=1))
                ax.set_xlabel(f"log10 {feat}")
            elif feat == "spot_488_counts":
                a_plot = np.log10(a.clip(lower=1))
                b_plot = np.log10(b.clip(lower=1))
                ax.set_xlabel(f"log10 {feat}")
            else:
                a_plot, b_plot = a, b
                ax.set_xlabel(feat)
            rng = (min(a_plot.min(), b_plot.min()),
                   max(a_plot.max(), b_plot.max()))
            ax.hist(b_plot, bins=bins, range=rng, color="tab:blue", alpha=0.6,
                    label=f"in-tissue (n={len(b)})", density=True)
            ax.hist(a_plot, bins=bins, range=rng, color="tab:red", alpha=0.6,
                    label=f"above-pia (n={len(a)})", density=True)
            ax.set_title(f"{sid} — {feat}")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stageA_feature_histograms.png", dpi=110)
    plt.close(fig)
    print("Saved stageA_feature_histograms.png")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "stageA_roi_feature_separability.csv", index=False)
    print("Saved stageA_roi_feature_separability.csv")
    return df


# ==============================================================================
# Stage B: simple-filter → shallowest-z plane fitter
# ==============================================================================
def _fit_shallowest_z_plane(
    xs_um: np.ndarray, ys_um: np.ndarray, zs_um: np.ndarray,
    xy_tile_um: float = 120.0, q: float = 0.02,
    min_per_tile: int = 3,
):
    """Tile in (x, y); take the q-quantile z per tile; IRLS-Huber plane fit."""
    if len(xs_um) < 50:
        return None
    x_edges = np.arange(xs_um.min(), xs_um.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(ys_um.min(), ys_um.max() + xy_tile_um, xy_tile_um)
    samples = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            m = ((xs_um >= x_edges[ix]) & (xs_um < x_edges[ix + 1])
                 & (ys_um >= y_edges[iy]) & (ys_um < y_edges[iy + 1]))
            if m.sum() < min_per_tile:
                continue
            z_sel = np.quantile(zs_um[m], q) if q > 0 else zs_um[m].min()
            cx = 0.5 * (x_edges[ix] + x_edges[ix + 1])
            cy = 0.5 * (y_edges[iy] + y_edges[iy + 1])
            samples.append((cx, cy, z_sel))
    if len(samples) < 10:
        return None
    arr = np.asarray(samples)
    fit = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])
    fit["n_tiles"] = int(len(arr))
    return fit


def _apply_filters(
    df: pd.DataFrame,
    density_radius_um: float = 30.0,
    density_min_neighbors: int = 6,
    use_volume: bool = True,
    volume_mode: str = "rel_median",  # "rel_median" | "iqr" | "log_valley"
    volume_rel_median: float = 0.3,
    use_connected_component: bool = True,
    cc_grid_um: float = 30.0,
    cc_min_density: int = 3,
) -> np.ndarray:
    """Return a boolean mask selecting ROIs judged to be in-tissue.

    `volume_mode`:
      - 'rel_median': keep vol > `volume_rel_median` × median(vol) (default).
      - 'iqr'       : keep vol within 1.5 × IQR of the median.
      - 'log_valley': find the valley between two modes of log10(vol) via a
                      cubic-spline-smoothed histogram; keep vol above valley.
    """
    mask = np.ones(len(df), dtype=bool)
    mask &= df["local_density_30um"].values >= density_min_neighbors

    if use_volume and "volume_px" in df.columns and df["volume_px"].notna().any():
        vol = df["volume_px"].fillna(-1).values
        keep_vol = vol > 0
        if volume_mode == "iqr":
            q25, q75 = np.nanpercentile(df["volume_px"], [25, 75])
            iqr = max(q75 - q25, 1.0)
            vmin = q25 - 1.5 * iqr
            vmax = q75 + 1.5 * iqr
            mask &= keep_vol & (vol >= vmin) & (vol <= vmax)
        elif volume_mode == "rel_median":
            med = float(np.nanmedian(df["volume_px"]))
            vmin = volume_rel_median * med
            mask &= keep_vol & (vol >= vmin)
        elif volume_mode == "log_valley":
            logs = np.log10(np.asarray(df["volume_px"].dropna().clip(lower=1)))
            hist, edges = np.histogram(logs, bins=60)
            # simple smoothing
            ker = np.array([1, 2, 4, 2, 1], dtype=float); ker /= ker.sum()
            hist_s = np.convolve(hist, ker, mode="same")
            # search for the first local minimum after the first peak
            peak = int(np.argmax(hist_s[:len(hist_s) // 2 + 5]))
            valley = peak + int(np.argmin(hist_s[peak:peak + 15])) if peak + 15 < len(hist_s) else peak + 1
            thr_log = 0.5 * (edges[valley] + edges[valley + 1])
            mask &= keep_vol & (vol >= 10 ** thr_log)

    # Largest connected component
    if use_connected_component and mask.any():
        from scipy.ndimage import label
        pts = df.loc[mask, ["x_um", "y_um", "z_um"]].values
        x0, y0, z0 = pts.min(axis=0)
        x1, y1, z1 = pts.max(axis=0)
        nx = int((x1 - x0) // cc_grid_um) + 2
        ny = int((y1 - y0) // cc_grid_um) + 2
        nz = int((z1 - z0) // cc_grid_um) + 2
        grid = np.zeros((nx, ny, nz), dtype=np.int32)
        ix = ((pts[:, 0] - x0) // cc_grid_um).astype(int)
        iy = ((pts[:, 1] - y0) // cc_grid_um).astype(int)
        iz = ((pts[:, 2] - z0) // cc_grid_um).astype(int)
        np.add.at(grid, (ix, iy, iz), 1)
        dense = grid >= cc_min_density
        labels, nlab = label(dense)
        if nlab >= 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            best = int(np.argmax(sizes))
            in_best = labels[ix, iy, iz] == best
            new_mask = mask.copy()
            mask_idx = np.where(mask)[0]
            new_mask[mask_idx] = in_best
            mask = new_mask
    return mask


def stage_B(subjects: dict[str, SubjectData], analyses: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for sid in subjects:
        s = subjects[sid]
        r = analyses[sid]
        ref_surf = r["hcr_surface_image"]
        if ref_surf is None:
            continue
        feats = load_hcr_roi_features(s)
        cen = feats[["x_um", "y_um", "z_um"]].values

        # Baseline: image-based (already computed)
        d = depth_from_surface(cen, ref_surf)
        fa_baseline = float((d < -5).sum()) / max(len(d), 1)
        rows.append({"subject": sid, "method": "baseline (image combined)",
                     "c": ref_surf["c"], "tilt_deg": ref_surf["tilt_deg"],
                     "rough_um": ref_surf["residual_std_um"],
                     "frac_above_pia": fa_baseline,
                     "n_roi_used": len(feats)})

        configs = [
            ("ROI density only", dict(density_min_neighbors=6,
                                       use_volume=False,
                                       use_connected_component=False)),
            ("ROI density+vol-iqr", dict(density_min_neighbors=6,
                                           use_volume=True, volume_mode="iqr",
                                           use_connected_component=False)),
            ("ROI density+vol-rel30", dict(density_min_neighbors=6,
                                             use_volume=True,
                                             volume_mode="rel_median",
                                             volume_rel_median=0.3,
                                             use_connected_component=False)),
            ("ROI density+vol-rel50", dict(density_min_neighbors=6,
                                             use_volume=True,
                                             volume_mode="rel_median",
                                             volume_rel_median=0.5,
                                             use_connected_component=False)),
            ("ROI density+vol-valley+LCC", dict(density_min_neighbors=6,
                                                  use_volume=True,
                                                  volume_mode="log_valley",
                                                  use_connected_component=True)),
        ]
        quantiles = [0.0, 0.01, 0.02, 0.05]
        for name, cfg in configs:
            mask = _apply_filters(feats, **cfg)
            if mask.sum() < 100:
                continue
            xs = feats.loc[mask, "x_um"].values
            ys = feats.loc[mask, "y_um"].values
            zs = feats.loc[mask, "z_um"].values
            for q in quantiles:
                fit = _fit_shallowest_z_plane(xs, ys, zs, q=q)
                if fit is None:
                    continue
                d2 = depth_from_surface(cen, fit)
                fa = float((d2 < -5).sum()) / max(len(d), 1)
                rows.append({
                    "subject": sid,
                    "method": f"{name}, q={q:.2f}",
                    "c": fit["c"], "tilt_deg": fit["tilt_deg"],
                    "rough_um": fit["residual_std_um"],
                    "frac_above_pia": fa,
                    "n_roi_used": int(mask.sum()),
                    "n_tiles": fit.get("n_tiles"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stageB_filter_plane_methods.csv", index=False)
    print("Saved stageB_filter_plane_methods.csv")
    return df


# ==============================================================================
# Stage C: ROI coarse prior + image-based refinement
# ==============================================================================
def _image_first_crossing_zmap(
    image: np.ndarray, z_um: float, xy_um: float,
    relative_margin: float = 0.05, min_signal_abs: float = 0.1,
    min_thick_um: float = 15.0, xy_stride_um: float = 10.0,
):
    """Return (ys_um, xs_um, first_z_um, mask_valid) at XY-subsampled columns.
    Mirrors the interior of `estimate_pia_surface_from_image` but returns the
    per-column first-crossing map so we can later apply a z-window restriction.
    """
    Z, Y, X = image.shape
    stride_x = max(1, int(round(xy_stride_um / xy_um)))
    stride_y = max(1, int(round(xy_stride_um / xy_um)))
    sub = image[:, ::stride_y, ::stride_x].astype(np.float32, copy=False)
    Zs, Ys, Xs = sub.shape

    col_bg = np.percentile(sub, 10, axis=0)
    col_top = np.percentile(sub, 95, axis=0)
    col_thr = col_bg + relative_margin * (col_top - col_bg)
    col_valid = (col_top - col_bg) >= min_signal_abs

    above = sub > col_thr[None, :, :]
    above &= col_valid[None, :, :]

    k = max(1, int(round(min_thick_um / z_um)))
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]
    is_start = win == k
    has_any = is_start.any(axis=0)
    first_z_idx = np.argmax(is_start, axis=0)

    # Broadcast back to um coords
    ys_grid = np.arange(Ys) * stride_y * xy_um
    xs_grid = np.arange(Xs) * stride_x * xy_um
    return ys_grid, xs_grid, first_z_idx.astype(float) * z_um, has_any


def stage_C(subjects: dict[str, SubjectData],
            analyses: dict[str, dict],
            stage_b_df: pd.DataFrame) -> pd.DataFrame:
    """For each subject pick the best Stage-B method (lowest frac_above_pia) as
    the coarse prior; restrict the image-based first-crossing map to
    z_prior ± 100 um; fit a plane."""
    rows = []
    for sid in subjects:
        s = subjects[sid]
        r = analyses[sid]
        ref_surf = r["hcr_surface_image"]
        if ref_surf is None:
            continue

        sub_b = stage_b_df[stage_b_df["subject"] == sid]
        sub_b = sub_b[sub_b["method"] != "baseline (image combined)"]
        if sub_b.empty:
            continue
        best_row = sub_b.sort_values("frac_above_pia").iloc[0]

        # Build prior plane using best-row coefficients — reuse _fit_shallowest_z_plane
        # output convention: a, b, c.  We need them — refit quickly.
        feats = load_hcr_roi_features(s)
        cen = feats[["x_um", "y_um", "z_um"]].values

        # Reproduce the same filters + quantile to get a, b, c
        name = best_row["method"]
        q = float(name.rsplit("q=", 1)[-1])
        cfg_base = dict(density_min_neighbors=6, use_volume=False,
                        use_connected_component=False)
        if "density+volume+LCC" in name:
            cfg_base.update(use_volume=True, use_connected_component=True)
        elif "density+volume" in name:
            cfg_base.update(use_volume=True)
        mask = _apply_filters(feats, **cfg_base)
        if mask.sum() < 100:
            continue
        xs = feats.loc[mask, "x_um"].values
        ys = feats.loc[mask, "y_um"].values
        zs = feats.loc[mask, "z_um"].values
        prior = _fit_shallowest_z_plane(xs, ys, zs, q=q)
        if prior is None:
            continue

        # Load combined image and compute first-crossing map
        try:
            combined, xy_um, z_um, _ = load_hcr_combined(s, level=4)
        except FileNotFoundError:
            continue
        ys_grid, xs_grid, first_z, has_any = _image_first_crossing_zmap(
            combined, z_um, xy_um
        )

        # Mesh of prior z at each column
        Xg, Yg = np.meshgrid(xs_grid, ys_grid)
        z_prior_grid = prior["a"] * Xg + prior["b"] * Yg + prior["c"]

        # Restrict to |first_z - z_prior| <= 100 um; then refit
        window_um = 100.0
        ok = has_any & (np.abs(first_z - z_prior_grid) <= window_um)
        if ok.sum() < 50:
            continue
        yi, xi = np.where(ok)
        fit = _robust_plane_fit(
            xs_grid[xi], ys_grid[yi], first_z[yi, xi]
        )

        d = depth_from_surface(cen, fit)
        fa = float((d < -5).sum()) / max(len(cen), 1)
        rows.append({
            "subject": sid,
            "method": f"hybrid (prior={name}, window={window_um}um)",
            "c": fit["c"], "tilt_deg": fit["tilt_deg"],
            "rough_um": fit["residual_std_um"],
            "frac_above_pia": fa,
            "n_roi_used": len(cen),
            "n_columns": int(ok.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stageC_hybrid_methods.csv", index=False)
    print("Saved stageC_hybrid_methods.csv")
    return df


# ==============================================================================
# Driver
# ==============================================================================
def main():
    print("Loading subjects and baseline analyses...")
    subjects = {}
    analyses = {}
    for sid in BENCHMARK_SUBJECTS:
        print(f"  {sid}")
        subjects[sid] = load_subject(sid)
        analyses[sid] = analyze_subject(subjects[sid])

    print("\n=== Stage A: ROI feature separability ===")
    dfA = stage_A(subjects, analyses)
    print(dfA.to_string())

    print("\n=== Stage B: filter-then-shallowest-z ===")
    dfB = stage_B(subjects, analyses)
    # Print compact summary: baseline + each subject's best method
    pivot = []
    for sid in BENCHMARK_SUBJECTS:
        sub = dfB[dfB["subject"] == sid]
        if sub.empty:
            continue
        base = sub[sub["method"].str.startswith("baseline")].iloc[0]
        best = sub[sub["method"] != base["method"]].sort_values("frac_above_pia").iloc[0]
        pivot.append({
            "subject": sid,
            "baseline_frac_above": base["frac_above_pia"],
            "best_method": best["method"],
            "best_frac_above": best["frac_above_pia"],
            "delta_pp": (best["frac_above_pia"] - base["frac_above_pia"]) * 100,
        })
    pd.DataFrame(pivot).to_csv(OUT_DIR / "stageB_best_per_subject.csv", index=False)
    print(pd.DataFrame(pivot).to_string())

    print("\n=== Stage C: hybrid ROI prior + image refinement ===")
    dfC = stage_C(subjects, analyses, dfB)
    print(dfC.to_string())

    # Combined comparison
    compare_rows = []
    for sid in BENCHMARK_SUBJECTS:
        base = dfB[(dfB["subject"] == sid)
                   & (dfB["method"].str.startswith("baseline"))]
        if base.empty:
            continue
        best_b = dfB[(dfB["subject"] == sid)
                     & (~dfB["method"].str.startswith("baseline"))] \
                     .sort_values("frac_above_pia")
        c_row = dfC[dfC["subject"] == sid]
        compare_rows.append({
            "subject": sid,
            "baseline": base["frac_above_pia"].iloc[0],
            "best_stageB": best_b["frac_above_pia"].iloc[0] if not best_b.empty else np.nan,
            "stageC_hybrid": c_row["frac_above_pia"].iloc[0] if not c_row.empty else np.nan,
        })
    comp = pd.DataFrame(compare_rows)
    comp.to_csv(OUT_DIR / "surface_exploration_methods.csv", index=False)
    print("\n=== Side-by-side frac_above_pia ===")
    print(comp.to_string())

    # Visual comparison on two representative subjects
    from benchmark_analysis import load_cz_y_slab, load_hcr_y_slab
    for sid in ("788406", "782149", "767018"):
        s = subjects[sid]
        r = analyses[sid]
        ref_surf = r["hcr_surface_image"]
        # Get hybrid surface for this subject by re-running
        sub_b = dfB[(dfB["subject"] == sid)
                    & (~dfB["method"].str.startswith("baseline"))] \
                    .sort_values("frac_above_pia")
        if sub_b.empty:
            continue
        best_name = sub_b.iloc[0]["method"]
        q = float(best_name.rsplit("q=", 1)[-1])
        feats = load_hcr_roi_features(s)
        cfg = dict(density_min_neighbors=6, use_volume=False,
                   use_connected_component=False)
        if "density+vol" in best_name:
            cfg["use_volume"] = True
        if "LCC" in best_name:
            cfg["use_connected_component"] = True
        mask = _apply_filters(feats, **cfg)
        prior = _fit_shallowest_z_plane(
            feats.loc[mask, "x_um"].values,
            feats.loc[mask, "y_um"].values,
            feats.loc[mask, "z_um"].values, q=q,
        )

        try:
            combined, xy_um, z_um, _ = load_hcr_combined(s, level=4)
        except FileNotFoundError:
            continue
        ys_grid, xs_grid, first_z, has_any = _image_first_crossing_zmap(
            combined, z_um, xy_um
        )
        Xg, Yg = np.meshgrid(xs_grid, ys_grid)
        z_prior_grid = prior["a"] * Xg + prior["b"] * Yg + prior["c"]
        ok = has_any & (np.abs(first_z - z_prior_grid) <= 100.0)
        yi, xi = np.where(ok)
        hybrid_surf = _robust_plane_fit(xs_grid[xi], ys_grid[yi], first_z[yi, xi])

        # Plot pia overlay with three surfaces
        try:
            hcr_mip, hcr_yc, hcr_z_um_v, hcr_x_um_v = load_hcr_y_slab(
                s, channel="405", half_width_um=30, level=2
            )
        except Exception as e:
            print(f"{sid}: skipping visual ({e})")
            continue
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        Zn, Xn = hcr_mip.shape
        extent = (0, Xn * hcr_x_um_v, Zn * hcr_z_um_v, 0)
        ax.imshow(hcr_mip, extent=extent, cmap="gray",
                  vmin=np.percentile(hcr_mip, 1),
                  vmax=np.percentile(hcr_mip, 99.5),
                  aspect="auto")
        xs = np.linspace(0, Xn * hcr_x_um_v, 200)
        y_slab = hcr_yc
        ax.plot(xs, ref_surf["a"] * xs + ref_surf["b"] * y_slab + ref_surf["c"],
                "red", lw=1.8,
                label=f"baseline image ({ref_surf['tilt_deg']:.1f}°, frac_above={r['summary']['gfp_fraction']:.2f} — see table)")
        ax.plot(xs, prior["a"] * xs + prior["b"] * y_slab + prior["c"],
                "yellow", lw=1.2, ls="--",
                label=f"Stage B prior ({prior['tilt_deg']:.1f}°)")
        ax.plot(xs, hybrid_surf["a"] * xs + hybrid_surf["b"] * y_slab + hybrid_surf["c"],
                "cyan", lw=1.5,
                label=f"Stage C hybrid ({hybrid_surf['tilt_deg']:.1f}°)")
        ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
        ax.set_title(f"{sid} HCR 405 MIP — surface method comparison")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"surface_method_overlay_{sid}.png", dpi=120)
        plt.close(fig)
        print(f"Saved surface_method_overlay_{sid}.png")


if __name__ == "__main__":
    main()
