"""Image-level depth-profile NCC estimator for sz (CZ ↔ HCR 488).

Design (session 07c' → 07d):
  1.  Apply R1 (identity scales) to every CZ voxel → HCR-frame µm.
  2.  Compute depth below HCR pia for every CZ voxel.
  3.  Histogram CZ voxel intensity by depth → 1D profile I_cz(d_cz).
  4.  Repeat for HCR 488 voxels (no R1, already in HCR frame).
  5.  For each candidate sz, stretch the CZ depth axis by sz and
      NCC against the HCR profile on the common interval.
  6.  Report argmax sz and compare to GT.

Why this might succeed where centroids failed: intensity is linear in
signal, so cells that failed the strict-GFP⁺ threshold still contribute
their full brightness to the HCR profile.  The threshold-induced
depth-dependent detection bias is bypassed entirely.

Diagnostic-only: GT is used only for scoring.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

from benchmark_analysis import (
    analyze_subject,
    fit_anisotropic_similarity,
    load_hcr_combined,
    load_hcr_volume,
)
from benchmark_data_loader import landmark_pairs_um, load_subject
from r1_revised import apply_coarse_affine, coarse_align_revised


DEPTH_BIN_UM = 10.0
MIN_OVERLAP_BINS = 20
DEFAULT_SZ_GRID = np.arange(1.5, 4.51, 0.02)


def _load_cz_zstack(s):
    cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not cz_tifs:
        cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
    if not cz_tifs:
        raise FileNotFoundError(f"No CZ zstack for {s.subject_id}")
    img = tifffile.imread(str(cz_tifs[0]))
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    return img.astype(np.float32, copy=False)


def _r1_fit(s):
    info = analyze_subject(s)
    fit = coarse_align_revised(
        info["cz_xyz"], info["gfp_xyz"],
        cz_surface=info["cz_surface"], hcr_surface=info["hcr_surface"],
    )
    return fit, info


def _gt_scales(s):
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    return np.asarray(fit.scales, dtype=float)


def build_cz_depth_profile(
    img_cz: np.ndarray,
    cz_xy_um: float,
    cz_z_um: float,
    r1_fit,
    hcr_pia: dict,
    depth_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (profile, nonzero_mask, xy_aabb_hcr).

    xy_aabb_hcr:  (2, 2) array = [[x_lo, y_lo], [x_hi, y_hi]] of the CZ
    footprint mapped into HCR frame.  Used to clip the HCR profile to
    the same xy region.
    """
    nz, ny, nx = img_cz.shape
    R = np.asarray(r1_fit.R, dtype=float)
    scales = np.asarray(r1_fit.scales, dtype=float)
    src_mean = np.asarray(r1_fit.src_mean, dtype=float)
    translation = np.asarray(r1_fit.translation, dtype=float)

    a = float(hcr_pia["a"]); b = float(hcr_pia["b"]); c = float(hcr_pia["c"])

    x_cz_1d = np.arange(nx, dtype=np.float32) * cz_xy_um
    y_cz_1d = np.arange(ny, dtype=np.float32) * cz_xy_um
    x_grid, y_grid = np.meshgrid(x_cz_1d, y_cz_1d, indexing="xy")

    xy_flat = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
    n_pix = xy_flat.shape[0]

    profile = np.zeros(len(depth_bins) - 1, dtype=np.float64)
    xy_lo = np.array([np.inf, np.inf])
    xy_hi = np.array([-np.inf, -np.inf])

    for k in range(nz):
        z_cz = np.float32(k * cz_z_um)
        p_cz = np.concatenate(
            [xy_flat, np.full((n_pix, 1), z_cz, dtype=np.float32)],
            axis=1,
        )
        p_hcr = (p_cz - src_mean) @ R * scales + translation
        d = p_hcr[:, 2] - (a * p_hcr[:, 0] + b * p_hcr[:, 1] + c)
        xy_lo = np.minimum(xy_lo, p_hcr[:, :2].min(axis=0))
        xy_hi = np.maximum(xy_hi, p_hcr[:, :2].max(axis=0))
        profile += np.histogram(
            d, bins=depth_bins,
            weights=img_cz[k].ravel().astype(np.float64),
        )[0]
    return profile, np.stack([xy_lo, xy_hi], axis=0)


def build_hcr_depth_profile(
    vol_hcr: np.ndarray,
    hcr_xy_um: float,
    hcr_z_um: float,
    hcr_pia: dict,
    depth_bins: np.ndarray,
    xy_aabb: np.ndarray | None = None,
) -> np.ndarray:
    nz, ny, nx = vol_hcr.shape
    a = float(hcr_pia["a"]); b = float(hcr_pia["b"]); c = float(hcr_pia["c"])

    x_1d = np.arange(nx, dtype=np.float32) * hcr_xy_um
    y_1d = np.arange(ny, dtype=np.float32) * hcr_xy_um
    x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing="xy")

    if xy_aabb is not None:
        xy_mask = ((x_grid >= xy_aabb[0, 0]) & (x_grid <= xy_aabb[1, 0])
                 & (y_grid >= xy_aabb[0, 1]) & (y_grid <= xy_aabb[1, 1]))
        if not xy_mask.any():
            raise RuntimeError("HCR xy AABB clip is empty")
    else:
        xy_mask = np.ones_like(x_grid, dtype=bool)

    d_xy = -(a * x_grid + b * y_grid + c)
    d_xy_masked = d_xy[xy_mask]
    flat_xy_int = lambda slab: slab[xy_mask].astype(np.float64)

    profile = np.zeros(len(depth_bins) - 1, dtype=np.float64)
    for k in range(nz):
        z_hcr = k * hcr_z_um
        d = z_hcr + d_xy_masked
        profile += np.histogram(
            d, bins=depth_bins, weights=flat_xy_int(vol_hcr[k]),
        )[0]
    return profile


def subtract_baseline(profile: np.ndarray, q: float = 10.0) -> np.ndarray:
    """Subtract the qth-percentile of the nonzero portion of ``profile``.

    The HCR profile always has a large autofluorescence / background
    pedestal; the CZ profile is usually cleaner, but may still sit on
    a nonzero base.  Subtracting the qth-percentile and clipping at 0
    removes this pedestal so both profiles start at zero in empty
    regions and NCC can focus on feature shape."""
    nz = profile[profile > 0]
    if nz.size == 0:
        return profile
    base = float(np.percentile(nz, q))
    return np.clip(profile - base, 0.0, None)


def ncc_over_overlap(p_cz: np.ndarray, p_hcr: np.ndarray) -> float:
    mask = (p_cz > 0) & (p_hcr > 0)
    if mask.sum() < MIN_OVERLAP_BINS:
        return np.nan
    a = p_cz[mask] - p_cz[mask].mean()
    b = p_hcr[mask] - p_hcr[mask].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(a @ b / denom)


def stretch_cz_profile(profile_cz: np.ndarray, depth_centers: np.ndarray,
                        sz: float, d_center: float) -> np.ndarray:
    """Stretch the CZ depth axis by ``sz`` around ``d_center``.

    Under R1-identity, the CZ cloud is placed centroid-aligned with HCR
    but not scaled.  The GT anisotropic-similarity map is
        d_GT = sz * (d_R1 - d_mean) + d_mean
    so to predict the GT-scaled CZ profile at hcr depth d we evaluate
    the raw CZ profile at the pre-image
        (d - d_mean) / sz + d_mean.
    """
    src = (depth_centers - d_center) / sz + d_center
    return np.interp(src, depth_centers, profile_cz, left=0.0, right=0.0)


def _r1_anchor_depth(r1_fit, hcr_pia: dict) -> float:
    """Depth of the CZ centroid under R1 → HCR.  This is the anchor
    point around which the GT anisotropic scale acts:
        hcr_GT = hcr_R1_anchor + scales · (hcr_R1 - hcr_R1_anchor)
    Using the intensity-weighted centroid of the depth profile as a
    stretch anchor is a lossy proxy — the CZ image intensity varies
    with depth (attenuation + non-uniform cell density), so the
    intensity-centroid is biased relative to the geometric centroid.
    """
    t = np.asarray(r1_fit.translation, dtype=float)
    a = float(hcr_pia["a"]); b = float(hcr_pia["b"]); c = float(hcr_pia["c"])
    return float(t[2] - (a * t[0] + b * t[1] + c))


def find_sz(
    profile_cz: np.ndarray, profile_hcr: np.ndarray,
    depth_centers: np.ndarray, sz_grid: np.ndarray,
    d_center: float,
) -> tuple[float, float, np.ndarray]:
    """Return (sz_best, ncc_best, ncc_curve)."""
    ncc_vals = np.full_like(sz_grid, np.nan, dtype=float)
    for idx, sz in enumerate(sz_grid):
        p_cz_scaled = stretch_cz_profile(profile_cz, depth_centers, sz, d_center)
        ncc_vals[idx] = ncc_over_overlap(p_cz_scaled, profile_hcr)
    if np.all(np.isnan(ncc_vals)):
        return np.nan, np.nan, ncc_vals
    best = int(np.nanargmax(ncc_vals))
    return float(sz_grid[best]), float(ncc_vals[best]), ncc_vals


def run_subject(
    sid: str,
    sz_grid: np.ndarray | None = None,
    hcr_level: int = 4,
    hcr_channel: str = "combined",
    baseline_subtract: bool = True,
) -> dict:
    """Run image-NCC sz estimator on one subject.

    Parameters
    ----------
    hcr_channel : {"488", "combined"}
        "488" — single HCR 488 channel (original); "combined" — all
        channels summed after per-channel background-subtraction and
        percentile normalisation (``load_hcr_combined``).  Combined
        is more robust to per-channel autofluorescence and saturation.
    baseline_subtract : bool
        If True, subtract the p10 of each profile before NCC.  Removes
        the DC pedestal so correlation operates on feature shape only.
    """
    s = load_subject(sid)
    r1, info = _r1_fit(s)
    gt_scales = _gt_scales(s)
    sz_gt = float(gt_scales[2])

    img_cz = _load_cz_zstack(s)
    if hcr_channel == "combined":
        vol_hcr, hcr_xy_um, hcr_z_um, hcr_channels_used = load_hcr_combined(
            s, level=hcr_level,
        )
    else:
        vol_hcr, hcr_xy_um, hcr_z_um = load_hcr_volume(
            s, channel=hcr_channel, level=hcr_level,
        )
        hcr_channels_used = [hcr_channel]

    # Depth grid wide enough for CZ native (~500 µm) *and* HCR (~1400 µm)
    depth_bins = np.arange(-300.0, 1700.0 + DEPTH_BIN_UM, DEPTH_BIN_UM)
    depth_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])

    p_cz_raw, cz_xy_aabb_hcr = build_cz_depth_profile(
        img_cz, s.cz_xy_um, s.cz_z_um, r1, info["hcr_surface"], depth_bins,
    )
    p_hcr_raw = build_hcr_depth_profile(
        vol_hcr, hcr_xy_um, hcr_z_um, info["hcr_surface"], depth_bins,
        xy_aabb=cz_xy_aabb_hcr,
    )

    p_cz = subtract_baseline(p_cz_raw) if baseline_subtract else p_cz_raw
    p_hcr = subtract_baseline(p_hcr_raw) if baseline_subtract else p_hcr_raw

    if sz_grid is None:
        sz_grid = DEFAULT_SZ_GRID
    d_center = _r1_anchor_depth(r1, info["hcr_surface"])
    sz_best, ncc_best, ncc_curve = find_sz(
        p_cz, p_hcr, depth_centers, sz_grid, d_center,
    )

    rel_err = (sz_best - sz_gt) / sz_gt * 100 if np.isfinite(sz_best) else np.nan
    print(f"{sid}   sz_gt = {sz_gt:.3f}   sz_ncc = {sz_best:.3f}   "
          f"NCC = {ncc_best:.3f}   err = {rel_err:+.2f}%   "
          f"d_center = {d_center:.1f} µm   chan = {hcr_channels_used}   "
          f"baseline = {baseline_subtract}")

    return {
        "subject": sid,
        "sz_gt": sz_gt,
        "sz_ncc": sz_best,
        "ncc_best": ncc_best,
        "rel_err_sz_pct": rel_err,
        "profile_cz_raw": p_cz_raw,
        "profile_hcr_raw": p_hcr_raw,
        "profile_cz": p_cz,
        "profile_hcr": p_hcr,
        "depth_centers": depth_centers,
        "sz_grid": sz_grid,
        "ncc_curve": ncc_curve,
        "d_center": d_center,
        "cz_xy_aabb_hcr": cz_xy_aabb_hcr,
        "r1_scales": np.asarray(r1.scales, dtype=float),
        "hcr_channel": hcr_channel,
        "hcr_channels_used": hcr_channels_used,
        "baseline_subtract": baseline_subtract,
    }


def plot_subject(res: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    d = res["depth_centers"]
    p_cz = res["profile_cz"]; p_hcr = res["profile_hcr"]
    p_cz_raw = res.get("profile_cz_raw", p_cz)
    p_hcr_raw = res.get("profile_hcr_raw", p_hcr)
    sz_best = res["sz_ncc"]; sz_gt = res["sz_gt"]
    hcr_label = "HCR " + ("+".join(res.get("hcr_channels_used", ["?"])))
    baseline_str = " (baseline-subtracted)" if res.get("baseline_subtract") else ""

    ax = axes[0]
    def _norm(v):
        return v / v.max() if v.max() > 0 else v
    # Stretched CZ under the best-NCC sz (around d_center)
    p_cz_stretched = stretch_cz_profile(p_cz, d, sz_best, res["d_center"])
    # Thin grey lines = raw profiles before baseline subtraction
    ax.plot(d, _norm(p_cz_raw), "-", color="#bdbdbd", lw=0.9,
            label="CZ raw (pre-baseline)")
    ax.plot(d, _norm(p_hcr_raw), "-", color="#c7e7e3", lw=0.9,
            label=f"{hcr_label} raw (pre-baseline)")
    ax.plot(d, _norm(p_cz), "-", color="#93a1a1", lw=1.4,
            label=f"CZ{baseline_str} (stretch=1)")
    ax.plot(d, _norm(p_cz_stretched), "-", color="#268bd2", lw=2.0,
            label=f"CZ stretched sz={sz_best:.3f}")
    ax.plot(d, _norm(p_hcr), "-", color="#2aa198", lw=1.8,
            label=f"{hcr_label}{baseline_str}")
    mask = (d >= -100) & (d <= 1500)
    ax.set_xlim(d[mask].min(), d[mask].max())
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("depth below HCR pia (µm)")
    ax.set_ylabel("intensity (max-normalised)")
    ax.set_title(
        f"{res['subject']}   sz_gt = {sz_gt:.3f}   "
        f"sz_NCC = {sz_best:.3f}   err = {res['rel_err_sz_pct']:+.1f}%"
    )
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    ax.plot(res["sz_grid"], res["ncc_curve"], "-", color="#268bd2", lw=1.8)
    ax.axvline(sz_gt, color="#cb4b16", lw=1.0, ls="--", label=f"GT sz = {sz_gt:.3f}")
    ax.axvline(sz_best, color="#268bd2", lw=1.0, ls=":", label=f"argmax NCC")
    ax.set_xlabel("sz")
    ax.set_ylabel("NCC (overlap only)")
    ax.set_title(f"{res['subject']}   NCC vs sz   max = {res['ncc_best']:.3f}")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import json

    SESSION_DIR = Path("/root/capsule/code/sessions/07d_image_ncc_scale")
    FIG_DIR = SESSION_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Argv:  [variant] [sid_or_ALL]
    #        variant ∈ VARIANT_CONFIG (see below) — default 488_baseline
    #        sid     — a benchmark subject ID, or "ALL" / missing for all 6
    from benchmark_data_loader import BENCHMARK_SUBJECTS
    sid_arg = sys.argv[2] if len(sys.argv) > 2 else "ALL"
    sids = list(BENCHMARK_SUBJECTS) if sid_arg in ("ALL", "") else [sid_arg]

    # Compare variants.  Each tuple is (variant_name, hcr_channel, baseline_subtract).
    variant_arg = sys.argv[1] if len(sys.argv) > 1 else "488_baseline"
    VARIANT_CONFIG = {
        "488_only":          ("488",      False),
        "488_baseline":      ("488",      True),
        "combined_baseline": ("combined", True),
    }
    if variant_arg not in VARIANT_CONFIG:
        raise SystemExit(
            f"Unknown variant {variant_arg}.  Options: {list(VARIANT_CONFIG)}"
        )
    hcr_channel, baseline_subtract = VARIANT_CONFIG[variant_arg]
    variant = variant_arg

    records = {}
    for sid in sids:
        try:
            res = run_subject(
                sid, hcr_channel=hcr_channel,
                baseline_subtract=baseline_subtract,
            )
        except Exception as e:
            print(f"{sid}   ERROR: {e}")
            records[sid] = {"subject": sid, "status": f"error: {e}"}
            continue
        plot_subject(res, FIG_DIR / f"sz_ncc_{variant}_{sid}.png")
        records[sid] = {
            "subject": sid,
            "status": "ok",
            "variant": variant,
            "hcr_channel": res["hcr_channel"],
            "hcr_channels_used": res["hcr_channels_used"],
            "baseline_subtract": res["baseline_subtract"],
            "sz_gt": res["sz_gt"],
            "sz_ncc": res["sz_ncc"],
            "ncc_best": res["ncc_best"],
            "rel_err_sz_pct": res["rel_err_sz_pct"],
            "d_center_um": res["d_center"],
            "sz_grid_lo": float(res["sz_grid"][0]),
            "sz_grid_hi": float(res["sz_grid"][-1]),
            "ncc_curve": res["ncc_curve"].tolist(),
            "cz_xy_aabb_hcr": res["cz_xy_aabb_hcr"].tolist(),
        }
    out = SESSION_DIR / f"sz_ncc_summary_{variant}.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    n_pass5 = sum(1 for r in records.values()
                  if r.get("status") == "ok" and abs(r["rel_err_sz_pct"]) <= 5)
    n_ok = sum(1 for r in records.values() if r.get("status") == "ok")
    print(f"\nImage-NCC sz ({variant}):  {n_pass5}/{n_ok} within ±5 %   "
          f"(out of {len(sids)} attempted)")
    print(f"Wrote {out}")
