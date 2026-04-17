"""Iterative HCR pia surface estimation experiments (session 03).

Goal: find a protocol where ROI density at the estimated pia is ~0
(i.e. the depth-from-surface profile has a clear floor at depth = 0).

Each method is registered with a name; running the script writes
`methods_results.csv` and per-subject depth-profile figures for the
methods we keep. Failed methods stay in the CSV with a comment so we
can refer back.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject
from benchmark_analysis import (
    _robust_plane_fit,
    depth_from_surface,
    depth_profile,
    estimate_pia_surface_from_image,
    estimate_pia_surface_hybrid,
    filter_in_tissue,
    hcr_level_resolution,
    load_hcr_combined,
    cz_px_to_um,
    hcr_px_to_um,
)

OUT = Path("/root/capsule/code/sessions/03_surface_estimation")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------
def quality_metrics(hcr_xyz_um: np.ndarray, surface: dict) -> dict:
    """Return several surface-quality metrics.

    - `frac_above_pia`: ROI count with depth < -5 um, normalized by total.
      These are the out-of-tissue/spike ROIs.
    - `r0_narrow`: mean ROI density in `[-3, 3] um` divided by mean density
      in `[50, 200] um`.  This is the user's main criterion: density right
      AT the surface, normalised to bulk.  Target near 0 (i.e. surface lies
      in the gap between the false-positive spike and the bulk).
    - `r0_broad`: same with a wider `[-10, 10]` window (catches edges of
      the spike if the surface sits inside it).
    - `spike_to_bulk`: peak density in `[-100, -10]` divided by bulk
      density.  >1 means a clear spike sits above the surface.
    - `gap_depth_um`: the depth (>= 0) where smoothed density first reaches
      50% of bulk.  How quickly bulk is reached after the surface.
    """
    if surface is None or len(hcr_xyz_um) == 0:
        return {"frac_above_pia": float("nan"), "r0_narrow": float("nan"),
                "r0_broad": float("nan"), "spike_to_bulk": float("nan"),
                "gap_depth_um": float("nan"), "n_above_pia": 0}
    d = depth_from_surface(hcr_xyz_um, surface)
    n_above = int((d < -5).sum())
    frac = n_above / len(d)
    # Density profile from -200 to 800 um in 5-um bins
    edges = np.arange(-200.0, 800.0 + 5.0, 5.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    dens = h.astype(float) / 5.0  # cells/um
    bulk = dens[(centers >= 50) & (centers <= 200)].mean()
    near_n = dens[(centers >= -3) & (centers <= 3)].mean()
    near_b = dens[(centers >= -10) & (centers <= 10)].mean()
    spike = dens[(centers >= -100) & (centers <= -10)].max() if (
        ((centers >= -100) & (centers <= -10)).any()) else 0.0
    # gap_depth: first depth >= 0 where smoothed density >= 0.5*bulk
    # smooth with 30um boxcar
    ksz = 6  # 6*5um=30um
    kernel = np.ones(ksz) / ksz
    smooth = np.convolve(dens, kernel, mode="same")
    pos_mask = centers >= 0
    bulk50 = 0.5 * bulk
    over = pos_mask & (smooth >= bulk50)
    gap_depth = float(centers[over][0]) if over.any() else float("nan")
    return {
        "frac_above_pia": float(frac),
        "r0_narrow": float(near_n / bulk) if bulk > 0 else float("nan"),
        "r0_broad": float(near_b / bulk) if bulk > 0 else float("nan"),
        "spike_to_bulk": float(spike / bulk) if bulk > 0 else float("nan"),
        "gap_depth_um": gap_depth,
        "n_above_pia": n_above,
    }


# ---------------------------------------------------------------
# M1: per-tile ROI z-density threshold
# ---------------------------------------------------------------
def estimate_pia_density_threshold(
    hcr_xyz_um: np.ndarray,
    *,
    xy_tile_um: float = 100.0,
    z_bin_um: float = 20.0,
    z_window_um: float = 60.0,
    plateau_frac: float = 0.30,
    plateau_quantile: float = 0.95,
    density_filter_radius: float = 30.0,
    density_min_neighbors: int = 6,
    z_search_low_um: float = -200.0,
    z_search_high_um: float = 1500.0,
):
    """For each (x, y) tile, find the first z where local ROI density (in a
    rolling z-window) first exceeds `plateau_frac` of that tile's plateau
    density. Plane-fit those per-tile pia z values.

    Steps:
      1. Density filter: drop ROIs with < density_min_neighbors within radius.
      2. For each tile of size `xy_tile_um`:
         - Build a histogram with bins of `z_bin_um` over the global z-search
           range.
         - Convolve with a window of size `z_window_um` (uniform).
         - Define plateau = quantile(window_density, `plateau_quantile`)
           (robust to outliers).
         - Pia z = first bin center where window_density >=
           plateau * plateau_frac.
         - Skip tiles with < 5 surviving ROIs in that tile.
      3. IRLS-Huber plane fit.

    Rationale: this directly implements the criterion ("density goes from
    near-0 to bulk at the pia") on the ROI data itself — no image needed.
    """
    pts = np.asarray(hcr_xyz_um, dtype=float)
    if len(pts) < 200:
        return None
    keep = filter_in_tissue(pts, density_filter_radius, density_min_neighbors)
    pts = pts[keep]
    if len(pts) < 200:
        return None

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_lo, z_hi = z_search_low_um, z_search_high_um
    edges = np.arange(z_lo, z_hi + z_bin_um, z_bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    win_bins = max(1, int(round(z_window_um / z_bin_um)))

    # Fast bin assignment per ROI
    iz = np.floor((z - z_lo) / z_bin_um).astype(int)
    valid = (iz >= 0) & (iz < len(centers))

    x_edges = np.arange(x.min(), x.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(y.min(), y.max() + xy_tile_um, xy_tile_um)
    ix_full = np.clip(((x - x_edges[0]) // xy_tile_um).astype(int),
                      0, len(x_edges) - 2)
    iy_full = np.clip(((y - y_edges[0]) // xy_tile_um).astype(int),
                      0, len(y_edges) - 2)

    samples = []
    Nx = len(x_edges) - 1
    Ny = len(y_edges) - 1
    # Vectorize over tiles: build a 2D index of ROIs per tile then loop.
    tile_key = iy_full * Nx + ix_full
    order = np.argsort(tile_key)
    sorted_keys = tile_key[order]
    sorted_iz = iz[order]
    sorted_valid = valid[order]
    # Where each tile's run starts/ends in the sorted arrays
    starts = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="left")
    ends = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="right")

    for tile in range(Nx * Ny):
        s, e = starts[tile], ends[tile]
        if e - s < 8:
            continue
        bins = sorted_iz[s:e][sorted_valid[s:e]]
        if len(bins) < 8:
            continue
        hist = np.bincount(bins, minlength=len(centers)).astype(float)
        # Rolling sum window
        c = np.cumsum(np.r_[0.0, hist])
        win = c[win_bins:] - c[:-win_bins]
        win_centers = centers[: len(win)] + (win_bins - 1) * z_bin_um / 2
        plateau = np.quantile(win, plateau_quantile)
        if plateau <= 0:
            continue
        thr = plateau * plateau_frac
        ge = win >= thr
        if not ge.any():
            continue
        first_idx = int(np.argmax(ge))
        z_pia = float(win_centers[first_idx])
        iy_t = tile // Nx
        ix_t = tile - iy_t * Nx
        x_c = 0.5 * (x_edges[ix_t] + x_edges[ix_t + 1])
        y_c = 0.5 * (y_edges[iy_t] + y_edges[iy_t + 1])
        samples.append((x_c, y_c, z_pia))

    if len(samples) < 12:
        return None
    arr = np.asarray(samples)
    fit = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])
    fit["method"] = "density_threshold"
    fit["n_tiles"] = int(len(samples))
    return fit


# ---------------------------------------------------------------
# M2: iterative outlier rejection on top of any base surface
# ---------------------------------------------------------------
def iterate_outlier_rejection(
    hcr_xyz_um: np.ndarray,
    base_surface: dict,
    *,
    base_refit: Callable[[np.ndarray], dict],
    max_iter: int = 6,
    above_cut_um: float = -5.0,
):
    """Iteratively refit `base_refit` after dropping ROIs above the surface.

    `base_refit(filtered_pts)` should return a dict with a, b, c. The base
    surface starts the iteration; we then drop ROIs with depth < above_cut_um,
    refit, and repeat until the surface stabilises (Δc < 1 um, Δtilt < 0.1°)
    or the population stops changing.
    """
    if base_surface is None or len(hcr_xyz_um) == 0:
        return None
    surf = base_surface
    pts = np.asarray(hcr_xyz_um, dtype=float)
    last_n = -1
    for it in range(max_iter):
        d = depth_from_surface(pts, surf)
        keep = d >= above_cut_um
        n = int(keep.sum())
        if n == last_n:
            break
        last_n = n
        if n < 200:
            break
        new_surf = base_refit(pts[keep])
        if new_surf is None:
            return surf
        dc = abs(new_surf["c"] - surf["c"])
        dt = abs(new_surf["tilt_deg"] - surf["tilt_deg"])
        surf = new_surf
        if dc < 1.0 and dt < 0.1:
            break
    surf = dict(surf)
    surf["n_iter"] = int(it + 1)
    return surf


# ---------------------------------------------------------------
# M3: density-threshold + image refinement (window-restricted)
# ---------------------------------------------------------------
def estimate_pia_density_then_image(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    z_window_um: float = 60.0,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    min_thick_um: float = 15.0,
    xy_stride_um: float = 10.0,
    density_kwargs: dict | None = None,
):
    """Use the density-threshold pia (M1) as a coarse prior, then run the
    image first-crossing fit but restrict each column to z within
    z_prior +/- z_window_um. This is the analog of `estimate_pia_surface_hybrid`
    but uses the density-threshold prior (instead of min-z) which sits closer
    to the real pia.
    """
    prior = estimate_pia_density_threshold(hcr_xyz_um, **(density_kwargs or {}))
    if prior is None:
        return None
    Z, Y, X = image_combined.shape
    sx = max(1, int(round(xy_stride_um / xy_um)))
    sy = max(1, int(round(xy_stride_um / xy_um)))
    sub = image_combined[:, ::sy, ::sx].astype(np.float32, copy=False)
    Zs, Ys, Xs = sub.shape

    col_bg = np.percentile(sub, 10, axis=0)
    col_top = np.percentile(sub, 95, axis=0)
    col_thr = col_bg + relative_margin * (col_top - col_bg)
    col_valid = (col_top - col_bg) >= min_signal_abs
    above = (sub > col_thr[None, :, :]) & col_valid[None, :, :]

    k = max(1, int(round(min_thick_um / z_um)))
    if Zs < k + 1:
        return None
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]
    is_start = win == k
    has_any = is_start.any(axis=0)
    first_z = np.argmax(is_start, axis=0).astype(float) * z_um

    ys_grid = np.arange(Ys) * sy * xy_um
    xs_grid = np.arange(Xs) * sx * xy_um
    Xg, Yg = np.meshgrid(xs_grid, ys_grid)
    z_prior_grid = prior["a"] * Xg + prior["b"] * Yg + prior["c"]
    ok = has_any & (np.abs(first_z - z_prior_grid) <= z_window_um)
    if ok.sum() < 50:
        return None
    yi, xi = np.where(ok)
    fit = _robust_plane_fit(xs_grid[xi], ys_grid[yi], first_z[yi, xi])
    fit["method"] = "density_then_image"
    fit["prior_c"] = float(prior["c"])
    fit["prior_tilt"] = float(prior["tilt_deg"])
    fit["n_columns"] = int(ok.sum())
    return fit


# ---------------------------------------------------------------
# M4: image surface + global trough offset
# ---------------------------------------------------------------
def estimate_pia_image_then_trough(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    search_depth_lo: float = -150.0,
    search_depth_hi: float = 150.0,
    smooth_um: float = 20.0,
):
    """Use the image-based surface as initial estimate, then apply a global
    z-offset that places the surface in the **trough** between the spike of
    out-of-tissue ROIs (around depth -50 um) and the bulk cortex (depth > 50).

    Steps:
      1. Image-based fit (combined channels, 5% relative margin).
      2. Compute global density profile of HCR ROIs vs depth-from-image-surface
         in 5-um bins, smoothed with a `smooth_um` boxcar.
      3. Within `[search_depth_lo, search_depth_hi]` find the **local minimum
         density** that lies *after* the in-window peak (the spike).  If no
         spike-then-trough pattern is detected, fall back to the image fit.
      4. Translate the surface plane by the trough offset.
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    d = depth_from_surface(pts, base)
    bin_um = 5.0
    edges = np.arange(search_depth_lo - 50.0, search_depth_hi + 50.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    dens = h.astype(float) / bin_um
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz
    smooth = np.convolve(dens, kernel, mode="same")
    in_search = (centers >= search_depth_lo) & (centers <= search_depth_hi)
    if in_search.sum() < 5:
        return base
    sc = centers[in_search]
    sd = smooth[in_search]
    # Find the spike peak (in the negative half of the search range)
    peak_mask = sc < 0
    if peak_mask.any():
        i_peak_local = int(np.argmax(sd[peak_mask]))
        i_peak = int(np.where(peak_mask)[0][i_peak_local])
    else:
        i_peak = 0
    # After the peak, find local min before density rises back to bulk
    after = slice(i_peak + 1, len(sd))
    if (after.stop - after.start) < 3:
        offset = 0.0
    else:
        seg = sd[after]
        # local min = first index where seg drops then increases
        # use a simple approach: argmin over [i_peak, i_peak + 100um]
        lookahead = min(len(seg), int(round(100.0 / bin_um)))
        i_min_local = int(np.argmin(seg[:lookahead]))
        offset = float(sc[i_peak + 1 + i_min_local])
    fit = dict(base)
    fit["c"] = float(fit["c"] + offset)
    fit["method"] = "image_then_trough"
    fit["trough_offset_um"] = offset
    return fit


# ---------------------------------------------------------------
# M5: per-tile trough finder (image-anchored)
# ---------------------------------------------------------------
def estimate_pia_image_then_trough_per_tile(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    xy_tile_um: float = 150.0,
    search_lo_um: float = -150.0,
    search_hi_um: float = 150.0,
    bin_um: float = 5.0,
    smooth_um: float = 25.0,
    min_cells_per_tile: int = 50,
    fallback_global_offset: bool = True,
):
    """Compute the image surface, then for each (x, y) tile find a per-tile
    trough offset and refit the plane through the corrected per-tile depths.

    This handles the case where the spike-then-trough pattern varies in
    depth across the volume (e.g., tilted pia or non-uniform agarose).
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    if len(pts) < 200:
        return base
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_img = base["a"] * x + base["b"] * y + base["c"]  # image-surface z per ROI

    edges = np.arange(search_lo_um - 50.0, search_hi_um + 50.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz
    in_search = (centers >= search_lo_um) & (centers <= search_hi_um)

    x_edges = np.arange(x.min(), x.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(y.min(), y.max() + xy_tile_um, xy_tile_um)
    Nx, Ny = len(x_edges) - 1, len(y_edges) - 1
    ix_full = np.clip(((x - x_edges[0]) // xy_tile_um).astype(int), 0, Nx - 1)
    iy_full = np.clip(((y - y_edges[0]) // xy_tile_um).astype(int), 0, Ny - 1)
    tile_key = iy_full * Nx + ix_full
    order = np.argsort(tile_key)
    sorted_keys = tile_key[order]
    starts = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="left")
    ends = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="right")
    sorted_d = (z - z_img)[order]

    tile_offsets = []
    samples = []  # (x, y, z_corrected) for plane fit
    for tile in range(Nx * Ny):
        s_, e_ = starts[tile], ends[tile]
        if e_ - s_ < min_cells_per_tile:
            continue
        d_t = sorted_d[s_:e_]
        h, _ = np.histogram(d_t, bins=edges)
        dens = h.astype(float) / bin_um
        smooth = np.convolve(dens, kernel, mode="same")
        sc = centers[in_search]
        sd = smooth[in_search]
        peak_mask = sc < 0
        if not peak_mask.any() or sd[peak_mask].max() <= 0:
            continue
        i_peak_local = int(np.argmax(sd[peak_mask]))
        i_peak = int(np.where(peak_mask)[0][i_peak_local])
        if i_peak + 1 >= len(sd):
            continue
        seg = sd[i_peak + 1:]
        lookahead = min(len(seg), int(round(100.0 / bin_um)))
        if lookahead < 3:
            continue
        i_min_local = int(np.argmin(seg[:lookahead]))
        off = float(sc[i_peak + 1 + i_min_local])
        # tile center in (x, y)
        iy_t = tile // Nx
        ix_t = tile - iy_t * Nx
        x_c = 0.5 * (x_edges[ix_t] + x_edges[ix_t + 1])
        y_c = 0.5 * (y_edges[iy_t] + y_edges[iy_t + 1])
        # corrected per-tile z: image_z(x_c, y_c) + off
        z_corr = base["a"] * x_c + base["b"] * y_c + base["c"] + off
        samples.append((x_c, y_c, z_corr))
        tile_offsets.append(off)

    if len(samples) < 8:
        if fallback_global_offset:
            # fall back to single global offset
            return estimate_pia_image_then_trough(
                hcr_xyz_um, image_combined, z_um, xy_um,
                relative_margin=relative_margin,
                min_signal_abs=min_signal_abs,
            )
        return base
    arr = np.asarray(samples)
    fit = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])
    fit["method"] = "image_then_trough_per_tile"
    fit["n_tiles"] = int(len(samples))
    fit["mean_offset_um"] = float(np.mean(tile_offsets))
    fit["median_offset_um"] = float(np.median(tile_offsets))
    return fit


# ---------------------------------------------------------------
# M6: spike-then-bulk two-component fit, global
# ---------------------------------------------------------------
def estimate_pia_spike_bulk_global(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    bin_um: float = 5.0,
    smooth_um: float = 20.0,
):
    """Fit a 2-component model to depth-from-image-surface density:
    a Gaussian spike (out-of-tissue) + a sigmoid rising to bulk.
    Surface = sigmoid midpoint - 1 sigma (left foot of bulk rise).
    Falls back to image surface if fit fails.
    """
    from scipy.optimize import curve_fit
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    d = depth_from_surface(pts, base)
    edges = np.arange(-200.0, 400.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    dens = h.astype(float) / bin_um
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz
    sm = np.convolve(dens, kernel, mode="same")

    def model(z, A_spike, mu_spike, sig_spike, A_bulk, mu_bulk, sig_bulk):
        spike = A_spike * np.exp(-0.5 * ((z - mu_spike) / sig_spike) ** 2)
        bulk = A_bulk / (1.0 + np.exp(-(z - mu_bulk) / sig_bulk))
        return spike + bulk

    p0 = [sm.max(), -50.0, 30.0, sm[(centers > 100) & (centers < 200)].mean()
          if ((centers > 100) & (centers < 200)).any() else sm.max(), 30.0, 30.0]
    try:
        popt, _ = curve_fit(model, centers, sm, p0=p0, maxfev=8000,
                            bounds=([0, -200, 5, 0, -100, 5],
                                    [np.inf, 50, 200, np.inf, 400, 200]))
    except Exception:
        return base
    A_sp, mu_sp, sig_sp, A_bk, mu_bk, sig_bk = popt
    # Surface: midpoint of bulk sigmoid (where bulk reaches half).
    # That's a robust "where does cortex actually start"
    offset = float(mu_bk)
    fit = dict(base)
    fit["c"] = float(fit["c"] + offset)
    fit["method"] = "spike_bulk_global"
    fit["spike_mu"] = float(mu_sp)
    fit["spike_sig"] = float(sig_sp)
    fit["bulk_mu"] = float(mu_bk)
    fit["bulk_sig"] = float(sig_bk)
    fit["offset_um"] = offset
    return fit


# ---------------------------------------------------------------
# Driver: run a list of methods on all subjects, write CSV + figures
# ---------------------------------------------------------------
@dataclass
class Method:
    name: str
    needs_image: bool
    fn: Callable  # signature differs by method


def run_methods(
    methods: list[Method],
    subjects: list[str] = BENCHMARK_SUBJECTS,
    out_csv_name: str = "methods_results.csv",
    log_intro: str = "",
):
    rows = []
    fits_by_subject = {}  # sid -> {method_name -> surface dict}

    for sid in subjects:
        print(f"\n=== {sid} ===")
        s = load_subject(sid)
        hcr_um = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
        hcr_xyz = hcr_um[:, [2, 1, 0]]
        # Lazily load image
        vol = z_um = xy_um = None

        fits_by_subject[sid] = {"_hcr_xyz": hcr_xyz}

        for m in methods:
            try:
                if m.needs_image:
                    if vol is None:
                        vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
                    surf = m.fn(hcr_xyz, vol, z_um, xy_um)
                else:
                    surf = m.fn(hcr_xyz)
            except Exception as e:
                print(f"  {m.name}: FAILED ({e})")
                rows.append({"subject": sid, "method": m.name,
                             "error": str(e)[:200]})
                continue
            if surf is None:
                print(f"  {m.name}: returned None")
                rows.append({"subject": sid, "method": m.name,
                             "error": "returned_none"})
                continue
            q = quality_metrics(hcr_xyz, surf)
            print(f"  {m.name}: c={surf.get('c'):.1f}  tilt={surf.get('tilt_deg'):.2f}  "
                  f"frac_above={q['frac_above_pia']*100:.2f}%  "
                  f"r0_n={q['r0_narrow']:.3f}  spike/bulk={q['spike_to_bulk']:.2f}  "
                  f"gap={q['gap_depth_um']}")
            rows.append({
                "subject": sid,
                "method": m.name,
                "c_um": surf.get("c"),
                "tilt_deg": surf.get("tilt_deg"),
                "rough_um": surf.get("residual_std_um"),
                "n_columns": surf.get("n_columns") or surf.get("n_tiles"),
                **q,
            })
            fits_by_subject[sid][m.name] = surf

    df = pd.DataFrame(rows)
    df.to_csv(OUT / out_csv_name, index=False)
    print(f"\nWrote {OUT / out_csv_name}")
    return df, fits_by_subject


# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------
def plot_depth_profiles(fits_by_subject: dict, methods: list[str],
                         out_path: Path, *, depth_range=(-200, 1500)):
    n = len(fits_by_subject)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    colors = plt.get_cmap("tab10").colors
    for ax, (sid, d) in zip(axes, fits_by_subject.items()):
        hcr_xyz = d["_hcr_xyz"]
        for j, m in enumerate(methods):
            surf = d.get(m)
            if surf is None:
                continue
            c, dd = depth_profile(hcr_xyz, surf, bin_um=20,
                                   depth_range=depth_range)
            if dd.max() > 0:
                ax.plot(c, dd / dd.max(), color=colors[j % len(colors)],
                        lw=1.3, label=m)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(sid); ax.set_xlabel("depth from pia (um)")
        ax.set_ylabel("normalized density")
        ax.set_xlim(depth_range)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(fits_by_subject):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------
# Method registries — round 1
# ---------------------------------------------------------------
def _baseline_hybrid(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_surface_hybrid(hcr_xyz, vol, z_um, xy_um)


def _baseline_image(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_surface_from_image(
        vol, z_um, xy_um, min_signal_abs=0.1, relative_margin=0.05
    )


def _m1_default(hcr_xyz):
    return estimate_pia_density_threshold(hcr_xyz)


def _m1_strict(hcr_xyz):
    # Stricter: smaller fraction of plateau, larger window (smoother)
    return estimate_pia_density_threshold(
        hcr_xyz, plateau_frac=0.20, z_window_um=80.0
    )


def _m1_loose(hcr_xyz):
    # Looser: higher fraction (deeper surface), smaller window
    return estimate_pia_density_threshold(
        hcr_xyz, plateau_frac=0.50, z_window_um=40.0
    )


def _m1_smalltile(hcr_xyz):
    return estimate_pia_density_threshold(
        hcr_xyz, xy_tile_um=60.0, plateau_frac=0.30, z_window_um=60.0
    )


def round1_methods() -> list[Method]:
    return [
        Method("baseline_image",  True,  _baseline_image),
        Method("baseline_hybrid", True,  _baseline_hybrid),
        Method("M1_density_default", False, _m1_default),
        Method("M1_density_strict",  False, _m1_strict),
        Method("M1_density_loose",   False, _m1_loose),
        Method("M1_density_smalltile", False, _m1_smalltile),
    ]


# ---------- round 2: gap-finders ----------------------------------
def _m4_global_trough(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_image_then_trough(hcr_xyz, vol, z_um, xy_um)


def _m5_per_tile_trough(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_image_then_trough_per_tile(hcr_xyz, vol, z_um, xy_um)


def _m5_per_tile_trough_smalltile(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_image_then_trough_per_tile(
        hcr_xyz, vol, z_um, xy_um, xy_tile_um=100.0
    )


def _m6_spike_bulk(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_bulk_global(hcr_xyz, vol, z_um, xy_um)


def round2_methods() -> list[Method]:
    return [
        Method("baseline_image",  True,  _baseline_image),
        Method("baseline_hybrid", True,  _baseline_hybrid),
        Method("M4_global_trough", True, _m4_global_trough),
        Method("M5_per_tile_trough", True, _m5_per_tile_trough),
        Method("M5_per_tile_trough_smalltile", True, _m5_per_tile_trough_smalltile),
        Method("M6_spike_bulk", True, _m6_spike_bulk),
    ]


# ---------- round 3: robust spike-edge finders --------------------
def estimate_pia_spike_edge_global(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    bin_um: float = 5.0,
    smooth_um: float = 25.0,
    spike_search_lo: float = -200.0,
    spike_search_hi: float = 0.0,
    decay_frac: float = 0.20,
    max_offset_um: float = 200.0,
    min_offset_um: float = 0.0,
):
    """Use the image surface as initial. Then find the global density
    profile's **right edge of the spike**: the first z (>= spike peak) at
    which smoothed density drops below `decay_frac * spike_peak_density`.
    Apply this as a positive offset (surface moves deeper, toward bulk).

    Robust against subjects without a clear spike (decay_frac fallback to
    tail of bulk also keeps offset in [min_offset_um, max_offset_um]).
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    d = depth_from_surface(pts, base)
    edges = np.arange(-300.0, 400.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    dens = h.astype(float) / bin_um
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz
    sm = np.convolve(dens, kernel, mode="same")

    # Spike search range
    sp_mask = (centers >= spike_search_lo) & (centers <= spike_search_hi)
    if not sp_mask.any():
        return base
    sp_idx = np.where(sp_mask)[0]
    sp_dens = sm[sp_idx]
    bulk = sm[(centers >= 100) & (centers <= 250)].mean()
    spike_peak = sp_dens.max()
    # If the "spike" is not actually elevated above bulk, no correction needed
    if spike_peak < 1.5 * bulk:
        fit = dict(base)
        fit["method"] = "spike_edge_global"
        fit["offset_um"] = 0.0
        fit["spike_peak"] = float(spike_peak)
        fit["bulk"] = float(bulk)
        fit["reason"] = "no_clear_spike"
        return fit
    i_peak = int(sp_idx[int(np.argmax(sp_dens))])
    threshold = decay_frac * spike_peak
    # Walk forward from the peak to find first z where sm <= threshold
    offset = None
    for j in range(i_peak + 1, len(sm)):
        if centers[j] - centers[i_peak] > 200.0:
            break
        if sm[j] <= threshold:
            offset = float(centers[j])
            break
    if offset is None:
        offset = 0.0
    offset = float(np.clip(offset, min_offset_um, max_offset_um))
    fit = dict(base)
    fit["c"] = float(fit["c"] + offset)
    fit["method"] = "spike_edge_global"
    fit["offset_um"] = offset
    fit["spike_peak"] = float(spike_peak)
    fit["bulk"] = float(bulk)
    return fit


def estimate_pia_spike_edge_per_tile(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    xy_tile_um: float = 200.0,
    min_cells_per_tile: int = 80,
    bin_um: float = 5.0,
    smooth_um: float = 30.0,
    spike_search_lo: float = -200.0,
    spike_search_hi: float = 0.0,
    decay_frac: float = 0.20,
    max_offset_um: float = 200.0,
    min_offset_um: float = 0.0,
    fallback_to_global: bool = True,
):
    """Per-tile version of the spike-edge finder. For each tile, computes the
    density profile relative to the image surface, finds the spike's right
    edge, and uses that as the per-tile pia z.  Plane-fits the corrections.
    Falls back to the global-offset surface if too few tiles converge.
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    if len(pts) < 200:
        return base
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_img = base["a"] * x + base["b"] * y + base["c"]

    edges = np.arange(-300.0, 400.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz

    # Compute global bulk for normalisation (used per-tile if local bulk is noisy)
    d_all = z - z_img
    h_all, _ = np.histogram(d_all, bins=edges)
    sm_all = np.convolve(h_all.astype(float) / bin_um, kernel, mode="same")
    global_bulk = sm_all[(centers >= 100) & (centers <= 250)].mean()

    x_edges = np.arange(x.min(), x.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(y.min(), y.max() + xy_tile_um, xy_tile_um)
    Nx, Ny = len(x_edges) - 1, len(y_edges) - 1
    ix_full = np.clip(((x - x_edges[0]) // xy_tile_um).astype(int), 0, Nx - 1)
    iy_full = np.clip(((y - y_edges[0]) // xy_tile_um).astype(int), 0, Ny - 1)
    tile_key = iy_full * Nx + ix_full
    order = np.argsort(tile_key)
    sorted_keys = tile_key[order]
    starts = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="left")
    ends = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="right")
    sorted_d = (z - z_img)[order]

    samples, offsets = [], []
    for tile in range(Nx * Ny):
        s_, e_ = starts[tile], ends[tile]
        if e_ - s_ < min_cells_per_tile:
            continue
        d_t = sorted_d[s_:e_]
        h, _ = np.histogram(d_t, bins=edges)
        sm = np.convolve(h.astype(float) / bin_um, kernel, mode="same")

        sp_mask = (centers >= spike_search_lo) & (centers <= spike_search_hi)
        sp_idx = np.where(sp_mask)[0]
        if len(sp_idx) == 0:
            continue
        sp_dens = sm[sp_idx]
        # bulk for this tile (use local 100-250)
        local_bulk_mask = (centers >= 100) & (centers <= 250)
        local_bulk = sm[local_bulk_mask].mean()
        # use whichever is larger (local can be 0 in sparse tile)
        bulk = max(local_bulk, 0.5 * global_bulk)
        spike_peak = sp_dens.max()
        if spike_peak < 1.5 * bulk:
            off = 0.0
        else:
            i_peak_local = int(np.argmax(sp_dens))
            i_peak = int(sp_idx[i_peak_local])
            thr = decay_frac * spike_peak
            off = None
            for j in range(i_peak + 1, len(sm)):
                if centers[j] - centers[i_peak] > 200.0:
                    break
                if sm[j] <= thr:
                    off = float(centers[j])
                    break
            if off is None:
                off = 0.0
        off = float(np.clip(off, min_offset_um, max_offset_um))
        offsets.append(off)
        iy_t = tile // Nx
        ix_t = tile - iy_t * Nx
        x_c = 0.5 * (x_edges[ix_t] + x_edges[ix_t + 1])
        y_c = 0.5 * (y_edges[iy_t] + y_edges[iy_t + 1])
        z_corr = base["a"] * x_c + base["b"] * y_c + base["c"] + off
        samples.append((x_c, y_c, z_corr))

    if len(samples) < 8 and fallback_to_global:
        return estimate_pia_spike_edge_global(
            hcr_xyz_um, image_combined, z_um, xy_um,
            relative_margin=relative_margin, min_signal_abs=min_signal_abs,
        )
    if len(samples) < 8:
        return base
    arr = np.asarray(samples)
    fit = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])
    fit["method"] = "spike_edge_per_tile"
    fit["n_tiles"] = int(len(samples))
    fit["mean_offset_um"] = float(np.mean(offsets))
    fit["median_offset_um"] = float(np.median(offsets))
    fit["frac_zero_offset"] = float(np.mean(np.array(offsets) == 0))
    return fit


def _m7_spike_edge_global(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_edge_global(hcr_xyz, vol, z_um, xy_um)


def _m7_spike_edge_strict(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_edge_global(
        hcr_xyz, vol, z_um, xy_um, decay_frac=0.10
    )


def _m7_spike_edge_loose(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_edge_global(
        hcr_xyz, vol, z_um, xy_um, decay_frac=0.40
    )


def _m8_per_tile(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_edge_per_tile(hcr_xyz, vol, z_um, xy_um)


def _m8_per_tile_smalltile(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_spike_edge_per_tile(
        hcr_xyz, vol, z_um, xy_um, xy_tile_um=150.0, min_cells_per_tile=50
    )


def estimate_pia_bulk_floor_global(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    bin_um: float = 5.0,
    smooth_um: float = 25.0,
    bulk_depth_lo: float = 200.0,
    bulk_depth_hi: float = 400.0,
    bulk_quantile: float = 0.50,  # use median, robust to layer fluctuations
    bulk_threshold_frac: float = 0.50,
    search_lo: float = 0.0,
    search_hi: float = 200.0,
    spike_required_ratio: float = 1.2,
):
    """Find global offset using "spike sustainment" criterion:

      bulk = quantile(density in [200, 400] um, q=bulk_quantile)
      threshold = bulk * bulk_threshold_frac

    The spike is defined as the contiguous run of depth bins (from the
    earliest negative depth where the smoothed density first exceeds the
    threshold) where density stays above threshold AT LEAST ONCE.  More
    precisely:

      1. smoothed density profile vs depth.
      2. Walk from `search_lo` (default 0) deeper. The surface offset is
         the FIRST depth >= search_lo where the smoothed density drops
         below threshold AND remains below for at least 15 um.
      3. If smoothed density at depth=search_lo is already below threshold,
         offset = 0 (we are already past the spike).
      4. If smoothed density never drops below threshold in the search
         range, fall back to LOCAL min in that range.
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    d = depth_from_surface(pts, base)
    edges = np.arange(-300.0, 600.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    sm = np.convolve(h.astype(float) / bin_um, np.ones(int(round(smooth_um / bin_um))) / int(round(smooth_um / bin_um)), mode="same")

    bulk_mask = (centers >= bulk_depth_lo) & (centers <= bulk_depth_hi)
    if not bulk_mask.any():
        return base
    bulk = float(np.quantile(sm[bulk_mask], bulk_quantile))
    threshold = bulk_threshold_frac * bulk

    search_mask = (centers >= search_lo) & (centers <= search_hi)
    if not search_mask.any():
        return base
    search_idx = np.where(search_mask)[0]
    sm_s = sm[search_idx]

    # check if there's even a spike to clear
    spike_in_search = sm_s.max() if len(sm_s) else 0.0
    if spike_in_search <= spike_required_ratio * bulk:
        # No clear spike - just place at search_lo
        offset = float(search_lo)
    else:
        # Walk forward and find first sustained drop below threshold
        sustain_bins = max(1, int(round(15.0 / bin_um)))
        offset = None
        for k in range(len(sm_s)):
            if sm_s[k] <= threshold:
                # check sustainment
                end = min(k + sustain_bins, len(sm_s))
                if (sm_s[k:end] <= threshold).all():
                    offset = float(centers[search_idx[k]])
                    break
        if offset is None:
            # Fall back to local min in search range
            i_min_local = int(np.argmin(sm_s))
            offset = float(centers[search_idx[i_min_local]])
    fit = dict(base)
    fit["c"] = float(fit["c"] + offset)
    fit["method"] = "bulk_floor_global"
    fit["offset_um"] = offset
    fit["bulk"] = bulk
    fit["threshold"] = threshold
    return fit


def estimate_pia_bulk_floor_per_tile(
    hcr_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    xy_tile_um: float = 200.0,
    min_cells_per_tile: int = 80,
    bin_um: float = 5.0,
    smooth_um: float = 30.0,
    bulk_depth_lo: float = 200.0,
    bulk_depth_hi: float = 400.0,
    bulk_threshold_frac: float = 0.50,
    search_lo: float = 0.0,
    search_hi: float = 200.0,
    fallback_to_global: bool = True,
):
    """Per-tile version of `estimate_pia_bulk_floor_global`.

    For each (x, y) tile, finds the first depth (>= search_lo) where the
    smoothed density falls below a tile-specific threshold (a fraction of
    the global bulk). The per-tile pia z is then `z_image_tile + offset`,
    and the plane is refit through these per-tile pia points.
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(hcr_xyz_um, dtype=float)
    if len(pts) < 200:
        return base
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z_img = base["a"] * x + base["b"] * y + base["c"]
    d_all = z - z_img

    edges = np.arange(-300.0, 600.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ksz = max(3, int(round(smooth_um / bin_um)))
    kernel = np.ones(ksz) / ksz

    # Global bulk reference
    h_all, _ = np.histogram(d_all, bins=edges)
    sm_all = np.convolve(h_all.astype(float) / bin_um, kernel, mode="same")
    global_bulk = float(np.median(sm_all[(centers >= bulk_depth_lo) & (centers <= bulk_depth_hi)]))
    global_threshold = bulk_threshold_frac * global_bulk

    x_edges = np.arange(x.min(), x.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(y.min(), y.max() + xy_tile_um, xy_tile_um)
    Nx, Ny = len(x_edges) - 1, len(y_edges) - 1
    ix_full = np.clip(((x - x_edges[0]) // xy_tile_um).astype(int), 0, Nx - 1)
    iy_full = np.clip(((y - y_edges[0]) // xy_tile_um).astype(int), 0, Ny - 1)
    tile_key = iy_full * Nx + ix_full
    order = np.argsort(tile_key)
    sorted_keys = tile_key[order]
    sorted_d = d_all[order]
    starts = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="left")
    ends = np.searchsorted(sorted_keys, np.arange(Nx * Ny), side="right")

    search_mask = (centers >= search_lo) & (centers <= search_hi)
    search_idx = np.where(search_mask)[0]
    sustain_bins = max(1, int(round(15.0 / bin_um)))

    samples, offsets = [], []
    for tile in range(Nx * Ny):
        s_, e_ = starts[tile], ends[tile]
        if e_ - s_ < min_cells_per_tile:
            continue
        d_t = sorted_d[s_:e_]
        h, _ = np.histogram(d_t, bins=edges)
        sm = np.convolve(h.astype(float) / bin_um, kernel, mode="same")
        local_bulk = float(np.median(sm[(centers >= bulk_depth_lo) & (centers <= bulk_depth_hi)]))
        # use blended bulk for robustness
        bulk = max(local_bulk, 0.5 * global_bulk)
        threshold = bulk_threshold_frac * bulk
        sm_s = sm[search_idx]
        if sm_s.max() <= 1.2 * bulk:
            off = float(search_lo)
        else:
            off = None
            for k in range(len(sm_s)):
                if sm_s[k] <= threshold:
                    end = min(k + sustain_bins, len(sm_s))
                    if (sm_s[k:end] <= threshold).all():
                        off = float(centers[search_idx[k]])
                        break
            if off is None:
                i_min_local = int(np.argmin(sm_s))
                off = float(centers[search_idx[i_min_local]])
        offsets.append(off)
        iy_t = tile // Nx
        ix_t = tile - iy_t * Nx
        x_c = 0.5 * (x_edges[ix_t] + x_edges[ix_t + 1])
        y_c = 0.5 * (y_edges[iy_t] + y_edges[iy_t + 1])
        z_corr = base["a"] * x_c + base["b"] * y_c + base["c"] + off
        samples.append((x_c, y_c, z_corr))

    if len(samples) < 8 and fallback_to_global:
        return estimate_pia_bulk_floor_global(
            hcr_xyz_um, image_combined, z_um, xy_um,
            relative_margin=relative_margin, min_signal_abs=min_signal_abs,
            bulk_threshold_frac=bulk_threshold_frac,
        )
    if len(samples) < 8:
        return base
    arr = np.asarray(samples)
    fit = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])
    fit["method"] = "bulk_floor_per_tile"
    fit["n_tiles"] = int(len(samples))
    fit["mean_offset_um"] = float(np.mean(offsets))
    fit["median_offset_um"] = float(np.median(offsets))
    return fit


def _m9_bulk_floor_default(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_bulk_floor_global(hcr_xyz, vol, z_um, xy_um)


def _m9_bulk_floor_strict(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_bulk_floor_global(
        hcr_xyz, vol, z_um, xy_um, bulk_threshold_frac=0.30
    )


def _m9_bulk_floor_loose(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_bulk_floor_global(
        hcr_xyz, vol, z_um, xy_um, bulk_threshold_frac=0.70
    )


def _m10_per_tile_default(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_bulk_floor_per_tile(hcr_xyz, vol, z_um, xy_um)


def _m10_per_tile_smalltile(hcr_xyz, vol, z_um, xy_um):
    return estimate_pia_bulk_floor_per_tile(
        hcr_xyz, vol, z_um, xy_um, xy_tile_um=150.0, min_cells_per_tile=60
    )


def round4_methods() -> list[Method]:
    return [
        Method("baseline_image",  True,  _baseline_image),
        Method("M4_global_trough", True, _m4_global_trough),
        Method("M9_bulk_floor_default(0.50)", True, _m9_bulk_floor_default),
        Method("M9_bulk_floor_strict(0.30)",  True, _m9_bulk_floor_strict),
        Method("M9_bulk_floor_loose(0.70)",   True, _m9_bulk_floor_loose),
        Method("M10_per_tile_default", True, _m10_per_tile_default),
        Method("M10_per_tile_smalltile", True, _m10_per_tile_smalltile),
    ]


def round3_methods() -> list[Method]:
    return [
        Method("baseline_image",  True,  _baseline_image),
        Method("M4_global_trough", True, _m4_global_trough),
        Method("M7_spike_edge_default(0.20)", True, _m7_spike_edge_global),
        Method("M7_spike_edge_strict(0.10)",  True, _m7_spike_edge_strict),
        Method("M7_spike_edge_loose(0.40)",   True, _m7_spike_edge_loose),
        Method("M8_per_tile_default", True, _m8_per_tile),
        Method("M8_per_tile_smalltile", True, _m8_per_tile_smalltile),
    ]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    subjects = args.subjects or BENCHMARK_SUBJECTS

    if args.round == 1:
        methods = round1_methods()
        df, fits = run_methods(methods, subjects, "round1_results.csv")
        plot_depth_profiles(
            fits,
            ["baseline_hybrid", "M1_density_default",
             "M1_density_strict", "M1_density_loose"],
            FIG / "round1_depth_profiles.png",
        )
    elif args.round == 2:
        methods = round2_methods()
        df, fits = run_methods(methods, subjects, "round2_results.csv")
        plot_depth_profiles(
            fits,
            ["baseline_image", "baseline_hybrid",
             "M4_global_trough", "M5_per_tile_trough", "M6_spike_bulk"],
            FIG / "round2_depth_profiles.png",
        )
    elif args.round == 3:
        methods = round3_methods()
        df, fits = run_methods(methods, subjects, "round3_results.csv")
        plot_depth_profiles(
            fits,
            ["baseline_image", "M4_global_trough",
             "M7_spike_edge_default(0.20)", "M7_spike_edge_strict(0.10)",
             "M7_spike_edge_loose(0.40)", "M8_per_tile_default"],
            FIG / "round3_depth_profiles.png",
        )
    elif args.round == 4:
        methods = round4_methods()
        df, fits = run_methods(methods, subjects, "round4_results.csv")
        plot_depth_profiles(
            fits,
            ["baseline_image", "M4_global_trough",
             "M9_bulk_floor_default(0.50)", "M9_bulk_floor_strict(0.30)",
             "M9_bulk_floor_loose(0.70)", "M10_per_tile_default"],
            FIG / "round4_depth_profiles.png",
        )
    else:
        raise ValueError(f"unknown round {args.round}")
