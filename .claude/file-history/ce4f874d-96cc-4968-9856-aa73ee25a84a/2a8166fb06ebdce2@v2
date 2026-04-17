"""Dump R1's intermediate 2D maps + xcorr for one subject."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject
from benchmark_data_loader import load_subject, landmark_pairs_um
from r1_coarse_align import (
    PRIOR_XY_EXPANSION,
    PRIOR_Z_EXPANSION,
    _density_map,
    _rotation_about_z,
    coarse_align,
)
from benchmark_analysis import depth_from_surface
from scipy.signal import fftconvolve


def main(sid: str):
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]

    fit = coarse_align(cz_xyz, gfp_xyz, cz_surface, hcr_surface)
    d = fit.diagnostics

    print(f"subject {sid}")
    print(f"  z_shift_um: {d['z_shift_um']:.1f}")
    print(f"  xy_shift_um (xcorr output): {d['xy_shift_um']}")
    print(f"  translation: {fit.translation}")

    # Independent prior transform for sanity.
    R = _rotation_about_z(180.0)
    scales = np.array([PRIOR_XY_EXPANSION, PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION])
    cz_mean = cz_xyz.mean(axis=0)
    cz_prior = (cz_xyz - cz_mean) @ R * scales

    # Cloud centroids.
    print(f"  CZ prior-frame centroid: {cz_prior.mean(0)} (should be 0)")
    print(f"  HCR GFP+ centroid:       {gfp_xyz.mean(0)}")
    print(f"  naive tx from centroids: {gfp_xyz.mean(0)[:2] - cz_prior.mean(0)[:2]}")

    # Landmark-based true translation target.
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    cz_center = cz_xyz.mean(0)
    src_mean = cz_lm.mean(0)
    dst_mean = hcr_lm.mean(0)
    # Use GT R = approx 180 about z, scales = gt scales from Procrustes.
    gt = info["procrustes"]
    gt_target = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean
    print(f"  GT prediction of CZ center in HCR: {gt_target}")

    # Peek at the xcorr map shape and peak.
    xc2d_shape = d.get("hcr_map_shape", None), d.get("cz_map_shape", None)
    print(f"  cz_map shape: {d['cz_map_shape']}, hcr_map shape: {d['hcr_map_shape']}")
    print(f"  xc_2d_peak idx: {d['xc_2d_peak']}")
    print(f"  n_cz_in_band: {d['n_cz_in_band']}, n_hcr_in_band: {d['n_hcr_in_band']}")

    # Recompute the maps to inspect the HCR density pattern.
    from r1_coarse_align import _density_map
    cz_depth_native = depth_from_surface(cz_xyz, cz_surface)
    hcr_depth = depth_from_surface(gfp_xyz, hcr_surface)
    cz_depth_scaled = cz_depth_native * PRIOR_Z_EXPANSION
    depth_band = (0.0, 800.0)
    cz_band = (cz_depth_scaled + d['z_shift_um'] >= depth_band[0]) & (
        cz_depth_scaled + d['z_shift_um'] <= depth_band[1]
    )
    hcr_band = (hcr_depth >= depth_band[0]) & (hcr_depth <= depth_band[1])
    cz_xy_prior = cz_prior[cz_band, :2]
    hcr_xy = gfp_xyz[hcr_band, :2]

    xy_bin = 20.0
    margin = 300.0
    hcr_x_edges = np.arange(hcr_xy[:, 0].min() - margin,
                             hcr_xy[:, 0].max() + margin + xy_bin, xy_bin)
    hcr_y_edges = np.arange(hcr_xy[:, 1].min() - margin,
                             hcr_xy[:, 1].max() + margin + xy_bin, xy_bin)
    cz_x_edges = np.arange(cz_xy_prior[:, 0].min() - margin,
                            cz_xy_prior[:, 0].max() + margin + xy_bin, xy_bin)
    cz_y_edges = np.arange(cz_xy_prior[:, 1].min() - margin,
                            cz_xy_prior[:, 1].max() + margin + xy_bin, xy_bin)

    hcr_map = _density_map(hcr_xy, hcr_x_edges, hcr_y_edges, 30.0 / xy_bin)
    cz_map = _density_map(cz_xy_prior, cz_x_edges, cz_y_edges, 30.0 / xy_bin)

    # Density along x and y for HCR
    hcr_x_marginal = hcr_map.sum(axis=1)
    hcr_y_marginal = hcr_map.sum(axis=0)
    x_centers = 0.5 * (hcr_x_edges[:-1] + hcr_x_edges[1:])
    y_centers = 0.5 * (hcr_y_edges[:-1] + hcr_y_edges[1:])
    print(f"  HCR x marginal peak at {x_centers[np.argmax(hcr_x_marginal)]:.0f} "
          f"(centroid={np.average(x_centers, weights=hcr_x_marginal):.0f})")
    print(f"  HCR y marginal peak at {y_centers[np.argmax(hcr_y_marginal)]:.0f} "
          f"(centroid={np.average(y_centers, weights=hcr_y_marginal):.0f})")
    print(f"  HCR map min/max/mean: {hcr_map.min():.2f}/{hcr_map.max():.2f}/{hcr_map.mean():.2f}")

    # Percentile density in sliding window of CZ size
    from scipy.signal import fftconvolve
    xc = fftconvolve(hcr_map, cz_map[::-1, ::-1], mode='valid')
    peak = np.unravel_index(int(np.argmax(xc)), xc.shape)
    # Convert peak to world CZ-center location
    cz_center_bin = (len(cz_x_edges) - 1) // 2
    world_x = hcr_x_edges[0] + (peak[0] + cz_center_bin) * xy_bin
    world_y = hcr_y_edges[0] + (peak[1] + cz_center_bin) * xy_bin
    print(f"  sliding-window peak places CZ center at ({world_x:.0f}, {world_y:.0f})")
    # Compare to xc top few peaks
    flat = xc.ravel()
    top_k = 5
    top_idx = np.argsort(flat)[-top_k:][::-1]
    print(f"  top-{top_k} xcorr peaks:")
    for idx in top_idx:
        p = np.unravel_index(int(idx), xc.shape)
        wx = hcr_x_edges[0] + (p[0] + cz_center_bin) * xy_bin
        wy = hcr_y_edges[0] + (p[1] + cz_center_bin) * xy_bin
        print(f"    val={flat[idx]:.1f} at ({wx:.0f}, {wy:.0f})")


if __name__ == "__main__":
    for sid in sys.argv[1:] or ["790322"]:
        main(sid)
