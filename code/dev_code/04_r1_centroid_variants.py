"""Compare XY-translation strategies for R1 on benchmark subjects.

Strategies tested:
  A. Full HCR GFP+ centroid (baseline).
  B. Iterative centroid restriction (N iters) with window W.
  C. Depth-banded full HCR GFP+ centroid.
  D. 2D density xcorr (valid mode, no refinement).
  E. Phase correlation (Fourier, magnitude-normalised).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, depth_from_surface
from benchmark_data_loader import load_subject, landmark_pairs_um
from r1_coarse_align import PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION, _rotation_about_z


def iter_centroid(xy, window, n_iters=5):
    c = xy.mean(axis=0)
    for _ in range(n_iters):
        mask = (np.abs(xy[:, 0] - c[0]) < window) & (np.abs(xy[:, 1] - c[1]) < window)
        if mask.sum() < 10:
            break
        c = xy[mask].mean(axis=0)
    return c


def density_map(xy, x_edges, y_edges, sigma_px):
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    if sigma_px > 0:
        h = gaussian_filter(h.astype(float), sigma=sigma_px)
    return h


def xcorr_peak_valid(hcr_map, cz_map, hcr_x_edges, hcr_y_edges, cz_x_edges, cz_y_edges, bin_um):
    xc = fftconvolve(hcr_map, cz_map[::-1, ::-1], mode="valid")
    peak = np.unravel_index(int(np.argmax(xc)), xc.shape)
    tx = float(hcr_x_edges[0] - cz_x_edges[0] + peak[0] * bin_um)
    ty = float(hcr_y_edges[0] - cz_y_edges[0] + peak[1] * bin_um)
    return tx, ty


def phase_correlation(hcr_map, cz_map, hcr_x_edges, hcr_y_edges, cz_x_edges, cz_y_edges, bin_um):
    from numpy.fft import fft2, ifft2, fftshift
    # Pad both to same large shape.
    H = max(hcr_map.shape[0], cz_map.shape[0]) * 2
    W = max(hcr_map.shape[1], cz_map.shape[1]) * 2
    Fh = fft2(hcr_map, s=(H, W))
    Fc = fft2(cz_map, s=(H, W))
    R = Fh * np.conj(Fc)
    R /= (np.abs(R) + 1e-8)
    r = np.real(ifft2(R))
    peak = np.unravel_index(int(np.argmax(r)), r.shape)
    # Shift of CZ relative to HCR in density-map bin space.
    # Interpret peak as shift (i, j). Account for wrap-around.
    sx = peak[0] if peak[0] < H//2 else peak[0] - H
    sy = peak[1] if peak[1] < W//2 else peak[1] - W
    tx = float(hcr_x_edges[0] - cz_x_edges[0] + sx * bin_um)
    ty = float(hcr_y_edges[0] - cz_y_edges[0] + sy * bin_um)
    return tx, ty


def main(subjects):
    header = f"{'sid':>7}  {'GT tgt':>18}  {'A full':>18}  {'B iter':>18}  {'C band':>18}  {'D xcorr':>18}  {'E phase':>18}"
    print(header)
    for sid in subjects:
        s = load_subject(sid)
        info = analyze_subject(s)
        cz_xyz = info["cz_xyz"]
        gfp_xyz = info["gfp_xyz"]
        cz_surface = info["cz_surface"]
        hcr_surface = info["hcr_surface"]

        cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
        gt = info["procrustes"]
        cz_center = cz_xyz.mean(axis=0)
        src_mean = cz_lm.mean(axis=0)
        dst_mean = hcr_lm.mean(axis=0)
        gt_target = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean

        # Prior transform for CZ.
        R_prior = _rotation_about_z(180.0)
        scales_prior = np.array([PRIOR_XY_EXPANSION, PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION])
        cz_prior = (cz_xyz - cz_center) @ R_prior * scales_prior  # mean = 0

        cz_depth = depth_from_surface(cz_xyz, cz_surface) * PRIOR_Z_EXPANSION
        hcr_depth = depth_from_surface(gfp_xyz, hcr_surface)

        # Estimate z shift.
        edges = np.arange(-100, 1500 + 20, 20)
        h_cz, _ = np.histogram(cz_depth, bins=edges)
        h_hcr, _ = np.histogram(hcr_depth, bins=edges)
        h_cz_zm = h_cz.astype(float) - h_cz.mean()
        h_hcr_zm = h_hcr.astype(float) - h_hcr.mean()
        xc = fftconvolve(h_hcr_zm, h_cz_zm[::-1], mode="full")
        peak_1d = int(np.argmax(xc))
        z_shift = (peak_1d - (len(h_cz) - 1)) * 20.0
        depth_band = (0.0, 800.0)

        # A. Full HCR GFP+ centroid.
        tA = gfp_xyz[:, :2].mean(axis=0)

        # B. Iterative restriction — start full, W=700
        tB = iter_centroid(gfp_xyz[:, :2], window=700, n_iters=5)

        # C. Depth-band (0..800) centroid minus cz-prior band centroid.
        m_hcr = (hcr_depth >= depth_band[0]) & (hcr_depth <= depth_band[1])
        m_cz = (cz_depth + z_shift >= depth_band[0]) & (cz_depth + z_shift <= depth_band[1])
        tC = gfp_xyz[m_hcr, :2].mean(axis=0) - cz_prior[m_cz, :2].mean(axis=0)

        # D. 2D xcorr (depth-banded).
        cz_xy = cz_prior[m_cz, :2]
        hcr_xy = gfp_xyz[m_hcr, :2]
        margin = 300.0
        bin_um = 20.0
        sigma_px = 30.0 / bin_um
        cze_x = np.arange(cz_xy[:, 0].min() - margin, cz_xy[:, 0].max() + margin + bin_um, bin_um)
        cze_y = np.arange(cz_xy[:, 1].min() - margin, cz_xy[:, 1].max() + margin + bin_um, bin_um)
        hce_x = np.arange(hcr_xy[:, 0].min() - margin, hcr_xy[:, 0].max() + margin + bin_um, bin_um)
        hce_y = np.arange(hcr_xy[:, 1].min() - margin, hcr_xy[:, 1].max() + margin + bin_um, bin_um)
        cz_m = density_map(cz_xy, cze_x, cze_y, sigma_px)
        hcr_m = density_map(hcr_xy, hce_x, hce_y, sigma_px)
        if any(t > s for t, s in zip(cz_m.shape, hcr_m.shape)):
            tD = np.array([np.nan, np.nan])
        else:
            tD = np.array(xcorr_peak_valid(hcr_m, cz_m, hce_x, hce_y, cze_x, cze_y, bin_um))

        # E. Phase correlation (shared grid).
        all_x_min = min(cz_xy[:, 0].min(), hcr_xy[:, 0].min()) - margin
        all_x_max = max(cz_xy[:, 0].max(), hcr_xy[:, 0].max()) + margin
        all_y_min = min(cz_xy[:, 1].min(), hcr_xy[:, 1].min()) - margin
        all_y_max = max(cz_xy[:, 1].max(), hcr_xy[:, 1].max()) + margin
        # Shared edges
        shared_x = np.arange(all_x_min, all_x_max + bin_um, bin_um)
        shared_y = np.arange(all_y_min, all_y_max + bin_um, bin_um)
        cz_m2 = density_map(cz_xy, shared_x, shared_y, sigma_px)
        hcr_m2 = density_map(hcr_xy, shared_x, shared_y, sigma_px)
        tE = np.array(phase_correlation(hcr_m2, cz_m2, shared_x, shared_y, shared_x, shared_y, bin_um))

        fmt = lambda p: f"({p[0]:6.0f},{p[1]:6.0f})"
        errs = [np.linalg.norm(p - gt_target[:2]) for p in [tA, tB, tC, tD, tE]]
        print(f"{sid:>7}  {fmt(gt_target[:2])}  "
              f"{fmt(tA)}  {fmt(tB)}  {fmt(tC)}  {fmt(tD)}  {fmt(tE)}")
        print(f"     errs: A={errs[0]:.0f}  B={errs[1]:.0f}  C={errs[2]:.0f}  "
              f"D={errs[3]:.0f}  E={errs[4]:.0f}")


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["788406", "790322", "767018", "782149", "755252", "767022"]
    main(subjects)
