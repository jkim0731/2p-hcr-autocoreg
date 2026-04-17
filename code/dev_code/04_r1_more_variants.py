"""More XY-translation variants for R1.

  F. Median(xy) of HCR GFP+ (robust to outliers).
  G. Mean(xy) of HCR GFP+ restricted to depth band [0, 800].
  H. Zero-mean xcorr with tight bounds (no padding beyond cells).
  I. Mean of HCR GFP+ trimmed to central 80% per axis.
  J. Peak of HCR GFP+ KDE (mode).
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


def density_map(xy, x_edges, y_edges, sigma_px):
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    if sigma_px > 0:
        h = gaussian_filter(h.astype(float), sigma=sigma_px)
    return h


def main(subjects):
    hdr = (f"{'sid':>7}  {'GT tgt':>14}  {'A mean':>14}  "
           f"{'F median':>14}  {'G band mean':>14}  {'H zm xcorr':>14}  "
           f"{'I trim80':>14}  {'J mode':>14}")
    print(hdr)
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

        R_prior = _rotation_about_z(180.0)
        scales_prior = np.array([PRIOR_XY_EXPANSION, PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION])
        cz_prior = (cz_xyz - cz_center) @ R_prior * scales_prior

        hcr_depth = depth_from_surface(gfp_xyz, hcr_surface)
        m_hcr = (hcr_depth >= 0) & (hcr_depth <= 800)

        tA = gfp_xyz[:, :2].mean(axis=0)
        tF = np.median(gfp_xyz[:, :2], axis=0)
        tG = gfp_xyz[m_hcr, :2].mean(axis=0)

        # H: zero-mean xcorr with TIGHT bounds.
        bin_um = 20.0
        sigma_px = 30.0 / bin_um
        cz_xy = cz_prior[:, :2]
        hcr_xy = gfp_xyz[m_hcr, :2]

        margin = 20.0  # barely any padding
        cze_x = np.arange(cz_xy[:, 0].min(), cz_xy[:, 0].max() + bin_um, bin_um)
        cze_y = np.arange(cz_xy[:, 1].min(), cz_xy[:, 1].max() + bin_um, bin_um)
        hce_x = np.arange(hcr_xy[:, 0].min(), hcr_xy[:, 0].max() + bin_um, bin_um)
        hce_y = np.arange(hcr_xy[:, 1].min(), hcr_xy[:, 1].max() + bin_um, bin_um)

        cz_m = density_map(cz_xy, cze_x, cze_y, sigma_px)
        hcr_m = density_map(hcr_xy, hce_x, hce_y, sigma_px)

        if any(t > s for t, s in zip(cz_m.shape, hcr_m.shape)):
            tH = np.array([np.nan, np.nan])
        else:
            cz_zm = cz_m - cz_m.mean()
            hcr_zm = hcr_m - hcr_m.mean()
            xc = fftconvolve(hcr_zm, cz_zm[::-1, ::-1], mode="valid")
            peak = np.unravel_index(int(np.argmax(xc)), xc.shape)
            off_x = hce_x[0] - cze_x[0]
            off_y = hce_y[0] - cze_y[0]
            tH = np.array([off_x + peak[0] * bin_um, off_y + peak[1] * bin_um])

        # I: trimmed mean (central 80% per axis).
        def trim_mean(xy, lo_pct=10, hi_pct=90):
            mx = (xy[:, 0] >= np.percentile(xy[:, 0], lo_pct)) & \
                 (xy[:, 0] <= np.percentile(xy[:, 0], hi_pct))
            my = (xy[:, 1] >= np.percentile(xy[:, 1], lo_pct)) & \
                 (xy[:, 1] <= np.percentile(xy[:, 1], hi_pct))
            return xy[mx & my].mean(axis=0)
        tI = trim_mean(gfp_xyz[:, :2], 10, 90)

        # J: mode of density map (smoothed).
        tight_x = np.arange(gfp_xyz[:, 0].min(), gfp_xyz[:, 0].max() + bin_um, bin_um)
        tight_y = np.arange(gfp_xyz[:, 1].min(), gfp_xyz[:, 1].max() + bin_um, bin_um)
        fm = density_map(gfp_xyz[:, :2], tight_x, tight_y, sigma_px=60.0/bin_um)  # bigger blur
        pk = np.unravel_index(int(np.argmax(fm)), fm.shape)
        tJ = np.array([0.5*(tight_x[pk[0]] + tight_x[pk[0]+1]),
                       0.5*(tight_y[pk[1]] + tight_y[pk[1]+1])])

        fmt = lambda p: f"({p[0]:5.0f},{p[1]:5.0f})"
        errs = [np.linalg.norm(p - gt_target[:2]) for p in [tA, tF, tG, tH, tI, tJ]]
        print(f"{sid:>7}  {fmt(gt_target[:2])}  "
              f"{fmt(tA)}  {fmt(tF)}  {fmt(tG)}  {fmt(tH)}  {fmt(tI)}  {fmt(tJ)}")
        print(f"     errs: A={errs[0]:.0f}  F={errs[1]:.0f}  G={errs[2]:.0f}  "
              f"H={errs[3]:.0f}  I={errs[4]:.0f}  J={errs[5]:.0f}")


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["788406", "790322", "767018", "782149", "755252", "767022"]
    main(subjects)
