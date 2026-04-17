"""Test hybrid XY strategies: band+mean, band+median, robust variants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, depth_from_surface
from benchmark_data_loader import load_subject, landmark_pairs_um
from r1_coarse_align import PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION, _rotation_about_z


def main(subjects):
    print(f"{'sid':>7}  {'GT tgt':>14}  "
          f"{'full mean':>12}  {'full med':>12}  {'band mean':>12}  "
          f"{'band med':>12}  {'band mean-dcz':>14}")
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
        cz_depth_scaled = depth_from_surface(cz_xyz, cz_surface) * PRIOR_Z_EXPANSION

        # Pretend z_shift from 1D xcorr — use 0 for simplicity; in reality z_shift matters.
        from scipy.signal import fftconvolve
        edges = np.arange(-100, 1500+20, 20)
        h_cz, _ = np.histogram(cz_depth_scaled, bins=edges)
        h_hcr, _ = np.histogram(hcr_depth, bins=edges)
        xc = fftconvolve(h_hcr - h_hcr.mean(), (h_cz - h_cz.mean())[::-1], mode="full")
        z_shift = (int(np.argmax(xc)) - (len(h_cz)-1)) * 20.0

        m_hcr = (hcr_depth >= 0) & (hcr_depth <= 800)
        m_cz = ((cz_depth_scaled + z_shift) >= 0) & ((cz_depth_scaled + z_shift) <= 800)

        t_full_mean = gfp_xyz[:, :2].mean(axis=0)
        t_full_med = np.median(gfp_xyz[:, :2], axis=0)
        t_band_mean = gfp_xyz[m_hcr, :2].mean(axis=0)
        t_band_med = np.median(gfp_xyz[m_hcr, :2], axis=0)
        t_band_diff = gfp_xyz[m_hcr, :2].mean(axis=0) - cz_prior[m_cz, :2].mean(axis=0)

        fmt = lambda p: f"({p[0]:4.0f},{p[1]:4.0f})"
        errs = [np.linalg.norm(p - gt_target[:2]) for p in
                [t_full_mean, t_full_med, t_band_mean, t_band_med, t_band_diff]]
        print(f"{sid:>7}  {fmt(gt_target[:2])}  "
              f"{fmt(t_full_mean)}  {fmt(t_full_med)}  {fmt(t_band_mean)}  "
              f"{fmt(t_band_med)}  {fmt(t_band_diff)}")
        print(f"     errs: fullmean={errs[0]:.0f}  fullmed={errs[1]:.0f}  "
              f"bandmean={errs[2]:.0f}  bandmed={errs[3]:.0f}  banddiff={errs[4]:.0f}")


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["788406", "790322", "767018", "782149", "755252", "767022"]
    main(subjects)
