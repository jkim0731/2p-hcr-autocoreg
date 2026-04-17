"""Test normalized cross-correlation (Pearson NCC) for XY translation in R1."""
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


def ncc_2d(image, template):
    """Normalized cross-correlation (Pearson), 'valid' mode."""
    image = image.astype(float)
    template = template.astype(float)
    h, w = template.shape
    N = h * w

    ones = np.ones_like(template)
    image_sum = fftconvolve(image, ones, mode="valid")
    image_sq_sum = fftconvolve(image ** 2, ones, mode="valid")
    image_mean = image_sum / N
    image_var = np.clip(image_sq_sum / N - image_mean ** 2, 0, None)
    image_std = np.sqrt(image_var)

    t_zm = template - template.mean()
    t_norm = np.sqrt(np.sum(t_zm ** 2))

    raw = fftconvolve(image, t_zm[::-1, ::-1], mode="valid")
    denom = image_std * np.sqrt(N) * t_norm + 1e-8
    return raw / denom


def density_map(xy, x_edges, y_edges, sigma_px):
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    if sigma_px > 0:
        h = gaussian_filter(h.astype(float), sigma=sigma_px)
    return h


def main(subjects):
    print(f"{'sid':>7}  {'GT tgt':>18}  {'A centroid':>18}  {'NCC peak':>18}  "
          f"{'NCC+refine100':>18}  {'NCC+refine300':>18}")
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

        cz_depth = depth_from_surface(cz_xyz, cz_surface) * PRIOR_Z_EXPANSION
        hcr_depth = depth_from_surface(gfp_xyz, hcr_surface)
        depth_band = (0.0, 800.0)
        m_cz = (cz_depth >= depth_band[0]) & (cz_depth <= depth_band[1])
        m_hcr = (hcr_depth >= depth_band[0]) & (hcr_depth <= depth_band[1])

        cz_xy = cz_prior[m_cz, :2]
        hcr_xy = gfp_xyz[m_hcr, :2]

        # Full HCR GFP+ centroid (strategy A).
        tA = gfp_xyz[:, :2].mean(axis=0)

        # Build density maps on a shared rectangle that fits both.
        margin = 300.0
        bin_um = 20.0
        sigma_px = 30.0 / bin_um
        cze_x = np.arange(cz_xy[:, 0].min() - margin, cz_xy[:, 0].max() + margin + bin_um, bin_um)
        cze_y = np.arange(cz_xy[:, 1].min() - margin, cz_xy[:, 1].max() + margin + bin_um, bin_um)
        hce_x = np.arange(hcr_xy[:, 0].min() - margin, hcr_xy[:, 0].max() + margin + bin_um, bin_um)
        hce_y = np.arange(hcr_xy[:, 1].min() - margin, hcr_xy[:, 1].max() + margin + bin_um, bin_um)
        cz_map = density_map(cz_xy, cze_x, cze_y, sigma_px)
        hcr_map = density_map(hcr_xy, hce_x, hce_y, sigma_px)

        if any(t > s for t, s in zip(cz_map.shape, hcr_map.shape)):
            ncc = np.zeros((1, 1))
            tNCC = np.array([np.nan, np.nan])
            tR100 = np.array([np.nan, np.nan])
            tR300 = np.array([np.nan, np.nan])
        else:
            ncc = ncc_2d(hcr_map, cz_map)
            off_x = hce_x[0] - cze_x[0]
            off_y = hce_y[0] - cze_y[0]
            x_shifts = np.arange(ncc.shape[0]) * bin_um + off_x
            y_shifts = np.arange(ncc.shape[1]) * bin_um + off_y

            # Global NCC peak.
            pk = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
            tNCC = np.array([x_shifts[pk[0]], y_shifts[pk[1]]])

            # Refine within ±100 µm of centroid.
            def _refine(window_um):
                xm = np.abs(x_shifts - tA[0]) <= window_um
                ym = np.abs(y_shifts - tA[1]) <= window_um
                if not xm.any() or not ym.any():
                    return tA
                sub = ncc[np.ix_(xm, ym)]
                sp = np.unravel_index(int(np.argmax(sub)), sub.shape)
                i_full = np.where(xm)[0][sp[0]]
                j_full = np.where(ym)[0][sp[1]]
                return np.array([x_shifts[i_full], y_shifts[j_full]])

            tR100 = _refine(100)
            tR300 = _refine(300)

        fmt = lambda p: f"({p[0]:6.0f},{p[1]:6.0f})"
        errs = [np.linalg.norm(p - gt_target[:2]) for p in [tA, tNCC, tR100, tR300]]
        print(f"{sid:>7}  {fmt(gt_target[:2])}  "
              f"{fmt(tA)}  {fmt(tNCC)}  {fmt(tR100)}  {fmt(tR300)}")
        print(f"     errs: A={errs[0]:.0f}  NCC={errs[1]:.0f}  R100={errs[2]:.0f}  R300={errs[3]:.0f}")


if __name__ == "__main__":
    subjects = sys.argv[1:] or ["788406", "790322", "767018", "782149", "755252", "767022"]
    main(subjects)
