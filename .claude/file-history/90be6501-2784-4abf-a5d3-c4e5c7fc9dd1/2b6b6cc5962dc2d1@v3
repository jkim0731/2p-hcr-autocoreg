"""M1 — 3D density-NCC coarse affine.

Revised: use CZ centroids (all) and HCR-GFP+ centroids rendered as
Gaussian-smoothed 3D density maps, then NCC-search on
(sxy, sz, t_z, t_y, t_x) after applying the 180° XY rotation prior.

The centroid-density representation was found (in the first pass of
M1, NCC peak 0.046) to be strictly stronger than comparing an outlined
single-blob CZ mask against a sparse HCR label mask.

Binding rule: scale grid ranges come from the plan's stated bounds
(no benchmark-tuned hyperparameters).

Output
------
`TransformDescriptor` (R=180°-XY, S=(sz, sxy, sxy), t=µm)
with intrinsic confidence = robust-z of the NCC peak over the whole
search surface.  Emits graceful-degradation (empty) if robust-z < 3.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.centroid_helpers import centroids_um


def _density3d(pts_um: np.ndarray, origin: np.ndarray, extent: np.ndarray,
               spacing_um: float, sigma_um: float) -> np.ndarray:
    """Histogram + Gaussian smooth a 3D cloud to a dense volume.

    ``pts_um`` is (N, 3) in (z, y, x) µm; ``origin`` and ``extent`` define
    the axis-aligned bbox (µm).
    """
    nz = max(1, int(np.ceil(extent[0] / spacing_um)))
    ny = max(1, int(np.ceil(extent[1] / spacing_um)))
    nx = max(1, int(np.ceil(extent[2] / spacing_um)))
    idx = np.floor((pts_um - origin) / spacing_um).astype(int)
    mask = ((idx[:, 0] >= 0) & (idx[:, 0] < nz)
            & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < nx))
    idx = idx[mask]
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    if sigma_um > 0:
        vol = gaussian_filter(vol, sigma=sigma_um / spacing_um)
    return vol


def _ncc3d_valid(image: np.ndarray, template: np.ndarray
                 ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """3D NCC with per-window local-variance normalisation (3D extension of
    Lewis 1995).  Returns (ncc_valid_map, best_offset_in_map).
    """
    I = image.astype(np.float32)
    T = template.astype(np.float32)
    D, H, W = I.shape
    d, h, w = T.shape
    if D < d or H < h or W < w:
        return np.array([[[-np.inf]]]), (0, 0, 0)
    T_zm = T - T.mean()
    T_norm = float(np.sqrt((T_zm ** 2).sum()))
    if T_norm <= 0:
        return np.zeros((D - d + 1, H - h + 1, W - w + 1)), (0, 0, 0)

    # 3D integral images
    I_sum = np.zeros((D + 1, H + 1, W + 1), dtype=np.float64)
    I2_sum = np.zeros((D + 1, H + 1, W + 1), dtype=np.float64)
    I_sum[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(I, axis=0), axis=1), axis=2)
    I2_sum[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(I * I, axis=0), axis=1), axis=2)

    def win(a_sum: np.ndarray) -> np.ndarray:
        # inclusion-exclusion over a (d, h, w) window
        return (a_sum[d:,    h:,    w:]
                - a_sum[:-d, h:,    w:]
                - a_sum[d:,    :-h, w:]
                - a_sum[d:,    h:,    :-w]
                + a_sum[:-d, :-h, w:]
                + a_sum[:-d, h:,    :-w]
                + a_sum[d:,    :-h, :-w]
                - a_sum[:-d, :-h, :-w])
    wsum = win(I_sum)
    w2sum = win(I2_sum)
    n = float(d * h * w)
    wmean = wsum / n
    wvar = np.clip(w2sum - n * wmean * wmean, 0.0, None)
    wstd = np.sqrt(wvar)

    num = fftconvolve(I, T_zm[::-1, ::-1, ::-1], mode="valid")
    denom = wstd * T_norm
    ncc = np.zeros_like(num)
    good = denom > 1e-9
    ncc[good] = num[good] / denom[good]
    ijk = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    return ncc, (int(ijk[0]), int(ijk[1]), int(ijk[2]))


def _sweep_ncc(hcr_vol, cz_rot, cz_c, hcr_lo, sxy_grid, sz_grid,
               spacing_um, sigma_um):
    """Run NCC sweep over a (sxy, sz) grid.

    Returns ``(peaks_list, best_dict)``.  The ranking metric is a per-scale
    z-score ``(peak - ncc_mean) / ncc_std`` computed over that scale's
    entire NCC map, which makes peaks comparable across scales (smaller
    scales have larger search volumes → higher raw max even for pure noise,
    so raw peak alone systematically biases toward small scales).
    """
    all_peaks = []
    best = dict(zscore=-np.inf, ncc=-np.inf, sxy=None, sz=None, t=None)
    for sxy in sxy_grid:
        for sz in sz_grid:
            S = np.array([sz, sxy, sxy])
            cz_scaled = cz_rot * S
            lo = cz_scaled.min(0) - 2 * sigma_um
            hi = cz_scaled.max(0) + 2 * sigma_um
            t_vol = _density3d(cz_scaled, lo, hi - lo, spacing_um, sigma_um)
            if any(t_vol.shape[i] >= hcr_vol.shape[i] for i in range(3)):
                continue
            ncc, off = _ncc3d_valid(hcr_vol, t_vol)
            if ncc.size == 0 or not np.isfinite(ncc).any():
                continue
            peak = float(ncc.max())
            # Per-scale z-score for cross-scale comparison
            vals = ncc[np.isfinite(ncc)]
            mu = float(vals.mean())
            sd = float(vals.std())
            zscore = (peak - mu) / sd if sd > 1e-9 else 0.0
            all_peaks.append((float(sxy), float(sz), peak, zscore))
            if zscore > best["zscore"]:
                top_lo_um = hcr_lo + np.array(off) * spacing_um
                tpl_c_um = top_lo_um + (np.array(t_vol.shape) * spacing_um) * 0.5
                best = dict(zscore=zscore, ncc=peak, sxy=float(sxy), sz=float(sz),
                            t=tpl_c_um, tpl_shape=tuple(t_vol.shape),
                            off=tuple(int(x) for x in off))
    return all_peaks, best


@register_candidate("M1")
def run_m1(s, *, spacing_um=20.0, sigma_xy_um=40.0, sigma_z_um=60.0,
           # Widened (S31) scale grid: covers 1.4..2.2 xy × 1.8..3.8 z
           sxy_grid=(1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2),
           sz_grid=(1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8),
           margin_um=50.0, min_robust_z=3.0,
           refine: bool = True) -> CoregResult:
    """M1 widened-scale variant (S31).

    Wider grid + optional fine-scale refinement around the coarse peak.
    Anisotropic Gaussian smoothing: sigma_z = 1.5 × sigma_xy because the HCR
    density is sparser in Z (no benchmark-tuning — this is a basic-geometry
    prior from the acquisition voxel size).  The smoothing is still applied
    isotropically in index space but on an anisotropic-sigma schedule.
    """
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    cz_c = cz_um.mean(0)
    hcr_lo = hcr_um.min(0) - margin_um
    hcr_hi = hcr_um.max(0) + margin_um

    # For simplicity (avoid anisotropic-sigma coding inside _density3d),
    # use the geometric mean of the sigmas for the 3D Gaussian.
    sigma_um = float(np.sqrt(sigma_xy_um * sigma_z_um))
    hcr_vol = _density3d(hcr_um, hcr_lo, hcr_hi - hcr_lo, spacing_um, sigma_um)

    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    cz_rot = (cz_um - cz_c) @ R0.T

    # Coarse sweep
    coarse_peaks, coarse_best = _sweep_ncc(
        hcr_vol, cz_rot, cz_c, hcr_lo, sxy_grid, sz_grid,
        spacing_um, sigma_um)

    if coarse_best["sxy"] is None:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"error": "no scale pair fits in HCR"})

    all_peaks = list(coarse_peaks)
    best = dict(coarse_best)

    # Fine-scale refinement around the best coarse peak.
    if refine:
        sxy_step = 0.05; sz_step = 0.1
        fine_sxy = [best["sxy"] + i * sxy_step for i in (-2, -1, 0, 1, 2)]
        fine_sz = [best["sz"] + i * sz_step for i in (-2, -1, 0, 1, 2)]
        fine_sxy = [max(0.5, s_) for s_ in fine_sxy]
        fine_sz = [max(0.5, s_) for s_ in fine_sz]
        # drop duplicates with the coarse grid
        fine_sxy = [s_ for s_ in fine_sxy if not any(abs(s_ - cs) < 1e-3 for cs in sxy_grid)]
        fine_sz = [s_ for s_ in fine_sz if not any(abs(s_ - cs) < 1e-3 for cs in sz_grid)]
        fine_peaks, fine_best = _sweep_ncc(
            hcr_vol, cz_rot, cz_c, hcr_lo, fine_sxy, fine_sz,
            spacing_um, sigma_um)
        all_peaks.extend(fine_peaks)
        if fine_best["ncc"] > best["ncc"]:
            best = fine_best

    # Report both the raw-NCC robust-z (may be biased by scale) and the
    # per-scale z-score actually used for ranking.
    peaks_arr = np.array([p for _, _, p, _ in all_peaks], dtype=float)
    zscores_arr = np.array([z for _, _, _, z in all_peaks], dtype=float)
    med = float(np.median(peaks_arr))
    q25 = float(np.quantile(peaks_arr, 0.25))
    q75 = float(np.quantile(peaks_arr, 0.75))
    iqr = max(q75 - q25, 1e-6)
    robust_z = (best["ncc"] - med) / (iqr / 1.349)
    zscore_med = float(np.median(zscores_arr))
    zscore_q75 = float(np.quantile(zscores_arr, 0.75))

    S = np.array([best["sz"], best["sxy"], best["sxy"]])
    transform = TransformDescriptor(
        R=R0, scales=S, translation=np.asarray(best["t"]),
        src_mean=cz_c, rotation_deg_z=180.0, kind="density-ncc",
    )

    return CoregResult(
        pairs_df=pd.DataFrame(),
        confidence=float(best["zscore"]),
        transform=transform,
        diagnostics=dict(
            best_ncc=best["ncc"], best_zscore=float(best["zscore"]),
            sxy=best["sxy"], sz=best["sz"],
            off=list(best["off"]), t_um=list(map(float, best["t"])),
            tpl_shape=list(best["tpl_shape"]),
            robust_z=float(robust_z),
            n_peaks=int(len(all_peaks)),
            peak_median=med, peak_q25=q25, peak_q75=q75,
            zscore_median=zscore_med, zscore_q75=zscore_q75,
            spacing_um=spacing_um, sigma_um=sigma_um,
            sxy_grid=list(map(float, sxy_grid)),
            sz_grid=list(map(float, sz_grid)),
            refined=bool(refine),
        ),
    )
