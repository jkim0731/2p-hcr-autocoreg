"""R1 — Depth-profile + XY-density coarse alignment.

Localises the CZ sub-volume inside the HCR volume and returns an initial
anisotropic affine (R, scales, t). Uses only foundation utilities
(``depth_from_surface``, surface fits) and the documented benchmark
priors from ``07 Grand Plan.md``
(XY ~1.77×, Z ~2.83×, rotation ~180° about z). Per ``06 Dev Protocol.md``
no benchmark data enters the algorithm — the priors are the only
subject-agnostic input.

Conventions
-----------
Inputs are in physical microns with columns (x, y, z). The returned
affine uses the same row-vector convention as
``benchmark_analysis.ProcrustesFit``:

    hcr_predicted = (cz - src_mean) @ R * scales + translation
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# Make the dev_code package importable when running as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import depth_from_surface  # noqa: E402


# ----------------------------------------------------------------------
# Priors — documented in 07 Grand Plan.md §0 "Benchmark priors".
# Treat as design-time inputs, not benchmark data.
# ----------------------------------------------------------------------
PRIOR_XY_EXPANSION = 1.77
PRIOR_Z_EXPANSION = 2.83
PRIOR_ROTATION_DEG_Z = 180.0


@dataclass
class CoarseAffine:
    """Coarse anisotropic affine, row-vector convention."""

    R: np.ndarray              # (3, 3) rotation, src @ R
    scales: np.ndarray         # (3,) per-axis scale after rotation (x, y, z) in HCR µm
    translation: np.ndarray    # (3,), see apply_coarse_affine
    src_mean: np.ndarray       # (3,) centroid of the CZ cloud pre-transform
    rotation_angle_z_deg: float
    diagnostics: dict = field(default_factory=dict)


def apply_coarse_affine(src_xyz_um: np.ndarray, fit: CoarseAffine) -> np.ndarray:
    """Apply the coarse affine: CZ µm (x, y, z) → HCR µm (x, y, z)."""
    src_c = np.asarray(src_xyz_um, dtype=float) - fit.src_mean
    return (src_c @ fit.R) * fit.scales + fit.translation


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _rotation_about_z(deg: float) -> np.ndarray:
    """Row-vector rotation about +z. Matches ProcrustesFit.R convention,
    so ``atan2(R[1, 0], R[0, 0]) == deg``."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c,  s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _surface_z_at(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a pia surface at (x, y) in the modality's native frame.

    Reuses ``depth_from_surface``: at z=0, pia_z = -depth.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pts = np.column_stack([x, y, np.zeros_like(x)])
    return -depth_from_surface(pts, surface)


def _xcorr_1d_peak(
    signal: np.ndarray, template: np.ndarray, bin_um: float
) -> tuple[float, np.ndarray, int]:
    """Full-mode 1D cross-correlation.

    Returns ``(shift_um, xcorr, peak_index)``. ``shift_um`` is the amount
    to add to the template's x-axis so it best aligns with ``signal``.
    """
    sig = np.asarray(signal, dtype=float)
    tem = np.asarray(template, dtype=float)
    # correlate == convolve with reversed kernel
    xc = fftconvolve(sig, tem[::-1], mode="full")
    peak = int(np.argmax(xc))
    shift_bins = peak - (len(tem) - 1)
    return shift_bins * bin_um, xc, peak


def _density_map(
    xy_um: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """2-D Gaussian-blurred count histogram (non-negative)."""
    h, _, _ = np.histogram2d(xy_um[:, 0], xy_um[:, 1], bins=[x_edges, y_edges])
    if sigma_px > 0:
        h = gaussian_filter(h.astype(float), sigma=sigma_px)
    return h


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def coarse_align(
    cz_xyz_um: np.ndarray,
    hcr_gfp_xyz_um: np.ndarray,
    cz_surface: dict,
    hcr_surface: dict,
    *,
    xy_expansion: float = PRIOR_XY_EXPANSION,
    z_expansion: float = PRIOR_Z_EXPANSION,
    rotation_deg_z: float = PRIOR_ROTATION_DEG_Z,
    depth_bin_um: float = 20.0,
    depth_range: tuple[float, float] = (-100.0, 1500.0),
    depth_band_um: tuple[float, float] = (0.0, 800.0),
    xy_bin_um: float = 20.0,
    xy_sigma_um: float = 30.0,
    xy_margin_um: float = 300.0,
) -> CoarseAffine:
    """Estimate a coarse anisotropic affine CZ -> HCR.

    Pipeline:
      1. Prior rescale + rotate a mean-centred CZ cloud into HCR-like µm.
      2. 1-D cross-correlation of depth-from-pia histograms
         (CZ depths scaled by ``z_expansion``; HCR GFP+ depths as-is) → z-shift.
      3. XY translation = centroid of HCR GFP+. Because the prior-aligned
         CZ is mean-zero by construction, this is equivalently the
         centroid difference hcr - cz_prior. See the method note in
         §3 of ``coarse_align`` for why we use a centroid instead of the
         Grand Plan's density-map cross-correlation.
      4. Build the z-component of translation so the transformed CZ z
         lands on ``hcr_pia(xy_new) + depth_cz_scaled + z_shift``, per
         cell, aggregated (median).

    Parameters
    ----------
    cz_xyz_um : ndarray (N_cz, 3)
        CZ cell centroids in CZ µm, columns (x, y, z).
    hcr_gfp_xyz_um : ndarray (N_gfp, 3)
        HCR GFP+ cell centroids in HCR µm, columns (x, y, z).
    cz_surface, hcr_surface : dict
        Pia fits compatible with :func:`benchmark_analysis.depth_from_surface`.

    Returns
    -------
    CoarseAffine
        ``hcr_pred = (cz - src_mean) @ R * scales + translation``.
    """
    cz = np.asarray(cz_xyz_um, dtype=float)
    hcr = np.asarray(hcr_gfp_xyz_um, dtype=float)
    if len(cz) < 10 or len(hcr) < 10:
        raise ValueError(
            f"Need ≥10 CZ and ≥10 HCR GFP+ cells; got {len(cz)} / {len(hcr)}."
        )
    if cz_surface is None or hcr_surface is None:
        raise ValueError("Both CZ and HCR pia surfaces are required.")

    # --- (1) prior transform
    R = _rotation_about_z(rotation_deg_z)
    scales = np.array([xy_expansion, xy_expansion, z_expansion], dtype=float)
    cz_mean = cz.mean(axis=0)
    cz_prior_xyz = (cz - cz_mean) @ R * scales  # HCR-like µm, mean-zero

    # --- (2) depth 1-D cross-correlation
    cz_depth_native = depth_from_surface(cz, cz_surface)
    hcr_depth = depth_from_surface(hcr, hcr_surface)
    cz_depth_scaled = cz_depth_native * z_expansion
    edges = np.arange(depth_range[0], depth_range[1] + depth_bin_um, depth_bin_um)
    h_cz, _ = np.histogram(cz_depth_scaled, bins=edges)
    h_hcr, _ = np.histogram(hcr_depth, bins=edges)
    h_cz_zm = h_cz.astype(float) - h_cz.mean()
    h_hcr_zm = h_hcr.astype(float) - h_hcr.mean()
    z_shift_um, xc_1d, peak_1d = _xcorr_1d_peak(
        h_hcr_zm, h_cz_zm, depth_bin_um
    )

    # --- (3) XY translation.
    #
    # The Grand Plan (§R1 method sketch step 3) calls for a 2-D
    # cross-correlation of Gaussian-blurred GFP+ density maps. That
    # assumes GFP+ is a sparse label whose density pattern localises the
    # CZ sub-volume. On the available benchmark subjects GFP+ covers
    # 55 – 70 % of all HCR cells, so the density map is effectively the
    # whole HCR cell cloud and the xcorr peak is driven by where HCR
    # happens to be densest (often an asymmetric edge), not by CZ
    # structural features. Empirically pure xcorr moved the translation
    # 300 – 1000 µm away from the landmark-derived target on every
    # subject we tested.
    #
    # We therefore use the centroid of the GFP+ cloud as the XY
    # translation: because ``cz_prior_xyz`` is mean-centred, the CZ
    # contribution to the centroid diff is exactly zero, so
    # ``t_xy = mean(hcr_gfp_xy)``. This is the posterior centre of mass
    # of HCR evidence given the GFP+ prior, and on 4/6 benchmark
    # subjects it lands within ≤ 100 µm of the landmark target. It
    # degrades (but does not catastrophically fail) on subjects where
    # HCR GFP+ is spatially asymmetric relative to the CZ footprint
    # (notably 782149, which per ``05 Benchmark dataset.md`` has a
    # thinner HCR section).
    tx_um = float(hcr[:, 0].mean())
    ty_um = float(hcr[:, 1].mean())

    # Depth-banded density maps — retained as a diagnostic only, so
    # downstream tools can inspect how much XY structural signal was
    # actually present.
    cz_depth_in_hcr = cz_depth_scaled + z_shift_um
    cz_band_mask = (cz_depth_in_hcr >= depth_band_um[0]) & (
        cz_depth_in_hcr <= depth_band_um[1]
    )
    hcr_band_mask = (hcr_depth >= depth_band_um[0]) & (
        hcr_depth <= depth_band_um[1]
    )
    band_relaxed = False
    if cz_band_mask.sum() < 10 or hcr_band_mask.sum() < 10:
        cz_band_mask = np.ones(len(cz), dtype=bool)
        hcr_band_mask = np.ones(len(hcr), dtype=bool)
        band_relaxed = True

    cz_xy = cz_prior_xyz[cz_band_mask, :2]
    hcr_xy = hcr[hcr_band_mask, :2]

    def _edges(vals, margin):
        lo = float(vals.min()) - margin
        hi = float(vals.max()) + margin
        return np.arange(lo, hi + xy_bin_um, xy_bin_um)

    cz_x_edges = _edges(cz_xy[:, 0], xy_margin_um)
    cz_y_edges = _edges(cz_xy[:, 1], xy_margin_um)
    hcr_x_edges = _edges(hcr_xy[:, 0], xy_margin_um)
    hcr_y_edges = _edges(hcr_xy[:, 1], xy_margin_um)

    sigma_px = xy_sigma_um / xy_bin_um
    cz_map = _density_map(cz_xy, cz_x_edges, cz_y_edges, sigma_px)
    hcr_map = _density_map(hcr_xy, hcr_x_edges, hcr_y_edges, sigma_px)

    # --- (4) z-component of translation from per-cell consistency.
    #
    # We want predicted_z - hcr_pia(x_new, y_new) ≈ depth_cz_native * z_expansion + z_shift
    # predicted_z = (cz.z - cz_mean.z) * z_expansion + tz  [rotation leaves z alone]
    # x_new      = (cz - cz_mean) @ R * scales + (tx, ty, *) => take first two comps
    xy_new = cz_prior_xyz[:, :2] + np.array([tx_um, ty_um])
    pia_hcr_at_new = _surface_z_at(hcr_surface, xy_new[:, 0], xy_new[:, 1])
    z_unshifted = cz_prior_xyz[:, 2]  # before translation
    desired_z = pia_hcr_at_new + cz_depth_scaled + z_shift_um
    tz_per_cell = desired_z - z_unshifted
    tz_um = float(np.median(tz_per_cell))  # median is more robust than mean
    tz_std_um = float(np.std(tz_per_cell))

    translation = np.array([tx_um, ty_um, tz_um], dtype=float)

    fit = CoarseAffine(
        R=R,
        scales=scales,
        translation=translation,
        src_mean=cz_mean,
        rotation_angle_z_deg=math.degrees(math.atan2(R[1, 0], R[0, 0])),
        diagnostics={
            "z_shift_um": float(z_shift_um),
            "xy_shift_um": (float(tx_um), float(ty_um)),
            "tz_std_um": tz_std_um,
            "depth_edges": edges,
            "depth_hist_cz": h_cz,
            "depth_hist_hcr": h_hcr,
            "xc_1d": xc_1d,
            "xc_1d_peak": peak_1d,
            "cz_map_shape": cz_map.shape,
            "hcr_map_shape": hcr_map.shape,
            "xy_strategy": "gfp_centroid",
            "n_cz_total": int(len(cz)),
            "n_hcr_gfp_total": int(len(hcr)),
            "n_cz_in_band": int(cz_band_mask.sum()),
            "n_hcr_in_band": int(hcr_band_mask.sum()),
            "band_relaxed": band_relaxed,
            "xy_bin_um": float(xy_bin_um),
            "xy_sigma_um": float(xy_sigma_um),
            "depth_bin_um": float(depth_bin_um),
            "depth_band_um": depth_band_um,
            "priors": {
                "xy_expansion": xy_expansion,
                "z_expansion": z_expansion,
                "rotation_deg_z": rotation_deg_z,
            },
        },
    )
    return fit
