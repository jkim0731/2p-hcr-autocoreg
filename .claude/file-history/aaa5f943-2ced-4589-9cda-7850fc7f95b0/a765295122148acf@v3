"""Stage B-Image — sz estimator via 1-D z-profile NCC sweep on the locked frame.

For a given subject, sweep ``sz ∈ [1.5, 4.0]`` at 0.05 step.  At each
candidate sz, warp the CZ raw 488 stack into the HCR µm frame using the
Stage A locked affine plus that sz, then compute the foreground-fraction
1-D z profile of both warped CZ and HCR raw 488 inside the
``surface_registration_v2`` ``crop_bbox`` and measure their Pearson NCC
with a sub-pixel z shift refinement (±100 µm).  Pick the sz that
maximises NCC.

Why foreground-fraction rather than voxel intensity?
Voxel-NCC is dominated by the gross intensity mismatch — HCR 488 raw is
sparse GFP+ cell bodies on a dark background, while CZ 488 raw is a
dense, structured GCaMP/autofluorescence signal.  Their pixel intensity
distributions almost never correlate.  The *spatial density* of bright
regions vs depth, however, does encode the depth-extent of the sample
and is what the sz parameter controls — so we threshold each volume at
its own p90 and correlate the z-profile of foreground fraction.

Pass criterion (per docs/09 §2 Stage B): unimodal peak with peak/median
≥ 1.10 and half-width at "half-elevation above median" ≤ 0.30 in sz
units (where elevation = peak − median).  Without subtracting the
baseline the HWHM of foreground-fraction NCC is always huge (whole
sweep) because ncc(sz) is everywhere > 0.5.

Hard stop on failure — no fallback.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

_THIS = Path(__file__).resolve().parent
_V2 = _THIS.parent
_CODE = _V2.parent
_V1 = _CODE / "full_automatic_execution_01"
for p in (_V1 / "lib", _CODE / "dev_code", _V2 / "lib",
          _CODE / "sessions" / "03c_onset_features" / "iterations"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from benchmark_analysis import depth_from_surface, load_hcr_volume  # noqa: E402
from iter08_cz_prior import load_cz_volume  # noqa: E402
from locked_prior_warm import (  # noqa: E402
    LockedPriorWarmStart,
    compute_locked_prior_warm_start,
)
from surface_registration_v2 import get_surface_registration  # noqa: E402
from surfaces_iter08 import get_cz_surface_iter08  # noqa: E402

SZ_LO = 1.5
SZ_HI = 4.0
SZ_STEP = 0.05
TZ_SEARCH_HALF_UM = 100.0   # ± window around per-sz tz centre (iter 4)
TZ_PAD_UM = 200.0           # extra HCR z padding beyond computed bounds
HCR_LEVEL = 4
FG_PERCENTILE = 90          # foreground threshold for fg_z scoring
SPOT_PERCENTILE = 90        # binary-spot threshold for spot_mask_3d scoring
DOG_SIGMA_SMALL = 1.5       # HCR-vox; small Gaussian for DoG (~1.5 µm)
DOG_SIGMA_LARGE = 6.0       # HCR-vox; large Gaussian for DoG (~6 µm cell-body)
SMOOTH_SIGMA = 5.0          # HCR-vox; cell-cluster scale low-pass for smoothed_voxel
PEAK_RATIO_MIN = 1.10
HALF_WIDTH_MAX = 0.30


@dataclass
class SzSweepResult:
    subject_id: str
    sz_grid: np.ndarray
    ncc_grid: np.ndarray
    sz_lp: float
    sz_peak: float | None
    ncc_peak: float | None
    ncc_median: float
    peak_ratio: float
    half_width: float
    passed: bool
    fail_reason: str
    tz_offset_um: float | None
    crop_bbox_um: tuple[float, float, float, float]
    diagnostics: dict


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson NCC of two same-shape arrays. NaN-safe via masked mean."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    av = a[m] - a[m].mean()
    bv = b[m] - b[m].mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if den < 1e-12:
        return float("nan")
    return float((av * bv).sum() / den)


def _build_affine(
    lp: LockedPriorWarmStart, sz: float, tz_offset_um: float,
    cz_xy_um: float, cz_z_um: float,
    hcr_xy_um: float, hcr_z_um: float,
):
    """Inverse map from HCR-output-pixel-index → CZ-input-pixel-index.

    Forward LP (zyx-µm convention): pred = ((cz - src_mean) * scales) @ R.T
                                          + translation
    => col-vec form: pred_col = R diag(scales) (cz_col - src_mean_col)
                              + translation_col
    Inverse: cz_col = diag(1/scales) R.T (pred_col - translation_col)
                    + src_mean_col

    With the candidate ``sz`` overriding lp.scales[0] and a tz offset
    on lp.translation[0], plus pixel-↔-µm conversions:

        D_hcr = diag(hcr_z_um, hcr_xy_um, hcr_xy_um)
        D_cz_inv = diag(1/cz_z_um, 1/cz_xy_um, 1/cz_xy_um)
        scales' = (sz, lp.sxy, lp.sxy)
        translation' = (lp.tz + tz_offset, lp.ty, lp.tx)

        A = D_cz_inv diag(1/scales') R.T D_hcr
        b = D_cz_inv (src_mean - diag(1/scales') R.T translation')
    """
    R = lp.R
    src_mean = lp.src_mean
    scales = np.array([sz, lp.scales[1], lp.scales[2]], dtype=float)
    translation = lp.translation.copy()
    translation[0] = translation[0] + tz_offset_um

    D_hcr = np.diag([hcr_z_um, hcr_xy_um, hcr_xy_um])
    D_cz_inv = np.diag([1.0 / cz_z_um, 1.0 / cz_xy_um, 1.0 / cz_xy_um])
    inv_scales = 1.0 / scales

    A = D_cz_inv @ np.diag(inv_scales) @ R.T @ D_hcr
    b_um = src_mean - np.diag(inv_scales) @ R.T @ translation
    b = D_cz_inv @ b_um
    return A, b


def _warp_cz_into_hcr_crop(
    cz_vol: np.ndarray,
    lp: LockedPriorWarmStart, sz: float, tz_offset_um: float,
    cz_xy_um: float, cz_z_um: float,
    hcr_xy_um: float, hcr_z_um: float,
    crop_bbox_px: tuple[int, int, int, int],
    z_lo_um: float, z_hi_um: float,
) -> np.ndarray:
    """Warp CZ raw onto the cropped HCR slab grid for given (sz, tz_offset).

    crop_bbox_px = (y0, y1, x0, x1) in level-4 HCR pixels.
    z_lo_um, z_hi_um = HCR µm range to render.
    Returns an array of shape (Z_out, Y_out, X_out) in HCR-level-4 pixel
    grid (xy from crop_bbox, z from [z_lo_um, z_hi_um] @ hcr_z_um).
    """
    A, b_root = _build_affine(
        lp, sz, tz_offset_um, cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um
    )

    y0, y1, x0, x1 = crop_bbox_px
    z0_idx = int(np.floor(z_lo_um / hcr_z_um))
    z1_idx = int(np.ceil(z_hi_um / hcr_z_um))
    out_shape = (z1_idx - z0_idx, y1 - y0, x1 - x0)

    # The output index in this cropped frame is (z_local, y_local, x_local);
    # the absolute HCR-level-4 pixel is (z0_idx + z_local, y0 + y_local,
    # x0 + x_local). Shift offset accordingly.
    crop_origin = np.array([z0_idx, y0, x0], dtype=float)
    b = b_root + A @ crop_origin

    warped = affine_transform(
        cz_vol, A, offset=b, output_shape=out_shape,
        order=1, mode="constant", cval=0.0,
    )
    return warped


def _foreground_z_profile(vol: np.ndarray, percentile: float = FG_PERCENTILE,
                          ignore_zero: bool = False) -> np.ndarray:
    """Per-z foreground fraction (fraction of voxels above per-volume pXX).

    For a warped CZ volume the zero-padded outside should not pull the
    threshold down — pass ``ignore_zero=True`` to compute the percentile
    only over voxels that are actually inside the CZ field of view.
    """
    if ignore_zero:
        nz = vol > 0.5  # warped CZ pads with 0; tiny noise excluded
        if nz.sum() < 100:
            return np.zeros(vol.shape[0], dtype=np.float32)
        thr = float(np.percentile(vol[nz], percentile))
    else:
        thr = float(np.percentile(vol, percentile))
    return (vol > thr).mean(axis=(1, 2)).astype(np.float32)


def _dog_spot_mask(vol: np.ndarray, sigma_small: float = 1.0,
                   sigma_large: float = 4.0,
                   percentile: float = 99.0,
                   ignore_zero: bool = False) -> np.ndarray:
    """Approximate spot mask via Difference-of-Gaussians, threshold at pXX.

    DoG suppresses the slowly-varying intensity bias and emphasises
    cell-body-scale bright structures.  Threshold at p99 keeps only the
    brightest spots (sparse mask).
    """
    small = gaussian_filter(vol, sigma=sigma_small)
    large = gaussian_filter(vol, sigma=sigma_large)
    dog = small - large
    if ignore_zero:
        nz = vol > 0.5
        if nz.sum() < 100:
            return np.zeros_like(vol, dtype=bool)
        thr = float(np.percentile(dog[nz], percentile))
    else:
        thr = float(np.percentile(dog, percentile))
    return (dog > thr) & (vol > 0)  # zero-padded warp pixels excluded


def _smoothed_voxel_ncc(
    warped: np.ndarray, hcr_smoothed: np.ndarray,
    sigma: float, max_shift: int,
) -> tuple[float, int]:
    """Pearson NCC of Gaussian-smoothed intensities in CZ FOV, with z-shift.

    HCR is pre-smoothed (passed in). CZ is smoothed inside the warped block.
    Smoothing at cell-cluster scale (sigma≈5–10 µm) allows two modalities
    that don't agree at the single-cell level to still co-localise at the
    region level.
    """
    nz = warped > 0.5
    if nz.sum() < 100:
        return float("nan"), 0
    cz_smoothed = gaussian_filter(warped, sigma=sigma).astype(np.float32)
    Z = warped.shape[0]
    best_ncc = -np.inf
    best_shift = 0
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            cz_s = cz_smoothed[sh:Z]
            hcr_s = hcr_smoothed[: Z - sh]
            mask_s = nz[sh:Z]
        else:
            cz_s = cz_smoothed[: Z + sh]
            hcr_s = hcr_smoothed[-sh:Z]
            mask_s = nz[: Z + sh]
        if int(mask_s.sum()) < 100:
            continue
        a = cz_s[mask_s]
        b = hcr_s[mask_s]
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den < 1e-12:
            continue
        v = float((a * b).sum() / den)
        if v > best_ncc:
            best_ncc = v
            best_shift = sh
    if not np.isfinite(best_ncc):
        return float("nan"), 0
    return float(best_ncc), best_shift


def _spot_mask_3d_ncc(
    warped: np.ndarray, hcr_target: np.ndarray,
    percentile: float, max_shift: int,
) -> tuple[float, int]:
    """Pearson NCC of binary spot masks ((vol > pXX)) inside CZ FOV, with z-shift search.

    Both volumes are thresholded at their pXX (HCR over the full slab; CZ
    over the in-FOV voxels only — zero padding is excluded). The NCC is
    computed over the in-FOV mask only, so out-of-CZ voxels do not pollute
    the score. The z-shift search is the analogue of the 1-D fg-NCC tz
    refinement; here it slides the binary mask volumes against each other
    in z and picks the offset maximising the spatial NCC.
    """
    nz = warped > 0.5
    if nz.sum() < 100:
        return float("nan"), 0
    cz_thr = float(np.percentile(warped[nz], percentile))
    hcr_thr = float(np.percentile(hcr_target, percentile))
    cz_m = (warped > cz_thr).astype(np.float32)
    hcr_m = (hcr_target > hcr_thr).astype(np.float32)
    Z = warped.shape[0]
    best_ncc = -np.inf
    best_shift = 0
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            cz_s = cz_m[sh:Z]
            hcr_s = hcr_m[: Z - sh]
            mask_s = nz[sh:Z]
        else:
            cz_s = cz_m[: Z + sh]
            hcr_s = hcr_m[-sh:Z]
            mask_s = nz[: Z + sh]
        n_valid = int(mask_s.sum())
        if n_valid < 100:
            continue
        a = cz_s[mask_s]
        b = hcr_s[mask_s]
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den < 1e-12:
            continue
        v = float((a * b).sum() / den)
        if v > best_ncc:
            best_ncc = v
            best_shift = sh
    if not np.isfinite(best_ncc):
        return float("nan"), 0
    return float(best_ncc), best_shift


def _dog_3d_ncc(
    warped: np.ndarray, hcr_dog: np.ndarray,
    sigma_small: float, sigma_large: float, max_shift: int,
) -> tuple[float, int]:
    """Pearson NCC of DoG-filtered intensities inside CZ FOV, with z-shift.

    HCR DoG is precomputed (passed in). CZ DoG is computed on the warped
    block. The mask is warped > 0.5 (in-FOV).
    """
    nz = warped > 0.5
    if nz.sum() < 100:
        return float("nan"), 0
    cz_dog = (
        gaussian_filter(warped, sigma=sigma_small)
        - gaussian_filter(warped, sigma=sigma_large)
    )
    Z = warped.shape[0]
    best_ncc = -np.inf
    best_shift = 0
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            cz_s = cz_dog[sh:Z]
            hcr_s = hcr_dog[: Z - sh]
            mask_s = nz[sh:Z]
        else:
            cz_s = cz_dog[: Z + sh]
            hcr_s = hcr_dog[-sh:Z]
            mask_s = nz[: Z + sh]
        if int(mask_s.sum()) < 100:
            continue
        a = cz_s[mask_s]
        b = hcr_s[mask_s]
        a = a - a.mean()
        b = b - b.mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den < 1e-12:
            continue
        v = float((a * b).sum() / den)
        if v > best_ncc:
            best_ncc = v
            best_shift = sh
    if not np.isfinite(best_ncc):
        return float("nan"), 0
    return float(best_ncc), best_shift


def _shifted_ncc_1d(
    cz_p: np.ndarray, hcr_p: np.ndarray, max_shift: int,
) -> tuple[float, int]:
    """Best Pearson NCC of cz_p sliding over hcr_p in shifts ∈ [-max, +max].

    Returns (best_ncc, best_shift). best_shift is the index shift applied
    to cz_p (positive = cz shifted right relative to hcr).
    """
    cp = cz_p - cz_p.mean()
    hp = hcr_p - hcr_p.mean()
    N = len(cp)
    if np.linalg.norm(cp) < 1e-9 or np.linalg.norm(hp) < 1e-9:
        return float("nan"), 0
    best_corr = -np.inf
    best_shift = 0
    for sh in range(-max_shift, max_shift + 1):
        if sh >= 0:
            a, b = cp[: N - sh], hp[sh:]
        else:
            a, b = cp[-sh:], hp[: N + sh]
        if a.size < 10:
            continue
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den < 1e-12:
            continue
        c = float((a * b).sum() / den)
        if c > best_corr:
            best_corr = c
            best_shift = int(sh)
    return float(best_corr), best_shift


def estimate_sz_image_ncc(
    s,
    *,
    lp: LockedPriorWarmStart | None = None,
    reg: dict | None = None,
    cz_vol: np.ndarray | None = None,
    sz_grid: np.ndarray | None = None,
    tz_search_half_um: float = TZ_SEARCH_HALF_UM,
    scoring: str = "spot_mask_3d",
    couple_tz: bool = True,
    verbose: bool = False,
) -> SzSweepResult:
    """Sweep sz on subject ``s`` and emit a SzSweepResult.

    ``scoring`` ∈ {``fg_z``, ``spot_mask_3d``, ``dog_3d``}.

    ``couple_tz`` (iter 4): centre the per-sz tz at the LP-formula's
    natural value ``(sz - sz_lp) * cz_mean_depth`` plus a small ±100 µm
    refinement window. This removes the sz/tz trade-off that lets the
    1-D fg-z NCC peak at multiple sz values.
    """
    sid = s.subject_id
    if lp is None:
        lp = compute_locked_prior_warm_start(s)
    if reg is None:
        reg = get_surface_registration(s)

    if cz_vol is None:
        cz_vol = load_cz_volume(s).astype(np.float32, copy=False)

    hcr_vol, hcr_xy_um, hcr_z_um = load_hcr_volume(
        s, channel="488", level=HCR_LEVEL
    )
    hcr_vol = np.asarray(hcr_vol, dtype=np.float32)

    cz_xy_um = float(s.cz_xy_um)
    cz_z_um = float(s.cz_z_um)

    crop_bbox_px = tuple(reg["crop_bbox"])  # (y0, y1, x0, x1) in level-4 px

    sz_lp = float(lp.scales[0])

    # cz_mean_depth_um = depth of the LP block centroid from the CZ pia
    # surface (iter08 image-based).  Used to centre tz at each sz under
    # the LP formula tz = pia_z + sz * cz_mean_depth.
    cz_surface = get_cz_surface_iter08(s)
    cz_mean_xyz_um = lp.src_mean[[2, 1, 0]]  # zyx → xyz
    cz_mean_depth_um = float(
        depth_from_surface(cz_mean_xyz_um[None, :], cz_surface)[0]
    )

    if sz_grid is None:
        sz_grid = np.arange(SZ_LO, SZ_HI + 1e-9, SZ_STEP)

    cz_z_extent_um = cz_vol.shape[0] * cz_z_um
    if couple_tz:
        tz_center_min = (float(sz_grid.min()) - sz_lp) * cz_mean_depth_um
        tz_center_max = (float(sz_grid.max()) - sz_lp) * cz_mean_depth_um
    else:
        tz_center_min = 0.0
        tz_center_max = 0.0
    half_warped_z = float(sz_grid.max()) * cz_z_extent_um / 2

    z_lo_um = (
        lp.translation[0] + tz_center_min - tz_search_half_um
        - half_warped_z - TZ_PAD_UM
    )
    z_hi_um = (
        lp.translation[0] + tz_center_max + tz_search_half_um
        + half_warped_z + TZ_PAD_UM
    )
    z_lo_um = max(0.0, z_lo_um)
    z_hi_um = min(hcr_vol.shape[0] * hcr_z_um, z_hi_um)

    z0_idx = int(np.floor(z_lo_um / hcr_z_um))
    z1_idx = int(np.ceil(z_hi_um / hcr_z_um))
    y0, y1, x0, x1 = crop_bbox_px
    hcr_target = hcr_vol[z0_idx:z1_idx, y0:y1, x0:x1].astype(np.float32)
    if verbose:
        print(f"  {sid}: HCR target slab={hcr_target.shape} "
              f"(z {z_lo_um:.0f}-{z_hi_um:.0f} µm, "
              f"crop {y1-y0}×{x1-x0} @ {hcr_xy_um:.2f} µm/px) "
              f"sz_lp={sz_lp:.2f}, cz_depth={cz_mean_depth_um:.0f} µm",
              flush=True)

    n_steps_z_search = int(round(tz_search_half_um / hcr_z_um))

    # Pre-compute scoring-specific HCR features
    hcr_fg_zp = None
    hcr_dog = None
    hcr_smoothed = None
    if scoring == "fg_z":
        hcr_fg_zp = _foreground_z_profile(hcr_target, FG_PERCENTILE,
                                          ignore_zero=False)
    elif scoring == "dog_3d":
        hcr_dog = (
            gaussian_filter(hcr_target, sigma=DOG_SIGMA_SMALL)
            - gaussian_filter(hcr_target, sigma=DOG_SIGMA_LARGE)
        ).astype(np.float32)
    elif scoring == "smoothed_voxel":
        hcr_smoothed = gaussian_filter(
            hcr_target, sigma=SMOOTH_SIGMA
        ).astype(np.float32)

    ncc_per_sz = np.full(sz_grid.shape, np.nan, dtype=float)
    tz_per_sz = np.full(sz_grid.shape, np.nan, dtype=float)
    diag_rows = []

    for i, sz in enumerate(sz_grid):
        if couple_tz:
            tz_center_um = (float(sz) - sz_lp) * cz_mean_depth_um
        else:
            tz_center_um = 0.0

        warped = _warp_cz_into_hcr_crop(
            cz_vol, lp, float(sz), tz_offset_um=tz_center_um,
            cz_xy_um=cz_xy_um, cz_z_um=cz_z_um,
            hcr_xy_um=hcr_xy_um, hcr_z_um=hcr_z_um,
            crop_bbox_px=crop_bbox_px,
            z_lo_um=z_lo_um, z_hi_um=z_hi_um,
        )

        if scoring == "fg_z":
            cz_fg_zp = _foreground_z_profile(warped, FG_PERCENTILE,
                                             ignore_zero=True)
            if cz_fg_zp.sum() < 1e-9:
                diag_rows.append({"sz": float(sz), "skip": "no_fg"})
                continue
            ncc_val, best_shift = _shifted_ncc_1d(cz_fg_zp, hcr_fg_zp,
                                                  n_steps_z_search)
        elif scoring == "spot_mask_3d":
            ncc_val, best_shift = _spot_mask_3d_ncc(
                warped, hcr_target, SPOT_PERCENTILE, n_steps_z_search,
            )
        elif scoring == "dog_3d":
            ncc_val, best_shift = _dog_3d_ncc(
                warped, hcr_dog, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE,
                n_steps_z_search,
            )
        elif scoring == "smoothed_voxel":
            ncc_val, best_shift = _smoothed_voxel_ncc(
                warped, hcr_smoothed, SMOOTH_SIGMA, n_steps_z_search,
            )
        else:
            raise ValueError(f"unknown scoring={scoring!r}")

        # Same sign convention as iter 2/3 fg_z: shifting cz right (sh>0)
        # corresponds to "warped CZ should sit shallower" → tz decrease.
        tz_refine_um = -best_shift * hcr_z_um
        tz_total_um = tz_center_um + tz_refine_um

        ncc_per_sz[i] = ncc_val
        tz_per_sz[i] = tz_total_um
        diag_rows.append({
            "sz": float(sz),
            "tz_center_um": float(tz_center_um),
            "tz_refine_um": float(tz_refine_um),
            "tz_offset_um": float(tz_total_um),
            "ncc": float(ncc_val),
        })
        if verbose:
            print(f"  {sid}: sz={sz:.2f} tz_c={tz_center_um:+5.0f} "
                  f"tz_r={tz_refine_um:+4.0f} NCC={ncc_val:.3f}", flush=True)

    common_diag = {
        "sweep_rows": diag_rows,
        "scoring": scoring,
        "couple_tz": bool(couple_tz),
        "cz_mean_depth_um": float(cz_mean_depth_um),
        "tz_search_half_um": float(tz_search_half_um),
    }
    finite = np.isfinite(ncc_per_sz)
    if finite.sum() < 5:
        return SzSweepResult(
            subject_id=sid, sz_grid=sz_grid, ncc_grid=ncc_per_sz,
            sz_lp=float(lp.scales[0]), sz_peak=None, ncc_peak=None,
            ncc_median=float("nan"), peak_ratio=float("nan"),
            half_width=float("nan"), passed=False,
            fail_reason="not_enough_finite_ncc",
            tz_offset_um=None,
            crop_bbox_um=(
                y0 * hcr_xy_um, y1 * hcr_xy_um,
                x0 * hcr_xy_um, x1 * hcr_xy_um,
            ),
            diagnostics=common_diag,
        )

    peak_i = int(np.nanargmax(ncc_per_sz))
    sz_peak = float(sz_grid[peak_i])
    ncc_peak = float(ncc_per_sz[peak_i])
    ncc_med = float(np.nanmedian(ncc_per_sz))
    ratio = ncc_peak / ncc_med if abs(ncc_med) > 1e-9 else float("inf")

    # "Half-width at half-max" measured above the median baseline:
    # half = (peak + median) / 2.  Without baseline subtraction the raw
    # foreground NCC is everywhere > 0.5 so peak/2 is meaningless.
    half_above = (ncc_peak + ncc_med) / 2.0
    above = ncc_per_sz >= half_above
    if above.sum() < 1:
        half_width = float("nan")
    else:
        idx = np.where(above)[0]
        half_width = float(sz_grid[idx[-1]] - sz_grid[idx[0]])

    passed = (ratio >= PEAK_RATIO_MIN) and (half_width <= HALF_WIDTH_MAX)
    fail_reason = ""
    if not passed:
        rs = []
        if ratio < PEAK_RATIO_MIN:
            rs.append(f"peak_ratio={ratio:.2f}<{PEAK_RATIO_MIN}")
        if not (half_width <= HALF_WIDTH_MAX):
            rs.append(f"half_width={half_width:.2f}>{HALF_WIDTH_MAX}")
        fail_reason = "; ".join(rs)

    return SzSweepResult(
        subject_id=sid, sz_grid=sz_grid, ncc_grid=ncc_per_sz,
        sz_lp=float(lp.scales[0]),
        sz_peak=sz_peak, ncc_peak=ncc_peak,
        ncc_median=ncc_med, peak_ratio=float(ratio),
        half_width=float(half_width),
        passed=bool(passed), fail_reason=fail_reason,
        tz_offset_um=float(tz_per_sz[peak_i]),
        crop_bbox_um=(
            y0 * hcr_xy_um, y1 * hcr_xy_um,
            x0 * hcr_xy_um, x1 * hcr_xy_um,
        ),
        diagnostics=common_diag,
    )
