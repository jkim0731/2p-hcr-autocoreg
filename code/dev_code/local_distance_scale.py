"""Local distance-based scaling estimator (session 06).

Tests the hypothesis that ``sxy`` and ``sz`` between CZ GCaMP+ and HCR
GFP+ populations can be recovered from ratios of local ROI-to-ROI
distance statistics, provided HCR GFP+ is localized to the CZ overlap
region via R1 minimal ``(R, t)``.

Methods:

* **M1 — axis-separated k-NN distance ratio (primary).**
  ``sxy = median(2D-kNN_xy)_hcr / median(2D-kNN_xy)_cz`` and
  ``sz  = median(1D-kNN_z)_hcr  / median(1D-kNN_z)_cz``.
  Separate projections are required: 3D-kNN selection biases neighbor
  sets into isotropic balls in physical space, so per-axis deltas
  drawn from 3D-kNN all scale by the volume factor
  ``(sxy**2 * sz) ** (1/3)`` regardless of which axis they came from
  — confirmed on synthetic stretched clouds.  2D xy-projection density
  scales as ``N / sxy**2`` so 2D-kNN distance scales as ``sxy``; 1D z
  density scales as ``N / sz`` so 1D-kNN distance scales as ``sz``.

* **M2 — local volume-density ratio (isotropic sanity).**
  ``rho = k / V_knn`` with ``V_knn`` the ball of radius equal to the
  mean k-th NN distance.  ``s_iso = (rho_cz / rho_hcr) ** (1/3)``.

Two practical complications the estimator handles:

1. **Quantized z.**  Centroid z is stored at 1-µm resolution in both
   modalities, so 1D-z k-NN distances collapse to zero when multiple
   cells share a z-plane.  We add ±0.5 µm uniform jitter (seeded) to
   z only, before computing the 1D-z statistic.  This reproduces the
   unbiased expectation under uniform sub-pixel placement.

2. **Localisation crop size.**  R1's feasibility upper bound
   ``L_hcr_xy / L_cz_xy`` is ≈ 5× for benchmark subjects — cropping at
   that bound effectively uses the whole HCR volume, inflating local
   density by a factor ≈ (crop_ratio / true_scale)**2 in xy.  We
   iterate: crop → estimate → re-crop at a tighter multiple of the
   current estimate, until the crop scales stabilise.  This converges
   even from a very loose starting crop because each iteration shrinks
   toward the true scale when the density is uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# Output dataclass
# ----------------------------------------------------------------------
@dataclass
class LocalDistanceScale:
    sxy: Optional[float]
    sz: Optional[float]
    s_iso: Optional[float]
    k: int
    n_cz: int
    n_hcr_local: int
    cz_summary: dict
    hcr_summary: dict
    iterations: int = 0
    crop_history: list = field(default_factory=list)   # list of (sxy_crop, sz_crop, n_hcr_local, sxy_est, sz_est)
    converged: bool = False
    reason_unknown: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def _jitter_z(xyz: np.ndarray, jitter_um: float, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(xyz, dtype=float).copy()
    out[:, 2] = out[:, 2] + rng.uniform(-jitter_um, jitter_um, size=out.shape[0])
    return out


def _axis_separated_knn_summary(points_xyz_um: np.ndarray, k: int) -> dict:
    """Separate 2D-xy k-NN and 1D-z k-NN distances.

    2D xy-plane: density scales as ``N / sxy**2`` → distance scales as
    ``sxy``.  1D z: density scales as ``N / sz`` → distance scales as
    ``sz``.  Separate projections are necessary: per-axis deltas drawn
    from a 3D-kNN neighbourhood all scale by ``(sxy**2 * sz) ** (1/3)``
    and do NOT recover anisotropy.
    """
    pts = np.asarray(points_xyz_um, dtype=float)
    if pts.shape[0] <= k:
        return {}
    xy = pts[:, :2]
    tree_xy = cKDTree(xy)
    d_xy, _ = tree_xy.query(xy, k=k + 1)
    d_xy = d_xy[:, 1:]
    z = pts[:, 2:3]
    tree_z = cKDTree(z)
    d_z, _ = tree_z.query(z, k=k + 1)
    d_z = d_z[:, 1:]
    return {
        "median_knn_xy2d": float(np.median(d_xy)),
        "median_knn_z1d": float(np.median(d_z)),
        "median_1nn_xy2d": float(np.median(d_xy[:, 0])),
        "median_1nn_z1d": float(np.median(d_z[:, 0])),
        "n_pairs": int(d_xy.size),
    }


def _mean_knn_density(points_xyz_um: np.ndarray, k: int) -> float:
    pts = np.asarray(points_xyz_um, dtype=float)
    if pts.shape[0] <= k:
        return float("nan")
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    r_k_mean = float(np.mean(d[:, -1]))
    if r_k_mean <= 0:
        return float("nan")
    V = (4.0 / 3.0) * np.pi * r_k_mean ** 3
    return k / V


# ----------------------------------------------------------------------
# Localization
# ----------------------------------------------------------------------
def _apply_r1(cz_xyz_um: np.ndarray, coarse_fit) -> np.ndarray:
    from r1_revised import apply_coarse_affine
    return apply_coarse_affine(np.asarray(cz_xyz_um, dtype=float), coarse_fit)


def _crop_box(mapped_cz: np.ndarray, sxy_crop: float, sz_crop: float,
              margin_um: float) -> tuple[np.ndarray, np.ndarray]:
    c_lo = mapped_cz.min(axis=0)
    c_hi = mapped_cz.max(axis=0)
    center = 0.5 * (c_lo + c_hi)
    half = 0.5 * (c_hi - c_lo)
    infl = np.array([sxy_crop, sxy_crop, sz_crop], dtype=float)
    box_lo = center - half * infl - margin_um
    box_hi = center + half * infl + margin_um
    return box_lo, box_hi


def _crop_mask(pts: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray) -> np.ndarray:
    return np.all((pts >= box_lo) & (pts <= box_hi), axis=1)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def estimate_local_distance_scale(
    cz_xyz_um: np.ndarray,
    hcr_gfp_xyz_um: np.ndarray,
    coarse_fit,
    sxy_upper_feasibility: float,
    *,
    k: int = 5,
    k_density: int = 10,
    crop_margin_um: float = 50.0,
    sz_upper: float = 6.0,
    sxy_lower_crop: float = 1.2,
    sz_lower_crop: float = 1.2,
    crop_margin_factor: float = 1.3,
    n_min_localized: int = 100,
    max_iter: int = 8,
    rel_tol: float = 0.02,
    z_jitter_um: float = 0.5,
    rng_seed: int = 0,
) -> LocalDistanceScale:
    """Estimate ``(sxy, sz)`` with iterative crop tightening.

    Iteration ``it``:
      1. Crop HCR GFP+ to the mapped CZ AABB inflated by
         ``(sxy_crop, sxy_crop, sz_crop)`` plus a constant margin.
      2. Apply z-jitter independently to the two clouds (to break the
         1-µm quantization ties in the z axis).
      3. Compute ``sxy = median_hcr / median_cz`` (2D-xy k-NN) and
         ``sz`` (1D-z k-NN).
      4. Next crop: ``crop_margin_factor × max(est, lower_crop)`` per
         axis, clipped to the feasibility envelope.
      5. Stop when per-axis crop changes by < ``rel_tol`` or after
         ``max_iter`` iterations.

    ``z_jitter_um`` should match the z voxel size (1 µm in this repo);
    the ±0.5 µm default is the unbiased choice under uniform sub-pixel
    placement.
    """
    rng = np.random.default_rng(rng_seed)

    cz_raw = np.asarray(cz_xyz_um, dtype=float)
    hcr_raw = np.asarray(hcr_gfp_xyz_um, dtype=float)

    mapped_cz = _apply_r1(cz_raw, coarse_fit)

    # z-jitter is seeded per-cloud so they don't share the same noise
    cz_j = _jitter_z(cz_raw, z_jitter_um, rng)
    hcr_j = _jitter_z(hcr_raw, z_jitter_um, rng)

    # Start from the widest feasible crop
    sxy_crop = float(sxy_upper_feasibility)
    sz_crop = float(sz_upper)

    history: list = []
    converged = False
    cz_sum: dict = {}
    hcr_sum: dict = {}
    sxy_est: Optional[float] = None
    sz_est: Optional[float] = None
    rho_cz: float = float("nan")
    rho_hcr: float = float("nan")
    n_local = 0

    for it in range(max_iter):
        box_lo, box_hi = _crop_box(mapped_cz, sxy_crop, sz_crop, crop_margin_um)
        mask = _crop_mask(hcr_j, box_lo, box_hi)
        hcr_local = hcr_j[mask]
        n_local = int(hcr_local.shape[0])

        if n_local < n_min_localized:
            history.append({
                "it": it, "sxy_crop": sxy_crop, "sz_crop": sz_crop,
                "n_hcr_local": n_local,
                "status": "too_few_localized",
            })
            break

        cz_sum = _axis_separated_knn_summary(cz_j, k=k)
        hcr_sum = _axis_separated_knn_summary(hcr_local, k=k)
        if not cz_sum or not hcr_sum:
            history.append({
                "it": it, "sxy_crop": sxy_crop, "sz_crop": sz_crop,
                "n_hcr_local": n_local,
                "status": "summary_failed",
            })
            break

        sxy_est = hcr_sum["median_knn_xy2d"] / cz_sum["median_knn_xy2d"]
        sz_est = hcr_sum["median_knn_z1d"] / cz_sum["median_knn_z1d"]

        rho_cz = _mean_knn_density(cz_j, k=k_density)
        rho_hcr = _mean_knn_density(hcr_local, k=k_density)

        new_sxy_crop = float(np.clip(
            crop_margin_factor * max(sxy_est, sxy_lower_crop),
            sxy_lower_crop, sxy_upper_feasibility,
        ))
        new_sz_crop = float(np.clip(
            crop_margin_factor * max(sz_est, sz_lower_crop),
            sz_lower_crop, sz_upper,
        ))

        history.append({
            "it": it, "sxy_crop": sxy_crop, "sz_crop": sz_crop,
            "n_hcr_local": n_local,
            "sxy_est": sxy_est, "sz_est": sz_est,
            "next_sxy_crop": new_sxy_crop, "next_sz_crop": new_sz_crop,
        })

        d_sxy = abs(new_sxy_crop - sxy_crop) / max(sxy_crop, 1e-9)
        d_sz = abs(new_sz_crop - sz_crop) / max(sz_crop, 1e-9)
        sxy_crop, sz_crop = new_sxy_crop, new_sz_crop
        if d_sxy < rel_tol and d_sz < rel_tol:
            converged = True
            break

    diagnostics = {
        "sxy_upper_feasibility": float(sxy_upper_feasibility),
        "sz_upper": float(sz_upper),
        "crop_margin_um": float(crop_margin_um),
        "crop_margin_factor": float(crop_margin_factor),
        "n_hcr_before_crop": int(hcr_raw.shape[0]),
        "z_jitter_um": float(z_jitter_um),
        "rho_cz": float(rho_cz),
        "rho_hcr_local": float(rho_hcr),
        "sxy_crop_final": float(sxy_crop),
        "sz_crop_final": float(sz_crop),
    }

    if sxy_est is None or sz_est is None or n_local < n_min_localized:
        return LocalDistanceScale(
            sxy=None, sz=None, s_iso=None,
            k=k, n_cz=int(cz_raw.shape[0]), n_hcr_local=n_local,
            cz_summary=cz_sum, hcr_summary=hcr_sum,
            iterations=len(history),
            crop_history=history, converged=converged,
            reason_unknown="too_few_localized" if n_local < n_min_localized else "no_estimate",
            diagnostics=diagnostics,
        )

    s_iso = None
    if rho_cz > 0 and rho_hcr > 0 and np.isfinite(rho_cz) and np.isfinite(rho_hcr):
        s_iso = float((rho_cz / rho_hcr) ** (1.0 / 3.0))

    return LocalDistanceScale(
        sxy=float(sxy_est),
        sz=float(sz_est),
        s_iso=s_iso,
        k=k,
        n_cz=int(cz_raw.shape[0]),
        n_hcr_local=n_local,
        cz_summary=cz_sum,
        hcr_summary=hcr_sum,
        iterations=len(history),
        crop_history=history,
        converged=converged,
        reason_unknown=None,
        diagnostics=diagnostics,
    )


# ----------------------------------------------------------------------
# Synthetic sanity check
# ----------------------------------------------------------------------
def _synthetic_sanity(seed: int = 0, n: int = 900,
                     sxy_true: float = 1.77, sz_true: float = 2.82,
                     extras_factor: float = 0.0) -> dict:
    """Stretch a CZ copy by known (sxy, sz); optionally add uniform
    "extras" inside the HCR bounding box at a given ratio to cells.
    """
    rng = np.random.default_rng(seed)
    cz = rng.uniform(0.0, 400.0, size=(n, 3))
    hcr = cz.copy()
    hcr[:, 0] *= sxy_true
    hcr[:, 1] *= sxy_true
    hcr[:, 2] *= sz_true

    if extras_factor > 0:
        n_extra = int(extras_factor * n)
        extras = np.column_stack([
            rng.uniform(0, 400 * sxy_true, size=n_extra),
            rng.uniform(0, 400 * sxy_true, size=n_extra),
            rng.uniform(0, 400 * sz_true, size=n_extra),
        ])
        hcr = np.vstack([hcr, extras])

    class _IdentityFit:
        R = np.eye(3)
        scales = np.array([1.0, 1.0, 1.0])
        src_mean = cz.mean(axis=0)
        translation = hcr.mean(axis=0)

    out = estimate_local_distance_scale(
        cz_xyz_um=cz,
        hcr_gfp_xyz_um=hcr,
        coarse_fit=_IdentityFit(),
        sxy_upper_feasibility=3.0,
        sz_upper=6.0,
        crop_margin_um=50.0,
    )
    return {
        "sxy_true": sxy_true,
        "sz_true": sz_true,
        "extras_factor": extras_factor,
        "sxy_est": out.sxy,
        "sz_est": out.sz,
        "s_iso_est": out.s_iso,
        "expected_s_iso": float((sxy_true ** 2 * sz_true) ** (1.0 / 3.0)),
        "n_cz": out.n_cz,
        "n_hcr_local": out.n_hcr_local,
        "iterations": out.iterations,
        "converged": out.converged,
    }


if __name__ == "__main__":
    import json
    checks = [
        _synthetic_sanity(extras_factor=0.0),
        _synthetic_sanity(extras_factor=1.0),
        _synthetic_sanity(extras_factor=3.0),
    ]
    print(json.dumps(checks, indent=2))
