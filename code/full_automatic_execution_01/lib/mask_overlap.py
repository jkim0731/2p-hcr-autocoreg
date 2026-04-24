"""F4 — Volumetric mask-overlap scorer.

Given two aligned 3-D label volumes on a common grid (output of F3), compute
Dice, Jaccard, a mask-NCC peak + offset, and an optional SDF-based cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


@dataclass
class MaskOverlapScores:
    dice: float
    jaccard: float
    ncc_peak: float
    ncc_offset_vox: tuple         # (dz, dy, dx)
    ncc_offset_um: tuple          # (dz, dy, dx) * spacing
    sdf_cost: Optional[float]     # None if skipped
    voxels_cz: int
    voxels_hcr: int


def _binarise(a: np.ndarray) -> np.ndarray:
    if a.dtype == bool:
        return a
    return a > 0


def mask_dice_jaccard_ncc(
    cz_mask: np.ndarray,
    hcr_mask: np.ndarray,
    spacing_um: float = 4.0,
    *,
    ncc_search_um: float = 60.0,
    compute_sdf: bool = False,
) -> MaskOverlapScores:
    """Compare two aligned label volumes.

    - Dice/Jaccard on binarised masks.
    - NCC peak searched in a ±`ncc_search_um` window around 0 shift.
    - Optional SDF cost (slow on large volumes).
    """
    if cz_mask.shape != hcr_mask.shape:
        raise ValueError(f"shape mismatch {cz_mask.shape} vs {hcr_mask.shape}")

    b_cz = _binarise(cz_mask)
    b_hcr = _binarise(hcr_mask)
    inter = int(np.logical_and(b_cz, b_hcr).sum())
    s_cz = int(b_cz.sum()); s_hcr = int(b_hcr.sum())
    union = s_cz + s_hcr - inter
    dice = 2.0 * inter / (s_cz + s_hcr) if (s_cz + s_hcr) else 0.0
    jaccard = inter / union if union else 0.0

    # NCC peak in a bounded window.
    r = max(1, int(round(ncc_search_um / spacing_um)))
    ncc_peak, (dz, dy, dx) = _ncc_window(b_cz.astype(np.float32),
                                         b_hcr.astype(np.float32),
                                         radius=r)
    off_um = (dz * spacing_um, dy * spacing_um, dx * spacing_um)

    sdf_cost = None
    if compute_sdf:
        sdf_cost = _sdf_cost(b_cz, b_hcr, spacing_um)

    return MaskOverlapScores(
        dice=dice, jaccard=jaccard,
        ncc_peak=float(ncc_peak),
        ncc_offset_vox=(dz, dy, dx),
        ncc_offset_um=off_um,
        sdf_cost=sdf_cost,
        voxels_cz=s_cz, voxels_hcr=s_hcr,
    )


def _ncc_window(a: np.ndarray, b: np.ndarray, radius: int) -> tuple[float, tuple]:
    """Zero-shift NCC evaluated over a small offset window by direct roll."""
    a = a - a.mean(); b = b - b.mean()
    a_std = a.std() + 1e-9
    b_std = b.std() + 1e-9
    best = -np.inf; best_off = (0, 0, 0)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                s = np.roll(b, (dz, dy, dx), axis=(0, 1, 2))
                c = float((a * s).mean() / (a_std * b_std))
                if c > best:
                    best = c; best_off = (dz, dy, dx)
    return best, best_off


def _sdf_cost(a: np.ndarray, b: np.ndarray, spacing_um: float) -> float:
    sdf_a = ndi.distance_transform_edt(~a) * spacing_um - ndi.distance_transform_edt(a) * spacing_um
    sdf_b = ndi.distance_transform_edt(~b) * spacing_um - ndi.distance_transform_edt(b) * spacing_um
    return float((np.abs(sdf_a) * (~b) + np.abs(sdf_b) * (~a)).mean())


def per_cell_dice(
    cz_mask: np.ndarray,
    hcr_mask: np.ndarray,
    pairs: "list[tuple[int, int]]",
) -> "dict[tuple[int, int], dict[str, float]]":
    """For each (cz_id, hcr_id) pair, compute Dice/Jaccard of that label pair
    in the co-registered volumes.

    `cz_mask` and `hcr_mask` are same-shape label volumes.
    """
    out = {}
    for cz_id, hcr_id in pairs:
        a = cz_mask == cz_id
        b = hcr_mask == hcr_id
        s_a = int(a.sum()); s_b = int(b.sum())
        if s_a == 0 or s_b == 0:
            out[(int(cz_id), int(hcr_id))] = dict(dice=0.0, jaccard=0.0,
                                                   voxels_cz=s_a, voxels_hcr=s_b)
            continue
        inter = int(np.logical_and(a, b).sum())
        union = s_a + s_b - inter
        out[(int(cz_id), int(hcr_id))] = dict(
            dice=2.0 * inter / (s_a + s_b),
            jaccard=inter / union if union else 0.0,
            voxels_cz=s_a, voxels_hcr=s_b,
        )
    return out


def _selftest():
    rng = np.random.default_rng(0)
    vol = np.zeros((20, 30, 30), dtype=np.uint8)
    vol[5:15, 10:20, 10:20] = 1
    vol2 = np.roll(vol, shift=(1, 0, 0), axis=(0, 1, 2))
    s = mask_dice_jaccard_ncc(vol, vol2, spacing_um=1.0, ncc_search_um=3.0)
    print("F4 selftest:", s)


if __name__ == "__main__":
    _selftest()
