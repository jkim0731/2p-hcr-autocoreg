"""S64 subgoal 1 — per-centroid 3D image patch extractor.

Given a `SubjectData` and an `(N, 3)` array of centroids in physical µm
(in the modality's native orientation), return `(N, 1, D, H, W)` float32
patches sampled on an isotropic µm lattice centred on each centroid.

Patches are sampled with `scipy.ndimage.map_coordinates` (linear); out of
volume coordinates pad with zero. Each patch is z-scored per-sample
(mean subtracted, divided by std + 1e-3, clipped to ±6 σ).

Intended consumers: C2 image-conditioned GNN (S64+). The CNN encoder
expects a 3D patch per centroid; this module produces those patches
without committing to an encoder architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, "/root/capsule/code/dev_code")
from benchmark_data_loader import SubjectData  # noqa: E402
from benchmark_analysis import load_hcr_volume  # noqa: E402


def _load_cz_volume(s: SubjectData) -> np.ndarray:
    """Load the full CZ z-stack as (Z, Y, X) uint16. Cached on SubjectData."""
    import tifffile

    cached = getattr(s, "_cz_volume_cache", None)
    if cached is not None:
        return cached

    files = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*zstack.tif"))
    if not files:
        raise FileNotFoundError(f"No CZ OME-TIFF in {s.coreg_dir}")
    with tifffile.TiffFile(files[0]) as tf:
        arr = tf.asarray()
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected CZ TIF shape: {arr.shape}")
    # Keep native dtype; normalise at the patch level.
    object.__setattr__(s, "_cz_volume_cache", arr)
    return arr


def _load_hcr_volume_cached(
    s: SubjectData, channel: str, level: int
) -> tuple[np.ndarray, float, float]:
    """Load and cache an HCR pyramid level on the subject."""
    key = f"_hcr_volume_cache_{channel}_L{level}"
    cached = getattr(s, key, None)
    if cached is not None:
        return cached
    vol, xy_um, z_um = load_hcr_volume(s, channel=channel, level=level)
    out = (vol, xy_um, z_um)
    object.__setattr__(s, key, out)
    return out


def _sample_patches(
    vol: np.ndarray,
    centroids_um: np.ndarray,
    voxel_spacing_um: tuple[float, float, float],
    patch_size: int,
    sample_spacing_um: float,
    orient: np.ndarray | None = None,
) -> np.ndarray:
    """Sample `(N, D, H, W)` patches from `vol` at physical µm centroids.

    `voxel_spacing_um` is `(z, y, x)` of the volume's own grid. Patches
    are sampled on an isotropic `sample_spacing_um` lattice. When
    `orient` is a `(3, 3)` matrix, the sampling lattice is pre-rotated
    by `orient` in physical space (e.g. for simulating a local warp
    Jacobian `(R·S)^{-1}` when extracting the "warped-view" patch of an
    anatomical point still located at the original centroid).
    """
    N = centroids_um.shape[0]
    D = H = W = patch_size
    offsets_um = (np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0) * sample_spacing_um
    zz, yy, xx = np.meshgrid(offsets_um, offsets_um, offsets_um, indexing="ij")
    rel_zyx = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)

    if orient is not None:
        rel_zyx = rel_zyx @ orient.astype(np.float32).T

    vz, vy, vx = voxel_spacing_um
    inv_voxel = np.array([1.0 / vz, 1.0 / vy, 1.0 / vx], dtype=np.float32)

    all_samples = centroids_um[:, None, :].astype(np.float32) + rel_zyx[None, :, :]
    all_samples *= inv_voxel[None, None, :]
    flat = all_samples.reshape(-1, 3).T

    vals = map_coordinates(vol, flat, order=1, mode="constant", cval=0.0)
    return vals.reshape(N, D, H, W).astype(np.float32)


def sample_patches_oriented(
    vol: np.ndarray,
    centroids_um: np.ndarray,
    voxel_spacing_um: tuple[float, float, float],
    patch_size: int = 16,
    sample_spacing_um: float = 4.0,
    orient: np.ndarray | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Public wrapper: `(N, 1, D, H, W)` patches with optional orientation."""
    patches = _sample_patches(
        vol, centroids_um, voxel_spacing_um, patch_size, sample_spacing_um, orient=orient
    )
    if normalize:
        patches = _zscore_per_patch(patches)
    return patches[:, None]


def _zscore_per_patch(patches: np.ndarray, clip: float = 6.0) -> np.ndarray:
    """Per-patch z-score: mean-subtract, divide by (std + 1e-3), clip."""
    axes = tuple(range(1, patches.ndim))
    mu = patches.mean(axis=axes, keepdims=True)
    sd = patches.std(axis=axes, keepdims=True)
    out = (patches - mu) / (sd + 1e-3)
    np.clip(out, -clip, clip, out=out)
    return out


@dataclass
class PatchExtractConfig:
    patch_size: int = 16
    sample_spacing_um: float = 4.0
    normalize: bool = True


def extract_cz_patches(
    s: SubjectData,
    centroids_um: np.ndarray,
    config: PatchExtractConfig | None = None,
) -> np.ndarray:
    """Extract `(N, 1, D, H, W)` CZ patches from a subject's z-stack."""
    cfg = config or PatchExtractConfig()
    vol = _load_cz_volume(s)
    patches = _sample_patches(
        vol,
        centroids_um,
        voxel_spacing_um=(s.cz_z_um, s.cz_xy_um, s.cz_xy_um),
        patch_size=cfg.patch_size,
        sample_spacing_um=cfg.sample_spacing_um,
    )
    if cfg.normalize:
        patches = _zscore_per_patch(patches)
    return patches[:, None]


def extract_hcr_patches(
    s: SubjectData,
    centroids_um: np.ndarray,
    channel: str = "488",
    level: int = 2,
    config: PatchExtractConfig | None = None,
) -> np.ndarray:
    """Extract `(N, 1, D, H, W)` HCR patches from a subject's 488 zarr."""
    cfg = config or PatchExtractConfig()
    vol, xy_um, z_um = _load_hcr_volume_cached(s, channel=channel, level=level)
    patches = _sample_patches(
        vol,
        centroids_um,
        voxel_spacing_um=(z_um, xy_um, xy_um),
        patch_size=cfg.patch_size,
        sample_spacing_um=cfg.sample_spacing_um,
    )
    if cfg.normalize:
        patches = _zscore_per_patch(patches)
    return patches[:, None]
