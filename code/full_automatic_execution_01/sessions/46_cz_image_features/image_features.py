"""Per-centroid image-feature helpers used by S46-b probes.

Extracts (mean, std, p90, |laplacian|_mean) from a small µm-sized bbox
around each centroid, for both CZ (`reg-dim-swapped.ome.tif`) and HCR
(channel 488 fused zarr at a chosen pyramid level).

Centroids are expected in native CZ / HCR-level-2 pixel coords
(matching `s.cz_centroids` / `s.hcr_centroids`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import laplace

sys.path.insert(0, "/root/capsule/code/dev_code")


def _cz_zstack_path(s) -> Path:
    files = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*zstack.tif"))
    if not files:
        raise FileNotFoundError(f"No CZ z-stack in {s.coreg_dir}")
    return files[0]


def _hcr_zarr(s, channel: str = "488"):
    zp = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    return zarr.open(str(zp), mode="r")


def _extract_bbox_features(vol, pts_zyx: np.ndarray,
                           half_z: int, half_y: int, half_x: int) -> np.ndarray:
    """For each integer voxel center in pts_zyx, extract a bbox and compute
    (mean, std, p90, lap_mean). Returns (N, 4).
    `vol` can be numpy ndarray or a zarr Array (anything slice-returning
    a numpy array)."""
    n = len(pts_zyx)
    out = np.full((n, 4), np.nan, dtype=float)
    Z, Y, X = vol.shape
    for i, (z, y, x) in enumerate(pts_zyx):
        z0 = max(0, int(z) - half_z); z1 = min(Z, int(z) + half_z + 1)
        y0 = max(0, int(y) - half_y); y1 = min(Y, int(y) + half_y + 1)
        x0 = max(0, int(x) - half_x); x1 = min(X, int(x) + half_x + 1)
        if z1 - z0 < 1 or y1 - y0 < 1 or x1 - x0 < 1:
            continue
        patch = np.asarray(vol[z0:z1, y0:y1, x0:x1]).astype(np.float32)
        if patch.size == 0:
            continue
        out[i, 0] = float(patch.mean())
        out[i, 1] = float(patch.std())
        out[i, 2] = float(np.percentile(patch, 90))
        out[i, 3] = float(np.abs(laplace(patch)).mean())
    return out


def cz_image_features(s, centroids_px: np.ndarray,
                      bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                      ) -> np.ndarray:
    path = _cz_zstack_path(s)
    with tifffile.TiffFile(path) as tf:
        vol = tf.asarray()
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = vol[0]
    half_z = max(1, int(round(bbox_um[0] / s.cz_z_um)))
    half_y = max(1, int(round(bbox_um[1] / s.cz_xy_um)))
    half_x = max(1, int(round(bbox_um[2] / s.cz_xy_um)))
    pts = np.asarray(centroids_px, dtype=int)
    return _extract_bbox_features(vol, pts, half_z, half_y, half_x)


def hcr_image_features(s, centroids_px: np.ndarray, *, channel: str = "488",
                       level: int = 2,
                       bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                       ) -> np.ndarray:
    arr = _hcr_zarr(s, channel=channel)
    vol = arr[str(level)][0, 0]
    factor_xy = 2 ** (level - 2); factor_z = 2 ** max(0, level - 2)
    xy_um = s.hcr_xy_um * factor_xy; z_um = s.hcr_z_um * factor_z
    half_z = max(1, int(round(bbox_um[0] / z_um)))
    half_y = max(1, int(round(bbox_um[1] / xy_um)))
    half_x = max(1, int(round(bbox_um[2] / xy_um)))
    pts = np.asarray(centroids_px, dtype=int)
    if factor_xy != 1 or factor_z != 1:
        pts = np.stack([pts[:, 0] // factor_z, pts[:, 1] // factor_xy,
                        pts[:, 2] // factor_xy], axis=1)
    return _extract_bbox_features(vol, pts, half_z, half_y, half_x)
