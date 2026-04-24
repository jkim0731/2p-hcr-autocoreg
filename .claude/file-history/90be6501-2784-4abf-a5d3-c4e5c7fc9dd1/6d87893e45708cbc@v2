"""HCR per-cell image-quality scalar (S46-b validated; S47 ships it).

`hcr_quality(s)` returns a 1-D float array keyed by the order of
`centroids_um(s, "hcr_gfp")` — i.e. aligned with P1's `hcr_um`/`hcr_ids`.
Values are the within-HCR z-scored sum of (mean, std, p90, |laplacian|_mean)
over a small µm bbox around each GFP+ centroid, read from the channel-488
fused zarr at pyramid level 2.

Callers (e.g. `_p1_teaser._seed_putative`) subtract `beta * hcr_quality[j]`
from the per-pair score to globally prefer high-quality HCR targets.

Caching: results are memoised per `(subject_id, channel, level, bbox_um)`
in a module-level dict. The per-centroid feature extraction is the
expensive step (~30–50 s per subject on the benchmark set).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import laplace

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import SubjectData  # noqa: E402


_CACHE: dict[tuple, np.ndarray] = {}


def _hcr_zarr(s: SubjectData, channel: str):
    zp = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    return zarr.open(str(zp), mode="r")


def _extract_bbox_features(vol, pts_zyx: np.ndarray,
                           half_z: int, half_y: int, half_x: int) -> np.ndarray:
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


def hcr_image_features(s: SubjectData, centroids_px: np.ndarray, *,
                       channel: str = "488", level: int = 2,
                       bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                       ) -> np.ndarray:
    """Per-centroid HCR image features (N, 4) = [mean, std, p90, |lap|]."""
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


def hcr_quality(s: SubjectData, *, channel: str = "488", level: int = 2,
                bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                ) -> np.ndarray:
    """Within-HCR z-scored sum of image features for GFP+ centroids.

    Return order matches `centroids_um(s, "hcr_gfp")` / `s.hcr_gfp_df`
    row order (i.e. aligned to P1's `hcr_um`, `hcr_ids`).
    """
    key = (s.subject_id, channel, level, bbox_um)
    if key in _CACHE:
        return _CACHE[key]

    hcr_px_all = s.hcr_centroids[["z_px", "y_px", "x_px"]].values
    hcr_ids_all = s.hcr_centroids["hcr_id"].astype(int).values
    gfp_ids_ordered = s.hcr_gfp_df["hcr_id"].astype(int).tolist()
    gfp_set = set(gfp_ids_ordered)
    keep = np.array([int(i) in gfp_set for i in hcr_ids_all])
    hcr_px_gfp = hcr_px_all[keep]
    hcr_ids_gfp = hcr_ids_all[keep]

    feats = hcr_image_features(s, hcr_px_gfp, channel=channel, level=level,
                               bbox_um=bbox_um)

    row_of = {int(h): i for i, h in enumerate(hcr_ids_gfp)}
    order = np.array([row_of[int(h)] for h in gfp_ids_ordered])
    feats = feats[order]

    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0) + 1e-6
    z = np.nan_to_num((feats - mu) / sd, nan=0.0)
    quality = z.sum(axis=1)
    _CACHE[key] = quality
    return quality
