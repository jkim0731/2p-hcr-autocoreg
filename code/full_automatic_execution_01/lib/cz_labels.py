"""Per-cell CZ labels via centroid-seeded voronoi + intensity cap.

S46-c established that the shipped `*_seg-mask-outline.tif` is binary uint8
{0,1} — not per-cell labels. Running cellpose 3D on CPU would take 30-60 min
per subject; instead we approximate per-cell footprints by (1) voronoi
assignment of each voxel to its nearest CZ centroid (anisotropic EDT with real
µm sampling), (2) capping the radius at `R_um` (default 8 µm, ≈ cortical-cell
radius), (3) masking out background via a per-volume intensity threshold
(midpoint between p50 and p90 of the lightly-smoothed z-stack).

On 788406 R=8 µm: 923/932 cells labeled, median 2038 µm³ (within cortical-cell
range ~1000–3500 µm³), p10/p90 = 1362/2120 µm³. Wall ~40 s/subject (EDT
dominates). Good enough for M4 per-cell IoU / per-pair bbox features.

Intended consumers: F2-style mask loader (future); M4 per-cell IoU candidate;
per-cell CZ volume/density features (future F6 augmentation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import SubjectData  # noqa: E402


_CACHE: dict[tuple, np.ndarray] = {}


def _load_zstack(s: SubjectData) -> np.ndarray:
    zs = next(Path(s.coreg_dir).glob("*reg-dim-swapped.ome.tif"))
    return tifffile.imread(zs).astype(np.float32)


def cz_voronoi_labels(s: SubjectData, *, R_um: float = 8.0,
                      smooth_sigma_zyx: tuple[float, float, float] = (1.0, 1.5, 1.5),
                      intensity_percentiles: tuple[float, float] = (50.0, 90.0),
                      intensity_mix: float = 0.3
                      ) -> np.ndarray:
    """Per-cell CZ label volume. Returns int32 (Z, Y, X); 0=background, else cz_id.

    Uses centroid-seeded voronoi bounded by `R_um` and an intensity mask
    `thr = p50 + intensity_mix * (p90 - p50)` of the smoothed z-stack.
    """
    key = (s.subject_id, R_um, smooth_sigma_zyx, intensity_percentiles, intensity_mix)
    if key in _CACHE:
        return _CACHE[key]

    vol = _load_zstack(s)
    vol_s = gaussian_filter(vol, sigma=smooth_sigma_zyx)

    Z, Y, X = vol.shape
    pts = s.cz_centroids[["z_px", "y_px", "x_px"]].astype(int).values
    ids = s.cz_centroids["cz_id"].astype(int).values
    pts[:, 0] = np.clip(pts[:, 0], 0, Z - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, Y - 1)
    pts[:, 2] = np.clip(pts[:, 2], 0, X - 1)

    markers = np.zeros(vol.shape, dtype=bool)
    markers[pts[:, 0], pts[:, 1], pts[:, 2]] = True

    sampling = (float(s.cz_z_um), float(s.cz_xy_um), float(s.cz_xy_um))
    dist, idx = distance_transform_edt(
        ~markers, sampling=sampling, return_indices=True
    )

    seed_lookup = np.zeros(vol.shape, dtype=np.int32)
    seed_lookup[pts[:, 0], pts[:, 1], pts[:, 2]] = ids
    labels = seed_lookup[idx[0], idx[1], idx[2]]

    labels[dist > R_um] = 0

    p_lo, p_hi = intensity_percentiles
    lo = np.percentile(vol_s, p_lo)
    hi = np.percentile(vol_s, p_hi)
    thr = lo + intensity_mix * (hi - lo)
    labels[vol_s < thr] = 0

    _CACHE[key] = labels
    return labels


def cz_cell_bboxes(s: SubjectData, *, R_um: float = 8.0) -> dict[int, tuple]:
    """Per-cell bounding box in CZ voxel coordinates.

    Returns {cz_id: (z0, z1, y0, y1, x0, x1)} for cells with any labeled voxels.
    """
    labels = cz_voronoi_labels(s, R_um=R_um)
    out: dict[int, tuple] = {}
    ids = s.cz_centroids["cz_id"].astype(int).values
    for cid in ids:
        where = np.where(labels == int(cid))
        if len(where[0]) == 0:
            continue
        out[int(cid)] = (
            int(where[0].min()), int(where[0].max()) + 1,
            int(where[1].min()), int(where[1].max()) + 1,
            int(where[2].min()), int(where[2].max()) + 1,
        )
    return out
