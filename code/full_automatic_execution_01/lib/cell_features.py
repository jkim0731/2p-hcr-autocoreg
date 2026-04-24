"""F6 — per-cell hand-crafted feature extractor.

Interpretable, rotation-invariant-after-180° features that feed the G-series
GNN, the P-series putative-correspondence generator, and any downstream
classifier/calibration utility.

Public API
----------

    extract_cell_features(s, modality, surface=None, *, k=6, n_dist_rank=10,
                          local_xy_radius_um=50.0, local_density_radius_um=30.0,
                          layer_bins=8, use_hcr_volume=True)

Returns ``(features, feature_names, ids)`` where ``features`` is ``(N, D)``,
``feature_names`` is length ``D``, and ``ids`` is length ``N`` (cz_id or hcr_id).

Values may be NaN where a feature is undefined (e.g. intensity on CZ side,
or local-density rank when <2 neighbours in the XY ring).

Rotation invariance note
------------------------

k-NN angles are stored as *sorted* sequences of dihedral angles after the
local frame is oriented with the pia normal as "up". This makes the angle
signature invariant to rotation around the pia-normal axis — which,
together with the 180°-XY structural prior, is what the grand plan calls
for.
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent.parent  # /root/capsule/code
if str(_ROOT / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "dev_code"))

from benchmark_data_loader import (  # noqa: E402
    SubjectData,
    cz_px_to_um,
    hcr_px_to_um,
)
from benchmark_analysis import (  # noqa: E402
    depth_from_surface,
    analyze_subject,
)


# ----------------------------------------------------------------------
# Small surface cache so repeated F6 calls do not reload HCR volumes.
# ----------------------------------------------------------------------
_SURFACE_CACHE: dict[str, dict] = {}


def get_surfaces(s: SubjectData) -> tuple[dict, dict]:
    """Return (cz_surface, hcr_surface) — image-ceiling / quantile-ceiling fits.

    Cached per subject_id. On a warm cache this call is ~microseconds.
    """
    key = s.subject_id
    if key in _SURFACE_CACHE:
        sc = _SURFACE_CACHE[key]
        return sc["cz"], sc["hcr"]
    out = analyze_subject(s)
    sc = {"cz": out["cz_surface"], "hcr": out["hcr_surface"]}
    _SURFACE_CACHE[key] = sc
    return sc["cz"], sc["hcr"]


# ----------------------------------------------------------------------
# Main helper
# ----------------------------------------------------------------------
def _modality_points(s: SubjectData, modality: str) -> tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Return ``(pts_xyz_um, ids, extras_df)`` for the requested modality.

    * ``cz``: all CZ centroids.
    * ``hcr_all``: all HCR centroids.
    * ``hcr_gfp``: HCR centroids restricted to the GFP+ subset; ``extras_df``
      contains ``density, count, mean, mean_minus_bg, log1p_count`` if
      available.
    """
    if modality == "cz":
        arr = cz_px_to_um(
            s.cz_centroids[["z_px", "y_px", "x_px"]].values.astype(float), s
        )
        xyz = arr[:, [2, 1, 0]]  # (x, y, z)
        ids = s.cz_centroids["cz_id"].astype(int).values
        return xyz, ids, None
    arr = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values.astype(float), s
    )
    xyz = arr[:, [2, 1, 0]]
    ids = s.hcr_centroids["hcr_id"].astype(int).values
    if modality == "hcr_all":
        return xyz, ids, None

    # hcr_gfp — restrict to GFP+ subset and return intensity/density columns.
    if s.hcr_gfp_df.empty:
        return xyz[:0], ids[:0], None
    gfp_ids = set(s.hcr_gfp_df["hcr_id"].astype(int).tolist())
    keep = np.asarray([int(i) in gfp_ids for i in ids])
    kept_xyz = xyz[keep]
    kept_ids = ids[keep]
    extras = s.hcr_gfp_df.set_index("hcr_id").reindex(kept_ids)
    # Normalise expected columns (loader uses ``counts`` on spot data,
    # ``mean`` + background on intensity data; accept either).
    if "count" not in extras.columns:
        if "counts" in extras.columns:
            extras["count"] = extras["counts"].astype(float)
        else:
            extras["count"] = np.nan
    for col in ["density", "mean", "mean_minus_bg"]:
        if col not in extras.columns:
            extras[col] = np.nan
    extras["log1p_count"] = np.log1p(extras["count"].fillna(0.0).astype(float))
    extras = extras.reset_index()
    return kept_xyz, kept_ids, extras


def _hcr_volume_lookup(s: SubjectData) -> dict[int, float]:
    """HCR per-cell volume from `metrics.pickle`, keyed by hcr_id."""
    p = s.hcr_dir / "cell_body_segmentation" / "metrics.pickle"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        data = pickle.load(f)
    return {int(k): float(v.get("volume", float("nan"))) for k, v in data.items()}


def _knn_angle_signature(
    xyz: np.ndarray,
    up: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each point, return `(N, 2*k)` sorted (elevation, azimuth-diff)
    signature of its k nearest neighbours in the local pia-normal frame.

    Elevations (angle above pia plane) are sorted ascending.  Azimuth-diffs
    are the *sorted* differences between consecutive azimuths wrapped to
    `[0, 2π)` — this is rotation-invariant around the up-axis.
    """
    n = len(xyz)
    out = np.full((n, 2 * k), np.nan, dtype=float)
    if n <= k:
        return out
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k + 1)  # 0-th is self
    idx = idx[:, 1:]
    for i in range(n):
        rel = xyz[idx[i]] - xyz[i]
        rn = np.linalg.norm(rel, axis=1)
        nz = rn > 0
        rel = rel[nz]
        if len(rel) < k:
            continue
        # Elevation: angle from pia plane, i.e. asin(cos(angle_to_up))
        dot_up = rel @ up
        with np.errstate(invalid="ignore"):
            elev = np.arcsin(np.clip(dot_up / np.linalg.norm(rel, axis=1), -1, 1))
        # Azimuth: project onto plane perpendicular to up, pick an arbitrary
        # in-plane basis e1/e2.
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(up @ e1) > 0.95:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - up * (e1 @ up)
        e1 /= np.linalg.norm(e1) + 1e-12
        e2 = np.cross(up, e1)
        az = np.arctan2(rel @ e2, rel @ e1)
        az = np.sort((az + 2 * math.pi) % (2 * math.pi))
        az_diffs = np.diff(np.r_[az, az[0] + 2 * math.pi])  # cyclic gaps, length k
        az_diffs.sort()
        elev.sort()
        out[i, :k] = elev[:k]
        out[i, k:] = az_diffs[:k]
    return out


def _pia_normal(surface: dict) -> np.ndarray:
    """Pia normal pointing from deep → shallow (up).

    For a planar fit z = a·x + b·y + c, the upward unit normal is
    `(-a, -b, 1) / ||·||`.
    """
    if surface is None:
        return np.array([0.0, 0.0, -1.0])
    a = float(surface.get("a", 0.0))
    b = float(surface.get("b", 0.0))
    n = np.array([-a, -b, 1.0])  # pointing in direction of decreasing z (pia side)
    n /= np.linalg.norm(n) + 1e-12
    # Want "up" pointing to the pia (depth = 0); depths are positive going deep.
    # The convention in depth_from_surface is depth = z - z_surf, so up is
    # direction of decreasing depth = -z direction → normal points to -z side.
    n = -n
    return n


def extract_cell_features(
    s: SubjectData,
    modality: str = "cz",
    *,
    k: int = 6,
    n_dist_rank: int = 10,
    local_xy_radius_um: float = 50.0,
    local_density_radius_um: float = 30.0,
    layer_bins: int = 8,
    use_hcr_volume: bool = True,
    surfaces: Optional[tuple[dict, dict]] = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute the (N, D) per-cell feature matrix.

    Parameters
    ----------
    modality : {"cz", "hcr_gfp", "hcr_all"}
    """
    if surfaces is None:
        cz_surf, hcr_surf = get_surfaces(s)
    else:
        cz_surf, hcr_surf = surfaces
    surf = cz_surf if modality == "cz" else hcr_surf

    xyz, ids, extras = _modality_points(s, modality)
    if len(xyz) == 0:
        return np.zeros((0, 0)), [], np.zeros((0,), dtype=int)

    # Also fetch the *full* population of this modality for density counts &
    # neighbour ranks (GFP+ subset plus unlabelled cells).
    if modality == "hcr_gfp":
        full_xyz = hcr_px_to_um(
            s.hcr_centroids[["z_px", "y_px", "x_px"]].values.astype(float), s
        )[:, [2, 1, 0]]
    else:
        full_xyz = xyz

    up = _pia_normal(surf)

    # ------- k-NN angles ------------------------------------------------
    ang_sig = _knn_angle_signature(xyz, up, k=k)   # (N, 2k)

    # ------- Depth-from-surface ----------------------------------------
    depths = depth_from_surface(xyz, surf)            # (N,)

    # ------- Depth-from-surface rank (local) ---------------------------
    N = len(xyz)
    xy = xyz[:, :2]
    xy_tree = cKDTree(xy)
    neighbours = xy_tree.query_ball_point(xy, r=local_xy_radius_um)
    depth_rank_local = np.full(N, np.nan)
    depth_rank_fullpop = np.full(N, np.nan)
    # For full-population rank we compare against all cells (not just GFP+ subset).
    full_xy = full_xyz[:, :2]
    full_depths = depth_from_surface(full_xyz, surf)
    full_xy_tree = cKDTree(full_xy)
    full_neighbours = full_xy_tree.query_ball_point(xy, r=local_xy_radius_um)

    for i in range(N):
        nbrs = neighbours[i]
        if len(nbrs) >= 2:
            ring_d = depths[nbrs]
            rank = float((ring_d < depths[i]).sum()) / (len(nbrs) - 1)
            depth_rank_local[i] = rank
        full_nbrs = full_neighbours[i]
        if len(full_nbrs) >= 2:
            rank = float((full_depths[full_nbrs] < depths[i]).sum()) / (len(full_nbrs) - 1)
            depth_rank_fullpop[i] = rank

    # ------- Local density (GFP vs all) --------------------------------
    count_gfp_ring = np.array([len(n) for n in xy_tree.query_ball_point(xy, r=local_density_radius_um)])
    count_all_ring = np.array([len(n) for n in full_xy_tree.query_ball_point(xy, r=local_density_radius_um)])
    density_ratio = count_gfp_ring / np.maximum(1, count_all_ring)

    # ------- Inter-ROI distance ranks ----------------------------------
    dist_tree = cKDTree(xyz)
    k_rank = min(n_dist_rank, max(0, N - 1))
    dist_knn = np.full((N, n_dist_rank), np.nan)
    if k_rank >= 1:
        d_arr, _ = dist_tree.query(xyz, k=k_rank + 1)
        d_arr = d_arr[:, 1:k_rank + 1]
        dist_knn[:, :k_rank] = d_arr
    # Global 1-NN median for z-score
    median_1nn = float(np.nanmedian(dist_knn[:, 0])) if k_rank >= 1 else 1.0
    if not np.isfinite(median_1nn) or median_1nn <= 0:
        median_1nn = 1.0
    dist_knn_z = dist_knn / median_1nn

    # ------- ROI volume ratio (HCR only) -------------------------------
    vol_ratio = np.full(N, np.nan)
    if modality in ("hcr_gfp", "hcr_all") and use_hcr_volume:
        vmap = _hcr_volume_lookup(s)
        if vmap:
            vols = np.array([vmap.get(int(i), np.nan) for i in ids], dtype=float)
            # Local median volume in XY ring
            for i in range(N):
                nbrs = neighbours[i]
                if len(nbrs) >= 2:
                    med = np.nanmedian(vols[nbrs])
                    if med > 0:
                        vol_ratio[i] = vols[i] / med

    # ------- GFP+ intensity / density features -------------------------
    gfp_density = np.full(N, np.nan)
    gfp_count = np.full(N, np.nan)
    gfp_log1p = np.full(N, np.nan)
    gfp_mean_bg = np.full(N, np.nan)
    if extras is not None and not extras.empty:
        if "density" in extras.columns:
            gfp_density = extras["density"].astype(float).values
        if "count" in extras.columns:
            gfp_count = extras["count"].astype(float).values
        if "log1p_count" in extras.columns:
            gfp_log1p = extras["log1p_count"].astype(float).values
        if "mean_minus_bg" in extras.columns:
            gfp_mean_bg = extras["mean_minus_bg"].astype(float).values

    # ------- Layer histogram ID (8-bin local depth hist) --------------
    layer_hist = np.full((N, layer_bins), np.nan)
    if np.isfinite(depths).any():
        # Modality-specific cortical thickness estimate: p95 - p5
        p5, p95 = np.nanpercentile(full_depths, [5, 95])
        thickness = max(1.0, p95 - p5)
        for i in range(N):
            full_nbrs = full_neighbours[i]
            if len(full_nbrs) >= 4:
                h, _ = np.histogram(full_depths[full_nbrs],
                                     bins=layer_bins, range=(p5, p95))
                total = h.sum()
                layer_hist[i] = h / max(1, total)

    # ------- Stack -----------------------------------------------------
    columns: list[np.ndarray] = []
    names: list[str] = []

    for j in range(k):
        columns.append(ang_sig[:, j]); names.append(f"knn_elev_{j}")
    for j in range(k):
        columns.append(ang_sig[:, k + j]); names.append(f"knn_azdiff_{j}")

    columns.append(depths); names.append("depth_um")
    columns.append(depth_rank_local); names.append("depth_rank_ring")
    columns.append(depth_rank_fullpop); names.append("depth_rank_fullpop")

    columns.append(count_gfp_ring.astype(float)); names.append("count_gfp_30um")
    columns.append(count_all_ring.astype(float)); names.append("count_all_30um")
    columns.append(density_ratio); names.append("density_ratio_30um")

    for j in range(n_dist_rank):
        columns.append(dist_knn_z[:, j]); names.append(f"dist_knn_z_{j+1}")

    columns.append(vol_ratio); names.append("vol_ratio")

    columns.append(gfp_density); names.append("gfp_density")
    columns.append(gfp_count); names.append("gfp_count")
    columns.append(gfp_log1p); names.append("gfp_log1p_count")
    columns.append(gfp_mean_bg); names.append("gfp_mean_minus_bg")

    for j in range(layer_bins):
        columns.append(layer_hist[:, j]); names.append(f"layer_hist_{j}")

    F = np.stack(columns, axis=1)
    return F, names, ids


# ----------------------------------------------------------------------
# Rotation-invariant subset helper
# ----------------------------------------------------------------------
def invariant_feature_mask(names: list[str]) -> np.ndarray:
    """Boolean mask selecting features that are safe to use for matching
    under unknown anisotropic scale (i.e. no absolute depth_um or
    dist_knn_z[0] which depends on the modality's scale)."""
    inv = []
    for n in names:
        if n.startswith("knn_elev_") or n.startswith("knn_azdiff_"):
            inv.append(True)
        elif n.startswith("depth_rank_"):
            inv.append(True)
        elif n.startswith("density_ratio"):
            inv.append(True)
        elif n.startswith("dist_knn_z_"):  # ratio to local median - scale invariant
            inv.append(True)
        elif n.startswith("layer_hist_"):
            inv.append(True)
        elif n.startswith("gfp_"):
            inv.append(True)
        else:
            inv.append(False)
    return np.asarray(inv)
