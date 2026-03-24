"""
coreg_verification.py
Multi-scale patch NCC + constellation similarity features + classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _slice_bounds(coord_px, patch_size_px, vol_shape):
    """Return (start, stop) slices for a 3D patch, clipped to volume."""
    half = np.array(patch_size_px) // 2
    starts = np.maximum(0, coord_px - half).astype(int)
    stops = np.minimum(vol_shape, coord_px + half + 1).astype(int)
    return starts, stops


def extract_patch_from_array(
    volume: np.ndarray,
    coord_px: np.ndarray,
    patch_size_px: np.ndarray,
) -> np.ndarray:
    """Extract a 3D sub-volume centred at coord_px (z, y, x order)."""
    starts, stops = _slice_bounds(coord_px.astype(int), patch_size_px, volume.shape)
    patch = volume[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]]
    # Pad if near boundary
    pad_widths = [(0, max(0, patch_size_px[i] - patch.shape[i])) for i in range(3)]
    if any(p[1] > 0 for p in pad_widths):
        patch = np.pad(patch, pad_widths, mode="constant")
    return patch


def extract_patch_from_zarr(
    zarr_array,            # zarr array at given level
    coord_px: np.ndarray,  # (z, y, x) in that level's pixel space
    patch_size_px: np.ndarray,
) -> np.ndarray:
    """Extract a 3D patch from a zarr array (shape: [1,1,Z,Y,X] or [Z,Y,X])."""
    if zarr_array.ndim == 5:
        shape = np.array(zarr_array.shape[2:])
        starts, stops = _slice_bounds(coord_px.astype(int), patch_size_px, shape)
        patch = zarr_array[0, 0,
                           starts[0]:stops[0],
                           starts[1]:stops[1],
                           starts[2]:stops[2]]
    elif zarr_array.ndim == 3:
        shape = np.array(zarr_array.shape)
        starts, stops = _slice_bounds(coord_px.astype(int), patch_size_px, shape)
        patch = zarr_array[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]]
    else:
        raise ValueError(f"Unexpected zarr array ndim: {zarr_array.ndim}")

    patch = np.asarray(patch, dtype=np.float32)
    pad_widths = [(0, max(0, patch_size_px[i] - patch.shape[i])) for i in range(3)]
    if any(p[1] > 0 for p in pad_widths):
        patch = np.pad(patch, pad_widths, mode="constant")
    return patch


# ---------------------------------------------------------------------------
# Normalized cross-correlation
# ---------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two arrays (scalar)."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_patch_ncc(
    czstack_vol: np.ndarray,
    hcr_zarr_path: str | Path,
    cz_coord_px: np.ndarray,
    hcr_coord_px: np.ndarray,
    czstack_res_um: np.ndarray,
    hcr_scales: dict,
    patch_size_um: float = 50.0,
    zarr_level: int = 3,
) -> float:
    """Compute NCC between czstack and HCR patches of given physical size.

    Parameters
    ----------
    czstack_vol    : 3D array (z, y, x) – full czstack volume already in memory
    hcr_zarr_path  : path to channel_488.zarr
    cz_coord_px    : (z, y, x) centroid in czstack pixels
    hcr_coord_px   : (z, y, x) centroid in HCR pixels (level-0)
    czstack_res_um : (z_um, y_um, x_um) per pixel
    hcr_scales     : dict with scale_z/y/x in µm/pixel (level-0)
    patch_size_um  : isotropic patch size in µm
    zarr_level     : which zarr pyramid level to use for HCR

    Returns
    -------
    ncc_value : float in [-1, 1]
    """
    import zarr

    # Czstack patch size in pixels
    cz_patch_px = np.round(patch_size_um / czstack_res_um).astype(int)
    cz_patch_px = np.maximum(cz_patch_px, 3)
    cz_patch = extract_patch_from_array(czstack_vol, cz_coord_px, cz_patch_px)

    # HCR zarr patch – account for pyramid downsampling
    z_grp = zarr.open(str(hcr_zarr_path), mode="r")
    level_str = str(zarr_level)
    if level_str not in z_grp:
        # Fall back to coarsest available level
        available = sorted(z_grp.keys(), key=int)
        level_str = available[-1]

    level_factor = 2 ** int(level_str)  # each level = 2× downsampling in XY
    hcr_res_level = np.array([
        hcr_scales["scale_z"],
        hcr_scales["scale_y"] * level_factor,
        hcr_scales["scale_x"] * level_factor,
    ])
    hcr_patch_px = np.round(patch_size_um / hcr_res_level).astype(int)
    hcr_patch_px = np.maximum(hcr_patch_px, 3)

    # HCR coord at the chosen level
    hcr_coord_level = hcr_coord_px.copy().astype(float)
    hcr_coord_level[1] /= level_factor  # y
    hcr_coord_level[2] /= level_factor  # x

    hcr_arr = z_grp[level_str]
    hcr_patch = extract_patch_from_zarr(hcr_arr, hcr_coord_level, hcr_patch_px)

    # Resize hcr_patch to match czstack patch shape for NCC
    from scipy.ndimage import zoom
    if hcr_patch.shape != tuple(cz_patch_px):
        factors = [cz_patch_px[i] / max(1, hcr_patch.shape[i]) for i in range(3)]
        hcr_patch = zoom(hcr_patch, factors, order=1)

    return _ncc(cz_patch, hcr_patch)


# ---------------------------------------------------------------------------
# Constellation similarity
# ---------------------------------------------------------------------------

def compute_constellation_similarity(
    czstack_cell_id: int,
    hcr_cell_id: int,
    czstack_projected_um: pd.DataFrame,
    hcr_centroids_um: pd.DataFrame,
    accepted_matches: pd.DataFrame,
    radius_um: float = 50.0,
    k_neighbors: int = 10,
) -> float:
    """Rotation-invariant constellation similarity at a given radius.

    For each cell, collect its K nearest already-matched neighbors within
    radius_um and compare the sorted distance vectors across modalities.

    Parameters
    ----------
    czstack_cell_id      : query czstack cell id
    hcr_cell_id          : query hcr cell id
    czstack_projected_um : DataFrame with czstack_cell_id + projected (z,y,x) in µm
    hcr_centroids_um     : DataFrame with hcr_cell_id + (z,y,x) in µm
    accepted_matches     : DataFrame with czstack_cell_id + hcr_cell_id (already matched pairs)
    radius_um            : neighbourhood radius
    k_neighbors          : max neighbors to use

    Returns
    -------
    similarity : float in [-1, 1], or NaN if < 3 neighbors
    """
    if len(accepted_matches) < 3:
        return np.nan

    # Filter to matched cells that have coordinates
    cz_cols = [c for c in czstack_projected_um.columns if c != "czstack_cell_id"]
    hcr_cols = [c for c in hcr_centroids_um.columns if c != "hcr_cell_id"]

    am = accepted_matches[["czstack_cell_id", "hcr_cell_id"]]
    cz_proj = czstack_projected_um.set_index("czstack_cell_id")
    hcr_cent = hcr_centroids_um.set_index("hcr_cell_id")

    matched_cz_ids = am["czstack_cell_id"].values
    matched_hcr_ids = am["hcr_cell_id"].values

    # Remove the query cell itself
    mask = (matched_cz_ids != czstack_cell_id) & (matched_hcr_ids != hcr_cell_id)
    matched_cz_ids = matched_cz_ids[mask]
    matched_hcr_ids = matched_hcr_ids[mask]

    if len(matched_cz_ids) < 3:
        return np.nan

    # Positions of matched neighbors in czstack projected space
    try:
        cz_query = cz_proj.loc[czstack_cell_id].values
        cz_neighbors = cz_proj.loc[matched_cz_ids].values
    except KeyError:
        return np.nan

    # Positions of matched neighbors in HCR space
    try:
        hcr_query = hcr_cent.loc[hcr_cell_id].values
        hcr_neighbors = hcr_cent.loc[matched_hcr_ids].values
    except KeyError:
        return np.nan

    # Distances from query to matched neighbors
    cz_dists = np.linalg.norm(cz_neighbors - cz_query, axis=1)
    hcr_dists = np.linalg.norm(hcr_neighbors - hcr_query, axis=1)

    # Filter to radius
    in_radius = (cz_dists <= radius_um) & (hcr_dists <= radius_um)
    if in_radius.sum() < 3:
        return np.nan

    cz_d_r = cz_dists[in_radius]
    hcr_d_r = hcr_dists[in_radius]

    # Use top-k by czstack distance
    if len(cz_d_r) > k_neighbors:
        order = np.argsort(cz_d_r)[:k_neighbors]
        cz_d_r = cz_d_r[order]
        hcr_d_r = hcr_d_r[order]

    # Sort both by czstack distance (rotation-invariant ordering)
    sort_idx = np.argsort(cz_d_r)
    cz_sorted = cz_d_r[sort_idx]
    hcr_sorted = hcr_d_r[sort_idx]

    if len(cz_sorted) < 2:
        return np.nan

    corr, _ = pearsonr(cz_sorted, hcr_sorted)
    return float(corr) if not np.isnan(corr) else np.nan


# ---------------------------------------------------------------------------
# Feature extraction for classifier
# ---------------------------------------------------------------------------

def extract_candidate_features(
    candidates_df: pd.DataFrame,
    czstack_projected_um: pd.DataFrame,
    hcr_centroids_um: pd.DataFrame,
    accepted_matches: pd.DataFrame,
    czstack_vol: Optional[np.ndarray],
    hcr_zarr_path: Optional[str | Path],
    czstack_res_um: np.ndarray,
    hcr_scales: dict,
    spot_counts: pd.DataFrame,
    hcr_metrics: pd.DataFrame,
    patch_sizes_um: list = [20, 50, 100],
    min_neighbors_for_scale: int = 3,
) -> pd.DataFrame:
    """Compute full feature set for a set of candidate matches.

    Parameters
    ----------
    candidates_df : DataFrame with columns czstack_cell_id, hcr_cell_id,
                    distance_um, nn_rank, is_mutual_nn
    czstack_projected_um : DataFrame with czstack_cell_id + projected_z/y/x columns (µm)
    hcr_centroids_um : DataFrame with hcr_cell_id + hcr_z/y/x columns (µm)
    accepted_matches : DataFrame with czstack_cell_id + hcr_cell_id (current landmarks)
    czstack_vol : None or 3D array (z,y,x)
    hcr_zarr_path : None or path to channel_488.zarr
    spot_counts : DataFrame with hcr_cell_id, counts, density
    hcr_metrics : DataFrame with hcr_cell_id, volume
    patch_sizes_um : list of patch sizes for NCC
    min_neighbors_for_scale : min neighbors needed to compute constellation similarity

    Returns
    -------
    DataFrame with one row per candidate and all feature columns
    """
    rows = []

    # Index lookup helpers
    spot_idx = spot_counts.set_index("hcr_cell_id") if len(spot_counts) else pd.DataFrame()
    metrics_idx = hcr_metrics.set_index("hcr_cell_id") if len(hcr_metrics) else pd.DataFrame()
    cz_proj_idx = czstack_projected_um.set_index("czstack_cell_id")
    hcr_cent_idx = hcr_centroids_um.set_index("hcr_cell_id")

    for _, row in candidates_df.iterrows():
        cz_id = int(row["czstack_cell_id"])
        hcr_id = int(row["hcr_cell_id"])
        feat = {
            "czstack_cell_id": cz_id,
            "hcr_cell_id": hcr_id,
            "distance_um": float(row.get("distance_um", np.nan)),
            "nn_rank": int(row.get("nn_rank", 1)),
            "is_mutual_nn": bool(row.get("is_mutual_nn", False)),
        }

        # Cell properties
        try:
            sp = spot_idx.loc[hcr_id]
            feat["gfp_counts"] = float(sp["counts"])
            feat["gfp_density"] = float(sp["density"])
        except KeyError:
            feat["gfp_counts"] = np.nan
            feat["gfp_density"] = np.nan

        try:
            mv = metrics_idx.loc[hcr_id]
            feat["hcr_volume_vox"] = float(mv["volume"])
        except KeyError:
            feat["hcr_volume_vox"] = np.nan

        # Constellation similarities at 3 scales
        for radius in [20, 50, 100]:
            key = f"constel_sim_{radius}um"
            feat[key] = compute_constellation_similarity(
                cz_id, hcr_id,
                czstack_projected_um, hcr_centroids_um,
                accepted_matches, radius_um=radius,
            )

        # Image patch NCC at 3 scales
        for ps in patch_sizes_um:
            key = f"ncc_{ps}um"
            if czstack_vol is not None and hcr_zarr_path is not None:
                try:
                    cz_px = cz_proj_idx.loc[cz_id, ["projected_z", "projected_y", "projected_x"]].values
                    hcr_px = hcr_cent_idx.loc[hcr_id, ["hcr_z", "hcr_y", "hcr_x"]].values
                    # Czstack vol indexing: (z,y,x) pixel coords
                    cz_vol_px = np.array([
                        cz_proj_idx.loc[cz_id].get("czstack_z", cz_px[0]),
                        cz_proj_idx.loc[cz_id].get("czstack_y", cz_px[1]),
                        cz_proj_idx.loc[cz_id].get("czstack_x", cz_px[2]),
                    ])
                    feat[key] = compute_patch_ncc(
                        czstack_vol, hcr_zarr_path,
                        cz_vol_px, hcr_px,
                        czstack_res_um, hcr_scales,
                        patch_size_um=float(ps),
                    )
                except Exception:
                    feat[key] = np.nan
            else:
                feat[key] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_match_classifier(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    random_state: int = 42,
) -> Pipeline:
    """Train a GradientBoosting classifier with NaN imputation.

    Parameters
    ----------
    features_df : DataFrame of numeric features (NaN allowed)
    labels      : binary array (1=match, 0=non-match)

    Returns
    -------
    Fitted sklearn Pipeline (imputer + classifier)
    """
    feature_cols = [c for c in features_df.columns
                    if c not in ("czstack_cell_id", "hcr_cell_id")]
    X = features_df[feature_cols].values.astype(float)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbt", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=random_state,
        )),
    ])
    clf.fit(X, labels)
    clf.feature_names_ = feature_cols
    return clf


def predict_match_probability(
    features_df: pd.DataFrame,
    clf: Pipeline,
) -> np.ndarray:
    """Predict match probability for each candidate.

    Returns array of shape (N,) with probability of class 1.
    Missing feature columns (e.g. subject_id used in training) are filled with NaN;
    the imputer handles them transparently.
    """
    feature_cols = clf.feature_names_
    df = features_df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_cols].values.astype(float)
    return clf.predict_proba(X)[:, 1]
