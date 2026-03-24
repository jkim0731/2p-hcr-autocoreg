"""
coreg_metrics.py
QC metric computation for the final co-registration table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Default czstack voxel resolution (z, y, x) in µm/pixel
CZ_RESOLUTION_UM = np.array([1.0, 0.78, 0.78])


# ---------------------------------------------------------------------------
# NN rank
# ---------------------------------------------------------------------------

def compute_nn_rank(
    czstack_projected: np.ndarray,
    matched_hcr_indices: np.ndarray,
    hcr_pool_coords: np.ndarray,
    k: int = 10,
) -> pd.Series:
    """For each matched pair, compute the rank of the matched HCR by proximity.

    Parameters
    ----------
    czstack_projected    : (N, 3) projected positions in HCR space
    matched_hcr_indices  : (N,) integer indices into hcr_pool_coords
    hcr_pool_coords      : (M, 3) HCR candidate positions in HCR space

    Returns
    -------
    Series of integer ranks (1 = nearest)
    """
    tree = cKDTree(hcr_pool_coords)
    k_actual = min(k, len(hcr_pool_coords))
    dists, indices = tree.query(czstack_projected, k=k_actual)

    ranks = []
    for i, j in enumerate(matched_hcr_indices):
        if j < 0:
            ranks.append(np.nan)
            continue
        row_idx = np.where(indices[i] == j)[0]
        if len(row_idx) == 0:
            ranks.append(k + 1)
        else:
            ranks.append(int(row_idx[0]) + 1)

    return pd.Series(ranks)


# ---------------------------------------------------------------------------
# Mutual NN (final check)
# ---------------------------------------------------------------------------

def compute_mutual_nn_final(
    matched_df: pd.DataFrame,
    czstack_projected_all: np.ndarray,
    hcr_all_coords: np.ndarray,
) -> pd.Series:
    """Final mutual-NN check for all matched pairs.

    Parameters
    ----------
    matched_df              : DataFrame with czstack row index and hcr row index
    czstack_projected_all   : (N_cz, 3) all czstack projected positions
    hcr_all_coords          : (N_hcr, 3) all HCR positions

    Returns
    -------
    Boolean Series
    """
    tree_P = cKDTree(czstack_projected_all)
    _, backward_idx = tree_P.query(hcr_all_coords, k=1)

    results = []
    for _, row in matched_df.iterrows():
        cz_idx = int(row.get("cz_row_idx", -1))
        hcr_idx = int(row.get("hcr_row_idx", -1))
        if cz_idx < 0 or hcr_idx < 0:
            results.append(False)
            continue
        results.append(bool(backward_idx[hcr_idx] == cz_idx))

    return pd.Series(results)


# ---------------------------------------------------------------------------
# TPS leave-one-out residuals
# ---------------------------------------------------------------------------

def compute_tps_loo_residuals(
    active_landmarks: pd.DataFrame,
    hcr_scales: dict,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
) -> pd.Series:
    """Leave-one-out TPS residuals for each active landmark.

    For each landmark i, fit TPS on the remaining landmarks and predict
    the HCR position of landmark i; compute Euclidean error in µm.

    Parameters
    ----------
    active_landmarks : DataFrame with ids, czstack_x/y/z, hcr_x/y/z (pixels)
    hcr_scales       : dict with scale_z/y/x in µm/px

    Returns
    -------
    Series of residuals in µm (indexed as active_landmarks)
    """
    from coreg_matching import fit_tps, project_czstack_to_hcr

    df = active_landmarks[active_landmarks["active"] == True].copy().reset_index(drop=True)
    if len(df) < 5:
        return pd.Series([np.nan] * len(df))

    hcr_res = np.array([hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]])
    residuals = []

    for i in range(len(df)):
        loo = df.drop(index=i)
        query = df.iloc[[i]]
        try:
            tps = fit_tps(loo, czstack_res_um)
            proj = project_czstack_to_hcr(query, tps, czstack_res_um)  # (1, 3) z/y/x pixels
            true_hcr = np.array([
                float(query.iloc[0]["hcr_z"]),
                float(query.iloc[0]["hcr_y"]),
                float(query.iloc[0]["hcr_x"]),
            ])
            err_px = proj[0] - true_hcr
            err_um = err_px * hcr_res
            residuals.append(float(np.linalg.norm(err_um)))
        except Exception:
            residuals.append(np.nan)

    return pd.Series(residuals, index=df.index)


# ---------------------------------------------------------------------------
# Full metric computation
# ---------------------------------------------------------------------------

def compute_match_metrics(
    matched_df: pd.DataFrame,
    czstack_df: pd.DataFrame,
    hcr_full_df: pd.DataFrame,
    spot_counts: pd.DataFrame,
    hcr_metrics: pd.DataFrame,
    active_landmarks: pd.DataFrame,
    hcr_scales: dict,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
) -> pd.DataFrame:
    """Compute all QC metrics for each matched pair.

    Parameters
    ----------
    matched_df       : DataFrame with czstack_cell_id, hcr_cell_id,
                       iter_matched, match_probability (from run_auto_matching)
    czstack_df       : full czstack centroid DataFrame
    hcr_full_df      : full HCR centroid DataFrame
    spot_counts      : hcr_cell_id, counts, density
    hcr_metrics      : hcr_cell_id, volume
    active_landmarks : final landmark set (for LOO residuals)
    hcr_scales       : dict with scale_z/y/x

    Returns
    -------
    DataFrame with all QC metric columns
    """
    from coreg_matching import fit_tps, project_czstack_to_hcr
    from landmark_filtering import grid_sample_landmarks

    hcr_res = np.array([hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]])

    # Fit final TPS — subsample landmarks if too many (O(N³) fit)
    try:
        active_lm = active_landmarks[active_landmarks["active"] == True].copy()
        if len(active_lm) > 400:
            result = grid_sample_landmarks(
                active_lm, interior_keep_proportion=0.15, edge_keep_proportion=0.3,
            )
            active_lm = result["sampled"]
        tps = fit_tps(active_lm, czstack_res_um)
        all_cz_proj = project_czstack_to_hcr(czstack_df, tps, czstack_res_um)
    except Exception:
        all_cz_proj = None

    # Index lookups
    spot_idx = spot_counts.set_index("hcr_cell_id")
    metrics_idx = hcr_metrics.set_index("hcr_cell_id")
    cz_idx_map = {int(r["czstack_cell_id"]): i
                  for i, r in czstack_df.iterrows()}
    hcr_idx_map = {int(r["hcr_cell_id"]): i
                   for i, r in hcr_full_df.iterrows()}

    rows = []
    for _, m in matched_df.iterrows():
        cz_id = int(m["czstack_cell_id"])
        hcr_id = int(m["hcr_cell_id"])
        row = {"czstack_cell_id": cz_id, "hcr_cell_id": hcr_id}

        # TPS projection error
        cz_i = cz_idx_map.get(cz_id, -1)
        hcr_i = hcr_idx_map.get(hcr_id, -1)
        if all_cz_proj is not None and cz_i >= 0 and hcr_i >= 0:
            proj_px = all_cz_proj[cz_i]
            true_hcr_px = hcr_full_df.iloc[hcr_i][["hcr_z", "hcr_y", "hcr_x"]].values.astype(float)
            err_um = (proj_px - true_hcr_px) * hcr_res
            row["distance_um"] = float(np.linalg.norm(err_um))
        else:
            row["distance_um"] = np.nan

        # Cell properties
        try:
            sp = spot_idx.loc[hcr_id]
            row["gfp_counts"] = float(sp["counts"])
            row["gfp_density"] = float(sp["density"])
        except KeyError:
            row["gfp_counts"] = np.nan
            row["gfp_density"] = np.nan

        try:
            mv = metrics_idx.loc[hcr_id]
            row["hcr_volume_vox"] = float(mv["volume"])
        except KeyError:
            row["hcr_volume_vox"] = np.nan

        row["iter_matched"] = int(m.get("iter_matched", -1))
        row["match_probability"] = float(m.get("match_probability", np.nan))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # LOO residuals (from landmarks)
    loo_residuals = compute_tps_loo_residuals(active_landmarks, hcr_scales, czstack_res_um)
    metrics_df["tps_loo_residual_um"] = np.nan  # default; computed separately for landmarks

    return metrics_df
