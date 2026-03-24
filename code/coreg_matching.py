"""
coreg_matching.py
TPS-based iterative matching: fit → project → candidate match → verify → accept.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from coreg_verification import extract_candidate_features, predict_match_probability
from landmark_filtering import grid_sample_landmarks
from manual_coreg_utils import choose_max_count_nearest_neighbor


# Default czstack voxel resolution (z, y, x) in µm/pixel
CZ_RESOLUTION_UM = np.array([1.0, 0.78, 0.78])


# ---------------------------------------------------------------------------
# TPS fitting and projection
# ---------------------------------------------------------------------------

def fit_tps(
    active_landmarks: pd.DataFrame,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    hcr_scales: Optional[dict] = None,
) -> tuple:
    """Fit three thin-plate-spline models (one per output dimension).

    Input coordinates: czstack pixels (x, y, z) → converted to µm
    Output coordinates: hcr pixels (x, y, z) – kept as pixels

    Parameters
    ----------
    active_landmarks : DataFrame with czstack_x/y/z, hcr_x/y/z (in pixels)

    Returns
    -------
    (rbf_x, rbf_y, rbf_z) – three Rbf objects mapping czstack µm → hcr pixels
    """
    df = active_landmarks[active_landmarks["active"] == True].copy()
    if len(df) < 4:
        raise ValueError(f"Need at least 4 active landmarks for TPS, got {len(df)}")

    # Convert czstack coords to µm for the RBF inputs
    cz_x_um = df["czstack_x"].values * czstack_res_um[2]
    cz_y_um = df["czstack_y"].values * czstack_res_um[1]
    cz_z_um = df["czstack_z"].values * czstack_res_um[0]

    hcr_x = df["hcr_x"].values
    hcr_y = df["hcr_y"].values
    hcr_z = df["hcr_z"].values

    rbf_x = Rbf(cz_x_um, cz_y_um, cz_z_um, hcr_x, function="thin_plate")
    rbf_y = Rbf(cz_x_um, cz_y_um, cz_z_um, hcr_y, function="thin_plate")
    rbf_z = Rbf(cz_x_um, cz_y_um, cz_z_um, hcr_z, function="thin_plate")

    return rbf_x, rbf_y, rbf_z


def project_czstack_to_hcr(
    czstack_centroids: pd.DataFrame,
    tps_models: tuple,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
) -> np.ndarray:
    """Project czstack centroids into HCR pixel space using TPS.

    Parameters
    ----------
    czstack_centroids : DataFrame with czstack_x/y/z columns (pixels)
    tps_models        : (rbf_x, rbf_y, rbf_z) from fit_tps()

    Returns
    -------
    projected : (N, 3) array – order (z, y, x) in HCR pixels
    """
    rbf_x, rbf_y, rbf_z = tps_models

    cz_x = czstack_centroids["czstack_x"].values * czstack_res_um[2]
    cz_y = czstack_centroids["czstack_y"].values * czstack_res_um[1]
    cz_z = czstack_centroids["czstack_z"].values * czstack_res_um[0]

    proj_x = rbf_x(cz_x, cz_y, cz_z)
    proj_y = rbf_y(cz_x, cz_y, cz_z)
    proj_z = rbf_z(cz_x, cz_y, cz_z)

    return np.stack([proj_z, proj_y, proj_x], axis=1)


# ---------------------------------------------------------------------------
# Mutual nearest-neighbour check
# ---------------------------------------------------------------------------

def check_mutual_nn(
    projected_P: np.ndarray,
    hcr_pool: np.ndarray,
    matched_hcr_indices: np.ndarray,
) -> np.ndarray:
    """For each forward match, check if the HCR cell also maps back to this czstack cell.

    Parameters
    ----------
    projected_P         : (N, 3) projected czstack positions in HCR space
    hcr_pool            : (M, 3) HCR pool positions in HCR space
    matched_hcr_indices : (N,) index into hcr_pool for each czstack cell

    Returns
    -------
    Boolean array (N,): True where match is mutual NN
    """
    tree_P = cKDTree(projected_P)
    _, backward_idx = tree_P.query(hcr_pool, k=1)

    is_mutual = np.zeros(len(projected_P), dtype=bool)
    for i, j in enumerate(matched_hcr_indices):
        if j >= 0 and backward_idx[j] == i:
            is_mutual[i] = True
    return is_mutual


# ---------------------------------------------------------------------------
# Per-iteration state
# ---------------------------------------------------------------------------

def _build_initial_state(
    seed_landmarks: pd.DataFrame,
    czstack_df: pd.DataFrame,
    hcr_df: pd.DataFrame,
    spot_counts: pd.DataFrame,
    hcr_scales: dict,
) -> dict:
    """Build the mutable state dict for the iterative matching loop."""
    return {
        "landmarks": seed_landmarks.copy(),
        "unmatched_cz_ids": set(czstack_df["czstack_cell_id"].values),
        "unmatched_hcr_ids": set(hcr_df["hcr_cell_id"].values),
        "all_matches": [],  # list of accepted match DataFrames
        "iteration": 0,
    }


def _update_unmatched(state: dict, new_matches: pd.DataFrame):
    for _, row in new_matches.iterrows():
        state["unmatched_cz_ids"].discard(int(row["czstack_cell_id"]))
        state["unmatched_hcr_ids"].discard(int(row["hcr_cell_id"]))


# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------

def run_one_iteration(
    state: dict,
    czstack_df: pd.DataFrame,
    hcr_df: pd.DataFrame,
    spot_counts: pd.DataFrame,
    hcr_metrics: pd.DataFrame,
    hcr_scales: dict,
    czstack_vol: Optional[np.ndarray],
    hcr_zarr_path: Optional[str],
    classifier,  # sklearn Pipeline or None
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    params: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Run one TPS-match-verify iteration.

    Parameters
    ----------
    state            : mutable state dict (modified in-place)
    czstack_df       : full czstack centroids DataFrame
    hcr_df           : full HCR centroids DataFrame
    spot_counts      : DataFrame with hcr_cell_id, counts, density
    hcr_metrics      : DataFrame with hcr_cell_id, volume
    hcr_scales       : dict with scale_z/y/x in µm/px
    czstack_vol      : 3D czstack volume (or None to skip NCC)
    hcr_zarr_path    : path to channel_488.zarr (or None)
    classifier       : trained Pipeline (or None → accept all MNN within threshold)
    czstack_res_um   : czstack voxel resolution
    params           : overrides for default algorithm parameters

    Returns
    -------
    new_matches : DataFrame of newly accepted matches
    state       : updated state (same dict)
    """
    p = {
        "k_neighbors": 5,
        "hcr_value_feature": "density",
        "filter_rate_from_min": 1.2,
        "accept_probability_threshold": 0.6,
        "sampling_threshold": 500,
        "patch_sizes_um": [20, 50, 100],
        "min_neighbors_for_scale": 3,
        "distance_threshold_um": 30.0,
    }
    if params:
        p.update(params)

    it = state["iteration"]
    landmarks = state["landmarks"]

    # 1. Fit TPS
    tps = fit_tps(landmarks, czstack_res_um)

    # 2. Get unmatched czstack cells
    unmatched_cz = czstack_df[
        czstack_df["czstack_cell_id"].isin(state["unmatched_cz_ids"])
    ].reset_index(drop=True)

    if len(unmatched_cz) == 0:
        return pd.DataFrame(), state

    # 3. Project unmatched czstack → HCR space
    projected = project_czstack_to_hcr(unmatched_cz, tps, czstack_res_um)
    projected_df = pd.DataFrame(
        projected, columns=["projected_z", "projected_y", "projected_x"]
    )
    projected_df["czstack_cell_id"] = unmatched_cz["czstack_cell_id"].values

    # Also include original pixel coords for NCC
    projected_df["czstack_z"] = unmatched_cz["czstack_z"].values
    projected_df["czstack_y"] = unmatched_cz["czstack_y"].values
    projected_df["czstack_x"] = unmatched_cz["czstack_x"].values

    # 4. Build HCR pool (unmatched, GFP+)
    sc_gfp = spot_counts[spot_counts["is_gfp"] == True]
    hcr_pool = hcr_df[
        hcr_df["hcr_cell_id"].isin(state["unmatched_hcr_ids"]) &
        hcr_df["hcr_cell_id"].isin(sc_gfp["hcr_cell_id"])
    ].reset_index(drop=True)

    if len(hcr_pool) == 0:
        return pd.DataFrame(), state

    # Merge density into hcr_pool for choose_max_count_nearest_neighbor
    # (skip if already present from caller who pre-merged hcr_df with spot_counts)
    missing = [c for c in ["counts", "density"] if c not in hcr_pool.columns]
    if missing:
        cols_to_merge = ["hcr_cell_id"] + [c for c in ["counts", "density"] if c in spot_counts.columns]
        hcr_pool = hcr_pool.merge(spot_counts[cols_to_merge], on="hcr_cell_id", how="left")
    hcr_pool["density"] = hcr_pool.get("density", pd.Series(0.0, index=hcr_pool.index)).fillna(0)
    hcr_pool["counts"] = hcr_pool.get("counts", pd.Series(0.0, index=hcr_pool.index)).fillna(0)

    # Rename for choose_max_count_nearest_neighbor (expects hcr_x/y/z)
    hcr_pool_coords = hcr_pool[["hcr_z", "hcr_y", "hcr_x"]].values

    # 5. Forward match
    # choose_max_count_nearest_neighbor expects (x,y,z) column order to match
    # leftover_HCR_df[['hcr_x','hcr_y','hcr_z']]; projected is (z,y,x) so reorder.
    proj_coords_xyz = projected[:, [2, 1, 0]]  # (z,y,x) → (x,y,z)
    chosen_idx, chosen_dist, _ = choose_max_count_nearest_neighbor(
        proj_coords_xyz,
        hcr_pool,
        feature=p["hcr_value_feature"],
        k=p["k_neighbors"],
        resolve_duplicates=True,
    )

    # 6. Mutual NN check
    is_mnn = check_mutual_nn(projected, hcr_pool_coords, chosen_idx)

    # 7. HCR scale for distance
    hcr_res = np.array([hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]])

    # Build candidates DataFrame
    accepted_matches_so_far = pd.concat(state["all_matches"]) if state["all_matches"] else pd.DataFrame(
        columns=["czstack_cell_id", "hcr_cell_id"]
    )

    # Build projected_um for constellation
    projected_um_df = pd.DataFrame({
        "czstack_cell_id": unmatched_cz["czstack_cell_id"].values,
        "projected_z": projected[:, 0] * hcr_res[0],
        "projected_y": projected[:, 1] * hcr_res[1],
        "projected_x": projected[:, 2] * hcr_res[2],
        "czstack_z": unmatched_cz["czstack_z"].values,
        "czstack_y": unmatched_cz["czstack_y"].values,
        "czstack_x": unmatched_cz["czstack_x"].values,
    })

    hcr_centroids_um = pd.DataFrame({
        "hcr_cell_id": hcr_pool["hcr_cell_id"].values,
        "hcr_z": hcr_pool["hcr_z"].values * hcr_res[0],
        "hcr_y": hcr_pool["hcr_y"].values * hcr_res[1],
        "hcr_x": hcr_pool["hcr_x"].values * hcr_res[2],
    })

    # Compute distances in µm
    dist_um = chosen_dist * hcr_res.mean()  # approximate

    candidates = []
    for i in range(len(unmatched_cz)):
        j = chosen_idx[i]
        if j < 0:
            continue
        hcr_row = hcr_pool.iloc[j]
        candidates.append({
            "czstack_cell_id": int(unmatched_cz.iloc[i]["czstack_cell_id"]),
            "hcr_cell_id": int(hcr_row["hcr_cell_id"]),
            "distance_um": float(dist_um[i]),
            "nn_rank": 1,
            "is_mutual_nn": bool(is_mnn[i]),
        })

    if not candidates:
        return pd.DataFrame(), state

    candidates_df = pd.DataFrame(candidates)

    # 8. Feature extraction + classifier
    if classifier is not None:
        feats = extract_candidate_features(
            candidates_df,
            projected_um_df,
            hcr_centroids_um,
            accepted_matches_so_far,
            czstack_vol,
            hcr_zarr_path,
            czstack_res_um,
            hcr_scales,
            spot_counts,
            hcr_metrics,
            patch_sizes_um=p["patch_sizes_um"],
        )
        probs = predict_match_probability(feats, classifier)
        candidates_df["match_probability"] = probs
        mask = probs >= p["accept_probability_threshold"]
    else:
        # Fallback: accept MNN within distance threshold
        candidates_df["match_probability"] = np.nan
        mask = (
            (candidates_df["is_mutual_nn"]) &
            (candidates_df["distance_um"] <= p["distance_threshold_um"])
        )

    new_matches = candidates_df[mask].copy()
    new_matches["iter_matched"] = it + 1

    if len(new_matches) == 0:
        state["iteration"] += 1
        return new_matches, state

    # 9. Build new landmark rows and update state
    new_lm_rows = []
    for _, m in new_matches.iterrows():
        cz_id = int(m["czstack_cell_id"])
        hcr_id = int(m["hcr_cell_id"])
        cz_row = czstack_df[czstack_df["czstack_cell_id"] == cz_id].iloc[0]
        hcr_row = hcr_pool[hcr_pool["hcr_cell_id"] == hcr_id].iloc[0]
        new_lm_rows.append({
            "ids": f"cz{cz_id}-hcr{hcr_id}",
            "active": True,
            "czstack_x": float(cz_row["czstack_x"]),
            "czstack_y": float(cz_row["czstack_y"]),
            "czstack_z": float(cz_row["czstack_z"]),
            "hcr_x": float(hcr_row["hcr_x"]),
            "hcr_y": float(hcr_row["hcr_y"]),
            "hcr_z": float(hcr_row["hcr_z"]),
        })

    new_lm_df = pd.DataFrame(new_lm_rows)
    state["landmarks"] = pd.concat([state["landmarks"], new_lm_df], ignore_index=True)

    # 10. Spatial subsampling if too many landmarks
    active_count = (state["landmarks"]["active"] == True).sum()
    if active_count > p["sampling_threshold"]:
        result = grid_sample_landmarks(
            state["landmarks"][state["landmarks"]["active"] == True],
            interior_keep_proportion=0.15,
            edge_keep_proportion=0.3,
        )
        kept_idx = result["sampled"].index
        state["landmarks"]["active"] = False
        state["landmarks"].loc[kept_idx, "active"] = True

    state["all_matches"].append(new_matches[["czstack_cell_id", "hcr_cell_id", "iter_matched", "match_probability"]])
    _update_unmatched(state, new_matches)
    state["iteration"] += 1

    return new_matches, state


# ---------------------------------------------------------------------------
# Full auto-matching loop
# ---------------------------------------------------------------------------

def run_auto_matching(
    seed_landmarks: pd.DataFrame,
    czstack_df: pd.DataFrame,
    hcr_df: pd.DataFrame,
    spot_counts: pd.DataFrame,
    hcr_metrics: pd.DataFrame,
    hcr_scales: dict,
    czstack_vol: Optional[np.ndarray],
    hcr_zarr_path: Optional[str],
    classifier=None,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """Run the full iterative matching loop.

    Parameters
    ----------
    seed_landmarks : initial landmark DataFrame (from step_2)
    czstack_df     : full czstack centroid DataFrame
    hcr_df         : full HCR centroid DataFrame
    spot_counts    : hcr_cell_id, counts, density, is_gfp
    hcr_metrics    : hcr_cell_id, volume
    hcr_scales     : dict with scale_z/y/x
    czstack_vol    : 3D array or None
    hcr_zarr_path  : path or None
    classifier     : trained Pipeline or None
    params         : algorithm parameter overrides

    Returns
    -------
    DataFrame with all accepted matches (czstack_cell_id, hcr_cell_id, iter_matched, match_probability)
    """
    p = {
        "max_iterations": 10,
        "convergence_rate": 0.005,
    }
    if params:
        p.update(params)

    n_cz = len(czstack_df)

    # Seed state
    state = _build_initial_state(seed_landmarks, czstack_df, hcr_df, spot_counts, hcr_scales)

    # Mark seed matches as matched
    seed_cz_ids = set()
    seed_hcr_ids = set()
    for _, row in seed_landmarks[seed_landmarks["active"] == True].iterrows():
        id_str = str(row["ids"])
        if "cz" in id_str and "-hcr" in id_str:
            try:
                cz_id = int(id_str.split("-")[0].replace("cz", ""))
                hcr_id = int(id_str.split("-")[1].replace("hcr", ""))
                seed_cz_ids.add(cz_id)
                seed_hcr_ids.add(hcr_id)
            except Exception:
                pass

    state["unmatched_cz_ids"] -= seed_cz_ids
    state["unmatched_hcr_ids"] -= seed_hcr_ids

    # Record seeds as iter_matched=0 so they appear in the output
    seed_pair_rows = []
    for _, row in seed_landmarks[seed_landmarks["active"] == True].iterrows():
        id_str = str(row["ids"])
        if "cz" in id_str and "-hcr" in id_str:
            try:
                cz_id = int(id_str.split("-")[0].replace("cz", ""))
                hcr_id = int(id_str.split("-")[1].replace("hcr", ""))
                seed_pair_rows.append({"czstack_cell_id": cz_id, "hcr_cell_id": hcr_id,
                                       "iter_matched": 0, "match_probability": float("nan")})
            except Exception:
                pass
    if seed_pair_rows:
        state["all_matches"].append(pd.DataFrame(seed_pair_rows))

    print(f"Starting auto-matching: {len(seed_cz_ids)} seed matches, "
          f"{len(state['unmatched_cz_ids'])} czstack cells remaining")

    # Subsample initial landmarks if they exceed sampling_threshold
    # (avoids O(N³) TPS fitting with too many seed landmarks)
    sampling_threshold = (params or {}).get("sampling_threshold", 500)
    active_count = (state["landmarks"]["active"] == True).sum()
    if active_count > sampling_threshold:
        result = grid_sample_landmarks(
            state["landmarks"][state["landmarks"]["active"] == True],
            interior_keep_proportion=0.15,
            edge_keep_proportion=0.3,
        )
        kept_idx = result["sampled"].index
        state["landmarks"]["active"] = False
        state["landmarks"].loc[kept_idx, "active"] = True
        print(f"  Subsampled initial landmarks: {active_count} → "
              f"{state['landmarks']['active'].sum()} active")

    for it in range(p["max_iterations"]):
        new_matches, state = run_one_iteration(
            state, czstack_df, hcr_df, spot_counts, hcr_metrics,
            hcr_scales, czstack_vol, hcr_zarr_path, classifier,
            czstack_res_um, params,
        )
        n_new = len(new_matches)
        n_total = n_cz - len(state["unmatched_cz_ids"])
        print(f"  Iteration {it + 1}: {n_new} new matches, "
              f"{n_total}/{n_cz} total ({100*n_total/n_cz:.1f}%)")

        if n_new == 0:
            print("  Converged: no new matches.")
            break
        if n_new < p["convergence_rate"] * n_cz:
            print("  Converged: below convergence rate.")
            break

    all_matches = state["all_matches"]
    if not all_matches:
        return pd.DataFrame(columns=["czstack_cell_id", "hcr_cell_id", "iter_matched", "match_probability"])

    return pd.concat(all_matches, ignore_index=True)
