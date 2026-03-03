import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def get_ids_from_landmarks(landmarks):
    columns = ['ids', 'active', 'czstack_x', 'czstack_y', 'czstack_z', 'hcr_x', 'hcr_y', 'hcr_z']
    assert len(landmarks.columns) == len(columns)
    if not all([a==b for a,b in zip(landmarks.columns, columns)]):
        landmarks.columns = columns
    matched_ids = landmarks['ids'].values
    def _get_ids(x):
        if x.startswith('cz'):
            cz_id = int(x.split('-')[0].split('cz')[-1])
            hcr_id = int(x.split('-')[1].split('hcr')[-1])
            return cz_id, hcr_id
        else:
            return -1
    
    ids = [_get_ids(x) for x in matched_ids]
    ids = [id for id in ids if id != -1]
    zstack_ids = [id[0] for id in ids]
    hcr_ids = [id[1] for id in ids]
    return zstack_ids, hcr_ids



def choose_max_count_nearest_neighbor(HCR_centroids_est, leftover_HCR_df, 
                                      feature='density', k=5, resolve_duplicates=True):
    """ Choose nearest neighbor with maximum spot counts among k nearest candidates.
    If resolve_duplicates is True, attempt to resolve many-to-one matches by reassigning duplicates

    Parameters:
    -----------
    HCR_centroids_est : ndarray of shape (M, 3)
        Estimated HCR centroids from leftover czstack centroids
    leftover_HCR_df : DataFrame of shape (N, 3)
        DataFrame containing leftover HCR centroids with columns ['hcr_x', 'hcr_y', 'hcr_z']
    feature : str
        Feature to use for matching. 'count' or 'density'
    k : int
        Number of nearest neighbors to consider
    resolve_duplicates : bool
        Whether to attempt to resolve many-to-one matches
    Returns:
    --------
    chosen_indices : ndarray of shape (M,)
        Index of matched target point for each source point (-1 if unmatched)
    chosen_distances : ndarray of shape (M,)
        Distance to matched target point for each source point (inf if unmatched)
    dist_matrix : ndarray of shape (M, N)
        Pairwise distance matrix between source and target points
    resolve_duplicates : bool
        Whether to attempt to resolve many-to-one matches
    Returns:
    --------
    chosen_indices : ndarray of shape (M,)
        Index of matched target point for each source point (-1 if unmatched)
    chosen_distances : ndarray of shape (M,)
        Distance to matched target point for each source point (inf if unmatched)
    dist_matrix : ndarray of shape (M, N)
        Pairwise distance matrix between source and target points
    """
    hcr_centroids = leftover_HCR_df[['hcr_x','hcr_y','hcr_z']].to_numpy()
    counts = leftover_HCR_df[feature].to_numpy()

    dist_matrix = distance_matrix(HCR_centroids_est, hcr_centroids)
    k = min(k, dist_matrix.shape[1])

    # Indices of k smallest distances per source (row)
    nearest_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(dist_matrix, nearest_indices, axis=1)

    # Counts for those candidates
    nearest_counts = counts[nearest_indices]

    # Pick candidate with max counts per row
    best_local_idx = np.argmax(nearest_counts, axis=1)
    row_idx = np.arange(nearest_indices.shape[0])
    chosen_indices = nearest_indices[row_idx, best_local_idx]
    chosen_distances = nearest_distances[row_idx, best_local_idx]

    # Detect many-to-one
    # targets mapped multiple times
    _, inverse, counts_per_target = np.unique(chosen_indices, return_inverse=True, return_counts=True)
    duplicate_sources_mask = counts_per_target[inverse] > 1

    if resolve_duplicates and duplicate_sources_mask.any():
        print(f'Resolving {np.sum(duplicate_sources_mask)} duplicate matches...')
        # Keep only the source with minimal distance for each duplicated target
        # For duplicates: build per-target minimal distance
        keep_mask = np.ones_like(chosen_indices, dtype=bool)
        dup_targets = np.unique(chosen_indices[duplicate_sources_mask])
        for t in dup_targets:
            sel = np.where(chosen_indices == t)[0]
            best = sel[np.argmin(chosen_distances[sel])]
            sel_remove = sel[sel != best]
            keep_mask[sel_remove] = False

        # Optionally attempt second-best for removed rows (still within their k candidates)
        removed_rows = np.where(~keep_mask)[0]
        if len(removed_rows):
            # For each removed row choose next best (by counts) that is not already taken
            taken = set(chosen_indices[keep_mask])
            # Mask out previously chosen best
            nearest_counts_removed = nearest_counts[removed_rows].copy()
            nearest_counts_removed[np.arange(len(removed_rows)), best_local_idx[removed_rows]] = -1
            # Iterate only over removed rows (small set)
            for j, r in enumerate(removed_rows):
                # Sort candidates by counts descending
                cand_order = np.argsort(-nearest_counts_removed[j])
                for cand_pos in cand_order:
                    cand_target = nearest_indices[r, cand_pos]
                    if cand_target not in taken:
                        chosen_indices[r] = cand_target
                        chosen_distances[r] = nearest_distances[r, cand_pos]
                        taken.add(cand_target)
                        break
                else:
                    # Could not find alternative unique target; mark unmatched
                    chosen_indices[r] = -1
                    chosen_distances[r] = np.inf

    return chosen_indices, chosen_distances, dist_matrix


# One-to-One Matching Algorithm
# Solves the many-to-one problem by iteratively assigning best matches

def one_to_one_matching(HCR_centroids_est, leftover_HCR_df):
    """
    Solve one-to-one matching from distance matrix.
    
    Parameters:
    -----------
    HCR_centroids_est : ndarray of shape (M, 3)
        Estimated HCR centroids from leftover czstack centroids
    leftover_HCR_df : DataFrame of shape (N, 3)
        DataFrame containing leftover HCR centroids with columns ['hcr_x', 'hcr_y', 'hcr_z']
    
    Returns:
    --------
    matched_indices : ndarray of shape (M,)
        Index of matched target point for each source point (-1 if unmatched)
    matched_distances : ndarray of shape (M,)
        Distance to matched target point for each source point (inf if unmatched)
    dist_matrix : ndarray of shape (M, N)
        Pairwise distance matrix between source and target points
    """
    dist_matrix = distance_matrix(HCR_centroids_est, leftover_HCR_df[['hcr_x','hcr_y','hcr_z']].values)
    M, N = dist_matrix.shape
    
    # Initialize output arrays
    matched_indices = np.full(M, -1, dtype=int)
    matched_distances = np.full(M, np.inf)
    
    # Keep track of which target points are already assigned
    used_targets = np.zeros(N, dtype=bool)
    
    # Create a copy of distance matrix that we can modify
    working_dist_matrix = dist_matrix.copy()
    
    # Process matches iteratively
    for iteration in range(min(M, N)):
        # Find the global minimum in the remaining matrix
        flat_idx = np.argmin(working_dist_matrix)
        source_idx, target_idx = np.unravel_index(flat_idx, working_dist_matrix.shape)
        
        # Record this match
        matched_indices[source_idx] = target_idx
        matched_distances[source_idx] = working_dist_matrix[source_idx, target_idx]
        
        # Mark this target as used
        used_targets[target_idx] = True
        
        # Remove this source row and target column from consideration
        working_dist_matrix[source_idx, :] = np.inf
        working_dist_matrix[:, target_idx] = np.inf
    
    return matched_indices, matched_distances, dist_matrix

