"""
coreg_alignment.py
Rigid alignment utilities: rotation search, density cross-correlation, seed extraction.

Coordinate convention throughout:
  - czstack space: (z, y, x) pixels; resolutions cz_res_um = (1.0, 0.78, 0.78) µm/px
  - HCR space:     (z, y, x) pixels; resolutions from load_hcr_scales()
  - All internal computations in µm
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Default czstack voxel resolution (z, y, x) in µm/pixel
CZ_RESOLUTION_UM = np.array([1.0, 0.78, 0.78])


# ---------------------------------------------------------------------------
# Rotation matrix helpers
# ---------------------------------------------------------------------------

def rotation_matrix_euler(
    theta_z_deg: float,
    theta_x_deg: float = 0.0,
    theta_y_deg: float = 0.0,
) -> np.ndarray:
    """Build a 3×3 rotation matrix from intrinsic Z-X-Y Euler angles (degrees).

    Convention: R = Ry @ Rx @ Rz  (applied right-to-left)
    """
    tz = np.deg2rad(theta_z_deg)
    tx = np.deg2rad(theta_x_deg)
    ty = np.deg2rad(theta_y_deg)

    Rz = np.array([
        [np.cos(tz), -np.sin(tz), 0],
        [np.sin(tz),  np.cos(tz), 0],
        [0, 0, 1],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tx), -np.sin(tx)],
        [0, np.sin(tx),  np.cos(tx)],
    ])
    Ry = np.array([
        [np.cos(ty), 0, np.sin(ty)],
        [0, 1, 0],
        [-np.sin(ty), 0, np.cos(ty)],
    ])
    return Ry @ Rx @ Rz


def euler_from_rotation_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix into Z-X-Y Euler angles (degrees).

    Returns (theta_z_deg, theta_x_deg, theta_y_deg).
    Assumes R = Ry @ Rx @ Rz convention.
    """
    # Ry @ Rx @ Rz expansion
    # R[2,0] = -sin(ty)*cos(tx)  → approximation for small tilts
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy > 1e-6:
        theta_y = np.arctan2(sy, cy)
        theta_x = np.arctan2(R[2, 1], R[2, 2])
        theta_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        theta_y = np.arctan2(sy, cy)
        theta_x = np.arctan2(-R[1, 2], R[1, 1])
        theta_z = 0.0
    return (
        float(np.rad2deg(theta_z)),
        float(np.rad2deg(theta_x)),
        float(np.rad2deg(theta_y)),
    )


# ---------------------------------------------------------------------------
# Apply rigid transform
# ---------------------------------------------------------------------------

def apply_rotation(
    centroids_um: np.ndarray,
    R: np.ndarray,
    center_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate centroid coordinates (N×3, order z/y/x) around a centre point.

    Parameters
    ----------
    centroids_um : (N, 3) array in µm, order (z, y, x)
    R            : 3×3 rotation matrix
    center_um    : rotation centre; defaults to centroid mean
    """
    if center_um is None:
        center_um = centroids_um.mean(axis=0)
    shifted = centroids_um - center_um
    rotated = (R @ shifted.T).T
    return rotated + center_um


def apply_rigid(
    centroids_um: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Apply R then translation t (both in µm).  result = R @ x + t"""
    return (R @ centroids_um.T).T + t


# ---------------------------------------------------------------------------
# Density volumes and cross-correlation
# ---------------------------------------------------------------------------

def centroids_to_density_volume(
    centroids_um: np.ndarray,
    voxel_um: float = 5.0,
    sigma_um: float = 5.0,
    bounds: Optional[tuple] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise centroid positions into a 3D Gaussian density volume.

    Parameters
    ----------
    centroids_um : (N, 3) array, order (z, y, x) in µm
    voxel_um     : voxel size in µm (isotropic)
    sigma_um     : Gaussian smoothing sigma in µm
    bounds       : optional ((z0,z1),(y0,y1),(x0,x1)) in µm; auto-computed if None

    Returns
    -------
    volume : 3D ndarray (float32)
    origin : (3,) array with the µm coordinate of voxel [0,0,0]
    """
    if bounds is None:
        mins = centroids_um.min(axis=0) - 3 * sigma_um
        maxs = centroids_um.max(axis=0) + 3 * sigma_um
    else:
        mins = np.array([b[0] for b in bounds], dtype=float)
        maxs = np.array([b[1] for b in bounds], dtype=float)

    shape = np.ceil((maxs - mins) / voxel_um).astype(int) + 1

    # Clip to sane size
    shape = np.minimum(shape, 512)

    vol = np.zeros(shape, dtype=np.float32)
    idx = ((centroids_um - mins) / voxel_um).astype(int)
    # Clip to volume bounds
    valid = np.all((idx >= 0) & (idx < shape), axis=1)
    idx = idx[valid]
    np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)

    sigma_vox = sigma_um / voxel_um
    vol = gaussian_filter(vol, sigma=sigma_vox).astype(np.float32)
    return vol, mins


def cross_correlate_translation(
    vol_moving: np.ndarray,
    vol_fixed: np.ndarray,
    voxel_um: float = 5.0,
) -> np.ndarray:
    """Find the translation that best aligns vol_moving to vol_fixed via 3D FFT.

    Returns (tz, ty, tx) in µm.
    """
    # Pad to same shape
    shape = np.maximum(vol_moving.shape, vol_fixed.shape)
    def pad(v, s):
        p = np.zeros(s, dtype=np.float32)
        sl = tuple(slice(0, d) for d in v.shape)
        p[sl] = v
        return p

    m = pad(vol_moving, shape)
    f = pad(vol_fixed, shape)

    # Normalise
    m = (m - m.mean()) / (m.std() + 1e-8)
    f = (f - f.mean()) / (f.std() + 1e-8)

    # FFT cross-correlation
    F = np.fft.fftn(f)
    M = np.fft.fftn(m)
    cc = np.fft.ifftn(F * np.conj(M)).real

    # Peak
    peak = np.unravel_index(np.argmax(cc), cc.shape)

    # Convert to signed shift (handle wrap-around)
    shift = np.array(peak, dtype=float)
    for i, s in enumerate(shape):
        if shift[i] > s / 2:
            shift[i] -= s

    return shift * voxel_um  # (tz, ty, tx) in µm


# ---------------------------------------------------------------------------
# Alignment scoring
# ---------------------------------------------------------------------------

def score_alignment(
    P_rot_um: np.ndarray,
    Q_um: np.ndarray,
    threshold_um: float = 15.0,
) -> int:
    """Count czstack cells in P_rot_um that have a GFP+ neighbour in Q_um
    within threshold_um µm."""
    if len(P_rot_um) == 0 or len(Q_um) == 0:
        return 0
    tree = cKDTree(Q_um)
    nn_dist, _ = tree.query(P_rot_um, k=1)
    return int((nn_dist <= threshold_um).sum())


def score_mutual_nn(
    P_rot_um: np.ndarray,
    Q_um: np.ndarray,
    threshold_um: float = 20.0,
) -> int:
    """Count mutual nearest-neighbour pairs within threshold_um.

    Far more discriminative than one-sided NN: a random alignment produces
    ~0 mutual NNs, while the correct alignment produces O(10–100).
    """
    if len(P_rot_um) == 0 or len(Q_um) == 0:
        return 0
    tree_Q = cKDTree(Q_um)
    tree_P = cKDTree(P_rot_um)
    dist_fwd, idx_fwd = tree_Q.query(P_rot_um, k=1)
    _, idx_bwd = tree_P.query(Q_um, k=1)
    # P_i is a mutual NN with Q[idx_fwd[i]] if:
    #   dist_fwd[i] <= threshold  AND  idx_bwd[idx_fwd[i]] == i
    valid_fwd = dist_fwd <= threshold_um
    mutual = idx_bwd[idx_fwd] == np.arange(len(P_rot_um))
    return int((valid_fwd & mutual).sum())


# ---------------------------------------------------------------------------
# Rotation search (Method A)
# ---------------------------------------------------------------------------

def _alignment_score_neg(
    params: np.ndarray,
    P_um: np.ndarray,
    Q_um: np.ndarray,
    voxel_um: float,
    threshold_um: float,
) -> float:
    """Nelder-Mead objective: negated one-sided NN score."""
    tz, tx, ty, dz, dy, dx = params
    R = rotation_matrix_euler(tz, tx, ty)
    P_rot = (R @ P_um.T).T + np.array([dz, dy, dx])
    return -float(score_alignment(P_rot, Q_um, threshold_um))


def _mnn_score_neg(
    params: np.ndarray,
    P_um: np.ndarray,
    Q_um: np.ndarray,
    voxel_um: float,
    threshold_um: float,
) -> float:
    """Nelder-Mead objective: negated mutual-NN score."""
    tz, tx, ty, dz, dy, dx = params
    R = rotation_matrix_euler(tz, tx, ty)
    P_rot = (R @ P_um.T).T + np.array([dz, dy, dx])
    return -float(score_mutual_nn(P_rot, Q_um, threshold_um))


def rotation_search(
    P_um: np.ndarray,
    Q_um: np.ndarray,
    template: Optional[dict] = None,
    voxel_um: float = 5.0,
    threshold_um: float = 20.0,
    n_refine: int = 3,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Grid search over rotation angles, then refine top candidates.

    Coordinate convention: centroids are (z, y, x) in µm.  In
    `rotation_matrix_euler(tz, tx, ty)` = Ry(ty) @ Rx(tx) @ Rz(tz):
      - Rx(tx) keeps dim-0 (z/depth) fixed  → tx ≈ ±173° is the main
        ~180° rotation around the tissue depth axis (proper rotation, det=+1, NOT a flip/mirror).
      - Rz(tz) and Ry(ty) are small tilts (pitch / roll).

    Template field mapping:
      - ``pitch_range_deg``       → main Rx rotation magnitude (|tx|); both ± searched
      - ``z_rotation_range_deg``  → theta_z (Rz) tilt range
      - ``roll_range_deg``        → theta_y (Ry) tilt range

    Grid search uses centroid-based translation (no FFT per candidate, fast).
    Nelder-Mead refines the top-N candidates with all 6 DOF.

    Parameters
    ----------
    P_um         : czstack centroids in µm, shape (N, 3), order (z, y, x)
    Q_um         : HCR GFP+ centroids in µm, shape (M, 3)
    template     : dict from coreg_transform_template.json
    voxel_um     : density voxel size (used only if CC fallback needed)
    threshold_um : mutual-NN threshold for scoring (µm)
    n_refine     : number of top rotation candidates to refine

    Returns
    -------
    R_best   : (3, 3) rotation matrix
    t_best   : (3,) translation vector in µm  (result = R @ P_um + t_best)
    score    : mutual-NN score at best alignment (vs all Q)
    """
    if template is None:
        template = {
            "z_rotation_range_deg": [-15, 15],
            "pitch_range_deg": [-186, 186],
            "roll_range_deg": [-15, 20],
        }

    # Template field mapping:
    #   pitch_range_deg      → theta_x (main ~180° rotation), abs values searched ±
    #   z_rotation_range_deg → theta_z tilt
    #   roll_range_deg       → theta_y tilt
    p_lo, p_hi = template["pitch_range_deg"]       # main rotation (abs → ±)
    z_lo, z_hi = template["z_rotation_range_deg"]  # theta_z tilt
    r_lo, r_hi = template["roll_range_deg"]         # theta_y tilt

    step = 5
    # Main rotation: search abs range of pitch with both signs
    tx_abs_lo = max(0.0, min(abs(p_lo), abs(p_hi)) - step)
    tx_abs_hi = max(abs(p_lo), abs(p_hi)) + step
    tx_pos = np.arange(tx_abs_lo, tx_abs_hi + 0.1, step)
    tx_main_angles = np.concatenate([-tx_pos, tx_pos])

    # Small tilts
    tz_tilt_angles = np.arange(z_lo, z_hi + 0.1, step)
    ty_tilt_angles = np.arange(r_lo, r_hi + 0.1, step)

    n_cands = len(tx_main_angles) * len(tz_tilt_angles) * len(ty_tilt_angles)

    if verbose:
        print(f"Rotation search: {n_cands} candidates")
        print(f"  tx (main ~180° rotation): {len(tx_main_angles)} values "
              f"[±{tx_abs_lo:.0f}..{tx_abs_hi:.0f}°, step={step}°]")
        print(f"  tz (tilt): {len(tz_tilt_angles)} values [{z_lo:.0f}..{z_hi:.0f}°]")
        print(f"  ty (tilt): {len(ty_tilt_angles)} values [{r_lo:.0f}..{r_hi:.0f}°]")

    # Grid search: centroid-based translation (fast — no FFT per candidate)
    Q_center = Q_um.mean(axis=0)

    results = []
    for tx in tx_main_angles:          # main ~180° rotation (Rx)
        for tz in tz_tilt_angles:      # small tilt (Rz)
            for ty in ty_tilt_angles:  # small tilt (Ry)
                R = rotation_matrix_euler(tz, tx, ty)
                P_rot = (R @ P_um.T).T
                # Centroid-alignment translation: fast and translation-agnostic
                t_cand = Q_center - P_rot.mean(axis=0)
                P_aligned = P_rot + t_cand
                # Score with mutual-NN against all Q
                sc = score_mutual_nn(P_aligned, Q_um, threshold_um)
                results.append((tz, tx, ty, t_cand.copy(), R.copy(), sc))

    results.sort(key=lambda r: -r[-1])

    if verbose:
        print(f"Top-5 grid candidates (mutual-NN @ {threshold_um:.0f} µm):")
        for r in results[:5]:
            tz_, tx_, ty_, t_, R_, sc_ = r
            print(f"  tz={tz_:.1f}° tx={tx_:.1f}° ty={ty_:.1f}° "
                  f"t=[{t_[0]:.0f},{t_[1]:.0f},{t_[2]:.0f}] "
                  f"mnn={sc_}/{len(P_um)}")

    # ------------------------------------------------------------------
    # Refine top-N unique rotation candidates with Nelder-Mead (all 6 DOF).
    # Score against all Q to avoid boundary artefacts.
    # ------------------------------------------------------------------
    seen_rotations: set = set()
    best_refined = []

    for tz0, tx0, ty0, t0, R0, sc0 in results:
        if len(best_refined) >= n_refine:
            break
        key = (round(tz0), round(tx0), round(ty0))
        if key in seen_rotations:
            continue
        seen_rotations.add(key)

        x0 = np.array([tz0, tx0, ty0, t0[0], t0[1], t0[2]])
        res = minimize(
            _mnn_score_neg,
            x0,
            args=(P_um, Q_um, voxel_um, threshold_um),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 0.3, "fatol": 1},
        )
        tz_r, tx_r, ty_r, dz, dy, dx = res.x
        R_r = rotation_matrix_euler(tz_r, tx_r, ty_r)
        t_r = np.array([dz, dy, dx])
        P_r = (R_r @ P_um.T).T + t_r
        sc_r = score_mutual_nn(P_r, Q_um, threshold_um)
        best_refined.append((R_r, t_r, sc_r))

        if verbose:
            tz_dec, tx_dec, ty_dec = euler_from_rotation_matrix(R_r)
            print(f"  Refined: tz={tz_dec:.2f}° tx={tx_dec:.2f}° ty={ty_dec:.2f}° "
                  f"t=[{t_r[0]:.0f},{t_r[1]:.0f},{t_r[2]:.0f}] "
                  f"mnn={sc_r}/{len(P_um)}")

    if not best_refined:
        tz0, tx0, ty0, t0, R0, sc0 = results[0]
        best_refined = [(R0, t0, sc0)]

    best_refined.sort(key=lambda r: -r[-1])
    R_best, t_best, score_best = best_refined[0]
    return R_best, t_best, score_best


# ---------------------------------------------------------------------------
# Seed landmark extraction
# ---------------------------------------------------------------------------

def extract_seed_landmarks(
    P_aligned_um: np.ndarray,
    czstack_df: pd.DataFrame,
    Q_um: np.ndarray,
    hcr_df: pd.DataFrame,
    hcr_scales: dict,
    czstack_resolution_um: np.ndarray = CZ_RESOLUTION_UM,
    threshold_um: float = 15.0,
) -> pd.DataFrame:
    """Find mutual-nearest-neighbour pairs after rigid alignment.

    Parameters
    ----------
    P_aligned_um : czstack centroids after rigid transform, µm (N, 3) z/y/x
    czstack_df   : DataFrame with czstack_cell_id, czstack_z/y/x in pixels
    Q_um         : HCR GFP+ centroids in µm (M, 3) z/y/x
    hcr_df       : DataFrame with hcr_cell_id, hcr_z/y/x in pixels
    hcr_scales   : dict with scale_z/y/x in µm/px
    threshold_um : MNN distance threshold

    Returns
    -------
    landmarks DataFrame with columns: ids, active, czstack_x/y/z, hcr_x/y/z
    """
    # Forward: P → nearest Q
    tree_Q = cKDTree(Q_um)
    dist_fwd, idx_fwd = tree_Q.query(P_aligned_um, k=1)

    # Backward: Q → nearest P
    tree_P = cKDTree(P_aligned_um)
    dist_bwd, idx_bwd = tree_P.query(Q_um, k=1)

    # Mutual nearest neighbours
    N = len(P_aligned_um)
    lm_rows = []
    for i in range(N):
        j = idx_fwd[i]
        if dist_fwd[i] <= threshold_um and idx_bwd[j] == i:
            cz_row = czstack_df.iloc[i]
            hcr_row = hcr_df.iloc[j]
            cz_id = int(cz_row["czstack_cell_id"])
            hcr_id = int(hcr_row["hcr_cell_id"])
            lm_rows.append({
                "ids": f"cz{cz_id}-hcr{hcr_id}",
                "active": True,
                "czstack_x": float(cz_row["czstack_x"]),
                "czstack_y": float(cz_row["czstack_y"]),
                "czstack_z": float(cz_row["czstack_z"]),
                "hcr_x": float(hcr_row["hcr_x"]),
                "hcr_y": float(hcr_row["hcr_y"]),
                "hcr_z": float(hcr_row["hcr_z"]),
            })

    return pd.DataFrame(lm_rows)


# ---------------------------------------------------------------------------
# Method C: RANSAC descriptor matching
# ---------------------------------------------------------------------------

def compute_cell_descriptors(
    centroids_um: np.ndarray,
    radii_um: tuple = (10, 20, 40, 80),
) -> np.ndarray:
    """Compute neighbor-count descriptors for each cell at multiple radii.

    Returns (N, len(radii_um)) array.
    """
    tree = cKDTree(centroids_um)
    descs = []
    for r in radii_um:
        counts = np.array([len(tree.query_ball_point(p, r)) - 1
                           for p in centroids_um], dtype=float)
        descs.append(counts)
    return np.stack(descs, axis=1)


# ---------------------------------------------------------------------------
# XY-only mutual NN scoring (robust to Z-scale difference)
# ---------------------------------------------------------------------------

def score_mutual_nn_xy(
    P_um: np.ndarray,
    Q_um: np.ndarray,
    xy_threshold_um: float = 18.0,
) -> int:
    """Mutual NN count using only Y,X coordinates.

    The main rotation (~180° around the tissue depth axis) preserves XY
    distances, while the Z scale differs by 2.34–3.46× between CZ and HCR
    due to tissue expansion from clearing.  Matching in 2D YX is therefore far more
    reliable than 3D matching during initial alignment.

    Parameters
    ----------
    P_um : (N, 3) array, order (z, y, x) in µm — CZ aligned
    Q_um : (M, 3) array, order (z, y, x) in µm — HCR GFP+
    xy_threshold_um : mutual-NN radius in the YX plane (µm)

    Returns
    -------
    Number of mutual NN pairs in 2D YX
    """
    if len(P_um) == 0 or len(Q_um) == 0:
        return 0
    P_yx = P_um[:, 1:]   # (N, 2)
    Q_yx = Q_um[:, 1:]   # (M, 2)
    tree_Q = cKDTree(Q_yx)
    tree_P = cKDTree(P_yx)
    dist_fwd, idx_fwd = tree_Q.query(P_yx, k=1)
    _, idx_bwd = tree_P.query(Q_yx, k=1)
    valid_fwd = dist_fwd <= xy_threshold_um
    mutual = idx_bwd[idx_fwd] == np.arange(len(P_um))
    return int((valid_fwd & mutual).sum())


# ---------------------------------------------------------------------------
# SVD Procrustes — optimal rigid fit helper
# ---------------------------------------------------------------------------

def _rigid_svd_3d(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform src → dst via SVD Procrustes.

    Returns (R, t) where R is (3, 3) and t is (3,), such that
    ``R @ src_i + t`` minimises squared distance to dst_i.
    """
    sc = src_pts.mean(0)
    dc = dst_pts.mean(0)
    H = (src_pts - sc).T @ (dst_pts - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dc - R @ sc
    return R, t


# ---------------------------------------------------------------------------
# Constellation matching: find HCR groups matching CZ XY pairwise distances
# ---------------------------------------------------------------------------

def _find_hcr_constellations(
    cz_yx: np.ndarray,
    hcr_yx: np.ndarray,
    tol_um: float,
    max_total: int = 50,
) -> list:
    """Find all groups of HCR cells whose XY pairwise distances match a CZ
    constellation within tolerance.

    Uses a pruned recursive search: the anchor h0 fixes the first CZ cell,
    then for each subsequent CZ cell the algorithm only considers HCR cells
    that satisfy ALL pairwise distance constraints against already-chosen
    HCR cells — branches are pruned early, keeping runtime low even for
    moderately large pools (M ~ 50–150).

    Parameters
    ----------
    cz_yx   : (k, 2) array, CZ cell YX positions in µm
    hcr_yx  : (M, 2) array, HCR candidate cell YX positions (search region)
    tol_um  : pairwise XY distance tolerance in µm
    max_total : stop after this many matches (prevents runaway for common patterns)

    Returns
    -------
    List of tuples of integer indices into hcr_yx
    """
    k = len(cz_yx)
    M = len(hcr_yx)
    if M < k:
        return []

    from scipy.spatial.distance import cdist as _cdist

    # Precompute pairwise distances with scipy (much faster than nested Python loops)
    cz_dists  = _cdist(cz_yx,  cz_yx)   # (k, k)
    hcr_dists = _cdist(hcr_yx, hcr_yx)  # (M, M)

    matches: list = []

    def _recurse(depth: int, chosen: list) -> None:
        if len(matches) >= max_total:
            return
        if depth == k:
            matches.append(tuple(chosen))
            return
        for h in range(M):
            if h in chosen:
                continue
            # Prune: check distance constraint against every already-chosen cell
            ok = True
            for prev_depth, prev_h in enumerate(chosen):
                if abs(hcr_dists[h, prev_h] - cz_dists[depth, prev_depth]) > tol_um:
                    ok = False
                    break
            if ok:
                _recurse(depth + 1, chosen + [h])

    _recurse(0, [])
    return matches


# ---------------------------------------------------------------------------
# Constellation-based seed finding — main initial alignment entry point
# ---------------------------------------------------------------------------

def find_seed_constellation(
    czstack_df: pd.DataFrame,
    hcr_gfp_df: pd.DataFrame,
    hcr_all_um: np.ndarray,
    hcr_scales: dict,
    template: Optional[dict] = None,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    surface_z_min_um: float = 5.0,
    surface_z_max_um: float = 60.0,
    constellation_size: int = 4,
    hcr_z_slack_frac: float = 0.15,
    hcr_xy_slack_um: float = 200.0,
    dist_tolerance_um: float = 10.0,
    mnn_threshold_um: float = 22.0,
    min_mnn_score: int = 10,
    max_hcr_pool: int = 100,
    n_try_constellations: int = 100,
    rng_seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Find initial seed landmarks by matching near-surface CZ constellations in HCR.

    The main rotation (~180° around the tissue-depth axis) preserves XY
    pairwise distances between cells, making them suitable fingerprints even
    when the Z scale is unknown (2.34–3.46×).  This function mimics the
    manual process: pick a distinctive cluster of 3–6 near-surface cells,
    search for the same spatial pattern in the HCR GFP+ field at the
    expected Z/XY location, then expand outward once seeds are confirmed.

    Algorithm
    ---------
    1. Select CZ cells near the imaging surface (z in [surface_z_min_um, surface_z_max_um]).
    2. Define a constrained HCR search region using the calibration template:
       - Z: margin_z_min_frac ± slack, plus the depth extent of near-surface cells × z_scale.
       - XY: centred on cz_center_frac_(y|x) of HCR volume, radius = hcr_xy_slack_um.
    3. Sample up to n_try_constellations random CZ groups of constellation_size cells.
    4. For each group, call _find_hcr_constellations (XY pairwise distance matching).
    5. Sort groups by ascending HCR match count (1 = most distinctive / unambiguous).
    6. For each candidate constellation (best first):
       a. Compute a 3D rigid transform from the k matched pairs (SVD Procrustes).
       b. Apply to near-surface CZ cells; score with score_mutual_nn_xy.
       c. Stop when XY-MNN score ≥ min_mnn_score.
    7. Extract seed landmarks with extract_seed_landmarks_xy on the winning transform.

    Parameters
    ----------
    czstack_df         : DataFrame with czstack_cell_id, czstack_z/y/x (pixels)
    hcr_gfp_df         : DataFrame with hcr_cell_id, hcr_z/y/x (pixels), GFP+ only
    hcr_all_um         : (N_all, 3) all HCR cells in µm — used for volume bounds
    hcr_scales         : dict with scale_z, scale_y, scale_x in µm/px
    template           : dict from coreg_transform_template.json
    czstack_res_um     : (3,) CZ voxel resolution in µm/px  (z, y, x)
    surface_z_min_um   : min CZ depth from top surface to consider (µm)
    surface_z_max_um   : max CZ depth from top surface to consider (µm)
    constellation_size : k — cells per candidate constellation (3–5 recommended)
    hcr_z_slack_frac   : extra Z slack on each side as fraction of HCR Z range
    hcr_xy_slack_um    : XY search radius around the expected CZ-in-HCR centre (µm)
    dist_tolerance_um  : XY pairwise distance tolerance for constellation matching (µm)
    mnn_threshold_um   : XY-MNN acceptance radius for scoring rigid candidates (µm)
    min_mnn_score      : minimum XY-MNN pairs to accept a solution
    max_hcr_pool       : max HCR GFP+ cells kept in the search pool (trim by Z proximity)
    n_try_constellations : max CZ constellations sampled and evaluated
    rng_seed           : random seed for reproducible constellation sampling
    verbose            : print progress messages

    Returns
    -------
    seed_df : DataFrame with columns ids, active, czstack_x/y/z, hcr_x/y/z
              (empty if no valid seed found)
    R_best  : (3, 3) rotation matrix — None if not found
    t_best  : (3,) translation in µm — None if not found
    score   : best XY-MNN score achieved (0 if nothing found)
    """
    from math import comb as _comb
    from itertools import combinations as _combinations

    rng = np.random.default_rng(rng_seed)
    k = constellation_size

    # ------------------------------------------------------------------
    # 1.  CZ centroids → µm; select near-surface cells
    # ------------------------------------------------------------------
    cz_res = np.asarray(czstack_res_um, dtype=float)
    cz_z_um = czstack_df["czstack_z"].values.astype(float) * cz_res[0]
    cz_y_um = czstack_df["czstack_y"].values.astype(float) * cz_res[1]
    cz_x_um = czstack_df["czstack_x"].values.astype(float) * cz_res[2]
    cz_um   = np.stack([cz_z_um, cz_y_um, cz_x_um], axis=1)   # (N, 3)

    # surface_z_min/max_um are *relative* to the shallowest CZ cell so that the
    # selection is invariant to the absolute z offset of the CZ stack.
    cz_z_min  = float(cz_z_um.min())
    surf_mask = (
        (cz_z_um - cz_z_min >= surface_z_min_um)
        & (cz_z_um - cz_z_min <= surface_z_max_um)
    )
    surf_idx  = np.where(surf_mask)[0]
    if len(surf_idx) < k:
        if verbose:
            print(f"[find_seed_constellation] Only {len(surf_idx)} near-surface CZ cells "
                  f"(need {k}). Falling back to shallowest cells.")
        order     = np.argsort(cz_z_um)
        surf_idx  = order[:max(k * 4, 20)]

    if verbose:
        z_lo_abs = cz_z_min + surface_z_min_um
        z_hi_abs = cz_z_min + surface_z_max_um
        print(f"Near-surface CZ cells: {len(surf_idx)} "
              f"(relative depth {surface_z_min_um:.0f}–{surface_z_max_um:.0f} µm "
              f"→ absolute z {z_lo_abs:.0f}–{z_hi_abs:.0f} µm)")

    # ------------------------------------------------------------------
    # 2.  HCR GFP+ centroids → µm
    # ------------------------------------------------------------------
    sc = np.array([hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]],
                  dtype=float)
    hcr_z_um    = hcr_gfp_df["hcr_z"].values.astype(float) * sc[0]
    hcr_y_um    = hcr_gfp_df["hcr_y"].values.astype(float) * sc[1]
    hcr_x_um    = hcr_gfp_df["hcr_x"].values.astype(float) * sc[2]
    hcr_gfp_um  = np.stack([hcr_z_um, hcr_y_um, hcr_x_um], axis=1)   # (M, 3)

    # ------------------------------------------------------------------
    # 3.  Define HCR search region from calibration template
    # ------------------------------------------------------------------
    all_z = hcr_all_um[:, 0]
    z_hcr_min, z_hcr_max = float(all_z.min()), float(all_z.max())
    z_hcr_range = z_hcr_max - z_hcr_min

    all_y = hcr_all_um[:, 1]; all_x = hcr_all_um[:, 2]
    y_hcr_min, y_hcr_max = float(all_y.min()), float(all_y.max())
    x_hcr_min, x_hcr_max = float(all_x.min()), float(all_x.max())

    z_scale = float((template or {}).get("z_scale_mean", 2.8))

    # Z search: near-surface CZ cells (0..surface_z_max_um above CZ z-min) map to
    # approximately [HCR_z_min, HCR_z_min + margin + surface_z_max * z_scale].
    # Use template margin if available; otherwise estimate from HCR GFP+ z distribution.
    mg_z_min_frac = (template or {}).get("margin_z_min_frac_mean", None)
    if mg_z_min_frac is not None:
        mg_z_min = float(mg_z_min_frac)
    else:
        # Estimate pool Z range from the CZ and HCR z extents + z_scale.
        # Model: hcr_z = cz_z * z_scale + z_offset
        # z_offset ≈ z_hcr_min + (hcr_z_range - cz_z_extent * z_scale) / 2
        # (assumes symmetric top/bottom margins in HCR around the CZ volume)
        cz_z_extent = float(
            czstack_df["czstack_z"].max() - czstack_df["czstack_z"].min()
        ) * float(czstack_res_um[0])
        z_margin_um = max(0.0, (z_hcr_range - cz_z_extent * z_scale) / 2)
        mg_z_min = z_margin_um / z_hcr_range  # fraction

    # Z range for the constellation pool:
    # near-surface CZ cells (z = cz_z_min .. cz_z_min + surface_z_max_um) appear in HCR at:
    #   z_hcr_min + z_margin + cz_z_min * z_scale  ..  + (cz_z_min + surface_z_max) * z_scale
    cz_z_min_um = float(czstack_df["czstack_z"].min()) * float(czstack_res_um[0])
    z_margin_um = mg_z_min * z_hcr_range
    z_pool_center_lo = z_hcr_min + z_margin_um + cz_z_min_um * z_scale
    z_pool_center_hi = z_pool_center_lo + surface_z_max_um * z_scale
    # Pool Z slack: absolute µm (not a fraction) so the window stays tight.
    # hcr_z_slack_frac × z_hcr_range would be ~139 µm for a 1160 µm volume —
    # far too wide.  Use a fixed ±60 µm slack instead.
    pool_z_abs_slack = 60.0
    z_pool_lo = z_pool_center_lo - pool_z_abs_slack
    z_pool_hi = z_pool_center_hi + pool_z_abs_slack

    z_search_lo = z_pool_lo   # used only for printing / fallback
    z_search_hi = z_pool_hi

    # XY search centre: use template fractions if available, else centre of HCR volume
    cz_frac_y = (template or {}).get("cz_center_frac_y_mean", None)
    cz_frac_x = (template or {}).get("cz_center_frac_x_mean", None)
    if cz_frac_y is not None and cz_frac_x is not None:
        y_center = y_hcr_min + float(cz_frac_y) * (y_hcr_max - y_hcr_min)
        x_center = x_hcr_min + float(cz_frac_x) * (x_hcr_max - x_hcr_min)
    else:
        # Assume CZ is roughly centred in the HCR XY field
        y_center = (y_hcr_min + y_hcr_max) / 2
        x_center = (x_hcr_min + x_hcr_max) / 2

    pool_mask = (
        (hcr_z_um >= z_pool_lo) & (hcr_z_um <= z_pool_hi)
        & (hcr_y_um >= y_center - hcr_xy_slack_um)
        & (hcr_y_um <= y_center + hcr_xy_slack_um)
        & (hcr_x_um >= x_center - hcr_xy_slack_um)
        & (hcr_x_um <= x_center + hcr_xy_slack_um)
    )
    pool_idx = np.where(pool_mask)[0]   # indices into hcr_gfp_df

    if verbose:
        print(f"HCR pool region: z=[{z_pool_lo:.0f}, {z_pool_hi:.0f}] µm "
              f"(estimated CZ surface at {z_pool_center_lo:.0f} µm in HCR), "
              f"XY centre=({y_center:.0f}, {x_center:.0f}) ±{hcr_xy_slack_um:.0f} µm"
              f" → {len(pool_idx)} GFP+ cells")

    if len(pool_idx) < k:
        if verbose:
            print("Too few HCR cells in pool region; "
                  "try increasing hcr_z_slack_frac or hcr_xy_slack_um.")
        return pd.DataFrame(), None, None, 0

    # Subsample to max_hcr_pool if needed; random subsample (not by z) to avoid z-bias
    if len(pool_idx) > max_hcr_pool:
        rng2 = np.random.default_rng(rng_seed + 999)
        pool_idx = pool_idx[rng2.choice(len(pool_idx), max_hcr_pool, replace=False)]
        if verbose:
            print(f"  Random subsample to {len(pool_idx)} pool cells")

    pool_yx = np.stack([hcr_y_um[pool_idx], hcr_x_um[pool_idx]], axis=1)   # (P, 2)

    # ------------------------------------------------------------------
    # 4.  Sample CZ constellations and score each against the HCR pool
    # ------------------------------------------------------------------
    surf_yx  = np.stack([cz_y_um[surf_idx], cz_x_um[surf_idx]], axis=1)   # (S, 2)
    n_surf   = len(surf_idx)
    n_combos = _comb(n_surf, k)

    if n_combos <= n_try_constellations:
        cz_combos = list(_combinations(range(n_surf), k))
    else:
        chosen_sets: set = set()
        cz_combos = []
        for _ in range(n_try_constellations * 20):
            if len(cz_combos) >= n_try_constellations:
                break
            sel = tuple(sorted(rng.choice(n_surf, size=k, replace=False).tolist()))
            if sel not in chosen_sets:
                chosen_sets.add(sel)
                cz_combos.append(sel)

    if verbose:
        print(f"Evaluating {len(cz_combos)} CZ constellations (k={k}) "
              f"against {len(pool_idx)} HCR pool cells ...")

    constellation_results = []   # (n_hcr_matches, combo_tuple, hcr_groups)
    for combo in cz_combos:
        cz_yx_k   = surf_yx[list(combo)]   # (k, 2)
        hcr_groups = _find_hcr_constellations(cz_yx_k, pool_yx, dist_tolerance_um)
        if len(hcr_groups) == 0:
            continue
        constellation_results.append((len(hcr_groups), combo, hcr_groups))

    if not constellation_results:
        if verbose:
            print("No matching HCR constellations found. "
                  "Try increasing dist_tolerance_um or hcr_xy_slack_um.")
        return pd.DataFrame(), None, None, 0

    # Sort ascending by match count (1 = most distinctive)
    constellation_results.sort(key=lambda r: r[0])

    if verbose:
        counts = [r[0] for r in constellation_results]
        print(f"Constellations with HCR matches: {len(constellation_results)} "
              f"(match counts: min={min(counts)}, "
              f"median={int(np.median(counts))}, max={max(counts)})")

    # ------------------------------------------------------------------
    # 5.  Try best constellations → rigid fit → XY-MNN validation
    # ------------------------------------------------------------------
    best_score = 0
    best_R: Optional[np.ndarray] = None
    best_t: Optional[np.ndarray] = None

    for n_hcr, combo, hcr_groups in constellation_results:
        cz_global_idx = surf_idx[list(combo)]   # indices into czstack_df
        cz_pts_3d     = cz_um[cz_global_idx]    # (k, 3) µm

        for hcr_group in hcr_groups:
            # hcr_group: tuple of local indices into pool_yx / pool_idx
            hcr_global_idx = pool_idx[list(hcr_group)]   # indices into hcr_gfp_df
            hcr_pts_3d     = hcr_gfp_um[hcr_global_idx]  # (k, 3) µm

            try:
                R_cand, t_cand = _rigid_svd_3d(cz_pts_3d, hcr_pts_3d)
            except Exception:
                continue

            # Validate: apply to all near-surface CZ → XY-MNN vs POOL cells only.
            # Using all 72K HCR cells gives a spuriously high score because the
            # density (~345 cells within 22 µm XY radius) guarantees mutual-NN
            # at any alignment.  The pool is Z-restricted and XY-bounded so its
            # density is much lower and the score is actually discriminative.
            surf_aligned = (R_cand @ cz_um[surf_idx].T).T + t_cand
            sc_xy = score_mutual_nn_xy(surf_aligned, hcr_gfp_um[pool_idx], mnn_threshold_um)

            if sc_xy > best_score:
                best_score = sc_xy
                best_R, best_t = R_cand.copy(), t_cand.copy()
                if verbose:
                    print(f"  Improved: n_hcr_matches={n_hcr}, "
                          f"XY-MNN={sc_xy}/{len(surf_idx)} "
                          f"(need ≥{min_mnn_score})")

        if best_score >= min_mnn_score:
            break   # Good enough — stop searching

    if best_score < min_mnn_score or best_R is None:
        if verbose:
            print(f"No seed found with XY-MNN ≥ {min_mnn_score}. "
                  f"Best = {best_score}. "
                  "Try adjusting dist_tolerance_um, mnn_threshold_um, or min_mnn_score.")
        return pd.DataFrame(), None, None, best_score

    if verbose:
        print(f"Seed found: XY-MNN = {best_score}/{len(surf_idx)}")

    # ------------------------------------------------------------------
    # 6.  Extract seed landmarks via Z-and-XY restricted matching
    # ------------------------------------------------------------------
    # For each CZ cell, estimate its expected HCR z from the rigid transform,
    # then restrict the HCR pool to a ±z_window around that value before
    # doing XY-MNN.  This avoids the density-inflation problem that arises
    # when matching against all 72K HCR cells with XY-only constraints.
    all_cz_aligned = (best_R @ cz_um.T).T + best_t
    seed_df = extract_seed_landmarks_zxy(
        P_aligned_um=all_cz_aligned,
        czstack_df=czstack_df,
        hcr_gfp_um=hcr_gfp_um,
        hcr_gfp_df=hcr_gfp_df,
        z_window_um=150.0,
        xy_threshold_um=mnn_threshold_um,
    )

    if verbose:
        print(f"Seed landmarks extracted: {len(seed_df)}")

    return seed_df, best_R, best_t, best_score


# ---------------------------------------------------------------------------
# XY-based seed landmark extraction
# ---------------------------------------------------------------------------

def extract_seed_landmarks_zxy(
    P_aligned_um: np.ndarray,
    czstack_df: pd.DataFrame,
    hcr_gfp_um: np.ndarray,
    hcr_gfp_df: pd.DataFrame,
    z_window_um: float = 150.0,
    xy_threshold_um: float = 22.0,
) -> pd.DataFrame:
    """Extract seed landmarks using Z-restricted XY mutual nearest neighbours.

    For each CZ cell (after rigid alignment) the HCR search is restricted to
    cells within ``z_window_um`` of the predicted HCR z.  This prevents the
    spurious high-score matches that occur when the HCR field is searched
    globally in XY (density ~345 HCR cells per 22 µm XY radius).

    Parameters
    ----------
    P_aligned_um  : (N, 3) CZ centroids after rigid transform, µm (z, y, x)
    czstack_df    : DataFrame with czstack_cell_id, czstack_z/y/x in pixels
    hcr_gfp_um    : (M, 3) HCR GFP+ centroids in µm (z, y, x)
    hcr_gfp_df    : DataFrame with hcr_cell_id, hcr_z/y/x in pixels
    z_window_um   : half-width of Z search window around predicted HCR z (µm)
    xy_threshold_um : MNN radius in 2D YX (µm)

    Returns
    -------
    DataFrame with columns: ids, active, czstack_x/y/z, hcr_x/y/z
    """
    if len(P_aligned_um) == 0 or len(hcr_gfp_um) == 0:
        return pd.DataFrame()

    # Build a KD-tree in 3D (z, y, x) but with z scaled so that z_window_um
    # acts as the z half-width and xy_threshold_um acts as the XY radius.
    # Scale z to be in the same units as XY: divide by (z_window_um / xy_threshold_um).
    z_scale_factor = xy_threshold_um / z_window_um   # z unit → XY-equivalent units
    P_scaled = P_aligned_um * np.array([z_scale_factor, 1.0, 1.0])
    Q_scaled = hcr_gfp_um   * np.array([z_scale_factor, 1.0, 1.0])

    tree_Q = cKDTree(Q_scaled)
    tree_P = cKDTree(P_scaled)

    # Forward: each CZ cell → nearest HCR within the anisotropic radius
    # In scaled space: Euclidean distance ≤ xy_threshold_um means the XY part
    # is ≤ xy_threshold_um AND the Z part is ≤ z_window_um.
    dist_fwd, idx_fwd = tree_Q.query(P_scaled, k=1)
    _, idx_bwd = tree_P.query(Q_scaled, k=1)

    lm_rows = []
    for i in range(len(P_aligned_um)):
        j = int(idx_fwd[i])
        # Check XY distance (not scaled) separately to enforce xy_threshold_um
        xy_dist = float(np.linalg.norm(P_aligned_um[i, 1:] - hcr_gfp_um[j, 1:]))
        if xy_dist <= xy_threshold_um and idx_bwd[j] == i:
            cz_row  = czstack_df.iloc[i]
            hcr_row = hcr_gfp_df.iloc[j]
            cz_id   = int(cz_row["czstack_cell_id"])
            hcr_id  = int(hcr_row["hcr_cell_id"])
            lm_rows.append({
                "ids":       f"cz{cz_id}-hcr{hcr_id}",
                "active":    True,
                "czstack_x": float(cz_row["czstack_x"]),
                "czstack_y": float(cz_row["czstack_y"]),
                "czstack_z": float(cz_row["czstack_z"]),
                "hcr_x":     float(hcr_row["hcr_x"]),
                "hcr_y":     float(hcr_row["hcr_y"]),
                "hcr_z":     float(hcr_row["hcr_z"]),
            })

    return pd.DataFrame(lm_rows)


def extract_seed_landmarks_xy(
    P_aligned_um: np.ndarray,
    czstack_df: pd.DataFrame,
    hcr_gfp_um: np.ndarray,
    hcr_gfp_df: pd.DataFrame,
    hcr_scales: dict,
    xy_threshold_um: float = 22.0,
) -> pd.DataFrame:
    """Extract seed landmark pairs using XY-only mutual nearest neighbours.

    Parameters
    ----------
    P_aligned_um : (N, 3) CZ centroids after rigid transform, µm (z, y, x)
    czstack_df   : DataFrame with czstack_cell_id, czstack_z/y/x in pixels
    hcr_gfp_um   : (M, 3) HCR GFP+ centroids in µm (z, y, x)
    hcr_gfp_df   : DataFrame with hcr_cell_id, hcr_z/y/x in pixels
    hcr_scales   : dict with scale_z/y/x in µm/px (unused; kept for API consistency)
    xy_threshold_um : MNN radius in 2D YX (µm)

    Returns
    -------
    DataFrame with columns: ids, active, czstack_x/y/z, hcr_x/y/z
    """
    if len(P_aligned_um) == 0 or len(hcr_gfp_um) == 0:
        return pd.DataFrame()

    P_yx = P_aligned_um[:, 1:]
    Q_yx = hcr_gfp_um[:, 1:]

    tree_Q = cKDTree(Q_yx)
    tree_P = cKDTree(P_yx)
    dist_fwd, idx_fwd = tree_Q.query(P_yx, k=1)
    _, idx_bwd = tree_P.query(Q_yx, k=1)

    lm_rows = []
    for i in range(len(P_aligned_um)):
        j = int(idx_fwd[i])
        if dist_fwd[i] <= xy_threshold_um and idx_bwd[j] == i:
            cz_row  = czstack_df.iloc[i]
            hcr_row = hcr_gfp_df.iloc[j]
            cz_id   = int(cz_row["czstack_cell_id"])
            hcr_id  = int(hcr_row["hcr_cell_id"])
            lm_rows.append({
                "ids":       f"cz{cz_id}-hcr{hcr_id}",
                "active":    True,
                "czstack_x": float(cz_row["czstack_x"]),
                "czstack_y": float(cz_row["czstack_y"]),
                "czstack_z": float(cz_row["czstack_z"]),
                "hcr_x":     float(hcr_row["hcr_x"]),
                "hcr_y":     float(hcr_row["hcr_y"]),
                "hcr_z":     float(hcr_row["hcr_z"]),
            })

    return pd.DataFrame(lm_rows)


# ---------------------------------------------------------------------------
# 2D XY cross-correlation — fast initial alignment
# ---------------------------------------------------------------------------

def _build_2d_density_image(
    centroids_yx: np.ndarray,
    yx_min: np.ndarray,
    shape: tuple,
    voxel_um: float,
    sigma_um: float,
) -> np.ndarray:
    """Build 2D Gaussian density image from (N, 2) YX centroids."""
    img = np.zeros(shape, dtype=np.float32)
    idx = ((centroids_yx - yx_min) / voxel_um).astype(int)
    valid = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx = idx[valid]
    if len(idx) > 0:
        np.add.at(img, (idx[:, 0], idx[:, 1]), 1)
    sigma_vox = sigma_um / voxel_um
    return gaussian_filter(img, sigma=sigma_vox).astype(np.float32)


def find_initial_alignment_xycorr(
    czstack_df: pd.DataFrame,
    hcr_gfp_df: pd.DataFrame,
    hcr_all_um: np.ndarray,
    hcr_scales: dict,
    template: Optional[dict] = None,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    # Stage 1 — coarse footprint CC (all Z-pool HCR cells, large sigma)
    voxel_coarse_um: float = 10.0,
    sigma_coarse_um: float = 40.0,
    # Stage 2 — fine CC (subsampled HCR, multiple runs, consensus)
    voxel_fine_um: float = 4.0,
    sigma_fine_um: float = 12.0,
    n_subsample_runs: int = 30,
    subsample_size: int = 1500,
    # Common
    z_roll_half_um: float = 7.5,
    xy_threshold_um: float = 22.0,
    z_window_um: float = 150.0,
    verbose: bool = True,
) -> tuple:
    """Find initial CZ→HCR alignment by two-stage 2D XY FFT cross-correlation.

    Algorithm
    ---------
    1. Apply the ~180° rotation from the calibration template to CZ centroids.
    2. Restrict HCR GFP+ cells to the estimated Z slab (from z_scale template)
       plus ±z_roll_half_um margins for rolling-max-in-Z effect.
    3. **Stage 1 — coarse footprint CC** (all Z-pool HCR cells, large sigma):
       Both density images are smoothed with sigma_coarse_um (40 µm) so they
       represent the *volume footprint* rather than individual cells.  The CC
       of two footprint images finds the translation between volume centres to
       within ~sigma_coarse accuracy.
    4. **Stage 2 — fine consensus CC** (subsampled HCR, multiple runs):
       After applying the coarse shift, subsample the HCR pool to ~1500 cells
       (≈CZ density) and run CC with finer sigma. Repeat n_subsample_runs times
       and take the trimmed-median residual correction. Averaging over many
       subsamples suppresses noise from random sparse draws.
    5. Z offset from centroid matching; no Nelder-Mead (prone to local optima
       with a dense HCR pool).
    6. Extract seed landmark pairs with Z-restricted MNN.

    Returns
    -------
    (seed_df, R_best, t_best, score)
      seed_df : DataFrame of initial seed landmark pairs
      R_best  : (3, 3) rotation matrix
      t_best  : (3,) translation vector in µm [z, y, x]
      score   : Z-restricted XY-MNN count after alignment
    """
    if template is None:
        template = {}

    rng = np.random.default_rng(42)
    cz_res = np.asarray(czstack_res_um, dtype=float)

    def _norm2d(img: np.ndarray) -> np.ndarray:
        return (img - img.mean()) / (img.std() + 1e-8)

    def _cc_peak_shift(img_cz_n, img_hcr_n):
        """2D FFT CC → shift = HCR_position − CZ_position (in voxels)."""
        cc = np.fft.ifft2(
            np.fft.fft2(img_hcr_n) * np.conj(np.fft.fft2(img_cz_n))
        ).real
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        shift = np.array(peak, dtype=float)
        for i, s in enumerate(cc.shape):
            if shift[i] > s / 2:
                shift[i] -= s
        return shift

    # ------------------------------------------------------------------
    # 1. CZ centroids → µm
    # ------------------------------------------------------------------
    cz_z_px = czstack_df["czstack_z"].values.astype(float)
    cz_y_px = czstack_df["czstack_y"].values.astype(float)
    cz_x_px = czstack_df["czstack_x"].values.astype(float)
    cz_um = np.stack(
        [cz_z_px * cz_res[0], cz_y_px * cz_res[1], cz_x_px * cz_res[2]], axis=1
    )  # (N, 3)

    # ------------------------------------------------------------------
    # 2. Rotation from template (default: ~180° pitch / X-axis rotation)
    # ------------------------------------------------------------------
    theta_x = float(template.get("pitch_mean_deg", 180.0))
    theta_z = float(template.get("z_rotation_mean_deg", 0.0))
    theta_y = float(template.get("roll_mean_deg", 0.0))
    R_init = rotation_matrix_euler(theta_z, theta_x, theta_y)
    P_rot  = (R_init @ cz_um.T).T   # (N, 3) — rotated CZ, no translation yet

    # ------------------------------------------------------------------
    # 3. HCR GFP+ → µm; Z slab + rolling-max margin
    # ------------------------------------------------------------------
    sc = np.array(
        [hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]],
        dtype=float,
    )
    hcr_z = hcr_gfp_df["hcr_z"].values.astype(float) * sc[0]
    hcr_y = hcr_gfp_df["hcr_y"].values.astype(float) * sc[1]
    hcr_x = hcr_gfp_df["hcr_x"].values.astype(float) * sc[2]
    hcr_gfp_um = np.stack([hcr_z, hcr_y, hcr_x], axis=1)  # (M, 3)

    all_z = hcr_all_um[:, 0]
    z_hcr_min, z_hcr_max = float(all_z.min()), float(all_z.max())
    z_hcr_range = z_hcr_max - z_hcr_min

    z_scale  = float(template.get("z_scale_mean", 2.8))
    cz_z_ext = float(cz_z_px.max() - cz_z_px.min()) * cz_res[0]
    z_margin = max(0.0, (z_hcr_range - cz_z_ext * z_scale) / 2.0)

    z_pool_lo = z_hcr_min + z_margin - z_roll_half_um
    z_pool_hi = z_hcr_min + z_margin + cz_z_ext * z_scale + z_roll_half_um

    pool_mask = (hcr_z >= z_pool_lo) & (hcr_z <= z_pool_hi)
    pool_idx  = np.where(pool_mask)[0]
    if len(pool_idx) < 20:
        extra = max(100.0, (z_pool_hi - z_pool_lo) * 0.5)
        pool_mask = (hcr_z >= z_pool_lo - extra) & (hcr_z <= z_pool_hi + extra)
        pool_idx  = np.where(pool_mask)[0]
        z_pool_lo -= extra
        z_pool_hi += extra

    if verbose:
        print(f"HCR Z pool: [{z_pool_lo:.0f}, {z_pool_hi:.0f}] µm "
              f"(z_scale={z_scale:.2f}, roll={z_roll_half_um:.1f} µm) "
              f"→ {len(pool_idx):,} / {len(hcr_z):,} GFP+ cells")

    # Z offset: centroid matching (used as tz throughout)
    tz_est = float(hcr_z[pool_idx].mean() - P_rot[:, 0].mean())

    # ------------------------------------------------------------------
    # STAGE 1 — Coarse footprint CC (large sigma, all pool cells)
    # ------------------------------------------------------------------
    P_yx = P_rot[:, 1:]  # (N, 2) YX µm, no translation
    Q_pool_yx = np.stack([hcr_y[pool_idx], hcr_x[pool_idx]], axis=1)

    # Image bounds: union of CZ and HCR pool, with sigma padding
    pad_s1 = 3 * sigma_coarse_um
    yx_min_s1 = np.minimum(P_yx.min(0), Q_pool_yx.min(0)) - pad_s1
    yx_max_s1 = np.maximum(P_yx.max(0), Q_pool_yx.max(0)) + pad_s1
    shape_s1 = tuple(
        np.clip(np.ceil((yx_max_s1 - yx_min_s1) / voxel_coarse_um).astype(int) + 1, 1, 512).tolist()
    )

    img_cz_s1  = _build_2d_density_image(P_yx,       yx_min_s1, shape_s1, voxel_coarse_um, sigma_coarse_um)
    img_hcr_s1 = _build_2d_density_image(Q_pool_yx,  yx_min_s1, shape_s1, voxel_coarse_um, sigma_coarse_um)

    shift_s1 = _cc_peak_shift(_norm2d(img_cz_s1), _norm2d(img_hcr_s1))
    ty_s1, tx_s1 = shift_s1 * voxel_coarse_um  # coarse YX translation in µm

    if verbose:
        print(f"Stage 1 (footprint CC, σ={sigma_coarse_um:.0f} µm): "
              f"ty={ty_s1:.0f} µm, tx={tx_s1:.0f} µm, tz={tz_est:.0f} µm")

    # ------------------------------------------------------------------
    # STAGE 2 — Fine consensus CC (subsampled HCR, multiple runs)
    # Idea: after applying the coarse shift, the CZ footprint is roughly
    # centred within the HCR volume.  We subsample the HCR pool to ~CZ
    # density and run CC many times; the median residual correction is
    # much more stable than a single run with all dense HCR cells.
    # ------------------------------------------------------------------
    t_coarse    = np.array([tz_est, ty_s1, tx_s1])
    P_coarse_yx = P_rot[:, 1:] + t_coarse[1:]  # CZ after coarse shift (YX only)

    # Build image bounds centred on the coarse-aligned CZ
    pad_s2  = sigma_coarse_um + sigma_fine_um  # leave room for residual
    cz_yx_c = P_coarse_yx.mean(0)
    yx_ext  = P_coarse_yx.max(0) - P_coarse_yx.min(0)
    yx_min_s2 = cz_yx_c - yx_ext / 2 - pad_s2
    yx_max_s2 = cz_yx_c + yx_ext / 2 + pad_s2
    shape_s2 = tuple(
        np.clip(np.ceil((yx_max_s2 - yx_min_s2) / voxel_fine_um).astype(int) + 1, 1, 1024).tolist()
    )

    img_cz_s2 = _build_2d_density_image(P_coarse_yx, yx_min_s2, shape_s2, voxel_fine_um, sigma_fine_um)
    img_cz_s2_n = _norm2d(img_cz_s2)

    n_sub = min(subsample_size, len(pool_idx))
    peaks_s2 = []
    for _ in range(n_subsample_runs):
        sub = rng.choice(len(pool_idx), size=n_sub, replace=False)
        Q_sub_yx = np.stack([hcr_y[pool_idx[sub]], hcr_x[pool_idx[sub]]], axis=1)
        img_hcr_s2 = _build_2d_density_image(Q_sub_yx, yx_min_s2, shape_s2, voxel_fine_um, sigma_fine_um)
        shift = _cc_peak_shift(img_cz_s2_n, _norm2d(img_hcr_s2))
        peaks_s2.append(shift * voxel_fine_um)

    peaks_arr = np.array(peaks_s2)  # (n_runs, 2) in µm

    # Trimmed median: discard top/bottom 20 % of each axis independently
    trim = max(1, int(0.2 * len(peaks_arr)))
    trim_mask = np.ones(len(peaks_arr), dtype=bool)
    for axis in range(2):
        order = np.argsort(peaks_arr[:, axis])
        trim_mask[order[:trim]] = False
        trim_mask[order[-trim:]] = False
    dty_fine, dtx_fine = peaks_arr[trim_mask].mean(axis=0)

    if verbose:
        std_yx = peaks_arr[trim_mask].std(axis=0)
        print(f"Stage 2 (fine CC, {n_sub} cells × {n_subsample_runs} runs): "
              f"Δty={dty_fine:.1f}±{std_yx[0]:.1f} µm, "
              f"Δtx={dtx_fine:.1f}±{std_yx[1]:.1f} µm")

    # ------------------------------------------------------------------
    # Final transform
    # ------------------------------------------------------------------
    t_best = np.array([tz_est, ty_s1 + dty_fine, tx_s1 + dtx_fine])
    R_best = R_init

    P_aligned = (R_best @ cz_um.T).T + t_best
    score = score_mutual_nn_xy(P_aligned, hcr_gfp_um[pool_idx], xy_threshold_um)

    if verbose:
        tz_r, tx_r, ty_r = euler_from_rotation_matrix(R_best)
        print(f"Final transform: rotation tz={tz_r:.2f}° tx={tx_r:.2f}° ty={ty_r:.2f}°")
        print(f"  t (µm): z={t_best[0]:.0f}  y={t_best[1]:.0f}  x={t_best[2]:.0f}")
        print(f"  XY-MNN vs pool ({len(pool_idx):,} cells): {score}/{len(cz_um)}")

    seed_df = extract_seed_landmarks_zxy(
        P_aligned_um=P_aligned,
        czstack_df=czstack_df,
        hcr_gfp_um=hcr_gfp_um,
        hcr_gfp_df=hcr_gfp_df,
        z_window_um=z_window_um,
        xy_threshold_um=xy_threshold_um,
    )

    if verbose:
        print(f"Seed landmarks extracted: {len(seed_df)}")

    return seed_df, R_best, t_best, score


# ---------------------------------------------------------------------------
# Spread-and-verify: validate alignment by expanding from seeds
# ---------------------------------------------------------------------------

def spread_and_verify(
    R: np.ndarray,
    t: np.ndarray,
    czstack_df: pd.DataFrame,
    hcr_gfp_df: pd.DataFrame,
    hcr_gfp_um: np.ndarray,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    xy_threshold_um: float = 22.0,
    z_window_um: float = 150.0,
    min_matches: int = 30,
    verbose: bool = True,
) -> tuple:
    """Apply a rigid transform to ALL CZ cells and find Z-restricted MNN matches.

    This is a fast quality check: a *correct* alignment produces many spatially
    distributed matches across the full CZ volume; an incorrect one produces
    few or spatially clustered matches.

    Call this after `find_initial_alignment_xycorr` (or after fitting SVD from a
    constellation seed) to decide whether to accept the alignment or restart.

    Parameters
    ----------
    R, t         : rigid transform (R·x + t maps CZ µm → HCR µm)
    czstack_df   : DataFrame with czstack_cell_id, czstack_z/y/x (pixels)
    hcr_gfp_df   : DataFrame with hcr_cell_id, hcr_z/y/x (pixels)
    hcr_gfp_um   : (M, 3) HCR GFP+ centroids in µm (z, y, x)
    czstack_res_um : (3,) CZ voxel resolution in µm/px
    xy_threshold_um : MNN radius in 2D YX (µm)
    z_window_um  : Z half-window for Z-restricted matching (µm)
    min_matches  : minimum accepted matches to consider alignment valid
    verbose      : print progress

    Returns
    -------
    matches_df   : DataFrame with columns ids, active, czstack_x/y/z, hcr_x/y/z
    quality      : fraction of CZ cells matched (0–1); below ~0.05 suggests bad alignment
    is_good      : bool — True if len(matches_df) >= min_matches
    """
    cz_res = np.asarray(czstack_res_um, dtype=float)
    cz_z = czstack_df["czstack_z"].values.astype(float) * cz_res[0]
    cz_y = czstack_df["czstack_y"].values.astype(float) * cz_res[1]
    cz_x = czstack_df["czstack_x"].values.astype(float) * cz_res[2]
    cz_um = np.stack([cz_z, cz_y, cz_x], axis=1)

    P_aligned = (R @ cz_um.T).T + t

    matches_df = extract_seed_landmarks_zxy(
        P_aligned_um=P_aligned,
        czstack_df=czstack_df,
        hcr_gfp_um=hcr_gfp_um,
        hcr_gfp_df=hcr_gfp_df,
        z_window_um=z_window_um,
        xy_threshold_um=xy_threshold_um,
    )

    quality  = len(matches_df) / max(len(czstack_df), 1)
    is_good  = len(matches_df) >= min_matches

    if verbose:
        print(f"Spread-and-verify: {len(matches_df)}/{len(czstack_df)} CZ cells matched "
              f"({100*quality:.1f}%) — {'GOOD' if is_good else 'POOR (alignment may be wrong)'}")

    return matches_df, quality, is_good


# ---------------------------------------------------------------------------
# Anchor-slice alignment: scan HCR Z levels with 2D CC
# ---------------------------------------------------------------------------

def estimate_hcr_tissue_surface(
    hcr_all_um: np.ndarray,
    n_edge: int = 100,
) -> tuple[float, float, float]:
    """Robustly estimate HCR tissue bottom and top surface positions.

    Uses only cells in the **center half** of the XY extent to avoid bias from
    oblique tissue edges (where the tissue may be thin or absent at the margins).
    Returns (z_bot, z_top, thickness) in µm.

    Parameters
    ----------
    hcr_all_um : (N, 3) array of all HCR centroids in µm (z, y, x)
    n_edge     : number of extreme-Z cells to take median of for surface estimate
    """
    z = hcr_all_um[:, 0]
    y = hcr_all_um[:, 1]
    x = hcr_all_um[:, 2]

    # Center-half XY mask
    y_mid = (y.max() + y.min()) / 2
    x_mid = (x.max() + x.min()) / 2
    y_qtr = (y.max() - y.min()) / 4
    x_qtr = (x.max() - x.min()) / 4
    mask = (np.abs(y - y_mid) <= y_qtr) & (np.abs(x - x_mid) <= x_qtr)
    z_ctr = z[mask] if mask.sum() >= n_edge * 2 else z

    sorted_z = np.sort(z_ctr)
    z_bot = float(np.median(sorted_z[:n_edge]))
    z_top = float(np.median(sorted_z[-n_edge:]))
    return z_bot, z_top, z_top - z_bot


def find_anchor_slice_alignment(
    czstack_df: pd.DataFrame,
    hcr_gfp_df: pd.DataFrame,
    hcr_all_um: np.ndarray,
    hcr_scales: dict,
    template: Optional[dict] = None,
    czstack_res_um: np.ndarray = CZ_RESOLUTION_UM,
    # CZ anchor scan — near the top of the CZ stack
    cz_slab_half_um: float = 20.0,
    cz_z_step_um: float = 15.0,
    cz_anchor_max_depth_um: float = 200.0,   # only scan top 200 µm of CZ
    min_cz_cells_in_slab: int = 5,
    # HCR Z scan — fraction-based bounds on tissue thickness from z_bot
    hcr_slab_half_um: float = 20.0,
    hcr_z_step_um: float = 15.0,
    # Fraction of tissue thickness from z_bot for HCR anchor search
    # Calibrated from 4 subjects: 0.17–0.27; use [0.10, 0.40] with margin.
    hcr_frac_min: float = 0.10,
    hcr_frac_max: float = 0.40,
    # 2D CC parameters
    sigma_um: float = 15.0,
    voxel_um: float = 5.0,
    # Surface estimation
    n_edge: int = 100,
    # Seed extraction
    xy_threshold_um: float = 22.0,
    z_window_um: float = 150.0,
    verbose: bool = True,
) -> tuple:
    """Find initial CZ→HCR alignment by scanning HCR Z levels with 2D CC.

    Mirrors the manual strategy: a human picks a single shallow CZ Z slice
    (near the pial surface, ~62–101 µm depth in the 3-subject calibration),
    then visually searches for the matching HCR Z level.  The absolute HCR
    depth of the match from the tissue bottom surface was 387–493 µm across
    subjects, giving the search range below.

    Algorithm
    ---------
    1. Estimate HCR tissue surface (z_bot, z_top) using center-half XY cells.
    2. Rotate CZ centroids by ~180° (from template).
    3. For each CZ anchor z in [CZ_z_min, CZ_z_min + cz_anchor_max_depth_um]:
       - Select CZ cells in ±cz_slab_half_um slab; skip if < min_cz_cells_in_slab.
    4. For each HCR z candidate in [z_bot + hcr_search_from_bot_um,
                                     z_bot + hcr_search_to_bot_um]:
       - Select HCR GFP+ cells in ±hcr_slab_half_um slab.
       - Build 2D density images on the **union** bounding box (same coordinate
         frame for CZ and HCR → CC gives the correct (ty, tx)).
       - 2D FFT CC → (ty, tx, cc_score).
    5. Select (cz_z_anchor, hcr_z_anchor, ty, tx) with highest cc_score.
    6. Derive tz = hcr_z_anchor − cz_z_anchor (approximate; non-rigid Z is
       handled by TPS in step 3).
    7. Extract seed landmark pairs with Z-restricted MNN on the full aligned
       volume.

    Returns
    -------
    (seed_df, R_best, t_best, score, info)
      seed_df  : DataFrame of initial seed landmark pairs
      R_best   : (3, 3) rotation matrix
      t_best   : (3,) translation in µm [z, y, x]
      score    : Z-restricted XY-MNN count
      info     : dict with 'cz_anchor_um', 'hcr_anchor_um', 'cc_score',
                 'z_bot', 'z_top', 'ty', 'tx'
    """
    if template is None:
        template = {}

    # Allow template to override fraction bounds
    hcr_frac_min = float(template.get("hcr_anchor_frac_min", hcr_frac_min))
    hcr_frac_max = float(template.get("hcr_anchor_frac_max", hcr_frac_max))

    cz_res = np.asarray(czstack_res_um, dtype=float)
    sc = np.array([hcr_scales["scale_z"], hcr_scales["scale_y"], hcr_scales["scale_x"]])

    # ------------------------------------------------------------------
    # 1. Tissue surface estimate from ALL HCR cells
    # ------------------------------------------------------------------
    z_bot, z_top, thickness = estimate_hcr_tissue_surface(hcr_all_um, n_edge=n_edge)
    if verbose:
        print(f"HCR tissue: z_bot={z_bot:.0f}  z_top={z_top:.0f}  "
              f"thickness={thickness:.0f} µm")

    # ------------------------------------------------------------------
    # 2. CZ centroids → µm, apply rotation
    # ------------------------------------------------------------------
    cz_z_px = czstack_df["czstack_z"].values.astype(float)
    cz_y_px = czstack_df["czstack_y"].values.astype(float)
    cz_x_px = czstack_df["czstack_x"].values.astype(float)
    cz_um = np.stack([cz_z_px * cz_res[0],
                      cz_y_px * cz_res[1],
                      cz_x_px * cz_res[2]], axis=1)

    theta_x = float(template.get("pitch_mean_deg", 180.0))
    theta_z = float(template.get("z_rotation_mean_deg", 0.0))
    theta_y = float(template.get("roll_mean_deg", 0.0))
    R_init  = rotation_matrix_euler(theta_z, theta_x, theta_y)
    P_rot   = (R_init @ cz_um.T).T   # (N, 3) rotated CZ in µm, no translation

    # CZ z in rotated frame.
    # With theta_x=180°, R=[1,0,0; 0,-1,0; 0,0,-1] in [z,y,x] coords: z is UNCHANGED.
    # So cz_rot_z.min() = pial surface (small z), cz_rot_z.max() = deep cortex.
    # Scan from the pial-surface end (cz_z_min) into the tissue.
    cz_rot_z = P_rot[:, 0]
    cz_z_min = float(cz_rot_z.min())
    cz_anchor_candidates = np.arange(
        cz_z_min + cz_slab_half_um,
        cz_z_min + cz_anchor_max_depth_um,
        cz_z_step_um,
    )

    # ------------------------------------------------------------------
    # 3. HCR GFP+ cells → µm YX (for CC)
    # ------------------------------------------------------------------
    hcr_z_all = hcr_gfp_df["hcr_z"].values.astype(float) * sc[0]
    hcr_y_all = hcr_gfp_df["hcr_y"].values.astype(float) * sc[1]
    hcr_x_all = hcr_gfp_df["hcr_x"].values.astype(float) * sc[2]

    hcr_anchor_lo = z_bot + hcr_frac_min * thickness
    hcr_anchor_hi = z_bot + hcr_frac_max * thickness
    hcr_anchor_candidates = np.arange(hcr_anchor_lo, hcr_anchor_hi, hcr_z_step_um)

    if verbose:
        print(f"CZ anchor scan: {len(cz_anchor_candidates)} levels "
              f"[{cz_anchor_candidates[0]:.0f}–{cz_anchor_candidates[-1]:.0f}] µm")
        print(f"HCR z scan:     {len(hcr_anchor_candidates)} levels "
              f"[{hcr_anchor_candidates[0]:.0f}–{hcr_anchor_candidates[-1]:.0f}] µm")

    # ------------------------------------------------------------------
    # 4. Inner helpers for CC
    # ------------------------------------------------------------------
    sigma_vox = sigma_um / voxel_um
    pad_um    = 3.0 * sigma_um

    def _build_img(yx_pts: np.ndarray, yx_min: np.ndarray, shape: tuple) -> np.ndarray:
        img = np.zeros(shape, dtype=np.float32)
        idx = ((yx_pts - yx_min) / voxel_um).astype(int)
        valid = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        if valid.any():
            np.add.at(img, (idx[valid, 0], idx[valid, 1]), 1)
        return gaussian_filter(img, sigma=sigma_vox).astype(np.float32)

    def _norm(img: np.ndarray) -> np.ndarray:
        return (img - img.mean()) / (img.std() + 1e-8)

    def _cc_shift(img_cz_n: np.ndarray, img_hcr_n: np.ndarray) -> np.ndarray:
        """FFT CC → shift (in voxels): HCR position − CZ position."""
        cc  = np.fft.ifft2(np.fft.fft2(img_hcr_n) * np.conj(np.fft.fft2(img_cz_n))).real
        pk  = np.array(np.unravel_index(np.argmax(cc), cc.shape), dtype=float)
        for i, s in enumerate(cc.shape):
            if pk[i] > s / 2:
                pk[i] -= s
        return pk

    # ------------------------------------------------------------------
    # 5. Grid search
    # Primary score: XY-MNN count (how many CZ cells match HCR cells within
    # xy_score_threshold_um after applying the CC-derived translation).
    # The CC uses one-sided normalization: CZ is fully normalised (zero-mean,
    # unit-std), HCR is only zero-mean (no std division) so that a uniform
    # HCR density image (dense slab) does NOT get its noise amplified.
    # Secondary score: raw CC peak / n_px (tie-break).
    # ------------------------------------------------------------------
    xy_score_threshold_um = float(xy_threshold_um) * 1.5  # loose radius for scoring

    best_mnn   = -1
    best_cc    = -np.inf   # secondary tie-break
    best_cz_z  = best_hcr_z = best_ty = best_tx = None
    all_mnn_zero = True    # flag for fallback

    for i_cz, cz_z in enumerate(cz_anchor_candidates):
        # CZ slab
        cz_mask = np.abs(cz_rot_z - cz_z) <= cz_slab_half_um
        if cz_mask.sum() < min_cz_cells_in_slab:
            continue
        P_yx = P_rot[cz_mask, 1:]   # (K, 2) YX µm

        for hcr_z in hcr_anchor_candidates:
            # HCR slab
            hcr_mask = np.abs(hcr_z_all - hcr_z) <= hcr_slab_half_um
            if hcr_mask.sum() < min_cz_cells_in_slab:
                continue
            Q_yx = np.stack([hcr_y_all[hcr_mask], hcr_x_all[hcr_mask]], axis=1)

            # Union bounding box — places both images in the same coordinate frame
            yx_min = np.minimum(P_yx.min(0), Q_yx.min(0)) - pad_um
            yx_max = np.maximum(P_yx.max(0), Q_yx.max(0)) + pad_um
            shape  = tuple(
                np.clip(
                    np.ceil((yx_max - yx_min) / voxel_um).astype(int) + 1,
                    1, 512,
                ).tolist()
            )

            img_cz  = _build_img(P_yx, yx_min, shape)
            img_hcr = _build_img(Q_yx, yx_min, shape)

            # One-sided normalised CC: normalise CZ fully; only remove HCR mean.
            # This prevents dense/uniform HCR slabs from having amplified noise.
            img_cz_n   = _norm(img_cz)
            img_hcr_zm = img_hcr - img_hcr.mean()   # zero-mean, original scale
            cc = np.fft.ifft2(
                np.fft.fft2(img_hcr_zm) * np.conj(np.fft.fft2(img_cz_n))
            ).real
            cc_peak = float(cc.max()) / float(cc.size)

            # CC peak → translation (both images share yx_min)
            pk = np.array(np.unravel_index(np.argmax(cc), cc.shape), dtype=float)
            for k, s in enumerate(cc.shape):
                if pk[k] > s / 2:
                    pk[k] -= s
            ty = float(pk[0] * voxel_um)
            tx = float(pk[1] * voxel_um)

            # Primary score: MNN count after applying CC-derived translation
            P_yx_aligned = P_yx + np.array([ty, tx])
            tree_Q = cKDTree(Q_yx)
            dists, _ = tree_Q.query(P_yx_aligned, k=1)
            mnn_score = int((dists <= xy_score_threshold_um).sum())

            if mnn_score > 0:
                all_mnn_zero = False

            better = (mnn_score > best_mnn) or (
                mnn_score == best_mnn and cc_peak > best_cc
            )
            if better:
                best_mnn   = mnn_score
                best_cc    = cc_peak
                best_cz_z  = cz_z
                best_hcr_z = hcr_z
                best_ty    = ty
                best_tx    = tx

    if best_cz_z is None:
        raise RuntimeError("Anchor-slice CC: no valid (cz_z, hcr_z) candidate found. "
                           "Try relaxing min_cz_cells_in_slab or widening the search range.")

    if all_mnn_zero and verbose:
        print("WARNING: all MNN scores were 0 — CC translation estimates may be poor. "
              "Falling back to best CC-score pair.")


    tz_local = best_hcr_z - best_cz_z
    R_best   = R_init

    if verbose:
        print(f"Best anchor: CZ_z={best_cz_z:.0f} µm  HCR_z={best_hcr_z:.0f} µm  "
              f"tz_local={tz_local:.0f}  ty={best_ty:.0f}  tx={best_tx:.0f}  "
              f"mnn={best_mnn}  cc={best_cc:.4f}")

    # ------------------------------------------------------------------
    # 5b. Use anchor-slice tz directly.
    #
    # The transform is fully non-rigid (z_scale ≈ 2.34–3.46× due to tissue
    # EXPANSION from clearing). A single global tz cannot correctly place
    # depth-varying cells. The anchor-slice tz (= hcr_z_anchor − cz_z_anchor)
    # is the best local estimate for seed extraction near the anchor depth.
    # TPS fitting in step_3 handles the full non-rigid deformation.
    # ------------------------------------------------------------------
    tz_best = tz_local
    t_best = np.array([tz_best, best_ty, best_tx])

    # ------------------------------------------------------------------
    # 6. Extract seed landmarks on the full aligned volume
    # ------------------------------------------------------------------
    hcr_gfp_um = np.stack([hcr_z_all, hcr_y_all, hcr_x_all], axis=1)
    P_aligned  = (R_best @ cz_um.T).T + t_best
    score      = score_mutual_nn_xy(P_aligned, hcr_gfp_um, xy_threshold_um)

    seed_df = extract_seed_landmarks_zxy(
        P_aligned_um   = P_aligned,
        czstack_df     = czstack_df,
        hcr_gfp_um     = hcr_gfp_um,
        hcr_gfp_df     = hcr_gfp_df,
        z_window_um    = z_window_um,
        xy_threshold_um= xy_threshold_um,
    )

    if verbose:
        print(f"XY-MNN score (full volume): {score}/{len(czstack_df)}")
        print(f"Seed landmarks extracted: {len(seed_df)}")

    info = dict(
        cz_anchor_um  = best_cz_z,
        hcr_anchor_um = best_hcr_z,
        mnn_score     = best_mnn,
        cc_score      = best_cc,
        z_bot         = z_bot,
        z_top         = z_top,
        ty            = best_ty,
        tx            = best_tx,
    )
    return seed_df, R_best, t_best, score, info


# ---------------------------------------------------------------------------
# Method C: RANSAC descriptor matching
# ---------------------------------------------------------------------------

def ransac_rigid_match(
    P_um: np.ndarray,
    Q_um: np.ndarray,
    desc_P: np.ndarray,
    desc_Q: np.ndarray,
    n_iter: int = 1000,
    threshold_um: float = 15.0,
    min_inliers: int = 5,
    rng_seed: int = 42,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """RANSAC rigid estimation from descriptor-matched correspondences.

    Returns (R, t, n_inliers) or (None, None, 0) on failure.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    rng = np.random.default_rng(rng_seed)

    # Normalise descriptors
    eps = 1e-8
    desc_P_n = desc_P / (np.linalg.norm(desc_P, axis=1, keepdims=True) + eps)
    desc_Q_n = desc_Q / (np.linalg.norm(desc_Q, axis=1, keepdims=True) + eps)

    # Build rough candidate matches: top-1 descriptor match per P cell
    D = euclidean_distances(desc_P_n, desc_Q_n)
    cands = np.argmin(D, axis=1)  # shape (N_P,)

    N = len(P_um)
    if N < 3:
        return None, None, 0

    best_inliers = 0
    best_R, best_t = None, None

    for _ in range(n_iter):
        # Sample 3 correspondences
        sel = rng.choice(N, size=3, replace=False)
        p3 = P_um[sel]
        q3 = Q_um[cands[sel]]

        # Fit rigid (rotation + translation) via SVD
        p_c = p3.mean(axis=0)
        q_c = q3.mean(axis=0)
        H = (p3 - p_c).T @ (q3 - q_c)
        U, _, Vt = np.linalg.svd(H)
        R_cand = Vt.T @ U.T
        if np.linalg.det(R_cand) < 0:
            Vt[-1] *= -1
            R_cand = Vt.T @ U.T
        t_cand = q_c - R_cand @ p_c

        P_aligned = (R_cand @ P_um.T).T + t_cand
        n_in = score_alignment(P_aligned, Q_um, threshold_um)
        if n_in > best_inliers:
            best_inliers = n_in
            best_R, best_t = R_cand, t_cand

    if best_inliers >= min_inliers:
        return best_R, best_t, best_inliers
    return None, None, 0


# ---------------------------------------------------------------------------
# CZ tissue surface estimation (mirrors estimate_hcr_tissue_surface for CZ)
# ---------------------------------------------------------------------------

def robust_tissue_bounds(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    n_edge: int = 5,
) -> tuple[float, float, float]:
    """Robustly estimate CZ tissue surface (z_min) and top (z_max).

    Uses only cells in the center-half XY extent to avoid bias from oblique
    tissue edges.  Returns (z_surface, z_top, thickness) in the same units
    as the input coordinates.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    y_mid = (y.max() + y.min()) / 2
    x_mid = (x.max() + x.min()) / 2
    y_qtr = (y.max() - y.min()) / 4
    x_qtr = (x.max() - x.min()) / 4
    mask = (np.abs(y - y_mid) <= y_qtr) & (np.abs(x - x_mid) <= x_qtr)
    z_ctr = z[mask] if mask.sum() >= n_edge * 2 else z

    sz = np.sort(z_ctr)
    z_surface = float(np.median(sz[:n_edge]))
    z_top = float(np.median(sz[-n_edge:]))
    return z_surface, z_top, z_top - z_surface


# ---------------------------------------------------------------------------
# Constellation selection — find k near-surface cells with empty XY hull
# ---------------------------------------------------------------------------

def select_constellation(
    cz_df_um: pd.DataFrame,
    cz_z_surface_um: float,
    dz_um: float = 15.0,
    dz_max_um: float = 60.0,
    n_cells: int = 6,
    kmeans_N_threshold: int = 35,
    max_subsets: int = 300_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find k near-surface CZ cells whose XY convex hull contains no other cells.

    Parameters
    ----------
    cz_df_um          : DataFrame with czstack_cell_id, czstack_z/y/x in µm
    cz_z_surface_um   : z-coordinate of tissue surface (from robust_tissue_bounds)
    dz_um             : initial z-slab depth above surface (µm); expands if needed
    dz_max_um         : maximum z-slab depth before giving up
    n_cells           : target constellation size; tries n_cells, n_cells-1, down to 4
    kmeans_N_threshold: if N_slab > this, pre-filter to ~25 cells via k-means
    max_subsets       : cap on number of subsets evaluated per (k, dz) combination
    verbose           : print progress

    Returns
    -------
    DataFrame subset of cz_df_um (k rows) representing the selected constellation,
    plus a column 'hull_intruders' with the intruder count (ideally 0).
    Empty DataFrame if no valid constellation found.
    """
    from itertools import combinations as _combinations
    from math import comb as _comb
    from scipy.spatial import ConvexHull, QhullError

    cz_z = cz_df_um["czstack_z"].values.astype(float)
    cz_y = cz_df_um["czstack_y"].values.astype(float)
    cz_x = cz_df_um["czstack_x"].values.astype(float)

    rng = np.random.default_rng(42)

    best_df = pd.DataFrame()
    best_intruders = np.inf
    best_hull_area = np.inf

    k_range = [k for k in range(n_cells, 3, -1)]  # n_cells … 4

    for dz_try in np.arange(dz_um, dz_max_um + 0.1, 5.0):
        slab_mask = (cz_z >= cz_z_surface_um) & (cz_z < cz_z_surface_um + dz_try)
        slab_pos = np.where(slab_mask)[0]  # positions into cz_df_um

        for k in k_range:
            if len(slab_pos) < k + 2:
                continue

            # --- optional k-means pre-filter ---
            cands_pos = slab_pos.copy()
            if len(cands_pos) > kmeans_N_threshold:
                try:
                    from sklearn.cluster import KMeans
                    xy_all = np.stack([cz_y[cands_pos], cz_x[cands_pos]], axis=1)
                    n_clust = min(25, len(cands_pos))
                    km = KMeans(n_clusters=n_clust, n_init=3, random_state=42)
                    km.fit(xy_all)
                    reduced = []
                    for c in range(n_clust):
                        members = np.where(km.labels_ == c)[0]
                        if len(members):
                            d = np.linalg.norm(xy_all[members] - km.cluster_centers_[c], axis=1)
                            reduced.append(cands_pos[members[np.argmin(d)]])
                    cands_pos = np.array(reduced)
                except ImportError:
                    pass  # sklearn not available; proceed with full set

            N = len(cands_pos)
            n_combos = _comb(N, k)

            # XY coords for candidates and all slab cells
            cands_xy = np.stack([cz_y[cands_pos], cz_x[cands_pos]], axis=1)  # (N, 2)
            slab_xy = np.stack([cz_y[slab_pos], cz_x[slab_pos]], axis=1)    # (S, 2)

            # Build list of subsets to evaluate
            if n_combos <= max_subsets:
                combos = list(_combinations(range(N), k))
            else:
                seen: set = set()
                combos = []
                for _ in range(max_subsets * 10):
                    if len(combos) >= max_subsets:
                        break
                    s = tuple(sorted(rng.choice(N, k, replace=False).tolist()))
                    if s not in seen:
                        seen.add(s)
                        combos.append(s)

            if verbose:
                print(f"  dz={dz_try:.0f} µm, k={k}: "
                      f"{len(slab_pos)} slab cells, {N} candidates, "
                      f"{len(combos):,} subsets to evaluate")

            for combo in combos:
                pts = cands_xy[list(combo)]  # (k, 2)
                sel_global = set(cands_pos[list(combo)])

                try:
                    hull = ConvexHull(pts)
                except QhullError:
                    continue  # degenerate (collinear)

                # Count slab cells not in the selected k that are inside the hull
                other_slab = np.array(
                    [i for i, g in enumerate(slab_pos) if g not in sel_global]
                )
                if len(other_slab) == 0:
                    intruders = 0
                else:
                    test_pts = slab_xy[other_slab]  # (M, 2)
                    # hull.equations: (n_edges, 3); inside if all rows <= 0
                    vals = hull.equations[:, :2] @ test_pts.T + hull.equations[:, 2:3]
                    intruders = int(np.all(vals <= 1e-10, axis=0).sum())

                hull_area = float(hull.volume)  # in 2D, ConvexHull.volume = area

                if intruders < best_intruders or (
                    intruders == best_intruders and hull_area < best_hull_area
                ):
                    best_intruders = intruders
                    best_hull_area = hull_area
                    best_df = cz_df_um.iloc[list(cands_pos[list(combo)])].copy()
                    best_df = best_df.assign(hull_intruders=intruders)

            if best_intruders == 0:
                break  # perfect constellation found at this k
        if best_intruders == 0:
            break  # no need to expand slab

    if verbose and not best_df.empty:
        print(f"Selected constellation: {len(best_df)} cells, "
              f"intruders={best_intruders}, hull_area={best_hull_area:.0f} µm²")
    elif verbose:
        print("No constellation found.")

    return best_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hough XY vote — find translation from constellation→HCR window
# ---------------------------------------------------------------------------

def hough_vote_xy(
    constellation_xy: np.ndarray,
    hcr_window_xy: np.ndarray,
    bin_size_um: float = 5.0,
) -> tuple[float, float, int]:
    """Find (t_y, t_x) translation via Hough voting over all cell pairs.

    For each pair (cz_i, hcr_j) the implied translation is
    (hcr_j_y − cz_i_y, hcr_j_x − cz_i_x).  The accumulator bin with the most
    votes is the consensus translation.

    Parameters
    ----------
    constellation_xy : (k, 2) array, YX in µm
    hcr_window_xy    : (M, 2) array, YX in µm
    bin_size_um      : accumulator bin size in µm

    Returns
    -------
    (t_y, t_x, peak_count)
    """
    # All k×M differences (hcr_j − cz_i) → (k, M, 2)
    diffs = hcr_window_xy[np.newaxis, :, :] - constellation_xy[:, np.newaxis, :]
    diffs_flat = diffs.reshape(-1, 2)  # (k*M, 2)

    d_min = diffs_flat.min(axis=0)
    bins = np.round((diffs_flat - d_min) / bin_size_um).astype(np.int32)  # (k*M, 2)

    max_bins = bins.max(axis=0) + 1
    lin_idx = bins[:, 0] * int(max_bins[1]) + bins[:, 1]
    counts = np.bincount(lin_idx, minlength=int(max_bins[0]) * int(max_bins[1]))

    peak_lin = int(np.argmax(counts))
    peak_count = int(counts[peak_lin])
    peak_y_bin, peak_x_bin = np.unravel_index(peak_lin, (int(max_bins[0]), int(max_bins[1])))

    t_y = float(d_min[0] + peak_y_bin * bin_size_um)
    t_x = float(d_min[1] + peak_x_bin * bin_size_um)
    return t_y, t_x, peak_count


# ---------------------------------------------------------------------------
# Constellation matching — rotation × Z scan with Hough XY
# ---------------------------------------------------------------------------

def match_constellation(
    constellation_um: np.ndarray,
    hcr_gfp_um: np.ndarray,
    template: Optional[dict] = None,
    W_um: float = 55.0,
    z_step_um: float = 25.0,
    match_threshold_um: float = 15.0,
    bin_size_um: float = 5.0,
    verbose: bool = True,
) -> list:
    """Search for the HCR location matching a CZ constellation.

    For each rotation candidate × HCR z-window, a Hough XY vote finds the
    best (t_y, t_x) translation.  Returns the top-5 hypotheses sorted by
    match score (descending).

    Parameters
    ----------
    constellation_um  : (k, 3) CZ constellation in µm, order (z, y, x)
    hcr_gfp_um        : (M, 3) HCR GFP+ centroids in µm
    template          : dict from coreg_transform_template.json
    W_um              : half-width of HCR z-window for each MIP level (µm)
    z_step_um         : step between HCR z-center candidates (µm)
    match_threshold_um: XY distance threshold for counting a match (µm)
    bin_size_um       : Hough accumulator bin size (µm)

    Returns
    -------
    List of up to 5 dicts, each with keys:
      R, z_center, t_y, t_x, score, peak_count
    Sorted by score descending.
    """
    if template is None:
        template = {
            "z_rotation_range_deg": [-15, 15],
            "pitch_range_deg": [-186, 186],
            "roll_range_deg": [-15, 15],
        }

    # Build rotation grid (same convention as rotation_search)
    p_lo, p_hi = template["pitch_range_deg"]
    z_lo, z_hi = template["z_rotation_range_deg"]
    r_lo, r_hi = template["roll_range_deg"]

    step = 5.0
    tx_abs_lo = max(0.0, min(abs(p_lo), abs(p_hi)) - step)
    tx_abs_hi = max(abs(p_lo), abs(p_hi)) + step
    tx_pos = np.arange(tx_abs_lo, tx_abs_hi + 0.1, step)
    tx_angles = np.concatenate([-tx_pos, tx_pos])
    tz_angles = np.arange(z_lo, z_hi + 0.1, step)
    ty_angles = np.arange(r_lo, r_hi + 0.1, step)

    # HCR z-scan
    hcr_z = hcr_gfp_um[:, 0]
    z_centers = np.arange(hcr_z.min() + W_um / 2, hcr_z.max() - W_um / 2 + z_step_um, z_step_um)

    n_rots = len(tx_angles) * len(tz_angles) * len(ty_angles)
    if verbose:
        print(f"match_constellation: {n_rots} rotations × {len(z_centers)} z-levels "
              f"= {n_rots * len(z_centers):,} candidates")

    results = []

    for tx in tx_angles:
        for tz in tz_angles:
            for ty in ty_angles:
                R = rotation_matrix_euler(tz, tx, ty)
                c_rot = (R @ constellation_um.T).T  # (k, 3)
                c_xy = c_rot[:, 1:]                  # (k, 2) YX

                for z_center in z_centers:
                    window_mask = np.abs(hcr_z - z_center) <= W_um / 2
                    if window_mask.sum() < 3:
                        continue
                    hcr_win_xy = hcr_gfp_um[window_mask, 1:]  # (M_w, 2)

                    t_y, t_x, peak_count = hough_vote_xy(c_xy, hcr_win_xy, bin_size_um)

                    # Score: count constellation cells matched within threshold
                    translated = c_xy + np.array([t_y, t_x])
                    tree = cKDTree(hcr_win_xy)
                    dists, _ = tree.query(translated, k=1)
                    score = int((dists <= match_threshold_um).sum())

                    if score > 0:
                        results.append({
                            "R": R.copy(),
                            "z_center": float(z_center),
                            "t_y": float(t_y),
                            "t_x": float(t_x),
                            "score": score,
                            "peak_count": peak_count,
                        })

    results.sort(key=lambda r: (-r["score"], -r["peak_count"]))

    if verbose:
        if results:
            top = results[0]
            print(f"Top hypothesis: score={top['score']}/{len(constellation_um)} "
                  f"z_center={top['z_center']:.0f} µm "
                  f"t=({top['t_y']:.0f}, {top['t_x']:.0f}) µm")
        else:
            print("No hypotheses found.")

    return results[:5]


# ---------------------------------------------------------------------------
# Constellation match verification
# ---------------------------------------------------------------------------

def verify_constellation_match(
    hypothesis: dict,
    constellation_um: np.ndarray,
    hcr_gfp_um: np.ndarray,
    dist_threshold_3d_um: float = 25.0,
    dist_consistency_um: float = 20.0,
) -> bool:
    """Check whether a constellation hypothesis is geometrically consistent.

    Applies the hypothesis transform to the constellation, finds the nearest
    HCR cell for each constellation cell, and verifies:
      1. All k cells matched within dist_threshold_3d_um (3D)
      2. Pairwise HCR distances match transformed CZ distances within
         dist_consistency_um

    Parameters
    ----------
    hypothesis        : dict with keys R, z_center, t_y, t_x (from match_constellation)
    constellation_um  : (k, 3) CZ constellation in µm
    hcr_gfp_um        : (M, 3) HCR GFP+ centroids in µm
    dist_threshold_3d_um : max 3D distance for a constellation cell to count as matched
    dist_consistency_um  : max allowed pairwise distance inconsistency (µm)

    Returns
    -------
    True if the hypothesis passes all checks.
    """
    R = hypothesis["R"]
    z_center = float(hypothesis["z_center"])
    t_y = float(hypothesis["t_y"])
    t_x = float(hypothesis["t_x"])

    c_rot = (R @ constellation_um.T).T  # (k, 3)
    # Estimate tz: the constellation's mean rotated-z should align to z_center
    tz = z_center - float(c_rot[:, 0].mean())
    t = np.array([tz, t_y, t_x])
    c_aligned = c_rot + t  # (k, 3)

    tree = cKDTree(hcr_gfp_um)
    dists_3d, nn_idx = tree.query(c_aligned, k=1)

    # Check 1: all cells matched within 3D threshold
    if not np.all(dists_3d <= dist_threshold_3d_um):
        return False

    # Check 2: pairwise distance consistency
    hcr_matched = hcr_gfp_um[nn_idx]  # (k, 3)
    k = len(constellation_um)
    for i in range(k):
        for j in range(i + 1, k):
            d_hcr = float(np.linalg.norm(hcr_matched[i] - hcr_matched[j]))
            d_cz = float(np.linalg.norm(c_aligned[i] - c_aligned[j]))
            if abs(d_hcr - d_cz) > dist_consistency_um:
                return False

    return True
