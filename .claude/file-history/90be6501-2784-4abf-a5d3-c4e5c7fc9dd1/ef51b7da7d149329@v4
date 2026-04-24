"""F8 — Synthetic warp pipeline.

Generate paired (source, warped, correspondence) samples from an HCR GFP+
centroid point cloud by applying a random anisotropic-scale + rigid rotation +
per-axis TPS warp to a sub-cube.  Used to train G-series matchers and other
learned methods without ground-truth labels.

Sampling bounds (benchmark-informed, for sampling only — not used as model
parameters):

- XY scale ∈ [1.5, 2.0]
- Z scale  ∈ [2.0, 3.5]
- XY rotation drawn in {0°, 180°} ± 10° jitter (reflects the structural prior)
- Z-axis jitter ±5°
- TPS: 6–10 control points, per-axis jitter 10–40 µm
- Drop rate 10–30 % to simulate partial overlap / segmentation error
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import Rbf


@dataclass
class WarpSample:
    source_um: np.ndarray          # (Ns, 3), (z,y,x)
    warped_um: np.ndarray          # (Nw, 3)
    correspondence: np.ndarray     # (N_match, 2) int — indices into (source, warped)
    R: np.ndarray
    scales: np.ndarray
    translation: np.ndarray
    tps_metadata: dict = field(default_factory=dict)


def _rot_xy(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # R in (z,y,x)


def sample_warped_pair(
    points_um: np.ndarray,          # (N, 3) in (z,y,x)
    rng: Optional[np.random.Generator] = None,
    cube_um: float = 400.0,
    *,
    xy_scale_range: "tuple[float, float]" = (1.5, 2.0),
    z_scale_range: "tuple[float, float]" = (2.0, 3.5),
    rot_jitter_xy_deg: float = 10.0,
    rot_jitter_z_deg: float = 5.0,
    tps_n_cp: int = 8,
    tps_jitter_um: float = 25.0,
    drop_rate: float = 0.2,
    always_180: bool = True,
) -> WarpSample:
    """Return a random source cube + warped version + GT correspondences."""
    rng = rng or np.random.default_rng()

    # 1) Choose centre so that the cube is inside the bbox (with margin).
    pmin = points_um.min(0) + cube_um / 2
    pmax = points_um.max(0) - cube_um / 2
    centre = rng.uniform(pmin, pmax)
    sel = np.all(np.abs(points_um - centre) < cube_um / 2, axis=1)
    src = points_um[sel]
    if len(src) < 10:
        return WarpSample(src, src.copy(), np.empty((0, 2), int),
                          R=np.eye(3), scales=np.ones(3), translation=np.zeros(3))

    # Centre the source around origin for the warp math.
    src_c = src - src.mean(0)

    # 2) Rigid rotation: 180° XY base (structural prior) + jitter in xy and z.
    yaw_deg = 180.0 if always_180 else 0.0
    yaw_deg += rng.uniform(-rot_jitter_xy_deg, rot_jitter_xy_deg)
    # (yaw is around z; in our (z,y,x) order, yaw rotates y/x)
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R_yaw = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])
    # z-jitter as small pitch
    pitch_deg = rng.uniform(-rot_jitter_z_deg, rot_jitter_z_deg)
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = R_yaw @ R_pitch

    # 3) Anisotropic scale.
    sz = rng.uniform(*z_scale_range)
    sxy = rng.uniform(*xy_scale_range)
    S = np.array([sz, sxy, sxy])

    # 4) Apply rigid + scale.
    rigid = (R @ (src_c * S).T).T

    # 5) TPS warp — sample control points on the cube interior.
    cp_idx = rng.choice(len(rigid), size=min(tps_n_cp, len(rigid)), replace=False)
    cps_src = rigid[cp_idx]
    disp = rng.normal(0, tps_jitter_um, size=cps_src.shape)
    cps_dst = cps_src + disp
    warped = rigid.copy()
    for axis in range(3):
        rbf = Rbf(cps_src[:, 0], cps_src[:, 1], cps_src[:, 2],
                  cps_dst[:, axis] - cps_src[:, axis],
                  function="thin_plate")
        delta = rbf(rigid[:, 0], rigid[:, 1], rigid[:, 2])
        warped[:, axis] = rigid[:, axis] + delta

    # 6) Drop random subset to simulate partial overlap.
    keep_src = rng.random(len(src)) > (drop_rate / 2.0)
    keep_dst = rng.random(len(warped)) > (drop_rate / 2.0)

    src_kept = src[keep_src]
    warped_kept = warped[keep_dst]
    # GT correspondences are over the intersection.
    idx_src_in = np.where(keep_src)[0]
    idx_dst_in = np.where(keep_dst)[0]
    src_pos = {i: k for k, i in enumerate(idx_src_in)}
    dst_pos = {i: k for k, i in enumerate(idx_dst_in)}
    common = sorted(set(idx_src_in) & set(idx_dst_in))
    corr = np.array([(src_pos[i], dst_pos[i]) for i in common], dtype=int)

    return WarpSample(
        source_um=src_kept,
        warped_um=warped_kept,
        correspondence=corr,
        R=R,
        scales=S,
        translation=src.mean(0),
        tps_metadata={"cps": cps_src, "dst": cps_dst, "yaw": yaw_deg, "pitch": pitch_deg},
    )


def sample_asymmetric_warped_pair(
    points_um: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    *,
    source_cube_um: float = 400.0,
    target_margin_um: float = 400.0,
    source_n_target: int = 900,
    target_n_cap: int = 12000,
    xy_scale_range: "tuple[float, float]" = (1.5, 2.0),
    z_scale_range: "tuple[float, float]" = (2.0, 3.5),
    rot_jitter_xy_deg: float = 10.0,
    rot_jitter_z_deg: float = 5.0,
    tps_n_cp: int = 8,
    tps_jitter_um: float = 25.0,
    source_drop_rate: float = 0.3,
    target_overlap_drop: float = 0.15,
    always_180: bool = True,
) -> WarpSample:
    """Asymmetric source/warped sampler — simulates CZ↔HCR count asymmetry.

    Source cube is small (`source_cube_um`, CZ-sized, ~700-1000 cells after
    subsample), target cube is larger by `target_margin_um` padding on each
    side, and source points are a *subset* of target points before warping.
    Target retains extra "distractor" points that have no source partner,
    mirroring HCR's 6-30× count excess over CZ.
    """
    rng = rng or np.random.default_rng()

    pmin = points_um.min(0) + source_cube_um / 2 + target_margin_um
    pmax = points_um.max(0) - source_cube_um / 2 - target_margin_um
    if np.any(pmax <= pmin):
        return sample_warped_pair(points_um, rng=rng, cube_um=source_cube_um,
                                   xy_scale_range=xy_scale_range,
                                   z_scale_range=z_scale_range,
                                   rot_jitter_xy_deg=rot_jitter_xy_deg,
                                   rot_jitter_z_deg=rot_jitter_z_deg,
                                   tps_n_cp=tps_n_cp, tps_jitter_um=tps_jitter_um,
                                   drop_rate=source_drop_rate, always_180=always_180)
    centre = rng.uniform(pmin, pmax)
    half_src = source_cube_um / 2
    half_tgt = source_cube_um / 2 + target_margin_um

    in_src = np.all(np.abs(points_um - centre) < half_src, axis=1)
    in_tgt = np.all(np.abs(points_um - centre) < half_tgt, axis=1)
    src_global_idx = np.where(in_src)[0]
    tgt_global_idx = np.where(in_tgt)[0]
    if len(src_global_idx) < 10 or len(tgt_global_idx) < 20:
        return WarpSample(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), int),
                          R=np.eye(3), scales=np.ones(3), translation=np.zeros(3))

    if len(src_global_idx) > source_n_target:
        src_sel = rng.choice(src_global_idx, size=source_n_target, replace=False)
    else:
        src_sel = src_global_idx
    keep_src = rng.random(len(src_sel)) > source_drop_rate
    src_sel = src_sel[keep_src]
    if len(tgt_global_idx) > target_n_cap:
        extra_idx = np.setdiff1d(tgt_global_idx, src_sel, assume_unique=False)
        need = target_n_cap - len(src_sel)
        if need > 0 and len(extra_idx) > 0:
            extra_sel = rng.choice(extra_idx, size=min(need, len(extra_idx)), replace=False)
            tgt_sel = np.concatenate([src_sel, extra_sel])
        else:
            tgt_sel = src_sel
    else:
        tgt_sel = tgt_global_idx
    keep_tgt = rng.random(len(tgt_sel)) > target_overlap_drop
    tgt_sel = tgt_sel[keep_tgt]

    src_pts = points_um[src_sel]
    tgt_pts = points_um[tgt_sel]

    mean0 = tgt_pts.mean(0)
    src_c = src_pts - mean0
    tgt_c = tgt_pts - mean0

    yaw_deg = 180.0 if always_180 else 0.0
    yaw_deg += rng.uniform(-rot_jitter_xy_deg, rot_jitter_xy_deg)
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R_yaw = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])
    pitch_deg = rng.uniform(-rot_jitter_z_deg, rot_jitter_z_deg)
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = R_yaw @ R_pitch

    sz = rng.uniform(*z_scale_range)
    sxy = rng.uniform(*xy_scale_range)
    S = np.array([sz, sxy, sxy])

    tgt_rigid = (R @ (tgt_c * S).T).T

    if len(tgt_rigid) >= tps_n_cp:
        cp_idx = rng.choice(len(tgt_rigid), size=tps_n_cp, replace=False)
        cps_src = tgt_rigid[cp_idx]
        disp = rng.normal(0, tps_jitter_um, size=cps_src.shape)
        cps_dst = cps_src + disp
        tgt_warped = tgt_rigid.copy()
        for axis in range(3):
            rbf = Rbf(cps_src[:, 0], cps_src[:, 1], cps_src[:, 2],
                      cps_dst[:, axis] - cps_src[:, axis],
                      function="thin_plate")
            delta = rbf(tgt_rigid[:, 0], tgt_rigid[:, 1], tgt_rigid[:, 2])
            tgt_warped[:, axis] = tgt_rigid[:, axis] + delta
    else:
        tgt_warped = tgt_rigid
        cps_src = np.empty((0, 3)); cps_dst = np.empty((0, 3))

    tgt_idx_of = {int(i): k for k, i in enumerate(tgt_sel)}
    corr_rows = []
    for k_src, g in enumerate(src_sel):
        if int(g) in tgt_idx_of:
            corr_rows.append((k_src, tgt_idx_of[int(g)]))
    corr = np.array(corr_rows, dtype=int) if corr_rows else np.empty((0, 2), int)

    return WarpSample(
        source_um=src_c,
        warped_um=tgt_warped,
        correspondence=corr,
        R=R,
        scales=S,
        translation=mean0,
        tps_metadata={"cps": cps_src, "dst": cps_dst,
                      "yaw": yaw_deg, "pitch": pitch_deg,
                      "asymmetric": True,
                      "n_src": int(len(src_c)), "n_tgt": int(len(tgt_warped)),
                      "src_cube_um": float(source_cube_um),
                      "tgt_margin_um": float(target_margin_um)},
    )


@dataclass
class VoxelWarpSample:
    """F8-image sample: paired source/warped centroids + image patches."""
    source_um: np.ndarray             # (Ns, 3), local frame (centred)
    warped_um: np.ndarray             # (Nw, 3), local frame
    correspondence: np.ndarray        # (Nc, 2) int: (k_src, k_warped) pairs
    source_patches: np.ndarray        # (Ns, 1, D, H, W) float32
    warped_patches: np.ndarray        # (Nw, 1, D, H, W) float32
    R: np.ndarray
    scales: np.ndarray
    translation: np.ndarray
    tps_metadata: dict = field(default_factory=dict)


def sample_voxel_warp_sample(
    volume: np.ndarray,
    voxel_spacing_um: "tuple[float, float, float]",
    centroids_um: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    *,
    source_cube_um: float = 400.0,
    target_margin_um: float = 400.0,
    source_n_target: int = 900,
    target_n_cap: int = 2000,
    xy_scale_range: "tuple[float, float]" = (1.5, 2.0),
    z_scale_range: "tuple[float, float]" = (2.0, 3.5),
    rot_jitter_xy_deg: float = 10.0,
    rot_jitter_z_deg: float = 5.0,
    tps_n_cp: int = 8,
    tps_jitter_um: float = 25.0,
    source_drop_rate: float = 0.3,
    target_overlap_drop: float = 0.15,
    always_180: bool = True,
    patch_size: int = 16,
    patch_spacing_um: float = 4.0,
    intensity_noise_std: float = 0.05,
) -> VoxelWarpSample:
    """Generate a single voxel-level F8 training sample for C2 stage-1.

    Source and target centroids are both sampled from a single HCR 488
    volume. Source patches are extracted axis-aligned at source-centroid
    positions. Target ("warped") patches are extracted from the SAME
    volume at the anatomically-corresponding original positions, but
    with a rotated/scaled sampling lattice whose orientation matches
    the inverse Jacobian `(R·S)^{-1}` of the synthetic warp — so the
    target patch looks like "the same cell viewed through the warp".

    The matcher learns (a) warp-invariant patch embeddings and (b) to
    pair patches whose spatial positions differ by the known warp. TPS
    is applied to positions only (not to the patch-sampling lattice),
    which is correct to first order for patches ≪ TPS control-point
    spacing.
    """
    from lib.image_patches import sample_patches_oriented

    rng = rng or np.random.default_rng()

    pmin = centroids_um.min(0) + source_cube_um / 2 + target_margin_um
    pmax = centroids_um.max(0) - source_cube_um / 2 - target_margin_um
    if np.any(pmax <= pmin):
        return VoxelWarpSample(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), int),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            R=np.eye(3), scales=np.ones(3), translation=np.zeros(3),
        )
    centre = rng.uniform(pmin, pmax)
    half_src = source_cube_um / 2
    half_tgt = source_cube_um / 2 + target_margin_um

    in_src = np.all(np.abs(centroids_um - centre) < half_src, axis=1)
    in_tgt = np.all(np.abs(centroids_um - centre) < half_tgt, axis=1)
    src_global = np.where(in_src)[0]
    tgt_global = np.where(in_tgt)[0]
    if len(src_global) < 10 or len(tgt_global) < 20:
        return VoxelWarpSample(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), int),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            R=np.eye(3), scales=np.ones(3), translation=np.zeros(3),
        )

    if len(src_global) > source_n_target:
        src_sel = rng.choice(src_global, size=source_n_target, replace=False)
    else:
        src_sel = src_global.copy()
    keep_src = rng.random(len(src_sel)) > source_drop_rate
    src_sel = src_sel[keep_src]

    if len(tgt_global) > target_n_cap:
        extra_idx = np.setdiff1d(tgt_global, src_sel, assume_unique=False)
        need = target_n_cap - len(src_sel)
        if need > 0 and len(extra_idx) > 0:
            extra_sel = rng.choice(extra_idx, size=min(need, len(extra_idx)), replace=False)
            tgt_sel = np.concatenate([src_sel, extra_sel])
        else:
            tgt_sel = src_sel.copy()
    else:
        tgt_sel = tgt_global.copy()
    keep_tgt = rng.random(len(tgt_sel)) > target_overlap_drop
    tgt_sel = tgt_sel[keep_tgt]

    if len(src_sel) < 2 or len(tgt_sel) < 2:
        return VoxelWarpSample(
            np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), int),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            np.empty((0, 1, patch_size, patch_size, patch_size), dtype=np.float32),
            R=np.eye(3), scales=np.ones(3), translation=np.zeros(3),
        )

    yaw_deg = 180.0 if always_180 else 0.0
    yaw_deg += rng.uniform(-rot_jitter_xy_deg, rot_jitter_xy_deg)
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    R_yaw = np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]])
    pitch_deg = rng.uniform(-rot_jitter_z_deg, rot_jitter_z_deg)
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = R_yaw @ R_pitch

    sz = rng.uniform(*z_scale_range)
    sxy = rng.uniform(*xy_scale_range)
    S_diag = np.array([sz, sxy, sxy])
    RS = R @ np.diag(S_diag)
    RS_inv = np.linalg.inv(RS)

    src_pts_orig = centroids_um[src_sel]
    tgt_pts_orig = centroids_um[tgt_sel]
    mean0 = tgt_pts_orig.mean(0)
    tgt_c = tgt_pts_orig - mean0
    src_c = src_pts_orig - mean0

    tgt_rigid = (RS @ tgt_c.T).T

    if len(tgt_rigid) >= tps_n_cp:
        cp_idx = rng.choice(len(tgt_rigid), size=tps_n_cp, replace=False)
        cps_src = tgt_rigid[cp_idx]
        disp = rng.normal(0, tps_jitter_um, size=cps_src.shape)
        cps_dst = cps_src + disp
        tgt_warped = tgt_rigid.copy()
        for axis in range(3):
            rbf = Rbf(cps_src[:, 0], cps_src[:, 1], cps_src[:, 2],
                      cps_dst[:, axis] - cps_src[:, axis],
                      function="thin_plate")
            delta = rbf(tgt_rigid[:, 0], tgt_rigid[:, 1], tgt_rigid[:, 2])
            tgt_warped[:, axis] = tgt_rigid[:, axis] + delta
    else:
        tgt_warped = tgt_rigid
        cps_src = np.empty((0, 3))
        cps_dst = np.empty((0, 3))

    src_patches = sample_patches_oriented(
        volume, src_pts_orig, voxel_spacing_um,
        patch_size=patch_size, sample_spacing_um=patch_spacing_um,
        orient=None, normalize=True,
    )
    tgt_patches = sample_patches_oriented(
        volume, tgt_pts_orig, voxel_spacing_um,
        patch_size=patch_size, sample_spacing_um=patch_spacing_um,
        orient=RS_inv, normalize=True,
    )

    if intensity_noise_std > 0:
        src_patches = src_patches + rng.normal(
            0, intensity_noise_std, src_patches.shape
        ).astype(np.float32)
        tgt_patches = tgt_patches + rng.normal(
            0, intensity_noise_std, tgt_patches.shape
        ).astype(np.float32)

    tgt_idx_of = {int(i): k for k, i in enumerate(tgt_sel)}
    corr_rows = [
        (k_src, tgt_idx_of[int(g)])
        for k_src, g in enumerate(src_sel) if int(g) in tgt_idx_of
    ]
    corr = np.array(corr_rows, dtype=int) if corr_rows else np.empty((0, 2), int)

    return VoxelWarpSample(
        source_um=src_c,
        warped_um=tgt_warped,
        correspondence=corr,
        source_patches=src_patches,
        warped_patches=tgt_patches,
        R=R,
        scales=S_diag,
        translation=mean0,
        tps_metadata={
            "cps": cps_src, "dst": cps_dst,
            "yaw": yaw_deg, "pitch": pitch_deg,
            "n_src": int(len(src_c)), "n_tgt": int(len(tgt_warped)),
            "src_cube_um": float(source_cube_um),
            "tgt_margin_um": float(target_margin_um),
            "patch_size": int(patch_size),
            "patch_spacing_um": float(patch_spacing_um),
        },
    )


def _selftest():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2000, 2000, size=(20000, 3))
    w = sample_warped_pair(pts, rng=rng)
    print(f"F8 symmetric: src={len(w.source_um)} warped={len(w.warped_um)} "
          f"corr={len(w.correspondence)} scales={w.scales}")
    a = sample_asymmetric_warped_pair(pts, rng=rng)
    ratio = len(a.warped_um) / max(len(a.source_um), 1)
    print(f"F8 asymmetric: src={len(a.source_um)} warped={len(a.warped_um)} "
          f"ratio={ratio:.1f}× corr={len(a.correspondence)} scales={a.scales}")


if __name__ == "__main__":
    _selftest()
