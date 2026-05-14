"""Resample HCR volumes into the CZ-stack voxel frame.

The locked-prior + sz transform is the canonical CZ→HCR map (in µm):

    hcr_um = R @ S @ (cz_um - src_mean) + T

For pull-resampling an HCR volume onto a CZ output grid we need the same
forward map: for each output voxel index in CZ, look up the corresponding
HCR voxel. Translated into voxel-index linear form:

    hcr_vox = A @ cz_out_vox + b
    A = D_hcr_inv @ R @ S @ D_cz
    b = D_hcr_inv @ (T - R @ S @ (margin_um + src_mean))

where `D_cz = diag(cz_z_um, cz_xy_um, cz_xy_um)`, similarly `D_hcr`, and
`margin_um` is the µm offset of the output grid origin relative to the CZ
stack origin (i.e. how many µm of padding sit before the CZ stack starts).

Sub-bbox loading: we compute the HCR-µm bbox spanned by the output grid
corners and only load that slab from the zarr, then offset `b` so it
indexes into the slab rather than the full volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import zarr
from scipy.ndimage import affine_transform

from benchmark_analysis import depth_from_surface
from locked_prior_warm import LockedPriorWarmStart
from surfaces_iter08 import get_cz_surface_iter08


@dataclass
class CzOutputGrid:
    """Output grid in CZ frame.

    Output voxel index 0 maps to CZ µm coordinate ``-margin_um[axis]``.
    So ``cz_um = D_cz @ cz_out_vox - margin_um``.
    """
    shape: tuple[int, int, int]      # (Z, Y, X) output voxels
    cz_xy_um: float
    cz_z_um: float
    margin_um: np.ndarray            # (3,) [z, y, x] µm padding before origin


def _diag(z, y, x) -> np.ndarray:
    return np.diag([float(z), float(y), float(x)])


def _effective_translation(
    lp: LockedPriorWarmStart, sz: float, cz_mean_depth_um: float
) -> np.ndarray:
    """Apply the sz-induced tz offset so the CZ pia stays pinned to HCR pia."""
    tz_offset = (float(sz) - float(lp.scales[0])) * float(cz_mean_depth_um)
    T = lp.translation.copy()
    T[0] += tz_offset
    return T


def cz_mean_depth_um(s, lp: LockedPriorWarmStart) -> float:
    cz_surface = get_cz_surface_iter08(s)
    cz_mean_xyz = lp.src_mean[[2, 1, 0]]
    return float(depth_from_surface(cz_mean_xyz[None, :], cz_surface)[0])


def build_cz_output_grid(
    s, *, margin_um: float, cz_shape: tuple[int, int, int]
) -> CzOutputGrid:
    """CZ grid + uniform µm margin on all sides."""
    cz_z_um = float(s.cz_z_um)
    cz_xy_um = float(s.cz_xy_um)
    margin_z_vox = int(np.round(margin_um / cz_z_um))
    margin_xy_vox = int(np.round(margin_um / cz_xy_um))
    out_shape = (
        cz_shape[0] + 2 * margin_z_vox,
        cz_shape[1] + 2 * margin_xy_vox,
        cz_shape[2] + 2 * margin_xy_vox,
    )
    margin_um_vec = np.array(
        [margin_z_vox * cz_z_um, margin_xy_vox * cz_xy_um, margin_xy_vox * cz_xy_um],
        dtype=float,
    )
    return CzOutputGrid(shape=out_shape, cz_xy_um=cz_xy_um, cz_z_um=cz_z_um,
                        margin_um=margin_um_vec)


def pad_cz_stack(cz_vol: np.ndarray, grid: CzOutputGrid, s) -> np.ndarray:
    """Zero-pad a CZ stack to match the output grid (margin on all sides)."""
    margin_z = int(np.round(grid.margin_um[0] / float(s.cz_z_um)))
    margin_xy = int(np.round(grid.margin_um[1] / float(s.cz_xy_um)))
    out = np.zeros(grid.shape, dtype=cz_vol.dtype)
    out[
        margin_z : margin_z + cz_vol.shape[0],
        margin_xy : margin_xy + cz_vol.shape[1],
        margin_xy : margin_xy + cz_vol.shape[2],
    ] = cz_vol
    return out


def _full_affine_voxel(
    lp: LockedPriorWarmStart, sz: float, T_eff: np.ndarray,
    grid: CzOutputGrid, hcr_xy_um: float, hcr_z_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (A, b) for the full HCR volume frame (no sub-bbox offset yet).

    For output index ``cz_out_vox`` (in CZ output grid), the source HCR
    voxel index is ``A @ cz_out_vox + b``.
    """
    S = np.diag([float(sz), float(lp.scales[1]), float(lp.scales[2])])
    D_cz = _diag(grid.cz_z_um, grid.cz_xy_um, grid.cz_xy_um)
    D_hcr_inv = _diag(1.0 / hcr_z_um, 1.0 / hcr_xy_um, 1.0 / hcr_xy_um)
    R = lp.R
    A = D_hcr_inv @ R @ S @ D_cz
    b = D_hcr_inv @ (T_eff - R @ S @ (grid.margin_um + lp.src_mean))
    return A, b


def _hcr_bbox_for_output(
    A: np.ndarray, b: np.ndarray, out_shape: tuple[int, int, int],
    hcr_full_shape: tuple[int, int, int], pad_vox: int = 4,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Compute HCR voxel bbox covering all CZ output corners.

    Returns (start, stop) zyx tuples, clamped to ``hcr_full_shape``.
    """
    Z, Y, X = out_shape
    corners = np.array([
        [0,     0,     0    ],
        [Z - 1, 0,     0    ],
        [0,     Y - 1, 0    ],
        [0,     0,     X - 1],
        [Z - 1, Y - 1, 0    ],
        [Z - 1, 0,     X - 1],
        [0,     Y - 1, X - 1],
        [Z - 1, Y - 1, X - 1],
    ], dtype=float)
    mapped = corners @ A.T + b
    lo = np.floor(mapped.min(axis=0)).astype(int) - pad_vox
    hi = np.ceil(mapped.max(axis=0)).astype(int) + pad_vox
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(hcr_full_shape))
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def _open_zarr_level(zarr_path: str, level: str | int):
    """Open a zarr array at the named level. Handles both pyramid groups and
    a plain 3D zarr (where level is ignored). Works with zarr v2 and v3."""
    root = zarr.open(str(zarr_path), mode="r")
    # zarr 3.x uses zarr.Group; zarr 2.x had zarr.hierarchy.Group. Sniff via
    # the presence of a `shape` attribute (Array has it, Group does not).
    if hasattr(root, "shape"):
        return root  # already an Array
    return root[str(level)]


def _slice_zarr(arr, slc_z, slc_y, slc_x) -> np.ndarray:
    """Read a 3D sub-bbox from a zarr that may have leading singleton dims."""
    nd = arr.ndim
    if nd == 3:
        return np.asarray(arr[slc_z, slc_y, slc_x])
    if nd == 4:
        return np.asarray(arr[0, slc_z, slc_y, slc_x])
    if nd == 5:
        return np.asarray(arr[0, 0, slc_z, slc_y, slc_x])
    raise ValueError(f"Unexpected zarr ndim={nd}, shape={arr.shape}")


def warp_hcr_zarr_to_cz_grid(
    zarr_path: str,
    level: str | int,
    *,
    hcr_xy_um: float,
    hcr_z_um: float,
    lp: LockedPriorWarmStart,
    sz: float,
    T_eff: np.ndarray,
    grid: CzOutputGrid,
    order: int = 1,
    cval: float = 0.0,
    label_keep_set: set | None = None,
    label_keep_sets: dict | None = None,
    dtype_out=None,
    chunk_out_z: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Pull-resample an HCR zarr onto ``grid`` using the locked-prior + sz
    transform.

    Filtering modes (use exactly one, or neither):
    - ``label_keep_set`` — single set; returns a single warped volume.
    - ``label_keep_sets`` — dict {name: set}; reuses the zarr reads across
      all keep sets in one chunk-loop pass; returns ``out`` as a dict
      keyed by the same names. Use ``order=0`` for labels.

    ``chunk_out_z`` chunks the work along the OUTPUT Z axis to bound memory.
    """
    if label_keep_set is not None and label_keep_sets is not None:
        raise ValueError("Pass either label_keep_set OR label_keep_sets, not both.")
    multi = label_keep_sets is not None
    keep_sets = label_keep_sets if multi else (
        {None: label_keep_set} if label_keep_set is not None else {None: None}
    )

    arr = _open_zarr_level(zarr_path, level)
    full_shape_3d = tuple(arr.shape[-3:])
    A, b = _full_affine_voxel(lp, sz, T_eff, grid, hcr_xy_um, hcr_z_um)
    info = {
        "A": A.tolist(),
        "b": b.tolist(),
        "chunks": [],
        "full_shape_3d_hcr": list(full_shape_3d),
        "keep_set_names": list(keep_sets.keys()),
    }
    outs: dict = {}
    out_dtype = dtype_out

    if chunk_out_z is None:
        z_chunks = [(0, grid.shape[0])]
    else:
        z_chunks = [(z, min(z + chunk_out_z, grid.shape[0]))
                    for z in range(0, grid.shape[0], chunk_out_z)]

    for zlo, zhi in z_chunks:
        sub_grid_shape = (zhi - zlo, grid.shape[1], grid.shape[2])
        b_chunk = b + A @ np.array([zlo, 0, 0], dtype=float)
        (z0, y0, x0), (z1, y1, x1) = _hcr_bbox_for_output(
            A, b_chunk, sub_grid_shape, full_shape_3d
        )
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            for name in keep_sets:
                if name not in outs:
                    outs[name] = np.zeros(grid.shape, dtype=out_dtype or np.float32)
            info["chunks"].append({"out_z": [zlo, zhi], "empty": True})
            continue

        sub = _slice_zarr(arr, slice(z0, z1), slice(y0, y1), slice(x0, x1))
        max_id = int(sub.max()) if sub.size > 0 else 0
        b_local = b_chunk - np.array([z0, y0, x0], dtype=float)

        for name, keep in keep_sets.items():
            if keep is None:
                sub_filt = sub
            elif max_id == 0:
                sub_filt = sub
            else:
                keep_arr = np.array(sorted(int(i) for i in keep
                                           if 0 < int(i) <= max_id),
                                    dtype=sub.dtype)
                lut = np.zeros(max_id + 1, dtype=sub.dtype)
                lut[keep_arr] = keep_arr
                sub_filt = lut[sub]

            sub_cast = (sub_filt.astype(np.float32) if order > 0
                        and not np.issubdtype(sub_filt.dtype, np.floating)
                        else sub_filt)
            warped_chunk = affine_transform(
                sub_cast, A, offset=b_local, output_shape=sub_grid_shape,
                order=order, mode="constant", cval=cval,
            )
            chunk_dtype = out_dtype or sub.dtype
            if name not in outs:
                outs[name] = np.empty(grid.shape, dtype=chunk_dtype)
            outs[name][zlo:zhi] = warped_chunk.astype(chunk_dtype, copy=False)
            del sub_cast, warped_chunk
            if keep is not None:
                del sub_filt

        info["chunks"].append({
            "out_z": [zlo, zhi],
            "hcr_bbox_zyx": [[int(z0), int(z1)], [int(y0), int(y1)], [int(x0), int(x1)]],
            "sub_shape": list(sub.shape),
        })
        del sub

    for name in keep_sets:
        if name not in outs:
            outs[name] = np.zeros(grid.shape, dtype=out_dtype or np.float32)

    if multi:
        return outs, info
    return outs[next(iter(outs))], info
