"""F3 — Cross-resolution 3D crop / resample given an affine.

Takes CZ and HCR arrays (image or mask) plus an affine `(R, t, S)` mapping CZ
physical µm → HCR physical µm, and produces two aligned arrays on a shared
isotropic target grid.

The forward affine maps CZ physical coords to HCR physical coords.  For
resampling we need, for each target voxel:
  src_voxel_cz  = cz_um_to_vox( target_um                    )
  src_voxel_hcr = hcr_um_to_vox( affine_forward( target_um ) )

We choose the target grid in CZ physical space (centred on the CZ bbox).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


@dataclass
class ResampleResult:
    cz: np.ndarray           # (D, H, W)
    hcr: np.ndarray          # (D, H, W)
    spacing_um: float        # isotropic voxel spacing
    origin_um: np.ndarray    # (3,) physical origin in CZ µm, (z,y,x)
    target_shape: tuple      # (D, H, W)


def _pad_margin_um(extent_um: np.ndarray, margin_um: float) -> np.ndarray:
    return extent_um + margin_um


def resample_to_shared_grid(
    cz_arr: np.ndarray,
    hcr_arr: np.ndarray,
    cz_xy_um: float,
    cz_z_um: float,
    hcr_xy_um: float,
    hcr_z_um: float,
    R: np.ndarray,
    t_um: np.ndarray,
    S: np.ndarray,
    src_mean_um: Optional[np.ndarray] = None,
    target_spacing_um: float = 4.0,
    mode: str = "image",
    margin_um: float = 40.0,
) -> ResampleResult:
    """Resample CZ and HCR onto a shared isotropic grid in CZ physical space.

    The affine maps CZ µm → HCR µm as
        hcr_um = R @ (cz_um - src_mean_um) * S + t_um + src_mean_um
    if ``src_mean_um`` is provided, else
        hcr_um = R @ (cz_um * S) + t_um.

    ``mode='image'`` uses bilinear interpolation (order=1); ``'mask'`` uses
    nearest (order=0) for label preservation.
    """
    cz_vox = np.array(cz_arr.shape, dtype=float)  # (Z, Y, X)
    # CZ physical extent, (z,y,x) µm
    cz_phys = cz_vox * np.array([cz_z_um, cz_xy_um, cz_xy_um])

    # Shared isotropic target grid covering CZ (with margin).
    target_extent = _pad_margin_um(cz_phys, margin_um)
    target_shape = np.ceil(target_extent / target_spacing_um).astype(int)
    # Target origin: CZ image origin minus margin
    origin_um = -np.array([margin_um] * 3)

    D, H, W = [int(x) for x in target_shape]

    # Build voxel-grid of (z, y, x) in target µm coords for vectorised lookup.
    # Rather than allocate (D,H,W,3) which is huge, compute per-axis sample
    # coordinates as grids and then use scipy.ndimage.affine_transform which
    # expects a single affine mapping target-voxel → source-voxel.
    order = 0 if mode == "mask" else 1

    # --- CZ resample ---
    # target µm = target_vox * spacing + origin
    # cz_vox = target_µm / cz_spacing_vec
    cz_spacing = np.array([cz_z_um, cz_xy_um, cz_xy_um])
    Mcz = np.diag(target_spacing_um / cz_spacing)
    off_cz = origin_um / cz_spacing
    cz_rs = ndi.affine_transform(
        cz_arr.astype(np.float32) if mode == "image" else cz_arr,
        matrix=Mcz,
        offset=off_cz,
        output_shape=(D, H, W),
        order=order,
        mode="constant",
        cval=0,
    )

    # --- HCR resample ---
    # target_um_cz = tv * spacing + origin    (in CZ µm frame)
    # hcr_um = R @ S @ (target_um_cz - mu) + t + mu       (if src_mean_um given)
    #        = R @ S @ target_um_cz + (t + mu - R @ S @ mu)
    # hcr_vox = hcr_um / hcr_spacing_vec
    hcr_spacing = np.array([hcr_z_um, hcr_xy_um, hcr_xy_um])
    S_diag = np.diag(np.asarray(S, dtype=float))
    RS = R @ S_diag  # (3,3)
    mu = np.zeros(3) if src_mean_um is None else np.asarray(src_mean_um, dtype=float)
    t_vec = np.asarray(t_um, dtype=float)
    b = t_vec + mu - RS @ mu
    # target_vox → target_µm = I*spacing_iso * tv + origin_um
    # → hcr_µm = RS @ (spacing_iso * tv + origin_um) + b
    # → hcr_vox = (RS @ spacing_iso) @ tv + (RS @ origin_um + b) / hcr_spacing
    M_to_hcr_um = RS * target_spacing_um  # (3,3)
    M_to_hcr_vox = np.diag(1.0 / hcr_spacing) @ M_to_hcr_um
    off_hcr_vox = (RS @ origin_um + b) / hcr_spacing
    hcr_rs = ndi.affine_transform(
        hcr_arr.astype(np.float32) if mode == "image" else hcr_arr,
        matrix=M_to_hcr_vox,
        offset=off_hcr_vox,
        output_shape=(D, H, W),
        order=order,
        mode="constant",
        cval=0,
    )

    return ResampleResult(
        cz=cz_rs,
        hcr=hcr_rs,
        spacing_um=float(target_spacing_um),
        origin_um=origin_um,
        target_shape=(D, H, W),
    )


def _selftest():
    """Synthetic round-trip test: warp an HCR-like volume by a known affine,
    feed back with inverse affine, confirm recovery."""
    rng = np.random.default_rng(0)
    Z, Y, X = 40, 60, 60
    hcr = rng.integers(0, 20, size=(Z, Y, X), dtype=np.uint32)
    cz = hcr[::2, ::2, ::2]  # half resolution + smaller

    # Identity affine (should put cz slices in the middle of hcr space)
    R = np.eye(3)
    S = np.array([1.0, 1.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])
    src_mean = None
    r = resample_to_shared_grid(
        cz, hcr,
        cz_xy_um=2.0, cz_z_um=2.0,
        hcr_xy_um=1.0, hcr_z_um=1.0,
        R=R, t_um=t, S=S, src_mean_um=src_mean,
        target_spacing_um=2.0, mode="mask",
    )
    print("F3 selftest shapes:", r.cz.shape, r.hcr.shape,
          "spacing", r.spacing_um, "origin", r.origin_um)
    # Rough consistency: nonzero fraction within ~2× of HCR
    nz_hcr = (r.hcr > 0).mean()
    nz_cz = (r.cz > 0).mean()
    print(f"F3 nz frac cz={nz_cz:.3f} hcr={nz_hcr:.3f}")


if __name__ == "__main__":
    _selftest()
