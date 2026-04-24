"""Render matched CZ/HCR-488 image overlays at the fitted ICP scales.

For each of a chosen set of subjects:
  1. Compute R1 (R, t) + ICP (sxy, sz)   -- same as the benchmark.
  2. Load HCR 488 (downsampled ome-zarr level) and CZ OME-TIFF.
  3. Resample CZ into HCR-µm frame using the fitted affine
     (scipy.ndimage.affine_transform on the voxel grid).
  4. Render MIPs (XY and XZ) as a 2-channel composite
     (green=HCR 488, magenta=CZ resampled) plus centroid scatter
     (HCR GFP+ green dots, mapped CZ red dots).

Output: ``sessions/07_scale_failure_diagnosis/figures/image_overlay_<sid>.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import affine_transform

from anisotropic_icp import estimate_scales_icp_multi_start
from benchmark_analysis import analyze_subject, load_hcr_volume
from benchmark_data_loader import load_subject
from r1_revised import coarse_align_revised

FIG_DIR = Path('/root/capsule/code/sessions/07_scale_failure_diagnosis/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ['788406', '767018', '755252', '782149']


def _pct_clip(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    """Percentile-clip to [0, 1] for display."""
    lo = np.percentile(img, lo_pct)
    hi = np.percentile(img, hi_pct)
    if hi <= lo:
        return np.zeros_like(img, dtype=float)
    out = (img.astype(float) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def resample_cz_into_hcr(
    cz_vol: np.ndarray,           # (Z, Y, X) int/float
    cz_spacing_zyx: np.ndarray,   # (z_um, y_um, x_um)
    hcr_shape_zyx: tuple,
    hcr_spacing_zyx: np.ndarray,  # (z_um, y_um, x_um)
    R: np.ndarray,                # 3x3 rotation acting on (X, Y, Z) coords
    src_mean: np.ndarray,         # (X, Y, Z) µm
    scales: np.ndarray,           # (sx, sy, sz)
    dst_mean: np.ndarray,         # (X, Y, Z) µm
    order: int = 1,
) -> np.ndarray:
    """Resample CZ into HCR voxel grid via the row-vec affine.

    Forward map (centroids):  hcr_xyz = (cz_xyz − src_mean) @ R * scales + dst_mean

    Inverse (for each HCR voxel, find the CZ voxel that maps into it):
      hcr_xyz = (hcr_vox_xyz) * hcr_spacing_xyz            [voxel→µm, xyz order]
      cz_xyz  = ((hcr_xyz − dst_mean) / scales) @ R.T + src_mean
      cz_vox  = cz_xyz / cz_spacing_xyz                     [µm→voxel, xyz order]

    scipy's ``affine_transform`` takes a matrix acting on output-voxel
    coordinates (ZYX convention for our arrays) and returns the
    corresponding input-voxel coordinates.  We build that matrix
    directly.
    """
    # Permutation: our voxel arrays are (Z, Y, X) but centroid coords are (X, Y, Z).
    # Let P = permutation matrix [[0,0,1],[0,1,0],[1,0,0]] so that XYZ = P @ ZYX.
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    # hcr_spacing in XYZ order = P @ hcr_spacing_zyx
    hcr_sp_xyz = P @ hcr_spacing_zyx
    cz_sp_xyz = P @ cz_spacing_zyx
    s = np.asarray(scales, dtype=float)

    # Compose the inverse map linear part acting on (ZYX) output voxel coords:
    #   zyx_out  --[Diag(hcr_sp_zyx)]--> zyx_µm
    #   zyx_µm   --[P]--> xyz_µm   (since P swaps Z↔X)
    #   xyz_µm   --[subtract dst_mean]-->
    #            --[divide scales]-->
    #            --[@ R.T]-->
    #            --[add src_mean]--> cz_xyz_µm
    #   cz_xyz_µm --[P.T == P]--> cz_zyx_µm
    #   cz_zyx_µm --[1/cz_sp_zyx]--> cz_vox_zyx
    #
    # Build matrix/offset numerically (simpler than symbolic).
    A_out_vox_to_xyz = P @ np.diag(hcr_spacing_zyx)      # 3x3, maps zyx_vox → xyz_µm
    # linear part of xyz→xyz map:
    M_xyz = (np.diag(1.0 / s)) @ R.T                      # xyz_µm_centered→xyz_µm_centered
    A_xyz_to_cz_vox = np.diag(1.0 / cz_spacing_zyx) @ P.T  # xyz_µm → zyx_vox

    matrix = A_xyz_to_cz_vox @ M_xyz @ A_out_vox_to_xyz
    offset = (A_xyz_to_cz_vox
              @ (M_xyz @ (-np.asarray(dst_mean, dtype=float))
                 + np.asarray(src_mean, dtype=float)))

    return affine_transform(
        cz_vol.astype(np.float32), matrix=matrix, offset=offset,
        output_shape=hcr_shape_zyx, order=order, mode='constant', cval=0.0,
    )


def render_subject(sid: str, hcr_level: int = 5) -> Path:
    s = load_subject(sid)
    info = analyze_subject(s)

    r1 = coarse_align_revised(info['cz_xyz'], info['gfp_xyz'],
                              info['cz_surface'], info['hcr_surface'])
    icp = estimate_scales_icp_multi_start(info['cz_xyz'], info['gfp_xyz'], r1)
    assert icp.sxy is not None and icp.sz is not None
    sxy, sz = float(icp.sxy), float(icp.sz)
    R = np.asarray(r1.R, dtype=float)
    src_mean = np.asarray(r1.src_mean, dtype=float)
    dst_mean = np.asarray(r1.translation, dtype=float)
    scales = np.array([sxy, sxy, sz], dtype=float)

    # Load volumes
    hcr, hcr_xy, hcr_z = load_hcr_volume(s, channel='488', level=hcr_level)
    cz_candidates = (list(s.coreg_dir.glob('*reg-dim-swapped.ome.tif'))
                     or list(s.coreg_dir.glob('*zstack.tif')))
    if not cz_candidates:
        raise FileNotFoundError(f"No CZ stack found under {s.coreg_dir}")
    cz_path = cz_candidates[0]
    cz = tifffile.imread(str(cz_path)).astype(np.float32)
    cz_sp_zyx = np.array([s.cz_z_um, s.cz_xy_um, s.cz_xy_um])
    hcr_sp_zyx = np.array([hcr_z, hcr_xy, hcr_xy])

    # Resample CZ into HCR frame
    cz_in_hcr = resample_cz_into_hcr(
        cz, cz_sp_zyx,
        tuple(hcr.shape), hcr_sp_zyx,
        R, src_mean, scales, dst_mean,
        order=1,
    )

    # Crop the HCR MIP region to where the mapped CZ lands (saves a lot of
    # blank pixels in the figure)
    mapped_cz = (info['cz_xyz'] - src_mean) @ R * scales + dst_mean
    # HCR voxel grid extents in µm
    def clip_to_vox(lo_um, hi_um, sp, n):
        lo = int(max(0, np.floor(lo_um / sp) - 5))
        hi = int(min(n, np.ceil(hi_um / sp) + 5))
        return lo, hi

    x_lo, x_hi = clip_to_vox(mapped_cz[:, 0].min(), mapped_cz[:, 0].max(),
                             hcr_xy, hcr.shape[2])
    y_lo, y_hi = clip_to_vox(mapped_cz[:, 1].min(), mapped_cz[:, 1].max(),
                             hcr_xy, hcr.shape[1])
    z_lo, z_hi = clip_to_vox(mapped_cz[:, 2].min(), mapped_cz[:, 2].max(),
                             hcr_z, hcr.shape[0])
    hcr_c = hcr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    cz_c = cz_in_hcr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

    # XY MIP (max over Z), XZ MIP (max over Y)
    hcr_xy_mip = hcr_c.max(axis=0)
    cz_xy_mip = cz_c.max(axis=0)
    hcr_xz_mip = hcr_c.max(axis=1)
    cz_xz_mip = cz_c.max(axis=1)

    # Normalise per-channel
    hcr_xy_n = _pct_clip(hcr_xy_mip)
    cz_xy_n = _pct_clip(cz_xy_mip)
    hcr_xz_n = _pct_clip(hcr_xz_mip)
    cz_xz_n = _pct_clip(cz_xz_mip)

    # Two-channel composite: HCR in green, CZ in magenta
    # Magenta = R+B, Green = G
    xy_rgb = np.stack([cz_xy_n, hcr_xy_n, cz_xy_n], axis=-1)
    xz_rgb = np.stack([cz_xz_n, hcr_xz_n, cz_xz_n], axis=-1)

    # Centroid overlays (in HCR µm)
    gfp_xyz = info['gfp_xyz']
    # Keep only centroids inside the crop window
    def inside(xyz):
        x_um_lo, x_um_hi = x_lo * hcr_xy, x_hi * hcr_xy
        y_um_lo, y_um_hi = y_lo * hcr_xy, y_hi * hcr_xy
        z_um_lo, z_um_hi = z_lo * hcr_z, z_hi * hcr_z
        return ((xyz[:, 0] >= x_um_lo) & (xyz[:, 0] < x_um_hi)
                & (xyz[:, 1] >= y_um_lo) & (xyz[:, 1] < y_um_hi)
                & (xyz[:, 2] >= z_um_lo) & (xyz[:, 2] < z_um_hi))

    gfp_in = gfp_xyz[inside(gfp_xyz)]
    map_in = mapped_cz[inside(mapped_cz)]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.imshow(xy_rgb, origin='lower',
              extent=[x_lo * hcr_xy, x_hi * hcr_xy,
                      y_lo * hcr_xy, y_hi * hcr_xy])
    ax.set_title(f"{sid}  XY MIP   HCR 488 (green) ⊕ CZ resampled (magenta)")
    ax.set_xlabel("x (µm, HCR)"); ax.set_ylabel("y (µm, HCR)")

    ax = axes[0, 1]
    ax.imshow(xz_rgb, origin='lower', aspect='auto',
              extent=[x_lo * hcr_xy, x_hi * hcr_xy,
                      z_lo * hcr_z, z_hi * hcr_z])
    ax.set_title(f"{sid}  XZ MIP  (Y collapsed)")
    ax.set_xlabel("x (µm, HCR)"); ax.set_ylabel("z (µm, HCR)")

    ax = axes[1, 0]
    ax.imshow(xy_rgb, origin='lower',
              extent=[x_lo * hcr_xy, x_hi * hcr_xy,
                      y_lo * hcr_xy, y_hi * hcr_xy])
    ax.scatter(gfp_in[:, 0], gfp_in[:, 1], s=6, facecolor='none',
               edgecolor='#55ff55', linewidth=0.6, label=f"HCR GFP+ (n={len(gfp_in)})")
    ax.scatter(map_in[:, 0], map_in[:, 1], s=6, facecolor='none',
               edgecolor='#ff4488', linewidth=0.6, label=f"mapped CZ (n={len(map_in)})")
    ax.set_title("XY + centroid overlay")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 1]
    ax.imshow(xz_rgb, origin='lower', aspect='auto',
              extent=[x_lo * hcr_xy, x_hi * hcr_xy,
                      z_lo * hcr_z, z_hi * hcr_z])
    ax.scatter(gfp_in[:, 0], gfp_in[:, 2], s=6, facecolor='none',
               edgecolor='#55ff55', linewidth=0.6)
    ax.scatter(map_in[:, 0], map_in[:, 2], s=6, facecolor='none',
               edgecolor='#ff4488', linewidth=0.6)
    ax.set_title("XZ + centroid overlay")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")

    sxy_gt = None; sz_gt = None
    try:
        from benchmark_analysis import fit_anisotropic_similarity
        from benchmark_data_loader import landmark_pairs_um
        cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
        if len(cz_lm) >= 4:
            gt_fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
            sxy_gt = float(np.sqrt(gt_fit.scales[0] * gt_fit.scales[1]))
            sz_gt = float(gt_fit.scales[2])
    except Exception:
        pass

    suptitle = (f"{sid}   ICP scales: sxy={sxy:.3f}, sz={sz:.3f}  |  "
                f"GT: sxy={sxy_gt:.3f}, sz={sz_gt:.3f}"
                if sxy_gt is not None
                else f"{sid}   ICP scales: sxy={sxy:.3f}, sz={sz:.3f}")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f"image_overlay_{sid}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {sid}  wrote {out.name}")
    return out


def main():
    for sid in SUBJECTS:
        try:
            render_subject(sid)
        except Exception as e:
            print(f"  {sid}  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
