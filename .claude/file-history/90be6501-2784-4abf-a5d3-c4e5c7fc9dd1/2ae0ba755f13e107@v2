"""S50 — Probe true seg-mask-NCC coarse alignment (M1 replacement).

Tests whether replacing M1's centroid-density volumes with real segmentation
masks (HCR: F1 on cell_body_segmentation/segmentation_mask.zarr restricted to
GFP+; CZ: S48 cz_voronoi_labels binarised) improves coarse-alignment accuracy
on 782149 (currently unreachable) without regressing other subjects.

Reuses M1's FFT-NCC + scale sweep; swaps volumes for real masks.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from lib.mask_loaders import load_hcr_seg_mask  # noqa: E402
from lib.cz_labels import cz_voronoi_labels  # noqa: E402
from bench.candidate_impls._m1_mask_ncc import _ncc3d_valid  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402


def _make_grid_from_mask(mask_zyx, xy_um, z_um, spacing_um, sigma_um):
    """Resample a label/mask volume to an isotropic µm grid; Gaussian-smooth.

    Returns (vol, origin_um) with origin_um = (z_min, y_min, x_min) of the
    cropped bounding box.  Trims to the nonzero bbox with a 50 µm margin."""
    bx = np.asarray(mask_zyx > 0, dtype=np.uint8)
    if bx.sum() == 0:
        return None, None
    zz = np.where(bx.sum(axis=(1, 2)))[0]
    yy = np.where(bx.sum(axis=(0, 2)))[0]
    xx = np.where(bx.sum(axis=(0, 1)))[0]
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    crop = bx[z0:z1, y0:y1, x0:x1].astype(np.float32)
    # Physical extent of cropped region
    ext_z = crop.shape[0] * z_um
    ext_y = crop.shape[1] * xy_um
    ext_x = crop.shape[2] * xy_um
    out = zoom(crop,
               (z_um / spacing_um, xy_um / spacing_um, xy_um / spacing_um),
               order=1)
    if sigma_um > 0:
        out = gaussian_filter(out, sigma=sigma_um / spacing_um)
    origin_um = np.array([z0 * z_um, y0 * xy_um, x0 * xy_um], dtype=float)
    return out, origin_um


def probe(subj, *, spacing_um=20.0, sigma_um=40.0, level=4,
          sxy_grid=(1.4, 1.6, 1.8, 2.0, 2.2),
          sz_grid=(1.8, 2.2, 2.6, 3.0, 3.4)):
    s = load_subject(subj)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    print(f"  {subj}: n_cz={len(cz_ids)} n_hcr_gfp={len(hcr_ids)}")

    t0 = time.time()
    hcr_gfp_ids = s.hcr_gfp_df["hcr_id"].astype(int).tolist()
    hcr_mask, hcr_xy, hcr_z = load_hcr_seg_mask(s, level=level, gfp_ids=hcr_gfp_ids)
    t_hcr = time.time() - t0
    print(f"    HCR mask (level={level}): shape={hcr_mask.shape} "
          f"xy={hcr_xy:.2f}µm z={hcr_z:.2f}µm nz={int((hcr_mask>0).sum())} "
          f"wall={t_hcr:.1f}s")

    t1 = time.time()
    cz_labels = cz_voronoi_labels(s, R_um=8.0)
    t_cz = time.time() - t1
    print(f"    CZ voronoi labels: shape={cz_labels.shape} "
          f"xy={s.cz_xy_um:.3f}µm z={s.cz_z_um:.2f}µm nz={int((cz_labels>0).sum())} "
          f"wall={t_cz:.1f}s")

    # Resample to shared isotropic grid
    hcr_vol, hcr_origin = _make_grid_from_mask(hcr_mask, hcr_xy, hcr_z,
                                                spacing_um, sigma_um)
    cz_vol_src, cz_origin = _make_grid_from_mask(cz_labels,
                                                  float(s.cz_xy_um),
                                                  float(s.cz_z_um),
                                                  spacing_um, sigma_um)
    if hcr_vol is None or cz_vol_src is None:
        print("    FAIL: empty mask")
        return

    print(f"    HCR vol: shape={hcr_vol.shape} origin_um={hcr_origin.round(1).tolist()}")
    print(f"    CZ  vol: shape={cz_vol_src.shape} origin_um={cz_origin.round(1).tolist()}")

    # Prepare rotated-CZ template; sweep (sxy, sz)
    # Instead of rotating+rescaling the raw CZ vol (which would resample-interpolate
    # every time), render the cz centroid cloud + cz-cell-size kernel at each
    # (sxy, sz), then NCC against HCR vol.  Use the same simple approach as M1 but
    # on the *voronoi-bounded* mask.
    best = dict(zscore=-np.inf, sxy=None, sz=None, peak=None, t_um=None)
    t_sweep = time.time()
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # 180° XY
    # Make a single oriented CZ template at base scale; zoom by (sz, sxy, sxy)
    # per combo to simulate expansion.
    cz_base = cz_vol_src  # already at spacing_um
    for sxy in sxy_grid:
        for sz in sz_grid:
            t_vol = zoom(cz_base, (sz, sxy, sxy), order=1)
            # Apply 180° XY rotation: flip YX axes
            t_vol = t_vol[:, ::-1, ::-1].copy()
            if any(t_vol.shape[i] >= hcr_vol.shape[i] for i in range(3)):
                continue
            ncc, off = _ncc3d_valid(hcr_vol, t_vol)
            if ncc.size == 0 or not np.isfinite(ncc).any():
                continue
            peak = float(ncc.max())
            vals = ncc[np.isfinite(ncc)]
            mu = float(vals.mean()); sd = float(vals.std())
            z = (peak - mu) / sd if sd > 1e-9 else 0.0
            if z > best["zscore"]:
                # Template center in HCR-aligned µm coords
                top_lo_um = hcr_origin + np.array(off) * spacing_um
                tpl_c_um = top_lo_um + (np.array(t_vol.shape) * spacing_um) * 0.5
                best = dict(zscore=z, sxy=float(sxy), sz=float(sz),
                            peak=peak, t_um=tpl_c_um.tolist(),
                            shape=tuple(t_vol.shape),
                            off=tuple(int(x) for x in off))
    t_s = time.time() - t_sweep
    print(f"    sweep wall={t_s:.1f}s best sxy={best['sxy']} sz={best['sz']} "
          f"ncc_peak={best['peak']:.3f} zscore={best['zscore']:.1f}")

    # Validate: compare predicted CZ center to GT-fit CZ center.
    # Fit anisotropic affine on coreg_table GT pairs → compute predicted mean CZ
    # position in HCR space; compare to best['t_um'].
    from benchmark_analysis import fit_anisotropic_similarity
    from lib.centroid_helpers import apply_aniso_fit
    gt = s.coreg_table
    cz_idx_of = {int(c): i for i, c in enumerate(cz_ids)}
    hcr_idx_of = {int(h): i for i, h in enumerate(hcr_ids)}
    hits_cz = np.array([cz_idx_of.get(int(c), -1) for c in gt["cz_id"]])
    hits_hcr = np.array([hcr_idx_of.get(int(h), -1) for h in gt["hcr_id"]])
    ok = (hits_cz >= 0) & (hits_hcr >= 0)
    if ok.sum() < 4:
        print("    SKIP: too few GT pairs for validation")
        return
    try:
        fit = fit_anisotropic_similarity(cz_um[hits_cz[ok]], hcr_um[hits_hcr[ok]])
        cz_warped_gt = apply_aniso_fit(cz_um, fit)
        gt_cz_center_in_hcr = cz_warped_gt.mean(0)
        gt_sxy = 0.5 * (fit.scales[1] + fit.scales[2])
        gt_sz = fit.scales[0]
        pred = np.asarray(best["t_um"])
        origin_err = np.linalg.norm(pred - gt_cz_center_in_hcr)
        print(f"    GT fit: sxy={gt_sxy:.3f} sz={gt_sz:.3f} "
              f"cz_center_in_hcr={gt_cz_center_in_hcr.round(1).tolist()}")
        print(f"    ORIGIN_ERR={origin_err:.1f} µm  "
              f"sxy_err={best['sxy']-gt_sxy:+.3f} sz_err={best['sz']-gt_sz:+.3f}")
    except Exception as e:
        print(f"    FIT_FAIL: {e}")


def main():
    subjects = sys.argv[1:] or ["788406", "782149"]
    for subj in subjects:
        print(f"\n===== {subj} =====")
        probe(subj)


if __name__ == "__main__":
    main()
