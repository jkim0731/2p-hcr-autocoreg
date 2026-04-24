"""Depth-resolved cell-density comparison in the GT-aligned overlap AABB.

Question: within the GT overlap box, does HCR strict-GFP+ exceed CZ at
every depth, or only at some depths? Is the density flat in depth, or
does the ratio vary systematically (e.g. HCR catches superficial cells
CZ misses)?

Uses the CZ→HCR GT affine from `fit_anisotropic_similarity`, identical
to 07e_gt_overlap_count.py. Depth is evaluated in the HCR µm frame
against the HCR pia surface for BOTH populations after mapping (they
now live in the same frame). A single depth axis is appropriate
because pia-normal distance is a property of tissue geometry, not
imaging modality.

Outputs:
  sessions/07e_sz_from_zvar_profile/gt_depth_density.json
  sessions/07e_sz_from_zvar_profile/figures_gt_depth/{sid}.png
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from benchmark_analysis import (
    analyze_subject,
    depth_from_surface,
    fit_anisotropic_similarity,
)
from benchmark_data_loader import (
    cz_px_to_um,
    hcr_px_to_um,
    landmark_pairs_um,
    load_subject,
)
from roi_area_sxy import D_SKIN_UM, SPOT_SUBJECTS

SESSION = Path("/root/capsule/code/sessions/07e_sz_from_zvar_profile")
FIG = SESSION / "figures_gt_depth"
FIG.mkdir(parents=True, exist_ok=True)

DEPTH_BIN_UM = 50.0


def _load_strict_gfp_module():
    spec = importlib.util.spec_from_file_location(
        "_gfp_thr", THIS / "07b_gfp_intersection_threshold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gfp_thr"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _cz_xyz_um(s, coreg_only=False):
    df = s.cz_centroids
    if coreg_only:
        ids = set(int(x) for x in s.coreg_table["cz_id"].values)
        df = df[df["cz_id"].astype(int).isin(ids)]
    arr = df[["z_px", "y_px", "x_px"]].to_numpy(float)
    return cz_px_to_um(arr, s)[:, [2, 1, 0]]


def _hcr_xyz_um_for_ids(s, ids):
    want = {int(x) for x in ids}
    if not want or s.hcr_centroids.empty:
        return np.zeros((0, 3))
    px = s.hcr_centroids.copy()
    keep = px["hcr_id"].astype(int).isin(want)
    arr = px.loc[keep, ["z_px", "y_px", "x_px"]].to_numpy(float)
    return hcr_px_to_um(arr, s)[:, [2, 1, 0]] if arr.size else np.zeros((0, 3))


def _apply_gt_affine(pts, fit, src_mean, dst_mean):
    return (pts - src_mean) @ fit.R * fit.scales + dst_mean


def _aabb(xyz):
    return [float(xyz[:, 0].min()), float(xyz[:, 0].max()),
            float(xyz[:, 1].min()), float(xyz[:, 1].max()),
            float(xyz[:, 2].min()), float(xyz[:, 2].max())]


def _intersect(a, b):
    return [max(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), min(a[3], b[3]),
            max(a[4], b[4]), min(a[5], b[5])]


def _in_box(xyz, box):
    return ((xyz[:, 0] >= box[0]) & (xyz[:, 0] <= box[1])
            & (xyz[:, 1] >= box[2]) & (xyz[:, 1] <= box[3])
            & (xyz[:, 2] >= box[4]) & (xyz[:, 2] <= box[5]))


def analyze(sid):
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    src_mean = cz_lm.mean(axis=0); dst_mean = hcr_lm.mean(axis=0)

    _gfp_thr = _load_strict_gfp_module()
    gi = _gfp_thr.analyze_subject(sid)
    strict_df = _gfp_thr.strict_gfp_df(sid, float(gi.cutoff_linear))
    strict_ids = set(int(x) for x in strict_df["hcr_id"].values)

    # all CZ + coreg CZ + HCR GFP+ in HCR frame
    cz_all_native = _cz_xyz_um(s, coreg_only=False)
    cz_cor_native = _cz_xyz_um(s, coreg_only=True)
    hcr_um = _hcr_xyz_um_for_ids(s, strict_ids)

    cz_all_gt = _apply_gt_affine(cz_all_native, fit, src_mean, dst_mean)
    cz_cor_gt = (_apply_gt_affine(cz_cor_native, fit, src_mean, dst_mean)
                 if len(cz_cor_native) else np.zeros((0, 3)))

    # cortex filter per modality in NATIVE frame
    cz_depth_native = depth_from_surface(cz_all_native, cz_surf)
    cz_p99 = float(np.nanpercentile(cz_depth_native, 99))
    cz_all_mask = (cz_depth_native >= D_SKIN_UM) & (cz_depth_native <= cz_p99)

    if len(cz_cor_native):
        cz_cor_depth = depth_from_surface(cz_cor_native, cz_surf)
        cz_cor_mask = (cz_cor_depth >= D_SKIN_UM) & (cz_cor_depth <= cz_p99)
    else:
        cz_cor_mask = np.zeros(0, dtype=bool)

    hcr_depth_native = depth_from_surface(hcr_um, hcr_surf)
    hcr_p99 = float(np.nanpercentile(hcr_depth_native, 99))
    hcr_mask = (hcr_depth_native >= D_SKIN_UM) & (hcr_depth_native <= hcr_p99)

    cz_all_gt_cx = cz_all_gt[cz_all_mask]
    cz_cor_gt_cx = cz_cor_gt[cz_cor_mask]
    hcr_cx = hcr_um[hcr_mask]

    # GT overlap AABB (using full CZ ∩ HCR)
    overlap = _intersect(_aabb(cz_all_gt_cx), _aabb(hcr_cx))

    # inside overlap (xyz all)
    cz_all_in = cz_all_gt_cx[_in_box(cz_all_gt_cx, overlap)]
    cz_cor_in = cz_cor_gt_cx[_in_box(cz_cor_gt_cx, overlap)] if len(cz_cor_gt_cx) else np.zeros((0, 3))
    hcr_in = hcr_cx[_in_box(hcr_cx, overlap)]

    # depth for each (use HCR surface since both live in HCR frame
    # after GT mapping — this is the relevant pia for the overlap box)
    d_cz_all = depth_from_surface(cz_all_in, hcr_surf) if len(cz_all_in) else np.zeros(0)
    d_cz_cor = depth_from_surface(cz_cor_in, hcr_surf) if len(cz_cor_in) else np.zeros(0)
    d_hcr = depth_from_surface(hcr_in, hcr_surf) if len(hcr_in) else np.zeros(0)

    d_max = float(max(d_cz_all.max() if d_cz_all.size else 0,
                      d_hcr.max() if d_hcr.size else 0))
    bin_edges = np.arange(D_SKIN_UM, d_max + DEPTH_BIN_UM, DEPTH_BIN_UM)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    c_all, _ = np.histogram(d_cz_all, bins=bin_edges)
    c_cor, _ = np.histogram(d_cz_cor, bins=bin_edges)
    c_hcr, _ = np.histogram(d_hcr, bins=bin_edges)

    # per-depth-bin density: counts per slab volume.
    # Slab volume = xy area of overlap AABB × bin thickness.
    # xy area of the box is same across depth; z extent is same. So
    # density is just count / (A_xy · DEPTH_BIN_UM).
    A_xy = (overlap[1] - overlap[0]) * (overlap[3] - overlap[2])
    V_bin = A_xy * DEPTH_BIN_UM
    rho_all = c_all / V_bin
    rho_cor = c_cor / V_bin
    rho_hcr = c_hcr / V_bin

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(bin_centers, rho_all * 1e6, 'o-', color='#268bd2',
            label=f'CZ all   (N∩={len(cz_all_in)})', lw=1.5)
    if len(cz_cor_in):
        ax.plot(bin_centers, rho_cor * 1e6, 'o--', color='#2aa198',
                label=f'CZ coreg (N∩={len(cz_cor_in)})', lw=1.2)
    ax.plot(bin_centers, rho_hcr * 1e6, 's-', color='#cb4b16',
            label=f'HCR GFP+ (N∩={len(hcr_in)})', lw=1.5)
    ax.set_xlabel('pia depth (µm) — HCR surface, HCR frame')
    ax.set_ylabel('density (cells / mm² / bin)')
    ax.set_title(f'{sid} — depth density in GT overlap AABB '
                 f'({DEPTH_BIN_UM:.0f} µm bins, A_xy={A_xy/1e6:.2f} mm²)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        r_all = np.where(c_hcr > 0, c_all / c_hcr, np.nan)
        r_cor = np.where(c_hcr > 0, c_cor / c_hcr, np.nan)
    ax.plot(bin_centers, r_all, 'o-', color='#268bd2', label='CZ_all / HCR+')
    if len(cz_cor_in):
        ax.plot(bin_centers, r_cor, 'o--', color='#2aa198', label='CZ_coreg / HCR+')
    ax.axhline(1.0, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('pia depth (µm)')
    ax.set_ylabel('count ratio CZ / HCR+')
    ax.set_ylim(0, 2)
    ax.set_title(f'{sid} — per-bin ratio vs depth')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG / f'{sid}.png'
    plt.savefig(out, dpi=120); plt.close(fig)

    # per-bin uniformity (CV across bins)
    def _cv(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.std(x) / np.mean(x)) if x.size >= 3 else float('nan')

    return {
        "sid": sid,
        "overlap_aabb_um": overlap,
        "A_xy_um2": A_xy,
        "bin_edges_um": bin_edges.tolist(),
        "bin_centers_um": bin_centers.tolist(),
        "counts_cz_all": c_all.tolist(),
        "counts_cz_coreg": c_cor.tolist(),
        "counts_hcr_gfp_strict": c_hcr.tolist(),
        "density_cz_all_per_mm3": (rho_all * 1e9).tolist(),
        "density_cz_coreg_per_mm3": (rho_cor * 1e9).tolist(),
        "density_hcr_per_mm3": (rho_hcr * 1e9).tolist(),
        "ratio_cz_all_over_hcr": r_all.tolist(),
        "ratio_cz_coreg_over_hcr": r_cor.tolist(),
        "cv_cz_all": _cv(c_all),
        "cv_cz_coreg": _cv(c_cor),
        "cv_hcr": _cv(c_hcr),
        "cv_ratio_cz_all": _cv(r_all),
        "n_cz_all_in_overlap": int(len(cz_all_in)),
        "n_cz_coreg_in_overlap": int(len(cz_cor_in)),
        "n_hcr_in_overlap": int(len(hcr_in)),
        "figure": str(out),
    }


def main():
    sids = sys.argv[1:] or sorted(SPOT_SUBJECTS)
    out = {}
    print(f"{'sid':<8} {'CV_CZall':>9} {'CV_HCR':>7} "
          f"{'CV(CZ/H)':>9} {'N_CZ∩':>6} {'N_HCR∩':>7} {'CZall/H':>8}")
    for sid in sids:
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"{sid}: ERROR {e}")
            import traceback; traceback.print_exc()
            continue
        ratio = (r['n_cz_all_in_overlap'] / r['n_hcr_in_overlap']
                 if r['n_hcr_in_overlap'] else float('nan'))
        print(f"{sid:<8} {r['cv_cz_all']:>9.3f} {r['cv_hcr']:>7.3f} "
              f"{r['cv_ratio_cz_all']:>9.3f} "
              f"{r['n_cz_all_in_overlap']:>6} {r['n_hcr_in_overlap']:>7} "
              f"{ratio:>8.3f}")
        out[sid] = r
    with open(SESSION / "gt_depth_density.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {SESSION/'gt_depth_density.json'}")


if __name__ == "__main__":
    main()
