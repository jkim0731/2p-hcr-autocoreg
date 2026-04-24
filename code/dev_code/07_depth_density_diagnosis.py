"""Depth-resolved density analysis inside the CZ→HCR overlap.

Diagnoses whether HCR GFP+ detection has a depth-dependent bias
relative to the *true* neuron population inside the overlap.

Ground-truth population at depth z (in HCR frame), using the coreg
table as the authoritative cell-match reference:

  pop_true(z) ≈ density_hcr_matched(z) + density_cz_mapped_unmatched(z)

- ``hcr_matched``        HCR cells that have a CZ partner in the
                        coreg table — known-true HCR detections.
- ``cz_unmatched``       CZ cells that have no HCR partner — true
                        neurons that HCR missed at that depth.
  (Mapped into HCR frame via the landmark-GT affine so that depth
  comparisons are on the same axis.)

We compare this baseline to:
  - ``hcr_gfp_plus``  v2.2 GFP+ threshold — what the estimator uses.
  - ``hcr_matched``   the known-matched subset alone.
  - ``cz_mapped``     the full CZ population, for coverage check.

All depths measured from the HCR pia plane (``hcr_surface`` a,b,c fit)
as ``z - (a*x + b*y + c)``.

Outputs per subject:
  - ``figures/depth_density_<sid>.png``
  - ``depth_density_summary.json`` with per-bin numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

import matplotlib.pyplot as plt
import numpy as np

from benchmark_analysis import analyze_subject, fit_anisotropic_similarity
from benchmark_data_loader import (BENCHMARK_SUBJECTS, cz_px_to_um,
                                    hcr_px_to_um, landmark_pairs_um,
                                    load_subject)

FIG_DIR = Path('/root/capsule/code/sessions/07_scale_failure_diagnosis/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = Path('/root/capsule/code/sessions/07_scale_failure_diagnosis/depth_density_summary.json')


def _centroids_xyz_um(df, s, kind: str) -> np.ndarray:
    """df has cz_id or hcr_id + (z_px, y_px, x_px) → (x_µm, y_µm, z_µm)."""
    px = df[['z_px', 'y_px', 'x_px']].to_numpy(dtype=float)
    um = cz_px_to_um(px, s) if kind == 'cz' else hcr_px_to_um(px, s)
    # um is (z_µm, y_µm, x_µm) — reorder to (x, y, z) to match cz_xyz convention
    return um[:, [2, 1, 0]]


def _gt_affine(s):
    """Return (R, src_mean, scales, dst_mean) from landmark Procrustes so
    mapped = (cz - src_mean) @ R * scales + dst_mean."""
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    if len(cz_lm) < 4:
        return None
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    R = np.asarray(fit.R, dtype=float)
    scales = np.asarray(fit.scales, dtype=float)
    src_mean = cz_lm.mean(axis=0)
    dst_mean = hcr_lm.mean(axis=0)
    return R, src_mean, scales, dst_mean


def _apply_affine(xyz: np.ndarray, R, src_mean, scales, dst_mean) -> np.ndarray:
    return (xyz - src_mean) @ R * scales + dst_mean


def _depth_from_pia(xyz_hcr: np.ndarray, pia) -> np.ndarray:
    """pia is hcr_surface dict with plane coefs a, b, c.
    z_pia(x,y) = a*x + b*y + c. Depth = z - z_pia."""
    a, b, c = pia['a'], pia['b'], pia['c']
    return xyz_hcr[:, 2] - (a * xyz_hcr[:, 0] + b * xyz_hcr[:, 1] + c)


def _density_profile(xyz_in_box: np.ndarray, depth: np.ndarray,
                     depth_bins_um: np.ndarray, xy_area_um2: float) -> np.ndarray:
    """cells / (µm² · µm depth) per bin.
    Uses XY area of the overlap (assumed constant across depth bins)."""
    counts, _ = np.histogram(depth, bins=depth_bins_um)
    dz = np.diff(depth_bins_um)
    vol = xy_area_um2 * dz
    return counts / vol  # cells / µm^3


def analyze_subject_depth(sid: str, depth_bin_um: float = 25.0) -> dict:
    s = load_subject(sid)
    info = analyze_subject(s)
    gt = _gt_affine(s)
    if gt is None:
        return {'subject': sid, 'status': 'no_landmarks'}
    R, src_mean, scales, dst_mean = gt

    # All HCR and CZ centroids in HCR µm-frame
    hcr_all_xyz = info['hcr_xyz']  # (x, y, z) µm, all HCR cells
    cz_all_xyz = info['cz_xyz']    # (x, y, z) µm, CZ native
    cz_mapped = _apply_affine(cz_all_xyz, R, src_mean, scales, dst_mean)

    # Match table
    coreg = s.coreg_table  # cz_id <-> hcr_id
    matched_hcr_ids = set(int(i) for i in coreg['hcr_id'].values)
    matched_cz_ids = set(int(i) for i in coreg['cz_id'].values)

    # HCR centroid table is indexed by hcr_id; pick matched subset
    hcr_rows = s.hcr_centroids
    hcr_rows = hcr_rows.assign(_is_matched=hcr_rows['hcr_id'].astype(int).isin(matched_hcr_ids))
    hcr_matched_xyz = _centroids_xyz_um(hcr_rows[hcr_rows['_is_matched']], s, 'hcr')

    cz_rows = s.cz_centroids
    cz_rows = cz_rows.assign(_is_matched=cz_rows['cz_id'].astype(int).isin(matched_cz_ids))
    # CZ unmatched (in HCR frame)
    cz_unmatched_px = cz_rows[~cz_rows['_is_matched']][['z_px', 'y_px', 'x_px']].to_numpy(float)
    cz_unmatched_native = cz_px_to_um(cz_unmatched_px, s)[:, [2, 1, 0]]
    cz_unmatched_mapped = _apply_affine(cz_unmatched_native, R, src_mean, scales, dst_mean)

    gfp_xyz = info['gfp_xyz']

    # Overlap box (AABB of mapped CZ, in HCR µm frame)
    pad = 10.0
    lo = cz_mapped.min(axis=0) - pad
    hi = cz_mapped.max(axis=0) + pad
    xy_area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))

    def _clip(xyz):
        m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        return xyz[m]

    # Clip all populations to the same box
    hcr_matched_box = _clip(hcr_matched_xyz)
    cz_unmatched_box = _clip(cz_unmatched_mapped)
    gfp_box = _clip(gfp_xyz)
    cz_mapped_box = _clip(cz_mapped)
    hcr_all_box = _clip(hcr_all_xyz)

    # Depth from pia (HCR pia plane)
    pia = info['hcr_surface']
    d_hcr_matched = _depth_from_pia(hcr_matched_box, pia)
    d_cz_unmatched = _depth_from_pia(cz_unmatched_box, pia)
    d_gfp = _depth_from_pia(gfp_box, pia)
    d_cz_mapped = _depth_from_pia(cz_mapped_box, pia)
    d_hcr_all = _depth_from_pia(hcr_all_box, pia)

    # Depth bins — same for all populations
    dmin = float(np.floor(min(d_hcr_matched.min(), d_cz_unmatched.min(),
                              d_gfp.min(), d_cz_mapped.min()) / depth_bin_um)
                  * depth_bin_um)
    dmax = float(np.ceil(max(d_hcr_matched.max(), d_cz_unmatched.max(),
                             d_gfp.max(), d_cz_mapped.max()) / depth_bin_um)
                  * depth_bin_um)
    bins = np.arange(dmin, dmax + depth_bin_um, depth_bin_um)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rho_hcr_matched = _density_profile(hcr_matched_box, d_hcr_matched, bins, xy_area)
    rho_cz_unmatched = _density_profile(cz_unmatched_box, d_cz_unmatched, bins, xy_area)
    rho_gfp = _density_profile(gfp_box, d_gfp, bins, xy_area)
    rho_cz_mapped = _density_profile(cz_mapped_box, d_cz_mapped, bins, xy_area)
    rho_hcr_all = _density_profile(hcr_all_box, d_hcr_all, bins, xy_area)
    rho_truth = rho_hcr_matched + rho_cz_unmatched

    # GFP+ detection ratio relative to truth baseline (per bin)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_gfp = np.where(rho_truth > 0, rho_gfp / rho_truth, np.nan)
        ratio_cz = np.where(rho_truth > 0, rho_cz_mapped / rho_truth, np.nan)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    scale = 1e6  # convert cells/µm^3 → cells / 1000·1000 µm²·µm

    ax = axes[0]
    ax.plot(centers, rho_truth * scale, '-', lw=2.2, color='black',
            label='truth = matched HCR + unmatched CZ (mapped)')
    ax.plot(centers, rho_gfp * scale, '-', lw=1.8, color='#2aa198',
            label='HCR GFP+ (v2.2 threshold)')
    ax.plot(centers, rho_hcr_matched * scale, '--', lw=1.2, color='#268bd2',
            label='HCR matched (coreg)')
    ax.plot(centers, rho_cz_unmatched * scale, ':', lw=1.2, color='#cb4b16',
            label='CZ unmatched (mapped)')
    ax.plot(centers, rho_cz_mapped * scale, '-.', lw=0.9, color='grey',
            label='CZ mapped (all)')
    ax.plot(centers, rho_hcr_all * scale, '-', lw=0.8, color='#93a1a1',
            alpha=0.6, label='HCR all (in overlap)')
    ax.set_xlabel('depth from HCR pia (µm)')
    ax.set_ylabel('density (cells / 10^6 µm³)')
    ax.set_title(f"{sid}  density vs depth  (overlap xy area = {xy_area/1e6:.2f} mm²)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)

    ax = axes[1]
    ax.plot(centers, ratio_gfp, '-', lw=1.8, color='#2aa198',
            label='GFP+ / truth')
    ax.plot(centers, ratio_cz, '-', lw=1.2, color='grey',
            label='CZ mapped / truth (coverage)')
    ax.axhline(1.0, color='black', lw=0.6)
    ax.set_xlabel('depth from HCR pia (µm)')
    ax.set_ylabel('ratio')
    ax.set_title(f"{sid}  detection / CZ-coverage ratios vs depth")
    ax.set_ylim(0, max(5.0, float(np.nanmax(ratio_gfp) * 1.1) if np.any(np.isfinite(ratio_gfp)) else 5.0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / f"depth_density_{sid}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    return {
        'subject': sid,
        'xy_area_um2': xy_area,
        'depth_bin_um': depth_bin_um,
        'depth_centers_um': centers.tolist(),
        'rho_truth_per_um3': rho_truth.tolist(),
        'rho_gfp_per_um3': rho_gfp.tolist(),
        'rho_hcr_matched_per_um3': rho_hcr_matched.tolist(),
        'rho_cz_unmatched_per_um3': rho_cz_unmatched.tolist(),
        'rho_cz_mapped_per_um3': rho_cz_mapped.tolist(),
        'rho_hcr_all_per_um3': rho_hcr_all.tolist(),
        'gfp_over_truth': ratio_gfp.tolist(),
        'cz_over_truth': ratio_cz.tolist(),
        'n_hcr_matched_in_box': int(hcr_matched_box.shape[0]),
        'n_cz_unmatched_in_box': int(cz_unmatched_box.shape[0]),
        'n_gfp_in_box': int(gfp_box.shape[0]),
        'n_cz_mapped_in_box': int(cz_mapped_box.shape[0]),
        'n_hcr_all_in_box': int(hcr_all_box.shape[0]),
        'figure': str(out),
    }


def main():
    out = []
    for sid in BENCHMARK_SUBJECTS:
        try:
            r = analyze_subject_depth(sid)
            print(f"  {sid} ok  GFP+/truth integrated ≈ "
                  f"{np.nanmean(r['gfp_over_truth']):.2f}  "
                  f"(xy_area = {r['xy_area_um2']/1e6:.2f} mm²)")
            out.append(r)
        except Exception as e:
            print(f"  {sid} FAILED: {type(e).__name__}: {e}")
            out.append({'subject': sid, 'status': f'error: {e}'})
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
