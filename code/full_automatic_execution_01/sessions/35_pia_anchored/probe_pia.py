"""S35 probe — pia-anchored alignment for 782149 and comparison subjects.

Hypothesis
----------
S34 established that on 782149 no ICP variant in centroid xyz space has a
local minimum at truth — the wrong basin compresses CZ's Z range to fit
HCR. Cause (S32 diag): 782149 has a 12° pia-plane tilt and only 55% of
CZ's axial extent actually lies inside HCR's Z extent at the true scale.
ICP's objective function cannot separate "squeeze CZ to fit HCR" from
"align CZ to HCR at true scale".

If we reparameterize z → depth-from-pia in both modalities, the pia
surface becomes "depth=0" in both, and the Z-offset + Z-tilt mismatch is
absorbed. The remaining problem is 2D XY + anisotropic depth scale —
a landscape ICP should handle better.

Approach
--------
For each subject:
  1. Fit CZ pia surface (image_ceiling) and HCR pia surface
     (image_autoselect, quadratic quantile-robust fit).
  2. Compute depth_from_surface for all CZ cells and HCR GFP+ cells.
  3. Build pia-anchored clouds: (x, y, depth).
  4. Run 6-seed multi-start reciprocal-NN ICP in anchored space,
     sweep trim ∈ {0.4, 0.6, 0.8, 0.9}.
  5. Convert each fit back to xyz: apply to (x_cz, y_cz, depth_cz),
     then add back the HCR pia plane evaluated at the predicted xy.
  6. Report OR n<50 and SS-ranker n<50 in xyz frame.

Comparison: baseline 6-seed multi-start in pure xyz (no pia anchoring).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from lib.centroid_helpers import centroids_um  # noqa
from benchmark_analysis import (
    estimate_pia_surface_image_ceiling,
    depth_from_surface,
    load_hcr_combined,
)  # noqa
from anisotropic_icp import estimate_scales_icp_multi_start  # noqa


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


R_XYZ = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)


def gt_pairs(s):
    ct = s.coreg_table
    cz = s.cz_centroids.set_index("cz_id")
    hc = s.hcr_centroids.set_index("hcr_id")
    mask = ct["cz_id"].isin(cz.index) & ct["hcr_id"].isin(hc.index)
    ct = ct[mask]
    cz_rows = cz.loc[ct["cz_id"].values]
    hc_rows = hc.loc[ct["hcr_id"].values]
    cz_um = cz_px_to_um(cz_rows[["z_px", "y_px", "x_px"]].values, s)
    hc_um = hcr_px_to_um(hc_rows[["z_px", "y_px", "x_px"]].values, s)
    return cz_um[:, [2, 1, 0]], hc_um[:, [2, 1, 0]]


def score_recip_unique(pred_xyz, hcr_xyz):
    th = cKDTree(hcr_xyz); tc = cKDTree(pred_xyz)
    d_c2h, idx_c2h = th.query(pred_xyz, k=1)
    _, idx_h2c = tc.query(hcr_xyz, k=1)
    recip = int(((idx_h2c[idx_c2h] == np.arange(len(pred_xyz))) & (d_c2h < 30)).sum())
    uniq = len(np.unique(idx_c2h)) / len(pred_xyz)
    return recip * uniq, recip, uniq


def fit_cz_pia(s):
    """Image-based CZ pia surface fit."""
    import tifffile
    from pathlib import Path
    files = list(Path(s.coreg_dir).glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(Path(s.coreg_dir).glob("*zstack.tif"))
    img = tifffile.imread(str(files[0]))
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    cz_xyz = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].values, s
    )[:, [2, 1, 0]]
    return estimate_pia_surface_image_ceiling(
        cz_xyz, img, s.cz_z_um, s.cz_xy_um,
        relative_margin=0.05, min_signal_abs=50.0,
    )


def surface_z(surf, xy):
    """Evaluate z_pia(x, y) for a quadratic surface dict."""
    x, y = xy[:, 0], xy[:, 1]
    a, b, c = surf["a"], surf["b"], surf["c"]
    z = a * x + b * y + c
    p = surf.get("p", 0.0); q = surf.get("q", 0.0); r = surf.get("r", 0.0)
    if p or q or r:
        z = z + p * x * x + q * x * y + r * y * y
    return z


def to_anchored(xyz, surf):
    """Replace z with depth-from-pia."""
    out = xyz.copy()
    out[:, 2] = depth_from_surface(xyz, surf)
    return out


def from_anchored(anchored_xyz, surf):
    """Convert pia-anchored (x, y, depth) back to (x, y, z)."""
    out = anchored_xyz.copy()
    out[:, 2] = anchored_xyz[:, 2] + surface_z(surf, anchored_xyz[:, :2])
    return out


def oracle_lt50_anchored(fit, cz_gt_xyz, hcr_gt_xyz, cz_surf, hcr_surf):
    """Apply an anchored-space fit to GT CZ, convert back, compare to GT HCR."""
    cz_gt_anch = to_anchored(cz_gt_xyz, cz_surf)
    pred_anch = (cz_gt_anch * fit.scales) @ fit.R.T + fit.translation
    pred_xyz = from_anchored(pred_anch, hcr_surf)
    d = np.linalg.norm(pred_xyz - hcr_gt_xyz, axis=1)
    return int((d < 50).sum()), float(np.median(d))


def oracle_lt50_xyz(fit, cz_gt_xyz, hcr_gt_xyz):
    pred = (cz_gt_xyz * fit.scales) @ fit.R.T + fit.translation
    d = np.linalg.norm(pred - hcr_gt_xyz, axis=1)
    return int((d < 50).sum()), float(np.median(d))


def run_multistart(cz_xyz, hcr_xyz, seeds: dict, trims=(0.4, 0.6, 0.8, 0.9)):
    """Returns per-(seed, trim) dict of fits + score."""
    results = []
    for name, t_seed in seeds.items():
        for trim in trims:
            f0 = _Fit(R=R_XYZ, src_mean=cz_xyz.mean(0),
                      translation=np.asarray(t_seed, float),
                      scales=np.ones(3))
            try:
                r = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f0,
                                                    inlier_residual_quantile=trim)
            except Exception:
                continue
            if r.fit is None:
                continue
            pred = (cz_xyz * r.fit.scales) @ r.fit.R.T + r.fit.translation
            rank, recip, uniq = score_recip_unique(pred, hcr_xyz)
            results.append(dict(seed=name, trim=trim, fit=r.fit, rank=rank))
    return results


def main():
    rows = []
    for subj in ["788406", "755252", "767022", "782149"]:
        print(f"\n=== {subj} ===", flush=True)
        s = load_subject(subj)
        cz_um, _ = centroids_um(s, "cz")
        hcr_um, _ = centroids_um(s, "hcr_gfp")
        cz_xyz = cz_um[:, [2, 1, 0]]
        hcr_xyz = hcr_um[:, [2, 1, 0]]
        cz_gt, hcr_gt = gt_pairs(s)
        print(f"  n_cz={len(cz_xyz)}, n_hcr_gfp={len(hcr_xyz)}, n_gt={len(cz_gt)}")

        # Fit pia surfaces (single-fit each; autoselect too slow for probe)
        t0 = time.time()
        cz_surf = fit_cz_pia(s)
        hcr_vol, hcr_xy, hcr_z, _ = load_hcr_combined(s, level=4)
        hcr_surf = estimate_pia_surface_image_ceiling(
            hcr_xyz, hcr_vol, hcr_z, hcr_xy,
            relative_margin=0.05, min_signal_abs=0.1,
        )
        print(f"  pia fits ({time.time()-t0:.1f}s):"
              f" CZ tilt={cz_surf.get('tilt_deg', float('nan')):.1f}°"
              f" c={cz_surf['c']:.0f};"
              f" HCR tilt={hcr_surf.get('tilt_deg', float('nan')):.1f}°"
              f" c={hcr_surf['c']:.0f}")

        # Pia-anchored clouds
        cz_anch = to_anchored(cz_xyz, cz_surf)
        hcr_anch = to_anchored(hcr_xyz, hcr_surf)
        print(f"  CZ depth: [{cz_anch[:,2].min():.0f}, {cz_anch[:,2].max():.0f}] µm;"
              f" HCR depth: [{hcr_anch[:,2].min():.0f}, {hcr_anch[:,2].max():.0f}] µm")

        # 6 default seeds in anchored space
        gfp_c_anch = hcr_anch.mean(0)
        seeds_anch = {
            "hcr_gfp":     gfp_c_anch,
            "gfp_dz+100":  gfp_c_anch + [0, 0, 100],
            "gfp_dz-100":  gfp_c_anch + [0, 0, -100],
            "gfp_dz+200":  gfp_c_anch + [0, 0, 200],
            "gfp_q25":     np.quantile(hcr_anch, 0.25, axis=0),
            "gfp_q75":     np.quantile(hcr_anch, 0.75, axis=0),
        }
        res_anch = run_multistart(cz_anch, hcr_anch, seeds_anch)
        best_or_anch = dict(n=0, seed=None, trim=None, med=float('nan'),
                            sxy=float('nan'), sz=float('nan'))
        best_ss_anch = dict(n=0, seed=None, trim=None, med=float('nan'), rank=-1)
        for r in res_anch:
            n50, med = oracle_lt50_anchored(r['fit'], cz_gt, hcr_gt,
                                             cz_surf, hcr_surf)
            if n50 > best_or_anch['n']:
                best_or_anch = dict(n=n50, seed=r['seed'], trim=r['trim'],
                                    med=med, sxy=r['fit'].scales[0],
                                    sz=r['fit'].scales[2])
            if r['rank'] > best_ss_anch['rank']:
                best_ss_anch = dict(n=n50, seed=r['seed'], trim=r['trim'],
                                    med=med, rank=r['rank'])
        print(f"  PIA-ANCHORED: OR n<50={best_or_anch['n']}"
              f" ({best_or_anch['seed']}, {best_or_anch['trim']})"
              f" sxy={best_or_anch['sxy']:.2f} sz={best_or_anch['sz']:.2f}"
              f" med={best_or_anch['med']:.0f};"
              f" SS n<50={best_ss_anch['n']} rank={best_ss_anch['rank']:.1f}")

        # Baseline: same 6 seeds in xyz frame (same as S33 seven_seeds but
        # without the I2 seed, for fair pia-vs-baseline comparison)
        gfp_c = hcr_xyz.mean(0)
        seeds_xyz = {
            "hcr_gfp":     gfp_c,
            "gfp_dz+100":  gfp_c + [0, 0, 100],
            "gfp_dz-100":  gfp_c + [0, 0, -100],
            "gfp_dz+200":  gfp_c + [0, 0, 200],
            "gfp_q25":     np.quantile(hcr_xyz, 0.25, axis=0),
            "gfp_q75":     np.quantile(hcr_xyz, 0.75, axis=0),
        }
        res_xyz = run_multistart(cz_xyz, hcr_xyz, seeds_xyz)
        best_or_xyz = dict(n=0, seed=None, trim=None, med=float('nan'),
                           sxy=float('nan'), sz=float('nan'))
        for r in res_xyz:
            n50, med = oracle_lt50_xyz(r['fit'], cz_gt, hcr_gt)
            if n50 > best_or_xyz['n']:
                best_or_xyz = dict(n=n50, seed=r['seed'], trim=r['trim'],
                                   med=med, sxy=r['fit'].scales[0],
                                   sz=r['fit'].scales[2])
        print(f"  BASELINE XYZ: OR n<50={best_or_xyz['n']}"
              f" ({best_or_xyz['seed']}, {best_or_xyz['trim']})"
              f" sxy={best_or_xyz['sxy']:.2f} sz={best_or_xyz['sz']:.2f}"
              f" med={best_or_xyz['med']:.0f}")

        rows.append(dict(
            subject=subj, n_gt=len(cz_gt),
            cz_tilt=cz_surf.get('tilt_deg', float('nan')),
            hcr_tilt=hcr_surf.get('tilt_deg', float('nan')),
            cz_depth_extent=cz_anch[:, 2].max() - cz_anch[:, 2].min(),
            hcr_depth_extent=hcr_anch[:, 2].max() - hcr_anch[:, 2].min(),
            pia_or=best_or_anch['n'], pia_seed=best_or_anch['seed'],
            pia_trim=best_or_anch['trim'], pia_med=best_or_anch['med'],
            pia_sxy=best_or_anch['sxy'], pia_sz=best_or_anch['sz'],
            pia_ss=best_ss_anch['n'],
            xyz_or=best_or_xyz['n'], xyz_seed=best_or_xyz['seed'],
            xyz_trim=best_or_xyz['trim'], xyz_med=best_or_xyz['med'],
        ))

    df = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))
    df.to_csv("/root/capsule/code/full_automatic_execution_01/"
              "sessions/35_pia_anchored/pia_sweep.csv", index=False)


if __name__ == "__main__":
    main()
