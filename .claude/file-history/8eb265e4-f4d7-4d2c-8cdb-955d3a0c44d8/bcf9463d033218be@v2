"""Session 07 — failure diagnosis for the session-06 local-distance estimator.

Session 06 rejected the local-distance kNN scale estimator (0/6 on sz,
3/6 on sxy). The dominant observed signal was a subject-specific excess
`f = N_hcr_local / N_cz ≈ 2.3–6.9` inside the R1 crop, plus a 1-µm
z-quantization floor that biased sz further down.

This script quantifies:

1. **Density disparity under the IDEAL crop.**  Use GT
   `ProcrustesFit` to place CZ centroids at their "coregistered"
   position in HCR.  Build a tight bounding box in that mapped volume,
   inflated by a symmetric 50-µm margin — the same margin session 06
   used.  Count HCR GFP+ inside.  f_ideal = N_gfp_in_ideal_crop /
   N_cz.  If f_ideal ≈ 1, the excess was driven by the R1 crop being
   too wide; if f_ideal ≫ 1, the excess is detection disparity within
   the true overlap.

2. **Depth distribution disparity.**  depth = z - surface(xy) for both
   clouds in HCR frame (HCR surface).  Compare depth-histograms of
   the mapped CZ centroids vs GFP+ inside the ideal crop.  Report
   p5/p50/p95 depth and the depth range.  If GFP+ extends deeper than
   the mapped CZ, the z-extent of the local HCR population is inflated
   by deep non-cortical labels.

3. **kNN under the ideal crop (control).**  Re-run axis-separated kNN
   with the ideal crop and the same z-jitter as session 06.  If this
   STILL fails on sz, the problem is structural (z quantization, not
   crop).  If it passes on sxy, the xy method is essentially correct
   and just needs a better crop strategy (or density-normalisation).

Outputs: `sessions/07_scale_failure_diagnosis/diagnosis.json` and a
rendered markdown block via the driver.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, depth_from_surface, fit_anisotropic_similarity
from benchmark_data_loader import load_subject, landmark_pairs_um

EVAL_ORDER = ["788406", "790322", "767018", "782149", "755252", "767022"]


def _fit_and_apply_gt(subject, cz_xyz: np.ndarray):
    """Refit anisotropic similarity from landmarks directly (avoids a
    latent bug in the stored ProcrustesFit.translation which uses
    column-vec convention inconsistent with the row-vec src @ R in the
    fit's own residual computation).

    Returns mapped CZ (x,y,z) in HCR frame, plus (R, scales, src_mean,
    dst_mean) for downstream use.
    """
    cz_lm, hcr_lm = landmark_pairs_um(subject, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    src_mean = cz_lm.mean(axis=0)
    dst_mean = hcr_lm.mean(axis=0)
    mapped = (np.asarray(cz_xyz, dtype=float) - src_mean) @ fit.R * fit.scales + dst_mean
    return mapped, fit, src_mean, dst_mean


def _axis_separated_knn(points: np.ndarray, k: int = 5) -> dict:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= k:
        return {}
    tree_xy = cKDTree(pts[:, :2])
    d_xy, _ = tree_xy.query(pts[:, :2], k=k + 1)
    d_xy = d_xy[:, 1:]
    tree_z = cKDTree(pts[:, 2:3])
    d_z, _ = tree_z.query(pts[:, 2:3], k=k + 1)
    d_z = d_z[:, 1:]
    return {
        "median_knn_xy2d": float(np.median(d_xy)),
        "median_knn_z1d": float(np.median(d_z)),
    }


def _per_subject(subject_id: str) -> dict:
    s = load_subject(subject_id)
    info = analyze_subject(s)

    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]

    gt = info["procrustes"]
    if gt is None:
        return {"subject": subject_id, "status": "no GT procrustes"}

    # 1. Map CZ → HCR under GT (refit from landmarks; see _fit_and_apply_gt).
    cz_in_hcr_gt, gt_fit, _src_mean, _dst_mean = _fit_and_apply_gt(s, cz_xyz)

    # 2. Ideal crop = tight AABB of mapped CZ + 50 µm margin.
    c_lo = cz_in_hcr_gt.min(axis=0) - 50.0
    c_hi = cz_in_hcr_gt.max(axis=0) + 50.0

    mask = np.all((gfp_xyz >= c_lo) & (gfp_xyz <= c_hi), axis=1)
    gfp_in_ideal = gfp_xyz[mask]

    N_cz = int(cz_xyz.shape[0])
    N_gfp_in_ideal = int(gfp_in_ideal.shape[0])
    f_ideal = N_gfp_in_ideal / max(N_cz, 1)

    # 3. Depth distribution (HCR surface) for both populations.
    depth_cz_mapped = depth_from_surface(cz_in_hcr_gt, hcr_surface)
    depth_gfp_local = depth_from_surface(gfp_in_ideal, hcr_surface)

    def _pct(arr, pcts):
        if arr.size == 0:
            return [None] * len(pcts)
        return [float(np.percentile(arr, p)) for p in pcts]

    pcts = [5, 50, 95]
    cz_p = _pct(depth_cz_mapped, pcts)
    gfp_p = _pct(depth_gfp_local, pcts)
    cz_range = (cz_p[2] - cz_p[0]) if cz_p[0] is not None else None
    gfp_range = (gfp_p[2] - gfp_p[0]) if gfp_p[0] is not None else None

    # 4. kNN under the ideal crop.
    # Use same z-jitter as session 06 to compare apples to apples.
    rng = np.random.default_rng(0)
    cz_j = cz_xyz.copy()
    cz_j[:, 2] += rng.uniform(-0.5, 0.5, size=cz_j.shape[0])
    gfp_j = gfp_in_ideal.copy()
    gfp_j[:, 2] += rng.uniform(-0.5, 0.5, size=gfp_j.shape[0])
    cz_sum = _axis_separated_knn(cz_j, k=5)
    gfp_sum = _axis_separated_knn(gfp_j, k=5)
    sxy_est_ideal = (gfp_sum["median_knn_xy2d"] / cz_sum["median_knn_xy2d"]
                     if gfp_sum and cz_sum else None)
    sz_est_ideal = (gfp_sum["median_knn_z1d"] / cz_sum["median_knn_z1d"]
                    if gfp_sum and cz_sum else None)

    sxy_gt = float(np.sqrt(gt_fit.scales[0] * gt_fit.scales[1]))
    sz_gt = float(gt_fit.scales[2])

    # 5. Volume of the ideal crop (for density), and CZ "theoretical
    # coregistered volume" — identical by construction (same bbox).
    V_crop = float(np.prod(c_hi - c_lo))
    rho_gfp_in_ideal = N_gfp_in_ideal / V_crop
    rho_cz_mapped = N_cz / V_crop

    return {
        "subject": subject_id,
        "N_cz": N_cz,
        "N_gfp_total": int(gfp_xyz.shape[0]),
        "N_gfp_in_ideal_crop": N_gfp_in_ideal,
        "f_ideal": float(f_ideal),
        "V_ideal_crop_um3": V_crop,
        "rho_cz_mapped_per_um3": float(rho_cz_mapped),
        "rho_gfp_local_per_um3": float(rho_gfp_in_ideal),
        "sxy_gt": sxy_gt,
        "sz_gt": sz_gt,
        "sxy_est_ideal_crop": sxy_est_ideal,
        "sz_est_ideal_crop": sz_est_ideal,
        "rel_err_sxy_ideal": (None if sxy_est_ideal is None
                              else (sxy_est_ideal - sxy_gt) / sxy_gt),
        "rel_err_sz_ideal": (None if sz_est_ideal is None
                             else (sz_est_ideal - sz_gt) / sz_gt),
        "depth_cz_mapped_p5_p50_p95": cz_p,
        "depth_gfp_local_p5_p50_p95": gfp_p,
        "depth_range_cz_mapped_um": cz_range,
        "depth_range_gfp_local_um": gfp_range,
        "depth_range_ratio": (gfp_range / cz_range
                              if cz_range and gfp_range else None),
        "cz_summary": cz_sum,
        "gfp_summary": gfp_sum,
    }


def run(subjects: list[str] | None = None) -> list[dict]:
    subjects = subjects or EVAL_ORDER
    results: list[dict] = []
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        t0 = time.time()
        try:
            r = _per_subject(sid)
        except Exception as e:
            r = {"subject": sid, "status": f"error: {type(e).__name__}: {e}"}
        print(f"  elapsed={time.time() - t0:.1f}s", flush=True)
        for key in ("N_cz", "N_gfp_in_ideal_crop", "f_ideal",
                    "sxy_gt", "sxy_est_ideal_crop", "rel_err_sxy_ideal",
                    "sz_gt", "sz_est_ideal_crop", "rel_err_sz_ideal",
                    "depth_range_cz_mapped_um", "depth_range_gfp_local_um",
                    "depth_range_ratio"):
            print(f"  {key}: {r.get(key)}")
        results.append(r)
    return results


if __name__ == "__main__":
    subj = sys.argv[1:] or None
    out = run(subj)
    out_path = _THIS_DIR.parent / "sessions" / "07_scale_failure_diagnosis" / "diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")
