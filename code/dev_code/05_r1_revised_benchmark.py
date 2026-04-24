"""R1-revised benchmark harness.

Runs :func:`r1_revised.coarse_align_revised` on the 6 benchmark subjects
and compares against the landmark-derived ground-truth affine.  Output is
written to ``sessions/05_R1_revised/r1_results.json``.

Metrics per grand plan §R1:
  * Minimal:   origin ≤ 100 µm, rotation ±5°.
  * Extended:  per-axis scale within ±20 % of GT (only when the axis's
               ``scale_known`` bit is set).

Per ``06 Dev Protocol.md`` this script validates; it does not design.
Algorithm parameters stay at the module defaults.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject
from benchmark_data_loader import landmark_pairs_um, load_subject
from r1_revised import apply_coarse_affine, coarse_align_revised

EVAL_ORDER = ["788406", "790322", "767018", "782149", "755252", "767022"]


def _rotation_err_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(abs(d))


def _per_subject(subject_id: str) -> dict:
    s = load_subject(subject_id)
    info = analyze_subject(s)

    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]
    if gfp_xyz.shape[0] < 10:
        return {"subject": subject_id,
                "status": f"too few GFP+ ({gfp_xyz.shape[0]})"}

    t0 = time.time()
    r1 = coarse_align_revised(cz_xyz, gfp_xyz, cz_surface, hcr_surface)
    elapsed = time.time() - t0

    # Ground truth from landmarks
    gt = info["procrustes"]
    origin_err_um = None
    rot_err_deg = None
    per_axis_scale_err = None
    gt_scales = None
    gt_angle = None
    gt_rms = None
    if gt is not None:
        cz_um_lm, hcr_um_lm = landmark_pairs_um(s, active_only=True)
        src_mean = cz_um_lm.mean(axis=0)
        dst_mean = hcr_um_lm.mean(axis=0)
        cz_center = cz_xyz.mean(axis=0, keepdims=True)
        gt_predict_center = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean
        r1_predict_center = apply_coarse_affine(cz_center, r1)
        origin_err_um = float(np.linalg.norm(r1_predict_center - gt_predict_center))
        rot_err_deg = _rotation_err_deg(r1.rotation_angle_z_deg, gt.rotation_angle_z_deg)
        per_axis_scale_err = (r1.scales - gt.scales).tolist()
        gt_scales = gt.scales.tolist()
        gt_angle = float(gt.rotation_angle_z_deg)
        gt_rms = float(gt.rms_um)

    diag = r1.diagnostics
    return {
        "subject": subject_id,
        "n_cz": int(cz_xyz.shape[0]),
        "n_hcr_gfp": int(gfp_xyz.shape[0]),
        "elapsed_s": float(elapsed),
        "cz_surface_method": cz_surface.get("method", "?"),
        "hcr_surface_method": hcr_surface.get("method", "?"),
        "r1_rotation_deg": float(r1.rotation_angle_z_deg),
        "r1_scales": r1.scales.tolist(),
        "r1_scale_known": r1.scale_known.tolist(),
        "r1_scale_confidence": r1.scale_confidence.tolist(),
        "r1_translation": r1.translation.tolist(),
        "r1_minimal_translation": r1.minimal_translation.tolist(),
        "r1_coverage_regime": r1.coverage_regime,
        "r1_cz_tilt_deg": diag["cz_tilt_deg"],
        "r1_hcr_tilt_deg": diag["hcr_tilt_deg"],
        "r1_sz_best": diag["sz_best"],
        "r1_tz_best_um": diag["tz_best_um"],
        "r1_sz_score_best": diag["sz_score_best"],
        "r1_sxy_best": diag["sxy_best"],
        "r1_sxy_score_best": diag["sxy_score_best"],
        "r1_aniso_done": diag["aniso_done"],
        "r1_tz_std_um": diag["tz_std_um"],
        "r1_L_cz_xy_um": diag["L_cz_xy_um"],
        "r1_L_hcr_xy_um": diag["L_hcr_xy_um"],
        "gt_rotation_deg": gt_angle,
        "gt_scales": gt_scales,
        "gt_landmark_rms_um": gt_rms,
        "origin_err_um": origin_err_um,
        "rotation_err_deg": rot_err_deg,
        "per_axis_scale_err": per_axis_scale_err,
    }


def run(subjects: list[str] | None = None) -> list[dict]:
    subjects = subjects or EVAL_ORDER
    results: list[dict] = []
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        try:
            r = _per_subject(sid)
        except Exception as e:  # noqa: BLE001
            r = {"subject": sid, "status": f"error: {type(e).__name__}: {e}"}
            print(r, flush=True)
            results.append(r)
            continue
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) > 6:
                print(f"  {k}: <len={len(v)}>")
            else:
                print(f"  {k}: {v}")
        results.append(r)
    return results


if __name__ == "__main__":
    subj = sys.argv[1:] or None
    out = run(subj)
    out_path = _THIS_DIR.parent / "sessions" / "05_R1_revised" / "r1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")
