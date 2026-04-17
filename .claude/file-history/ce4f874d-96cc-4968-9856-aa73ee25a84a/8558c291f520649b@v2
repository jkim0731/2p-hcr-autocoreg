"""R1 benchmark harness.

For each benchmark subject:
  1. Load subject data; compute CZ and HCR pia surfaces via
     ``analyze_subject`` (CZ=image_ceiling, HCR=quantile_ceiling).
  2. Run ``r1_coarse_align.coarse_align`` on the GFP+ centroids.
  3. Build the ground-truth anisotropic affine from the active
     landmark pairs via ``fit_anisotropic_similarity``.
  4. Compare: translation / origin error at the CZ-cloud centre,
     rotation-about-z error, per-axis scale, tz consistency.

Per ``06 Dev Protocol.md`` this script validates; it does not design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, fit_anisotropic_similarity
from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject
from r1_coarse_align import apply_coarse_affine, coarse_align

# Order listed in 05 Benchmark dataset.md priorities + stress cases.
EVAL_ORDER = ["788406", "790322", "767018", "782149", "755252", "767022"]


def _rotation_err_deg(a: float, b: float) -> float:
    """Shortest angular difference (deg) between two rotations about z."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(abs(d))


def _per_subject(subject_id: str) -> dict:
    s = load_subject(subject_id)
    info = analyze_subject(s)

    cz_xyz = info["cz_xyz"]  # CZ µm (x, y, z)
    gfp_xyz = info["gfp_xyz"]  # HCR µm (x, y, z), GFP+ only
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]

    if gfp_xyz.shape[0] < 10:
        return {
            "subject": subject_id,
            "status": f"too few GFP+ cells ({gfp_xyz.shape[0]}) — falling back",
        }

    r1 = coarse_align(cz_xyz, gfp_xyz, cz_surface, hcr_surface)

    # Ground truth from landmarks (x, y, z µm in each modality).
    gt = info["procrustes"]  # may be None
    gt_translation = None
    gt_scales = None
    gt_angle = None
    origin_err_um = None
    rot_err_deg = None
    per_axis_scale_err = None
    gt_rms_um = None
    if gt is not None:
        gt_scales = gt.scales.copy()
        gt_angle = gt.rotation_angle_z_deg
        gt_rms_um = gt.rms_um

        # Pick CZ-cloud centre as the reference probe point.
        cz_center = cz_xyz.mean(axis=0, keepdims=True)
        # Apply GT affine using its own convention:
        # hcr_pred = (cz - mean_src) @ R * scales + translation,
        # where translation = dst_mean - R @ (src_mean * scales).
        # Reconstruct the ProcrustesFit-style prediction.
        src_c = cz_center - cz_xyz.mean(axis=0, keepdims=True)  # = 0
        gt_pred = src_c @ gt.R * gt.scales + (
            gt.translation if gt.translation.shape == (3,) else np.zeros(3)
        )
        # ProcrustesFit.translation = dst_mean - R @ (src_mean * scales)
        # Correct application: predicted = (cz - src_mean) @ R * scales + dst_mean
        # We don't have src_mean/dst_mean directly from ProcrustesFit, so
        # reconstruct by applying on full landmark pairs:
        # Simpler: use the ProcrustesFit residuals to compute implied dst.
        # Fallback: re-derive src_mean / dst_mean via landmark_pairs_um.
        from benchmark_data_loader import landmark_pairs_um

        cz_um_lm, hcr_um_lm = landmark_pairs_um(s, active_only=True)
        src_mean = cz_um_lm.mean(axis=0)
        dst_mean = hcr_um_lm.mean(axis=0)
        gt_predict_center = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean

        r1_predict_center = apply_coarse_affine(cz_center, r1)
        origin_err_um = float(
            np.linalg.norm(r1_predict_center - gt_predict_center)
        )
        rot_err_deg = _rotation_err_deg(r1.rotation_angle_z_deg, gt_angle)
        per_axis_scale_err = (r1.scales - gt_scales).tolist()
        gt_translation = (dst_mean - src_mean @ gt.R * gt.scales).tolist()

    return {
        "subject": subject_id,
        "n_cz": int(cz_xyz.shape[0]),
        "n_hcr_gfp": int(gfp_xyz.shape[0]),
        "cz_surface_method": cz_surface.get("method", "?") if cz_surface else None,
        "hcr_surface_method": hcr_surface.get("method", "?") if hcr_surface else None,
        "r1_rotation_deg": float(r1.rotation_angle_z_deg),
        "r1_scales": r1.scales.tolist(),
        "r1_translation": r1.translation.tolist(),
        "r1_z_shift_um": float(r1.diagnostics["z_shift_um"]),
        "r1_xy_shift_um": r1.diagnostics["xy_shift_um"],
        "r1_tz_std_um": float(r1.diagnostics["tz_std_um"]),
        "r1_n_cz_in_band": int(r1.diagnostics["n_cz_in_band"]),
        "r1_n_hcr_in_band": int(r1.diagnostics["n_hcr_in_band"]),
        "r1_band_relaxed": bool(r1.diagnostics["band_relaxed"]),
        "gt_rotation_deg": gt_angle,
        "gt_scales": gt_scales.tolist() if gt_scales is not None else None,
        "gt_landmark_rms_um": gt_rms_um,
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
            print(f"  {k}: {v}")
        results.append(r)
    return results


if __name__ == "__main__":
    import json

    subj = sys.argv[1:] or None
    out = run(subj)
    out_path = _THIS_DIR.parent / "sessions" / "04_R1_coarse_align" / "r1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")
