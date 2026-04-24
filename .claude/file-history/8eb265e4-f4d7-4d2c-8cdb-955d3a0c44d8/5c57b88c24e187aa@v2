"""Session 06 benchmark harness — local distance-based scaling estimator.

Runs :func:`local_distance_scale.estimate_local_distance_scale` on the 6
benchmark subjects using R1-revised minimal ``(R, t)`` for localization,
and compares estimated ``(sxy, sz)`` against the landmark-derived
ground truth (same numbers as ``code/docs/01 Data Description.md``
anisotropic-expansion table).

Pass criterion (per plan):
  ``|sxy_est - sxy_gt| / sxy_gt <= 0.20``  AND
  ``|sz_est  - sz_gt | / sz_gt  <= 0.20``  on **all 6 subjects**.
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
from benchmark_data_loader import load_subject
from local_distance_scale import estimate_local_distance_scale
from r1_revised import coarse_align_revised

EVAL_ORDER = ["788406", "790322", "767018", "782149", "755252", "767022"]


def _per_subject(subject_id: str) -> dict:
    s = load_subject(subject_id)
    info = analyze_subject(s)

    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    if gfp_xyz.shape[0] < 10:
        return {"subject": subject_id,
                "status": f"too few GFP+ ({gfp_xyz.shape[0]})"}

    # R1 minimal (R, t)
    t0 = time.time()
    r1 = coarse_align_revised(cz_xyz, gfp_xyz, info["cz_surface"], info["hcr_surface"])
    t_r1 = time.time() - t0

    # R1 feasibility bound for xy (same expression R1 uses internally)
    L_cz_xy = float(r1.diagnostics.get("L_cz_xy_um", np.nan))
    L_hcr_xy = float(r1.diagnostics.get("L_hcr_xy_um", np.nan))
    sxy_upper = L_hcr_xy / L_cz_xy if (L_cz_xy > 0 and np.isfinite(L_hcr_xy)) else 3.0

    t0 = time.time()
    est = estimate_local_distance_scale(
        cz_xyz_um=cz_xyz,
        hcr_gfp_xyz_um=gfp_xyz,
        coarse_fit=r1,
        sxy_upper_feasibility=sxy_upper,
    )
    t_est = time.time() - t0

    # Ground truth
    gt = info["procrustes"]
    if gt is None:
        gt_scales = None
        sxy_gt = sz_gt = None
        rel_err_sxy = rel_err_sz = rel_err_siso = None
    else:
        gt_scales = gt.scales.tolist()
        sxy_gt = float(np.sqrt(gt.scales[0] * gt.scales[1]))
        sz_gt = float(gt.scales[2])
        rel_err_sxy = None if est.sxy is None else float((est.sxy - sxy_gt) / sxy_gt)
        rel_err_sz = None if est.sz is None else float((est.sz - sz_gt) / sz_gt)
        s_iso_gt = float((gt.scales[0] * gt.scales[1] * gt.scales[2]) ** (1.0 / 3.0))
        rel_err_siso = (None if est.s_iso is None
                        else float((est.s_iso - s_iso_gt) / s_iso_gt))

    return {
        "subject": subject_id,
        "n_cz": est.n_cz,
        "n_hcr_gfp_total": int(gfp_xyz.shape[0]),
        "n_hcr_local": est.n_hcr_local,
        "k": est.k,
        "sxy_est": est.sxy,
        "sz_est": est.sz,
        "s_iso_est": est.s_iso,
        "sxy_gt": sxy_gt,
        "sz_gt": sz_gt,
        "gt_scales": gt_scales,
        "rel_err_sxy": rel_err_sxy,
        "rel_err_sz": rel_err_sz,
        "rel_err_siso": rel_err_siso,
        "pass_sxy_20": (rel_err_sxy is not None
                        and abs(rel_err_sxy) <= 0.20),
        "pass_sz_20": (rel_err_sz is not None
                       and abs(rel_err_sz) <= 0.20),
        "sxy_upper_feasibility": sxy_upper,
        "cz_summary": est.cz_summary,
        "hcr_summary": est.hcr_summary,
        "estimator_diagnostics": est.diagnostics,
        "reason_unknown": est.reason_unknown,
        "elapsed_r1_s": float(t_r1),
        "elapsed_est_s": float(t_est),
    }


def run(subjects: list[str] | None = None) -> list[dict]:
    subjects = subjects or EVAL_ORDER
    results: list[dict] = []
    for sid in subjects:
        print(f"\n=== {sid} ===", flush=True)
        try:
            r = _per_subject(sid)
        except Exception as e:
            r = {"subject": sid, "status": f"error: {type(e).__name__}: {e}"}
            print(r, flush=True)
            results.append(r)
            continue
        for key in ("n_cz", "n_hcr_local", "sxy_est", "sz_est", "s_iso_est",
                    "sxy_gt", "sz_gt", "rel_err_sxy", "rel_err_sz",
                    "pass_sxy_20", "pass_sz_20", "reason_unknown"):
            print(f"  {key}: {r.get(key)}")
        results.append(r)
    return results


if __name__ == "__main__":
    subj = sys.argv[1:] or None
    out = run(subj)
    out_path = _THIS_DIR.parent / "sessions" / "06_local_distance_scale" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")

    # Aggregate pass/fail
    n = len(out)
    n_pass_xy = sum(1 for r in out if r.get("pass_sxy_20"))
    n_pass_z = sum(1 for r in out if r.get("pass_sz_20"))
    print(f"\nPass sxy (20%): {n_pass_xy}/{n}")
    print(f"Pass sz  (20%): {n_pass_z}/{n}")
    print(f"Both pass:       {sum(1 for r in out if r.get('pass_sxy_20') and r.get('pass_sz_20'))}/{n}")
