"""Session 07b driver — depth-density gate + M1 k-NN + M3 density-ratio on strict GFP+.

Pipeline per subject:
  1. load_subject (v2.2 defaults, untouched).
  2. Compute strict cutoff via 07b_gfp_intersection_threshold.analyze_subject.
  3. Monkey-patch ``SubjectData.hcr_gfp_df`` to the strict id set
     (keep column ``hcr_id`` only; analyze_subject uses it to mask hcr_xyz).
  4. Run ``07_depth_density_diagnosis.analyze_subject_depth`` against the
     patched subject by monkey-patching its ``load_subject`` binding.
  5. Apply Gate A (per-bin CV ≤ 0.20) and Gate B (integrated ratio ∈ [0.8, 1.25]).
  6. Run M1 (estimate_local_distance_scale) and M3 (global xy/z density ratio).
  7. Score vs landmark-Procrustes GT at ±5 %.
  8. Dump results.json + depth_density_clean_summary.json; update plots.

Note: ``fit_anisotropic_similarity`` and ``landmark_pairs_um`` appear only
inside ``_gt_scales`` — scoring block, no leakage into estimators.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))


def _load_module_numeric_prefix(basename: str, mod_alias: str):
    spec = importlib.util.spec_from_file_location(mod_alias, _DEV / basename)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module so @dataclass's sys.modules lookup works
    sys.modules[mod_alias] = module
    spec.loader.exec_module(module)
    return module


gfp_thr = _load_module_numeric_prefix(
    "07b_gfp_intersection_threshold.py", "gfp_intersection_threshold_07b"
)
depth_diag = _load_module_numeric_prefix(
    "07_depth_density_diagnosis.py", "depth_density_diagnosis_07"
)

from benchmark_analysis import (
    analyze_subject,
    fit_anisotropic_similarity,
)
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS,
    cz_px_to_um,
    hcr_px_to_um,
    landmark_pairs_um,
    load_subject,
)
from local_distance_scale import estimate_local_distance_scale
from r1_revised import coarse_align_revised

SESSION_DIR = Path("/root/capsule/code/sessions/07b_scale_clean_gfp")
FIG_DIR = SESSION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REL_ERR_BAR = 0.05  # 5 % acceptance
DEPTH_BIN_UM = 25.0


# ----------------------------------------------------------------------
# GT (scoring only)
# ----------------------------------------------------------------------
def _gt_scales(subject) -> tuple[float, float]:
    cz_lm, hcr_lm = landmark_pairs_um(subject, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    scales = np.asarray(fit.scales, dtype=float)
    sxy_gt = float(np.sqrt(scales[0] * scales[1]))
    sz_gt = float(scales[2])
    return sxy_gt, sz_gt


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def _xyz_um_from_hcr_centroids(s, hcr_ids: set[int]) -> np.ndarray:
    hcr_px = s.hcr_centroids.copy()
    hcr_px["_keep"] = hcr_px["hcr_id"].astype(int).isin(hcr_ids)
    arr = hcr_px[hcr_px["_keep"]][["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    um = hcr_px_to_um(arr, s)
    return um[:, [2, 1, 0]]  # (x, y, z)


def _cz_xyz_um(s) -> np.ndarray:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    um = cz_px_to_um(arr, s)
    return um[:, [2, 1, 0]]


def _r1_apply(cz_xyz, coarse_fit):
    from r1_revised import apply_coarse_affine
    return apply_coarse_affine(np.asarray(cz_xyz, dtype=float), coarse_fit)


def _m3_scale(cz_xyz_um, gfp_xyz_um, coarse_fit) -> tuple[float, float]:
    """Global xy/z density ratio — each cloud uses its own native span.

    Derivation (under ideal anisotropic stretch ``(sxy, sxy, sz)``
    applied to CZ, assuming the same cell population is sampled by
    both so ``N_cz ≈ N_hcr``):

    * Native CZ volume ``V_cz`` (in CZ-µm³). In the HCR µm frame after
      R1 with identity scales, the mapped CZ still occupies ``V_cz``.
    * HCR true tissue volume ``V_hcr = sxy²·sz·V_cz`` (scaled).
    * Populations sampled at the same underlying density ``ρ₀``:
      ``N_cz = ρ₀·V_cz``,  ``N_hcr = ρ₀·V_hcr = ρ₀·sxy²·sz·V_cz``.
    * ``N_hcr / N_cz = sxy²·sz``.
    * Along z: ``N_hcr / Lz_hcr = ρ₀·(sxy²·Lxy_cz²)`` is xy-area; the
      robust per-axis identity is extent ratios of the full AABBs:
        sz = Lz_hcr / Lz_cz_mapped
        sxy²·sz = (V_hcr / V_cz)  where V is the full-cloud AABB volume.
    * We use the ``p5–p95`` span per axis (outlier-robust) rather than
      min–max. Volume is the product.

    With N matched (idealised), this gives:
      sxy = sqrt(V_hcr / V_cz / sz)
      sz  = Lz_hcr / Lz_cz
    But our two populations may have *different* counts (detection
    bias, coverage). So we keep the general identity:
      sxy²·sz = (N_hcr/N_cz) × (V_cz / V_hcr)⁻¹? — no. Correct form:
      (N_hcr/V_hcr) / (N_cz/V_cz) = 1  under matched sampling.
      If N_cz ≠ N_hcr, the ratio ``(N_hcr/V_hcr) / (N_cz/V_cz)`` is
      the *detection-coverage ratio*, not scale. We therefore cannot
      separate scale from coverage here. The simplest scale-only
      estimate is the **span ratio**:
        sz_m3  = Lz_hcr / Lz_cz_mapped
        sxy_m3 = sqrt( (Lx_hcr · Ly_hcr) / (Lx_cz_mapped · Ly_cz_mapped) )
    These reduce to the volume-ratio form when sampling matches.
    """
    mapped_cz = _r1_apply(cz_xyz_um, coarse_fit)
    if len(mapped_cz) < 20 or len(gfp_xyz_um) < 20:
        return float("nan"), float("nan")

    def _span(arr, axis):
        return float(np.percentile(arr[:, axis], 95) - np.percentile(arr[:, axis], 5))

    Lx_cz, Ly_cz, Lz_cz = _span(mapped_cz, 0), _span(mapped_cz, 1), _span(mapped_cz, 2)
    Lx_h, Ly_h, Lz_h = _span(gfp_xyz_um, 0), _span(gfp_xyz_um, 1), _span(gfp_xyz_um, 2)
    if min(Lx_cz, Ly_cz, Lz_cz, Lx_h, Ly_h, Lz_h) <= 0:
        return float("nan"), float("nan")

    sz_est = Lz_h / Lz_cz
    sxy_sq = (Lx_h * Ly_h) / (Lx_cz * Ly_cz)
    sxy_est = float(np.sqrt(sxy_sq)) if sxy_sq > 0 else float("nan")
    return sxy_est, float(sz_est)


# ----------------------------------------------------------------------
# Monkey-patched load_subject for depth_diagnosis
# ----------------------------------------------------------------------
def _patched_loader(patched_subjects: dict):
    """Return a function that returns our patched SubjectData for any sid in patched_subjects."""
    original = depth_diag.load_subject

    def _f(sid: str):
        if sid in patched_subjects:
            return patched_subjects[sid]
        return original(sid)

    return _f


# ----------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------
def _evaluate_depth_gate(dd_record: dict) -> dict:
    ratios = np.asarray(dd_record["gfp_over_truth"], dtype=float)
    rho_truth = np.asarray(dd_record["rho_truth_per_um3"], dtype=float)
    mask = np.isfinite(ratios) & (rho_truth > 0)
    if mask.any():
        thr = float(np.percentile(rho_truth[mask], 25))
        m2 = mask & (rho_truth > thr)
        r = ratios[m2]
        cv = float(np.nanstd(r) / np.nanmean(r)) if r.size else float("nan")
        integrated = float(np.nanmean(ratios))
    else:
        cv = float("nan")
        integrated = float("nan")
    gate_a = (cv <= 0.20)
    gate_b = (0.8 <= integrated <= 1.25)
    return {
        "per_bin_cv": cv,
        "integrated_ratio": integrated,
        "gate_a_uniformity": bool(gate_a),
        "gate_b_offset": bool(gate_b),
        "passed": bool(gate_a and gate_b),
    }


# ----------------------------------------------------------------------
# Synthetic sanity
# ----------------------------------------------------------------------
def _synthetic_check() -> dict:
    """Stretch a random CZ cloud by (1.77, 1.77, 2.82) and verify M1/M3 recover it."""
    rng = np.random.default_rng(0)
    cz = rng.uniform(-300, 300, size=(4000, 3))
    cz[:, 2] = rng.uniform(-50, 300, size=4000)
    scales = np.array([1.77, 1.77, 2.82])
    hcr = cz * scales  # identity rotation, no translation, no sub-sampling

    # CoarseAffineV2 attributes used by apply_coarse_affine:
    #   R, scales, translation, src_mean. Identity fit for the synthetic check.
    from types import SimpleNamespace
    r1_fake = SimpleNamespace(
        R=np.eye(3),
        scales=np.array([1.0, 1.0, 1.0]),
        translation=np.zeros(3),
        src_mean=np.zeros(3),
    )
    try:
        result = estimate_local_distance_scale(
            cz, hcr, r1_fake, sxy_upper_feasibility=3.0
        )
        m1 = (float(result.sxy), float(result.sz))
    except Exception as e:
        m1 = (float("nan"), float("nan"))
    # M3 synthetic: use the same helper; fake coarse_fit
    m3 = _m3_scale(cz, hcr, r1_fake)
    return {
        "sxy_expected": 1.77,
        "sz_expected": 2.82,
        "m1": {"sxy": m1[0], "sz": m1[1]},
        "m3": {"sxy": m3[0], "sz": m3[1]},
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # (1) GMM-intersection per subject (already persisted)
    gmm_results = []
    patched_subjects: dict = {}
    strict_gfp_map: dict = {}
    for sid in BENCHMARK_SUBJECTS:
        gi = gfp_thr.analyze_subject(sid)
        gmm_results.append(
            {
                "subject": sid,
                "cutoff_linear": gi.cutoff_linear,
                "cutoff_v22_linear": gi.cutoff_v22_linear,
                "n_strict": gi.n_strict,
                "n_v22": gi.n_v22_reference,
                "sanity_passed": gi.sanity_passed,
                "sanity_notes": gi.sanity_notes,
                "coreg_coverage_strict": gi.coreg_coverage_strict,
            }
        )
        # Build strict GFP+ DataFrame (just hcr_id column for analyze_subject's mask)
        if gi.sanity_passed:
            strict_df = gfp_thr.strict_gfp_df(sid, gi.cutoff_linear)
            strict_df = strict_df[["hcr_id"]].copy()
            # Patch subject
            s = load_subject(sid)
            s.hcr_gfp_df = strict_df
            patched_subjects[sid] = s
            strict_gfp_map[sid] = strict_df

    # (2) depth-density gate, on subjects that pass sanity
    # Monkey-patch depth_diag.load_subject to return our patched objects
    original_loader = depth_diag.load_subject
    depth_diag.load_subject = _patched_loader(patched_subjects)
    try:
        dd_records = []
        gate_records = {}
        for sid in BENCHMARK_SUBJECTS:
            if sid not in patched_subjects:
                dd_records.append({"subject": sid, "status": "skipped: sanity fail"})
                continue
            r = depth_diag.analyze_subject_depth(sid, depth_bin_um=DEPTH_BIN_UM)
            dd_records.append(r)
            gate_records[sid] = _evaluate_depth_gate(r)
            print(
                f"  {sid}  depth-gate  CV={gate_records[sid]['per_bin_cv']:.3f}  "
                f"int_ratio={gate_records[sid]['integrated_ratio']:.3f}  "
                f"A={gate_records[sid]['gate_a_uniformity']} B={gate_records[sid]['gate_b_offset']}"
            )
        with open(SESSION_DIR / "depth_density_clean_summary.json", "w") as f:
            json.dump(
                dd_records, f, indent=2,
                default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v),
            )
    finally:
        depth_diag.load_subject = original_loader

    # (3) Synthetic sanity
    synth = _synthetic_check()
    print(
        f"  synth  M1=({synth['m1']['sxy']:.3f}, {synth['m1']['sz']:.3f}) "
        f"M3=({synth['m3']['sxy']:.3f}, {synth['m3']['sz']:.3f})  "
        f"expected=(1.77, 2.82)"
    )

    # (4) Scale estimators
    scale_records = []
    for sid in BENCHMARK_SUBJECTS:
        row = {"subject": sid}
        if sid not in patched_subjects:
            row["status"] = "skipped_sanity"
            scale_records.append(row)
            continue
        s = patched_subjects[sid]
        # Inputs via analyze_subject — same pattern as sessions 05/06/07
        info = analyze_subject(s)
        cz_xyz = info["cz_xyz"]
        gfp_xyz = info["gfp_xyz"]  # strict GFP+ after monkey-patch of s.hcr_gfp_df
        # R1 minimal (R, t)
        r1_fit = coarse_align_revised(
            cz_xyz, gfp_xyz, info["cz_surface"], info["hcr_surface"]
        )
        # Feasibility upper bound on sxy: L_hcr_xy / L_cz_xy (extent ratio).
        hcr_all_px = s.hcr_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
        hcr_all_um = hcr_px_to_um(hcr_all_px, s)[:, [2, 1, 0]]
        cz_span_xy = 0.5 * (
            float(cz_xyz[:, 0].max() - cz_xyz[:, 0].min())
            + float(cz_xyz[:, 1].max() - cz_xyz[:, 1].min())
        )
        hcr_span_xy = 0.5 * (
            float(hcr_all_um[:, 0].max() - hcr_all_um[:, 0].min())
            + float(hcr_all_um[:, 1].max() - hcr_all_um[:, 1].min())
        )
        sxy_upper = max(hcr_span_xy / cz_span_xy, 2.0) if cz_span_xy > 0 else 3.0
        # M1
        try:
            m1 = estimate_local_distance_scale(
                cz_xyz, gfp_xyz, r1_fit, sxy_upper_feasibility=sxy_upper
            )
            row["sxy_m1"] = float(m1.sxy) if m1.sxy is not None else float("nan")
            row["sz_m1"] = float(m1.sz) if m1.sz is not None else float("nan")
            row["m1_converged"] = bool(m1.converged)
            row["m1_n_hcr_local"] = int(m1.n_hcr_local)
            row["m1_iterations"] = int(m1.iterations)
        except Exception as e:
            row["m1_error"] = f"{type(e).__name__}: {e}"
            row["sxy_m1"] = float("nan")
            row["sz_m1"] = float("nan")

        # M3
        sxy_m3, sz_m3 = _m3_scale(cz_xyz, gfp_xyz, r1_fit)
        row["sxy_m3"] = float(sxy_m3)
        row["sz_m3"] = float(sz_m3)

        # GT + rel errors
        sxy_gt, sz_gt = _gt_scales(s)
        row["sxy_gt"] = sxy_gt
        row["sz_gt"] = sz_gt
        for method in ("m1", "m3"):
            sxy = row.get(f"sxy_{method}", float("nan"))
            sz = row.get(f"sz_{method}", float("nan"))
            row[f"rel_err_sxy_{method}"] = (sxy - sxy_gt) / sxy_gt if np.isfinite(sxy) else float("nan")
            row[f"rel_err_sz_{method}"] = (sz - sz_gt) / sz_gt if np.isfinite(sz) else float("nan")
            row[f"pass5_{method}"] = bool(
                np.isfinite(sxy) and np.isfinite(sz)
                and abs(row[f"rel_err_sxy_{method}"]) <= REL_ERR_BAR
                and abs(row[f"rel_err_sz_{method}"]) <= REL_ERR_BAR
            )
        row["pass5_any"] = bool(row.get("pass5_m1") or row.get("pass5_m3"))
        scale_records.append(row)
        print(
            f"  {sid}  M1=({row.get('sxy_m1', float('nan')):.3f},{row.get('sz_m1', float('nan')):.3f})  "
            f"M3=({row.get('sxy_m3', float('nan')):.3f},{row.get('sz_m3', float('nan')):.3f})  "
            f"GT=({sxy_gt:.3f},{sz_gt:.3f})  "
            f"pass5={row['pass5_any']}"
        )

    # (5) Aggregate
    summary = {
        "rel_err_bar": REL_ERR_BAR,
        "n_subjects": len(BENCHMARK_SUBJECTS),
        "n_sanity_passed": sum(1 for r in gmm_results if r["sanity_passed"]),
        "n_depth_gate_passed": sum(1 for v in gate_records.values() if v["passed"]),
        "n_scale_pass5": sum(1 for r in scale_records if r.get("pass5_any")),
        "pass_6of6": all(r.get("pass5_any") for r in scale_records),
        "synthetic": synth,
    }
    out = {
        "summary": summary,
        "gmm_intersection": gmm_results,
        "depth_density_gate": gate_records,
        "scales": scale_records,
    }
    with open(SESSION_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda v: v.tolist()
                  if hasattr(v, "tolist") else str(v))
    print(f"\nSummary: sanity={summary['n_sanity_passed']}/6  "
          f"depth-gate={summary['n_depth_gate_passed']}/6  "
          f"scale-pass5={summary['n_scale_pass5']}/6  pass_6of6={summary['pass_6of6']}")
    print(f"Wrote {SESSION_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
