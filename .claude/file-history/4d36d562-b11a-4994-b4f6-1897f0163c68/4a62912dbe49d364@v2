"""Session 07c driver — BIC-sweep GFP+ + CZ-density uniformity gate.

Differences from 07b:
  1. GFP+ threshold uses BIC-selected K∈[2,6] (module updated in-place).
  2. Validation gate compares HCR GFP+ density profile to **all CZ
     centroids mapped via R1** (no coreg / matched-HCR subset). Gate
     passes when CV of per-depth-bin ``N_gfp+_bin / N_cz_bin`` is
     ≤ 0.20 on informative bins. No absolute-ratio constraint (the
     integrated ratio encodes the unknown scale factor).
  3. Otherwise same pipeline: M1 (k-NN) + M3 (span ratio) vs
     landmark-Procrustes GT at ±5 % both axes.

Scale estimators are unchanged from 07b. If the new threshold makes
GFP+ vs CZ density uniform across depth, M1 should work because the
bias term cancels.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DEV = Path(__file__).resolve().parent
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))


def _load_module_numeric_prefix(basename: str, mod_alias: str):
    spec = importlib.util.spec_from_file_location(mod_alias, _DEV / basename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_alias] = module
    spec.loader.exec_module(module)
    return module


gfp_thr = _load_module_numeric_prefix(
    "07b_gfp_intersection_threshold.py", "gfp_intersection_threshold_07b"
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
from r1_revised import apply_coarse_affine, coarse_align_revised

SESSION_DIR = Path("/root/capsule/code/sessions/07c_gfp_bic_cz_density")
FIG_DIR = SESSION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REL_ERR_BAR = 0.05
DEPTH_BIN_UM = 25.0
CV_GATE = 0.20


# ----------------------------------------------------------------------
def _gt_scales(subject) -> tuple[float, float]:
    cz_lm, hcr_lm = landmark_pairs_um(subject, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    scales = np.asarray(fit.scales, dtype=float)
    sxy_gt = float(np.sqrt(scales[0] * scales[1]))
    sz_gt = float(scales[2])
    return sxy_gt, sz_gt


def _cz_xyz_um(s) -> np.ndarray:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    um = cz_px_to_um(arr, s)
    return um[:, [2, 1, 0]]


def _xyz_um_from_hcr_centroids(s, hcr_ids: set[int]) -> np.ndarray:
    hcr_px = s.hcr_centroids.copy()
    hcr_px["_keep"] = hcr_px["hcr_id"].astype(int).isin(hcr_ids)
    arr = hcr_px[hcr_px["_keep"]][["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    if len(arr) == 0:
        return np.zeros((0, 3))
    um = hcr_px_to_um(arr, s)
    return um[:, [2, 1, 0]]


# ----------------------------------------------------------------------
# CZ-density validation gate
# ----------------------------------------------------------------------
def _pia_depth(xyz_um: np.ndarray, pia: dict) -> np.ndarray:
    """Depth below the HCR pia plane (µm) — positive = deeper."""
    a = float(pia["a"]); b = float(pia["b"]); c = float(pia["c"])
    return xyz_um[:, 2] - (a * xyz_um[:, 0] + b * xyz_um[:, 1] + c)


def _density_profile(xyz_um: np.ndarray, depths: np.ndarray,
                     bins: np.ndarray, xy_area_um2: float) -> np.ndarray:
    h, _ = np.histogram(depths, bins=bins)
    bin_widths = np.diff(bins)
    vols = xy_area_um2 * bin_widths
    return h.astype(float) / np.maximum(vols, 1.0)


def cz_density_gate(sid: str, s, hcr_gfp_xyz_um: np.ndarray, r1_fit,
                    info: dict, depth_bin_um: float = DEPTH_BIN_UM) -> dict:
    """Compare HCR GFP+ depth-density profile against mapped-all-CZ profile.

    Gate = CV ≤ 0.20 of per-bin ratio ``rho_gfp+ / rho_cz_mapped`` over
    bins where ``rho_cz > p25(rho_cz)`` (informative bins).

    Writes a depth-density figure per subject. Returns a dict with:
    ``cv_ratio``, ``integrated_ratio``, ``gate_pass``, per-bin arrays.
    """
    cz_all_um = _cz_xyz_um(s)
    cz_mapped = apply_coarse_affine(cz_all_um, r1_fit)

    # Overlap AABB in HCR-µm frame
    if len(hcr_gfp_xyz_um) < 20 or len(cz_mapped) < 20:
        return {"subject": sid, "status": "too_few_cells"}
    lo = np.maximum(cz_mapped.min(axis=0), hcr_gfp_xyz_um.min(axis=0))
    hi = np.minimum(cz_mapped.max(axis=0), hcr_gfp_xyz_um.max(axis=0))
    if np.any(hi <= lo):
        return {"subject": sid, "status": "empty_overlap"}

    def _mask(p):
        return np.all((p >= lo) & (p <= hi), axis=1)

    cz_in = cz_mapped[_mask(cz_mapped)]
    gfp_in = hcr_gfp_xyz_um[_mask(hcr_gfp_xyz_um)]
    if len(cz_in) < 20 or len(gfp_in) < 20:
        return {"subject": sid, "status": "overlap_too_sparse"}

    xy_area_um2 = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    pia = info["hcr_surface"]
    d_cz = _pia_depth(cz_in, pia)
    d_gfp = _pia_depth(gfp_in, pia)

    dmin = float(np.floor(min(d_cz.min(), d_gfp.min()) / depth_bin_um) * depth_bin_um)
    dmax = float(np.ceil(max(d_cz.max(), d_gfp.max()) / depth_bin_um) * depth_bin_um)
    bins = np.arange(dmin, dmax + depth_bin_um, depth_bin_um)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rho_cz = _density_profile(cz_in, d_cz, bins, xy_area_um2)
    rho_gfp = _density_profile(gfp_in, d_gfp, bins, xy_area_um2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rho_cz > 0, rho_gfp / rho_cz, np.nan)

    # Informative bins: CZ density above its own 25th percentile
    finite_cz = rho_cz[np.isfinite(rho_cz)]
    if finite_cz.size == 0:
        return {"subject": sid, "status": "no_finite_cz_density"}
    thr = float(np.percentile(finite_cz[finite_cz > 0], 25)) if np.any(finite_cz > 0) else 0.0
    good = (rho_cz > thr) & np.isfinite(ratio)
    if not np.any(good):
        cv = float("nan")
    else:
        r = ratio[good]
        cv = float(np.nanstd(r) / np.nanmean(r)) if np.nanmean(r) != 0 else float("nan")
    integrated = float(np.nanmean(ratio[good])) if np.any(good) else float("nan")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    scale = 1e6
    ax = axes[0]
    ax.plot(centers, rho_cz * scale, "-", lw=2.0, color="black",
            label="CZ mapped (all cells, R1 identity-scale)")
    ax.plot(centers, rho_gfp * scale, "-", lw=1.8, color="#2aa198",
            label="HCR GFP+ strict (BIC-K)")
    ax.set_xlabel("depth from HCR pia (µm)")
    ax.set_ylabel("density (cells / 10^6 µm³)")
    ax.set_title(
        f"{sid}  density vs depth  (overlap xy = {xy_area_um2/1e6:.2f} mm²)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(centers, ratio, "-", lw=1.8, color="#268bd2",
            label="GFP+ / CZ (per bin)")
    if np.any(good):
        ax.axhline(np.nanmean(ratio[good]), color="#cc3333", lw=1.2, ls="--",
                   label=f"mean = {integrated:.3f}")
    ax.set_xlabel("depth from HCR pia (µm)")
    ax.set_ylabel("GFP+ / CZ ratio")
    ax.set_title(f"{sid}  CV over informative bins = {cv:.3f}  (gate bar {CV_GATE})")
    ax.axhspan(integrated * 0.8, integrated * 1.2, color="black", alpha=0.08)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"depth_density_cz_{sid}.png", dpi=140)
    plt.close(fig)

    return {
        "subject": sid,
        "status": "ok",
        "xy_area_um2": xy_area_um2,
        "depth_bin_um": depth_bin_um,
        "depth_centers_um": centers.tolist(),
        "rho_cz_per_um3": rho_cz.tolist(),
        "rho_gfp_per_um3": rho_gfp.tolist(),
        "gfp_over_cz": ratio.tolist(),
        "cv_ratio": cv,
        "integrated_ratio": integrated,
        "gate_pass": bool(np.isfinite(cv) and cv <= CV_GATE),
        "n_cz_in_box": int(len(cz_in)),
        "n_gfp_in_box": int(len(gfp_in)),
    }


# ----------------------------------------------------------------------
# M3 (same as 07b — span ratio per cloud)
# ----------------------------------------------------------------------
def _m3_scale(cz_xyz_um, gfp_xyz_um, coarse_fit) -> tuple[float, float]:
    mapped_cz = apply_coarse_affine(np.asarray(cz_xyz_um, dtype=float), coarse_fit)
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
# Main
# ----------------------------------------------------------------------
def main():
    # (1) BIC-sweep threshold per subject
    gmm_results = []
    patched_subjects: dict = {}
    strict_ids_map: dict = {}
    for sid in BENCHMARK_SUBJECTS:
        gi = gfp_thr.analyze_subject(sid)
        gmm_results.append({
            "subject": sid,
            "n_components_best": gi.n_components,
            "cutoff_linear": gi.cutoff_linear,
            "cutoff_v22_linear": gi.cutoff_v22_linear,
            "n_strict": gi.n_strict,
            "n_v22": gi.n_v22_reference,
            "coreg_coverage_strict": gi.coreg_coverage_strict,
            "sanity_passed": gi.sanity_passed,
            "sanity_notes": gi.sanity_notes,
            "bic": gi.bic,
            "bic_sweep": gi.bic_sweep,
        })
        # For the CZ-density gate, we want to include ALL subjects (the
        # coreg-coverage sanity was a heuristic; the CZ-density check is
        # independent). Skip only if n_strict is tiny.
        if gi.n_strict >= 300:
            strict_df = gfp_thr.strict_gfp_df(sid, gi.cutoff_linear)
            strict_df = strict_df[["hcr_id"]].copy()
            s = load_subject(sid)
            s.hcr_gfp_df = strict_df
            patched_subjects[sid] = s
            strict_ids_map[sid] = set(int(x) for x in strict_df["hcr_id"].values)

    # (2) CZ-density gate + scale estimators
    gate_records = {}
    scale_records = []
    for sid in BENCHMARK_SUBJECTS:
        row = {"subject": sid}
        if sid not in patched_subjects:
            row["status"] = "skipped_too_few"
            scale_records.append(row)
            continue
        s = patched_subjects[sid]
        info = analyze_subject(s)
        cz_xyz = info["cz_xyz"]
        gfp_xyz = info["gfp_xyz"]

        # R1 minimal
        r1_fit = coarse_align_revised(
            cz_xyz, gfp_xyz, info["cz_surface"], info["hcr_surface"]
        )

        # CZ-density gate
        gate = cz_density_gate(sid, s, gfp_xyz, r1_fit, info)
        gate_records[sid] = gate
        print(
            f"  {sid}  K*={gmm_results[[g['subject'] for g in gmm_results].index(sid)]['n_components_best']}  "
            f"cv_gfp/cz={gate.get('cv_ratio', float('nan')):.3f}  "
            f"mean_gfp/cz={gate.get('integrated_ratio', float('nan')):.3f}  "
            f"gate={gate.get('gate_pass', False)}"
        )

        # Feasibility upper bound on sxy for M1
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
        for m in ("m1", "m3"):
            sxy = row.get(f"sxy_{m}", float("nan"))
            sz = row.get(f"sz_{m}", float("nan"))
            row[f"rel_err_sxy_{m}"] = (sxy - sxy_gt) / sxy_gt if np.isfinite(sxy) else float("nan")
            row[f"rel_err_sz_{m}"] = (sz - sz_gt) / sz_gt if np.isfinite(sz) else float("nan")
            row[f"pass5_{m}"] = bool(
                np.isfinite(sxy) and np.isfinite(sz)
                and abs(row[f"rel_err_sxy_{m}"]) <= REL_ERR_BAR
                and abs(row[f"rel_err_sz_{m}"]) <= REL_ERR_BAR
            )
        row["pass5_any"] = bool(row.get("pass5_m1") or row.get("pass5_m3"))
        scale_records.append(row)
        print(
            f"    M1=({row.get('sxy_m1', float('nan')):.3f},{row.get('sz_m1', float('nan')):.3f})  "
            f"M3=({row.get('sxy_m3', float('nan')):.3f},{row.get('sz_m3', float('nan')):.3f})  "
            f"GT=({sxy_gt:.3f},{sz_gt:.3f})  pass5={row['pass5_any']}"
        )

    # (3) Summary
    summary = {
        "rel_err_bar": REL_ERR_BAR,
        "cv_gate_bar": CV_GATE,
        "n_subjects": len(BENCHMARK_SUBJECTS),
        "n_sanity_passed": sum(1 for r in gmm_results if r["sanity_passed"]),
        "n_cz_gate_passed": sum(1 for v in gate_records.values()
                                 if v.get("gate_pass")),
        "n_scale_pass5": sum(1 for r in scale_records if r.get("pass5_any")),
        "pass_6of6": all(r.get("pass5_any") for r in scale_records),
    }
    out = {
        "summary": summary,
        "gmm_intersection": gmm_results,
        "cz_density_gate": gate_records,
        "scales": scale_records,
    }
    with open(SESSION_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    with open(SESSION_DIR / "cz_density_summary.json", "w") as f:
        json.dump(gate_records, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(
        f"\nSummary: sanity={summary['n_sanity_passed']}/6  "
        f"cz-gate={summary['n_cz_gate_passed']}/6  "
        f"scale-pass5={summary['n_scale_pass5']}/6  "
        f"pass_6of6={summary['pass_6of6']}"
    )
    print(f"Wrote {SESSION_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
