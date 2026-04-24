"""Re-run the 07c CZ-density diagnostic using GT-anisotropic Procrustes
to map CZ → HCR (instead of R1-identity). This puts CZ cells in the
TRUE coregistered volume, which is what the subgoal asks for.

Subgoal (user, verbatim from this conversation):
    "check if the density of HCR GFP+ and czstack ROIs only within the
     known overlapping volume after cz to hcr registration in the
     ground-truth dataset"

Diagnostic only — no feedback into M1/M3 scale estimators, no GT leak
into the threshold-selection pipeline (both done upstream).

Writes:
  sessions/07c_gfp_bic_cz_density/cz_density_gt_summary.json
  sessions/07c_gfp_bic_cz_density/figures/depth_density_gt_<sid>.png
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

from benchmark_analysis import analyze_subject, fit_anisotropic_similarity
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS,
    cz_px_to_um,
    landmark_pairs_um,
    load_subject,
)

SESSION_DIR = Path("/root/capsule/code/sessions/07c_gfp_bic_cz_density")
FIG_DIR = SESSION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEPTH_BIN_UM = 25.0
CV_GATE = 0.20


def _cz_xyz_um(s) -> np.ndarray:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    um = cz_px_to_um(arr, s)
    return um[:, [2, 1, 0]]


def _gt_affine(s):
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    if len(cz_lm) < 4:
        return None
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    return {
        "R": np.asarray(fit.R, dtype=float),
        "scales": np.asarray(fit.scales, dtype=float),
        "src_mean": cz_lm.mean(axis=0),
        "dst_mean": hcr_lm.mean(axis=0),
    }


def _apply_gt(xyz, gt) -> np.ndarray:
    return (xyz - gt["src_mean"]) @ gt["R"] * gt["scales"] + gt["dst_mean"]


def _pia_depth(xyz_um: np.ndarray, pia: dict) -> np.ndarray:
    a = float(pia["a"]); b = float(pia["b"]); c = float(pia["c"])
    return xyz_um[:, 2] - (a * xyz_um[:, 0] + b * xyz_um[:, 1] + c)


def _density_profile(xyz_um: np.ndarray, depths: np.ndarray,
                     bins: np.ndarray, xy_area_um2: float) -> np.ndarray:
    h, _ = np.histogram(depths, bins=bins)
    vols = xy_area_um2 * np.diff(bins)
    return h.astype(float) / np.maximum(vols, 1.0)


def gt_density_gate(sid: str, strict_hcr_ids: set[int]) -> dict:
    s = load_subject(sid)
    info = analyze_subject(s)
    gt = _gt_affine(s)
    if gt is None:
        return {"subject": sid, "status": "no_landmarks"}

    cz_all_um = _cz_xyz_um(s)
    cz_mapped = _apply_gt(cz_all_um, gt)

    hcr_centroids = s.hcr_centroids.copy()
    mask_strict = hcr_centroids["hcr_id"].astype(int).isin(strict_hcr_ids)
    hcr_strict = hcr_centroids[mask_strict]
    if len(hcr_strict) < 20:
        return {"subject": sid, "status": "too_few_gfp"}

    from benchmark_data_loader import hcr_px_to_um
    hcr_gfp_px = hcr_strict[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    hcr_gfp_um = hcr_px_to_um(hcr_gfp_px, s)[:, [2, 1, 0]]

    # True coregistered overlap: xy AABB where GT-mapped CZ and HCR GFP+ overlap
    lo = np.maximum(cz_mapped.min(axis=0)[:2], hcr_gfp_um.min(axis=0)[:2])
    hi = np.minimum(cz_mapped.max(axis=0)[:2], hcr_gfp_um.max(axis=0)[:2])
    if np.any(hi <= lo):
        return {"subject": sid, "status": "empty_xy_overlap"}

    def _xy_mask(p):
        return np.all((p[:, :2] >= lo) & (p[:, :2] <= hi), axis=1)

    cz_in = cz_mapped[_xy_mask(cz_mapped)]
    gfp_in = hcr_gfp_um[_xy_mask(hcr_gfp_um)]
    if len(cz_in) < 20 or len(gfp_in) < 20:
        return {"subject": sid, "status": "xy_overlap_too_sparse"}

    pia = info["hcr_surface"]
    d_cz = _pia_depth(cz_in, pia)
    d_gfp = _pia_depth(gfp_in, pia)

    d_lo = max(d_cz.min(), d_gfp.min())
    d_hi = min(d_cz.max(), d_gfp.max())
    if d_hi <= d_lo:
        return {"subject": sid, "status": "empty_z_overlap"}

    z_mask_cz = (d_cz >= d_lo) & (d_cz <= d_hi)
    z_mask_gfp = (d_gfp >= d_lo) & (d_gfp <= d_hi)
    cz_in = cz_in[z_mask_cz]; d_cz = d_cz[z_mask_cz]
    gfp_in = gfp_in[z_mask_gfp]; d_gfp = d_gfp[z_mask_gfp]

    xy_area_um2 = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    dmin = float(np.floor(d_lo / DEPTH_BIN_UM) * DEPTH_BIN_UM)
    dmax = float(np.ceil(d_hi / DEPTH_BIN_UM) * DEPTH_BIN_UM)
    bins = np.arange(dmin, dmax + DEPTH_BIN_UM, DEPTH_BIN_UM)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rho_cz = _density_profile(cz_in, d_cz, bins, xy_area_um2)
    rho_gfp = _density_profile(gfp_in, d_gfp, bins, xy_area_um2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rho_cz > 0, rho_gfp / rho_cz, np.nan)

    finite_cz = rho_cz[np.isfinite(rho_cz)]
    thr = float(np.percentile(finite_cz[finite_cz > 0], 25)) if np.any(finite_cz > 0) else 0.0
    good = (rho_cz > thr) & np.isfinite(ratio)
    if not np.any(good):
        cv = float("nan"); integrated = float("nan")
    else:
        r = ratio[good]
        cv = float(np.nanstd(r) / np.nanmean(r)) if np.nanmean(r) != 0 else float("nan")
        integrated = float(np.nanmean(r))

    # Plot — axes driven by data so 782149 shows its own range
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    scale = 1e6
    ax = axes[0]
    ax.plot(centers, rho_cz * scale, "-", lw=2.0, color="black",
            label="CZ mapped via GT Procrustes (all cells)")
    ax.plot(centers, rho_gfp * scale, "-", lw=1.8, color="#2aa198",
            label="HCR GFP+ strict (BIC-K)")
    ax.set_xlabel("depth from HCR pia (µm)")
    ax.set_ylabel("density (cells / 10^6 µm³)")
    ax.set_title(
        f"{sid}  GT-coreg overlap  "
        f"xy = {xy_area_um2/1e6:.2f} mm²  "
        f"depth {dmin:.0f}–{dmax:.0f} µm"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(dmin, dmax)

    ax = axes[1]
    ax.plot(centers, ratio, "-", lw=1.8, color="#268bd2", label="GFP+ / CZ per bin")
    if np.any(good):
        ax.axhline(integrated, color="#cc3333", lw=1.2, ls="--",
                   label=f"mean = {integrated:.3f}")
    ax.set_xlabel("depth from HCR pia (µm)")
    ax.set_ylabel("GFP+ / CZ ratio")
    ax.set_title(f"{sid}  informative-bin CV = {cv:.3f}  (bar {CV_GATE})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(dmin, dmax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"depth_density_gt_{sid}.png", dpi=140)
    plt.close(fig)

    return {
        "subject": sid,
        "status": "ok",
        "xy_area_um2": xy_area_um2,
        "depth_bin_um": DEPTH_BIN_UM,
        "depth_lo_um": dmin,
        "depth_hi_um": dmax,
        "depth_centers_um": centers.tolist(),
        "rho_cz_per_um3": rho_cz.tolist(),
        "rho_gfp_per_um3": rho_gfp.tolist(),
        "gfp_over_cz": ratio.tolist(),
        "cv_ratio": cv,
        "integrated_ratio": integrated,
        "gate_pass": bool(np.isfinite(cv) and cv <= CV_GATE),
        "gt_scales": gt["scales"].tolist(),
        "n_cz_in_box": int(len(cz_in)),
        "n_gfp_in_box": int(len(gfp_in)),
    }


def main():
    # Reuse the BIC-selected strict GFP+ set per subject
    records = {}
    for sid in BENCHMARK_SUBJECTS:
        gi = gfp_thr.analyze_subject(sid)
        if gi.n_strict < 300:
            records[sid] = {"subject": sid, "status": "too_few_strict"}
            continue
        strict_df = gfp_thr.strict_gfp_df(sid, gi.cutoff_linear)
        strict_ids = set(int(x) for x in strict_df["hcr_id"].values)
        res = gt_density_gate(sid, strict_ids)
        res["n_components_best"] = gi.n_components
        res["cutoff_strict_linear"] = gi.cutoff_linear
        records[sid] = res
        if res.get("status") == "ok":
            print(
                f"  {sid}  K*={gi.n_components}  "
                f"xy={res['xy_area_um2']/1e6:.2f}mm²  "
                f"depth {res['depth_lo_um']:.0f}–{res['depth_hi_um']:.0f}µm  "
                f"CV={res['cv_ratio']:.3f}  mean={res['integrated_ratio']:.3f}  "
                f"gate={res['gate_pass']}  gt_s={res['gt_scales']}"
            )
        else:
            print(f"  {sid}  status={res.get('status')}")

    with open(SESSION_DIR / "cz_density_gt_summary.json", "w") as f:
        json.dump(records, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    passed = sum(1 for r in records.values() if r.get("gate_pass"))
    print(f"\nGT-based gate: {passed}/{len(BENCHMARK_SUBJECTS)} pass CV ≤ {CV_GATE}")
    print(f"Wrote {SESSION_DIR / 'cz_density_gt_summary.json'}")


if __name__ == "__main__":
    main()
