"""Quick scalar-σ_z check: does σ_z_HCR / σ_z_CZ (native) recover sz?

Two scales:
(a) Window-limited σ_z (W=100 µm): dominated by window (σ ≈ W/√12) —
    should give ratio ≈ 1 regardless of stretch. Sanity check.
(b) Global σ_z of all centroids in a *matched depth band*, in each
    modality's native frame: sz ≈ σ_z_HCR / σ_z_CZ.

Matched depth band: [d_skin, min(d_CZ_native_max, d_HCR_max)]
where d_HCR_max is HCR cortex extent measured pia-to-bottom.
CZ is in its NATIVE frame (no stretch applied).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "sess07e", THIS / "07e_sz_from_zvar_profile.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules["sess07e"] = m
spec.loader.exec_module(m)  # type: ignore

from benchmark_analysis import depth_from_surface, fit_anisotropic_similarity
from benchmark_data_loader import landmark_pairs_um, load_subject
from surfaces_iter08 import get_hcr_bottom_surface_iter08


def scalar_check(sid: str) -> dict:
    s = load_subject(sid)
    cz_um = m._cz_xyz_um(s)
    hcr_gfp_um = m._hcr_gfp_xyz_um(s)

    from benchmark_analysis import analyze_subject
    info = analyze_subject(s)
    cz_surf = info["cz_surface"]
    hcr_surf = info["hcr_surface"]
    hcr_bot = get_hcr_bottom_surface_iter08(s)

    # Depths in each modality's NATIVE frame
    cz_depth = depth_from_surface(cz_um, cz_surf)
    hcr_depth = depth_from_surface(hcr_gfp_um, hcr_surf)

    d_skin = 100.0

    # CZ cortex extent (native)
    d_cz_max = float(np.nanpercentile(cz_depth, 99))

    # HCR cortex extent (pia to bottom) from bottom surface
    hcr_all_px = s.hcr_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
    hcr_all_um = m.hcr_px_to_um(hcr_all_px, s)[:, [2, 1, 0]]
    probe_xyz = np.column_stack([
        hcr_all_um[:, 0], hcr_all_um[:, 1], np.zeros(len(hcr_all_um))
    ])
    pia_z = -depth_from_surface(probe_xyz, hcr_surf)
    bot_z = -depth_from_surface(probe_xyz, hcr_bot)
    d_hcr_max = float(np.nanpercentile(bot_z - pia_z, 90))

    # GT
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    sxy_gt = float(np.sqrt(fit.scales[0] * fit.scales[1]))
    sz_gt = float(fit.scales[2])

    # --- Scheme A: global σ_z of ALL centroids in each native frame
    #     (no matched band) — bounded by respective imaging depths
    cz_band_A = (cz_depth >= d_skin) & (cz_depth <= d_cz_max)
    hcr_band_A = (hcr_depth >= d_skin) & (hcr_depth <= d_hcr_max)
    # σ_z in native frame is σ of (pia-depth) of cell centroids
    # (equiv. to σ of z-relative-to-pia)
    sigma_cz_A = float(np.nanstd(cz_depth[cz_band_A]))
    sigma_hcr_A = float(np.nanstd(hcr_depth[hcr_band_A]))
    sz_A = sigma_hcr_A / sigma_cz_A if sigma_cz_A > 0 else float("nan")

    # --- Scheme B: matched physical depth band using GT-stretched CZ
    #     [d_skin, min(d_cz_max · sz_gt, d_hcr_max)] in HCR frame
    #     CZ band in native: [d_skin/sz_gt, (d_hcr_matched)/sz_gt]
    d_match_hcr = min(d_cz_max * sz_gt, d_hcr_max)
    # CZ native band:
    d_cz_lo = d_skin / sz_gt
    d_cz_hi = d_match_hcr / sz_gt
    cz_band_B = (cz_depth >= d_cz_lo) & (cz_depth <= d_cz_hi)
    hcr_band_B = (hcr_depth >= d_skin) & (hcr_depth <= d_match_hcr)
    sigma_cz_B = float(np.nanstd(cz_depth[cz_band_B]))
    sigma_hcr_B = float(np.nanstd(hcr_depth[hcr_band_B]))
    sz_B = sigma_hcr_B / sigma_cz_B if sigma_cz_B > 0 else float("nan")
    # This uses GT — just to verify the best-case scalar ratio.

    # --- Scheme C: iterated scheme B without GT (seed sz_0=2, iterate)
    sz_C = 2.0
    for _ in range(20):
        d_match = min(d_cz_max * sz_C, d_hcr_max)
        cz_band = (cz_depth >= d_skin / sz_C) & (cz_depth <= d_match / sz_C)
        hcr_band = (hcr_depth >= d_skin) & (hcr_depth <= d_match)
        sig_cz = float(np.nanstd(cz_depth[cz_band]))
        sig_hcr = float(np.nanstd(hcr_depth[hcr_band]))
        if sig_cz <= 0:
            break
        sz_new = sig_hcr / sig_cz
        if abs(sz_new - sz_C) < 0.005:
            sz_C = sz_new
            break
        sz_C = 0.5 * sz_C + 0.5 * sz_new

    return {
        "sid": sid,
        "sz_gt": sz_gt,
        "d_cz_max": d_cz_max,
        "d_hcr_max": d_hcr_max,
        "sz_gt_cz_stretched": d_cz_max * sz_gt,
        "hcr_truncated": d_hcr_max < d_cz_max * sz_gt,
        "A_global_native": {
            "sigma_cz": sigma_cz_A,
            "sigma_hcr": sigma_hcr_A,
            "sz_estimated": sz_A,
            "err_pct": 100 * (sz_A - sz_gt) / sz_gt,
        },
        "B_GT_matched_band": {
            "d_match_hcr": d_match_hcr,
            "sigma_cz": sigma_cz_B,
            "sigma_hcr": sigma_hcr_B,
            "sz_estimated": sz_B,
            "err_pct": 100 * (sz_B - sz_gt) / sz_gt,
            "note": "uses GT to define band — best case",
        },
        "C_iterated_matched": {
            "sz_estimated": sz_C,
            "err_pct": 100 * (sz_C - sz_gt) / sz_gt,
        },
    }


if __name__ == "__main__":
    import json
    sids = sys.argv[1:] or ["755252", "767018", "767022", "782149", "788406", "790322"]
    results = {}
    for sid in sids:
        print(f"=== {sid} ===")
        try:
            r = scalar_check(sid)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        print(f"  GT sz = {r['sz_gt']:.3f}")
        print(f"  d_cz_max = {r['d_cz_max']:.0f}, d_hcr_max = {r['d_hcr_max']:.0f}, "
              f"HCR {'TRUNCATED' if r['hcr_truncated'] else 'ok'}")
        for key in ("A_global_native", "B_GT_matched_band", "C_iterated_matched"):
            d = r[key]
            print(f"  {key:25s}: sz_est = {d['sz_estimated']:.3f}  "
                  f"err = {d['err_pct']:+.1f}%")
        results[sid] = r

    out = Path("/root/capsule/code/sessions/07e_sz_from_zvar_profile/scalar_sz_check.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")
