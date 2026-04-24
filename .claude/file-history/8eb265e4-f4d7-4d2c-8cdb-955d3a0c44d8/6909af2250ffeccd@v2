"""Simple σ_z ratio sz estimator.

Idea: both modalities image cortex pia-to-bottom. σ_z of cell depths
(native frame) measures tissue z-spread. Under isotropic stretch,
σ_z_HCR = sz · σ_z_CZ. No iteration, no matched band — just a scalar
ratio. For HCR, restrict to the xy center matching CZ footprint
after R1 so we aren't summing over HCR regions that don't overlap
CZ.

Two variants:
  (i) σ_z of CZ depths (all cells past skin), σ_z of HCR depths
      (all cells past skin within xy center window).
  (ii) Same but with an additional bottom-truncation on HCR to
       drop cells below the HCR bottom surface (iter08).
"""
from __future__ import annotations

import json
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

from benchmark_analysis import analyze_subject, depth_from_surface, fit_anisotropic_similarity
from benchmark_data_loader import landmark_pairs_um, load_subject, hcr_px_to_um
from surfaces_iter08 import get_hcr_bottom_surface_iter08
from r1_revised import coarse_align_revised


D_SKIN = 100.0
S0 = np.array([2.0, 2.0, 2.0])


def _cz_in_hcr(cz_um, hcr_gfp_um, cz_surf, hcr_surf, s0=S0):
    fit = coarse_align_revised(cz_um, hcr_gfp_um, cz_surf, hcr_surf,
                                aniso_refine=False)
    src_c = cz_um - fit.src_mean
    return (src_c @ fit.R) * s0 + fit.minimal_translation, fit


def estimate_sz(sid: str) -> dict:
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf = info["cz_surface"]
    hcr_surf = info["hcr_surface"]
    hcr_bot = get_hcr_bottom_surface_iter08(s)

    cz_um = m._cz_xyz_um(s)
    hcr_gfp_um = m._hcr_gfp_xyz_um(s)

    # ---- σ_z for CZ: native depth, skin filter, inside CZ tissue
    cz_depth = depth_from_surface(cz_um, cz_surf)
    cz_mask = cz_depth >= D_SKIN
    sigma_cz = float(np.nanstd(cz_depth[cz_mask]))
    n_cz = int(cz_mask.sum())

    # ---- HCR xy center window matching CZ footprint after R1 ----
    cz_in_hcr, fit = _cz_in_hcr(cz_um, hcr_gfp_um, cz_surf, hcr_surf, S0)
    x0, x1 = float(cz_in_hcr[:, 0].min()), float(cz_in_hcr[:, 0].max())
    y0, y1 = float(cz_in_hcr[:, 1].min()), float(cz_in_hcr[:, 1].max())

    # HCR depths of GFP+ cells; and HCR ALL centroid depths for ref
    hcr_depth = depth_from_surface(hcr_gfp_um, hcr_surf)

    # Variant (i): GFP+ inside xy window, depth ≥ skin
    xy = hcr_gfp_um[:, :2]
    xy_mask = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    m_i = xy_mask & (hcr_depth >= D_SKIN)
    sigma_hcr_i = float(np.nanstd(hcr_depth[m_i])) if m_i.sum() >= 4 else float("nan")
    n_hcr_i = int(m_i.sum())

    # Variant (ii): add bottom truncation — drop cells deeper than HCR bottom
    bot_depth = depth_from_surface(hcr_gfp_um, hcr_bot)  # distance from bottom surface
    # cells inside tissue have bot_depth < 0 (they're above the bottom surface)
    m_ii = m_i & (bot_depth <= 0)
    sigma_hcr_ii = float(np.nanstd(hcr_depth[m_ii])) if m_ii.sum() >= 4 else float("nan")
    n_hcr_ii = int(m_ii.sum())

    # ---- GT ----
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    gt = fit_anisotropic_similarity(cz_lm, hcr_lm)
    sz_gt = float(gt.scales[2])

    sz_i = sigma_hcr_i / sigma_cz if sigma_cz > 0 else float("nan")
    sz_ii = sigma_hcr_ii / sigma_cz if sigma_cz > 0 else float("nan")
    return {
        "sid": sid,
        "sz_gt": sz_gt,
        "sigma_cz": sigma_cz, "n_cz": n_cz,
        "variant_i": {
            "sigma_hcr": sigma_hcr_i, "n_hcr": n_hcr_i,
            "sz": sz_i,
            "err_pct": 100 * (sz_i - sz_gt) / sz_gt,
        },
        "variant_ii_bottom_trunc": {
            "sigma_hcr": sigma_hcr_ii, "n_hcr": n_hcr_ii,
            "sz": sz_ii,
            "err_pct": 100 * (sz_ii - sz_gt) / sz_gt,
        },
        "xy_window_hcr": [x0, x1, y0, y1],
    }


if __name__ == "__main__":
    sids = sys.argv[1:] or ["755252", "767018", "767022", "782149", "788406", "790322"]
    results = {}
    print(f"{'sid':<8} {'GT':>6} {'σ_CZ':>7} {'σ_HCR_i':>9} {'sz_i':>7} {'err_i%':>8}  "
          f"{'σ_HCR_ii':>9} {'sz_ii':>7} {'err_ii%':>8}")
    for sid in sids:
        try:
            r = estimate_sz(sid)
        except Exception as e:
            print(f"{sid}: ERROR {e}")
            continue
        print(f"{sid:<8} {r['sz_gt']:>6.3f} {r['sigma_cz']:>7.1f} "
              f"{r['variant_i']['sigma_hcr']:>9.1f} {r['variant_i']['sz']:>7.3f} "
              f"{r['variant_i']['err_pct']:>+8.1f}  "
              f"{r['variant_ii_bottom_trunc']['sigma_hcr']:>9.1f} "
              f"{r['variant_ii_bottom_trunc']['sz']:>7.3f} "
              f"{r['variant_ii_bottom_trunc']['err_pct']:>+8.1f}")
        results[sid] = r

    pass_i = [s for s, r in results.items()
              if abs(r['variant_i']['err_pct']) <= 5.0]
    pass_ii = [s for s, r in results.items()
               if abs(r['variant_ii_bottom_trunc']['err_pct']) <= 5.0]
    print(f"\nvariant (i):  {len(pass_i)}/6 within ±5 %  {pass_i}")
    print(f"variant (ii): {len(pass_ii)}/6 within ±5 %  {pass_ii}")

    out = Path("/root/capsule/code/sessions/07e_sz_from_zvar_profile/simple_sigma_ratio.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")
