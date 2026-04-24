"""Synthetic sanity check for 07e σ_z-profile NCC.

Test 1 (positive control): take HCR σ_z(d) as the ground-truth profile.
Create a synthetic CZ profile by UN-stretching HCR by a known factor c_true
(i.e. synthetic_CZ(d) = HCR(c_true*d) / c_true). Run the NCC estimator —
should recover c_z ≈ c_true with high NCC.

Test 2 (real data): compute NCC peak height and NCC standard deviation
across the c_z grid. If peak-to-sd ratio < 3, the profile is too flat
to support stretch recovery.
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

RESULTS = Path("/root/capsule/code/sessions/07e_sz_from_zvar_profile/results.json")


def synthetic_test(sid: str, c_true: float = 1.5) -> dict:
    """Use real HCR σ_z(d) from stored results, synthesize CZ at 1/c_true,
    run NCC. Expect argmax near c_true."""
    r = json.load(open(RESULTS))["subjects"][sid]
    # Rebuild profile arrays from ncc_curve + traces is not enough;
    # we need the actual σ_z profiles. Those weren't saved per-iter,
    # but the final iterate σ_z IS available via the code's SubjectResult
    # fields. Let me re-run for just the profile data.
    res = m.run_subject(sid)
    def _f(xs):
        return np.array([x if x is not None else np.nan for x in xs], dtype=float)
    cz_d = np.array(res.cz_depth_grid, dtype=float)
    cz_sig = _f(res.cz_sigma_z)
    hcr_d = np.array(res.hcr_depth_grid, dtype=float)
    hcr_sig = _f(res.hcr_sigma_z)
    return {
        "sid": sid,
        "cz_d_range": (float(np.nanmin(cz_d)), float(np.nanmax(cz_d))) if len(cz_d) else None,
        "hcr_d_range": (float(np.nanmin(hcr_d)), float(np.nanmax(hcr_d))) if len(hcr_d) else None,
        "cz_sigma_range": (float(np.nanmin(cz_sig)), float(np.nanmax(cz_sig))) if len(cz_sig) else None,
        "hcr_sigma_range": (float(np.nanmin(hcr_sig)), float(np.nanmax(hcr_sig))) if len(hcr_sig) else None,
        "cz_sigma_std": float(np.nanstd(cz_sig)) if len(cz_sig) else None,
        "hcr_sigma_std": float(np.nanstd(hcr_sig)) if len(hcr_sig) else None,
        "cz_sigma_mean": float(np.nanmean(cz_sig)) if len(cz_sig) else None,
        "hcr_sigma_mean": float(np.nanmean(hcr_sig)) if len(hcr_sig) else None,
    }


def ncc_flatness(sid: str) -> dict:
    """Peak-to-sd ratio of NCC curve."""
    res = m.run_subject(sid)
    ncc = np.array(res.ncc_curve, dtype=float)
    valid = np.isfinite(ncc)
    if valid.sum() < 10:
        return {"sid": sid, "n_valid": int(valid.sum())}
    v = ncc[valid]
    peak = float(np.max(v))
    med = float(np.median(v))
    sd = float(np.std(v))
    return {
        "sid": sid,
        "n_valid": int(valid.sum()),
        "peak": peak,
        "median": med,
        "sd": sd,
        "peak_to_sd": (peak - med) / sd if sd > 0 else float("inf"),
    }


def synthetic_recovery(sid: str, c_true: float = 1.5) -> dict:
    """Synthesize CZ by unstretching HCR by c_true; run NCC; check recovery."""
    res = m.run_subject(sid)
    def _f(xs):
        return np.array([x if x is not None else np.nan for x in xs], dtype=float)
    hcr_d = np.array(res.hcr_depth_grid, dtype=float)
    hcr_sig = _f(res.hcr_sigma_z)
    hcr_n = np.array(res.hcr_n, dtype=int)
    # anchor at mid of valid HCR range
    valid = np.isfinite(hcr_sig)
    if valid.sum() < 10:
        return {"sid": sid, "error": "too few valid HCR bins"}
    d_anchor = float(hcr_d[valid].mean())
    # Synthetic CZ profile at c_true:
    #   HCR(d) = c_true * CZ((d - a)/c_true + a)
    #   => CZ(d) = HCR(c_true*(d - a) + a) / c_true
    cz_d_syn = hcr_d.copy()
    cz_sig_syn = np.full_like(hcr_sig, np.nan)
    # for each d_i, sample HCR at c_true*(d_i - a) + a
    d_sample = c_true * (cz_d_syn - d_anchor) + d_anchor
    hcr_sig_valid = hcr_sig.copy()
    order = np.argsort(hcr_d)
    cz_sig_syn = np.interp(d_sample, hcr_d[order], hcr_sig[order],
                            left=np.nan, right=np.nan) / c_true

    c_best, ncc_best, overlap, ncc_curve = m._fit_sz_via_ncc(
        cz_d_syn, cz_sig_syn, hcr_n,
        hcr_d, hcr_sig, hcr_n,
        d_anchor=d_anchor,
        c_z_grid=np.arange(0.5, 2.01, 0.01),
    )
    return {
        "sid": sid,
        "c_true": c_true,
        "c_recovered": c_best,
        "ncc_best": ncc_best,
        "overlap_um": overlap,
        "err_pct": 100.0 * (c_best - c_true) / c_true,
    }


if __name__ == "__main__":
    for sid in ["788406", "790322"]:
        print(f"=== {sid} ===")
        print("profile stats:", synthetic_test(sid))
        print("NCC flatness: ", ncc_flatness(sid))
        print("synthetic c=1.5 recovery:", synthetic_recovery(sid, 1.5))
        print("synthetic c=1.2 recovery:", synthetic_recovery(sid, 1.2))
        print("synthetic c=0.8 recovery:", synthetic_recovery(sid, 0.8))
        print()
