"""One-off diagnostic: dump Z and XY score curves for a single subject."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject
from benchmark_data_loader import load_subject
from r1_revised import coarse_align_revised

subject_id = sys.argv[1] if len(sys.argv) > 1 else "788406"
s = load_subject(subject_id)
info = analyze_subject(s)
cz_xyz = info["cz_xyz"]
gfp_xyz = info["gfp_xyz"]

r1 = coarse_align_revised(cz_xyz, gfp_xyz, info["cz_surface"], info["hcr_surface"])

sz_grid = r1.diagnostics["sz_grid"]
sz_scores = r1.diagnostics["sz_score_curve"]
sxy_grid = r1.diagnostics["sxy_grid"]
sxy_scores = r1.diagnostics["sxy_score_curve"]

print(f"\n{subject_id}  (cz={len(cz_xyz)}, hcr_gfp={len(gfp_xyz)})")
print(f"Tilt: cz={r1.diagnostics['cz_tilt_deg']:.2f}°  "
      f"hcr={r1.diagnostics['hcr_tilt_deg']:.2f}°")
print(f"\nZ-score curve (sz vs max-over-tz NCC):")
print(f"{'sz':>6} {'score':>8}")
for sz, sc in zip(sz_grid, sz_scores):
    if np.isfinite(sc):
        print(f"{sz:>6.2f} {sc:>8.4f}")

print(f"\nXY-score curve (sxy vs max-over-(tx,ty) NCC):")
print(f"{'sxy':>6} {'score':>8} {'tx':>8} {'ty':>8}")
for sxy, sc, tx, ty in zip(
    sxy_grid, sxy_scores,
    r1.diagnostics["sxy_score_curve"],  # placeholder
    r1.diagnostics["sxy_score_curve"],  # placeholder
):
    if np.isfinite(sc):
        print(f"{sxy:>6.2f} {sc:>8.4f}")

print(f"\nBest sz = {r1.diagnostics['sz_best']}, tz = {r1.diagnostics['tz_best_um']:.1f} µm")
print(f"Best sxy = {r1.diagnostics['sxy_best']}, tx = {r1.diagnostics['xy_tx_best_um']:.1f}, ty = {r1.diagnostics['xy_ty_best_um']:.1f}")
print(f"Coverage regime: {r1.coverage_regime}")
print(f"L_cz={r1.diagnostics['L_cz_xy_um']:.0f}, L_hcr={r1.diagnostics['L_hcr_xy_um']:.0f} µm")

# Depth ranges
from r1_revised import depth_from_surface
cz_depth = depth_from_surface(cz_xyz, info["cz_surface"])
hcr_depth = depth_from_surface(gfp_xyz, info["hcr_surface"])
print(f"\nCZ depth: min={cz_depth.min():.1f} max={cz_depth.max():.1f} med={np.median(cz_depth):.1f}")
print(f"HCR depth: min={hcr_depth.min():.1f} max={hcr_depth.max():.1f} med={np.median(hcr_depth):.1f}")
print(f"CZ range: {cz_depth.max() - cz_depth.min():.1f} µm")
print(f"HCR range: {hcr_depth.max() - hcr_depth.min():.1f} µm")
