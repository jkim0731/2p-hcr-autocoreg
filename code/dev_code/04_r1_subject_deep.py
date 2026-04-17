"""Deep dive into 782149 — where is CZ relative to HCR GFP+?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject, depth_from_surface
from benchmark_data_loader import load_subject, landmark_pairs_um
from r1_coarse_align import PRIOR_XY_EXPANSION, PRIOR_Z_EXPANSION, _rotation_about_z


def analyze(sid):
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    hcr_xyz = info["hcr_xyz"]  # all HCR cells
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]

    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    gt = info["procrustes"]
    cz_center = cz_xyz.mean(axis=0)
    src_mean = cz_lm.mean(axis=0)
    dst_mean = hcr_lm.mean(axis=0)
    gt_target = (cz_center - src_mean) @ gt.R * gt.scales + dst_mean

    print(f"=== {sid} ===")
    print(f"  GT target (CZ center -> HCR): {gt_target}")
    print(f"  HCR GFP+ centroid (all):       {gfp_xyz.mean(axis=0)}")
    print(f"  HCR GFP+ count: {len(gfp_xyz)}")
    print(f"  HCR total count: {len(hcr_xyz)}")
    print(f"  HCR all centroid: {hcr_xyz.mean(axis=0)}")
    print(f"  HCR landmarks centroid: {hcr_lm.mean(axis=0)}")

    # Histogram of GFP+ x, y
    for ax, name in [(0, 'x'), (1, 'y')]:
        vals = gfp_xyz[:, ax]
        pct = [np.percentile(vals, p) for p in [10, 25, 50, 75, 90]]
        print(f"  GFP+ {name}: min={vals.min():.0f} p10={pct[0]:.0f} p25={pct[1]:.0f} "
              f"p50={pct[2]:.0f} p75={pct[3]:.0f} p90={pct[4]:.0f} max={vals.max():.0f}")
        lm_vals = hcr_lm[:, ax]
        print(f"  LM  {name}: min={lm_vals.min():.0f} p50={np.median(lm_vals):.0f} max={lm_vals.max():.0f}")

    # Depth profile of HCR GFP+
    hcr_depth = depth_from_surface(gfp_xyz, hcr_surface)
    cz_depth = depth_from_surface(cz_xyz, cz_surface)
    print(f"  HCR GFP+ depth: min={hcr_depth.min():.0f} median={np.median(hcr_depth):.0f} max={hcr_depth.max():.0f}")
    print(f"  CZ depth (scaled): min={(cz_depth*PRIOR_Z_EXPANSION).min():.0f} median={np.median(cz_depth)*PRIOR_Z_EXPANSION:.0f} max={(cz_depth*PRIOR_Z_EXPANSION).max():.0f}")

    # Density in quadrants
    cx, cy = np.median(gfp_xyz[:, 0]), np.median(gfp_xyz[:, 1])
    quads = {
        'Q1 (high x, high y)': ((gfp_xyz[:, 0] > cx) & (gfp_xyz[:, 1] > cy)).sum(),
        'Q2 (low x, high y)':  ((gfp_xyz[:, 0] < cx) & (gfp_xyz[:, 1] > cy)).sum(),
        'Q3 (low x, low y)':   ((gfp_xyz[:, 0] < cx) & (gfp_xyz[:, 1] < cy)).sum(),
        'Q4 (high x, low y)':  ((gfp_xyz[:, 0] > cx) & (gfp_xyz[:, 1] < cy)).sum(),
    }
    print(f"  HCR GFP+ quadrants at ({cx:.0f},{cy:.0f}): {quads}")

    # Where is GT target relative to centroid?
    centroid_xy = gfp_xyz[:, :2].mean(axis=0)
    offset = gt_target[:2] - centroid_xy
    print(f"  GT target offset from centroid: ({offset[0]:.0f}, {offset[1]:.0f})")
    print()


if __name__ == "__main__":
    for sid in sys.argv[1:] or ["782149", "767018"]:
        analyze(sid)
