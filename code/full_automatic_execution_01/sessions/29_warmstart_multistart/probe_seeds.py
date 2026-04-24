"""Probe seeding scenarios on the 3 failing subjects to see if GFP+ centroid
or a coarse XY grid offers a better initial translation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from lib.centroid_helpers import centroids_um


def seed_inlier_score(cz_um_zyx, hcr_um_zyx, translation_zyx, radius=50.0):
    """After 180° rotation about CZ centroid, how many CZ land within `radius`
    of any HCR under the given HCR-centroid translation."""
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    cz_c = cz_um_zyx.mean(0)
    pts = (cz_um_zyx - cz_c) @ R.T + translation_zyx
    tree = cKDTree(hcr_um_zyx)
    d, _ = tree.query(pts, k=1)
    return int((d < radius).sum())


for sid in ["767022", "782149", "755252", "788406", "767018", "790322"]:
    s = load_subject(sid)
    cz, _ = centroids_um(s, "cz")
    hcr_gfp, _ = centroids_um(s, "hcr_gfp")
    hcr_all, _ = centroids_um(s, "hcr_all")
    n_cz = len(cz)
    print(f"\n--- {sid} (n_cz={n_cz}) ---")

    seeds = {
        "hcr_all_cent": hcr_all.mean(0),
        "hcr_gfp_cent": hcr_gfp.mean(0),
        "hcr_gfp_q25":  np.quantile(hcr_gfp, 0.25, axis=0),
        "hcr_gfp_q75":  np.quantile(hcr_gfp, 0.75, axis=0),
    }
    # Add z-shifted seeds around the GFP+ centroid (cortical-depth grid)
    gc = hcr_gfp.mean(0)
    for dz in (-200, -100, 100, 200):
        seeds[f"gfp_cent_dz{dz:+d}"] = gc + np.array([dz, 0, 0])

    scored = []
    for name, t in seeds.items():
        sc = seed_inlier_score(cz, hcr_gfp, t, radius=50.0)
        scored.append((sc, name, t))
    scored.sort(reverse=True)
    for sc, name, t in scored[:6]:
        frac = sc / n_cz
        print(f"  {name:20s} inlier@50 = {sc:4d} / {n_cz:4d} ({frac:.3f})  t_zyx={t}")
