"""After ICP fixes scales, search over translation to maximise the number
of warped CZ cells that have an HCR neighbour within 50 um.

The HCR-all centroid may be ~150 um off the true CZ-subset centroid when
the CZ slab is offset from the bulk HCR cloud.
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from lib.centroid_helpers import centroids_um
from anisotropic_icp import estimate_scales_icp_multi_start


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


def search_translation(cz_warped_xyz, hcr_xyz, radius_um=30.0,
                       dz_range=(-300, 300, 30), dxy_range=(-200, 200, 50)):
    """Grid search translation offsets; score by count of warped CZ cells
    having an HCR neighbour within radius.
    """
    tree = cKDTree(hcr_xyz)
    best_score = -1
    best_dxyz = np.zeros(3)
    for dz in range(dz_range[0], dz_range[1] + 1, dz_range[2]):
        for dy in range(dxy_range[0], dxy_range[1] + 1, dxy_range[2]):
            for dx in range(dxy_range[0], dxy_range[1] + 1, dxy_range[2]):
                pts = cz_warped_xyz + np.array([dx, dy, dz])
                d, _ = tree.query(pts, k=1)
                score = int(np.sum(d < radius_um))
                if score > best_score:
                    best_score = score; best_dxyz = np.array([dx, dy, dz])
    return best_dxyz, best_score


def main(sid="788406"):
    s = load_subject(sid)
    cz_zyx, cz_ids = centroids_um(s, "cz")
    hcr_zyx, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_zyx[:, [2, 1, 0]]
    hcr_xyz = hcr_zyx[:, [2, 1, 0]]

    R0 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    fit0 = _Fit(R=R0, src_mean=cz_xyz.mean(0), translation=hcr_xyz.mean(0),
                scales=np.ones(3))
    res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
    fit = res.fit
    cz_warp = (cz_xyz * fit.scales) @ fit.R.T + fit.translation
    print(f"ICP scales = {fit.scales}")

    # GT pairs
    id_to_cz = {int(i): k for k, i in enumerate(cz_ids)}
    id_to_hcr = {int(i): k for k, i in enumerate(hcr_ids)}
    rows = [(id_to_cz[int(r.cz_id)], id_to_hcr[int(r.hcr_id)])
            for _, r in s.coreg_table.iterrows()
            if int(r.cz_id) in id_to_cz and int(r.hcr_id) in id_to_hcr]
    idx_cz, idx_hcr = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    d0 = np.linalg.norm(cz_warp[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    print(f"Before translation search — per-GT median={np.median(d0):.1f} um, "
          f"<50um={(d0<50).sum()}/{len(d0)}")

    # Coarse search
    dxyz1, sc1 = search_translation(cz_warp, hcr_xyz, radius_um=50.0,
                                    dz_range=(-300, 300, 30),
                                    dxy_range=(-200, 200, 50))
    print(f"Coarse best dxyz={dxyz1}, score={sc1}")
    # Fine search around the coarse best
    dxyz2, sc2 = search_translation(cz_warp + dxyz1, hcr_xyz, radius_um=30.0,
                                    dz_range=(-60, 60, 10),
                                    dxy_range=(-40, 40, 10))
    t_refine = dxyz1 + dxyz2
    print(f"Fine refinement dxyz={dxyz2}, extra score={sc2}")
    print(f"Total refined offset (xyz): {t_refine}")
    cz_warp2 = cz_warp + t_refine
    d1 = np.linalg.norm(cz_warp2[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    print(f"After translation search — per-GT median={np.median(d1):.1f} um, "
          f"<50um={(d1<50).sum()}/{len(d1)}, <20um={(d1<20).sum()}, "
          f"<10um={(d1<10).sum()}")


if __name__ == "__main__":
    main()
