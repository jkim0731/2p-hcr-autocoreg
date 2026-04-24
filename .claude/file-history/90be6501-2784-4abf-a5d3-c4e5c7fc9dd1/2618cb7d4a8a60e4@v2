"""Restart ICP from the translation-search-refined position and see if it
converges to a tighter fit than the single-pass ICP.
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


def main(sid="788406"):
    s = load_subject(sid)
    cz_zyx, cz_ids = centroids_um(s, "cz")
    hcr_zyx, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_xyz = cz_zyx[:, [2, 1, 0]]
    hcr_xyz = hcr_zyx[:, [2, 1, 0]]

    R0 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    fit0 = _Fit(R=R0, src_mean=cz_xyz.mean(0), translation=hcr_xyz.mean(0),
                scales=np.ones(3))
    res1 = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
    fit1 = res1.fit
    pred1 = (cz_xyz * fit1.scales) @ fit1.R.T + fit1.translation
    # Translation search
    tree = cKDTree(hcr_xyz)
    best = np.zeros(3); best_sc = (tree.query(pred1, k=1)[0] < 50.0).sum()
    for dz in range(-300, 301, 30):
        for dy in range(-200, 201, 50):
            for dx in range(-200, 201, 50):
                c = np.array([dx, dy, dz], float)
                sc = (tree.query(pred1 + c, k=1)[0] < 50.0).sum()
                if sc > best_sc:
                    best_sc = sc; best = c
    pred1r = pred1 + best
    print(f"pass1 ICP rms={fit1.rms_um:.1f}, scales={fit1.scales}, translation_refine_xyz={best}, inlier@50={best_sc}")

    # Restart ICP from the refined transform
    fit_restart = _Fit(R=fit1.R, src_mean=cz_xyz.mean(0),
                        translation=fit1.translation + best, scales=fit1.scales)
    res2 = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit_restart)
    fit2 = res2.fit
    pred2 = (cz_xyz * fit2.scales) @ fit2.R.T + fit2.translation
    print(f"pass2 ICP rms={fit2.rms_um:.1f}, scales={fit2.scales}, iters={res2.iterations}, matched={res2.n_matched}")

    # GT comparisons
    id_to_cz = {int(i): k for k, i in enumerate(cz_ids)}
    id_to_hcr = {int(i): k for k, i in enumerate(hcr_ids)}
    rows = [(id_to_cz[int(r.cz_id)], id_to_hcr[int(r.hcr_id)]) for _, r in s.coreg_table.iterrows() if int(r.cz_id) in id_to_cz and int(r.hcr_id) in id_to_hcr]
    idx_cz, idx_hcr = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    d1 = np.linalg.norm(pred1r[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    d2 = np.linalg.norm(pred2[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    print(f"pass1 per-GT: median={np.median(d1):.1f}, <20={(d1<20).sum()}, <30={(d1<30).sum()}, <50={(d1<50).sum()} / {len(d1)}")
    print(f"pass2 per-GT: median={np.median(d2):.1f}, <20={(d2<20).sum()}, <30={(d2<30).sum()}, <50={(d2<50).sum()} / {len(d2)}")


if __name__ == "__main__":
    main()
