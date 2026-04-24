"""Multi-start ICP: run ICP from multiple initial translations per subject;
score converged warps by post-ICP inlier@30µm on GFP+ HCR; return the winner.
"""
from __future__ import annotations
import sys, json
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


def run_icp_from_seed(cz_xyz: np.ndarray, hcr_xyz: np.ndarray, seed_t_xyz: np.ndarray):
    """Run ICP with given initial translation. Returns (pred_xyz, fit, rms)."""
    R0 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    fit0 = _Fit(R=R0, src_mean=cz_xyz.mean(0),
                translation=seed_t_xyz, scales=np.ones(3))
    try:
        res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
        if res.fit is None:
            return None
        fit = res.fit
        pred = (cz_xyz * fit.scales) @ fit.R.T + fit.translation
        return dict(pred_xyz=pred, fit=fit, rms=float(fit.rms_um),
                    matched=int(res.n_matched), iters=int(res.iterations))
    except Exception as e:
        return None


def score_converged(pred_xyz, hcr_xyz, radius=30.0):
    tree = cKDTree(hcr_xyz)
    d, _ = tree.query(pred_xyz, k=1)
    return int((d < radius).sum()), float(np.median(d))


def main():
    # Scan subjects + seed translations. Seeds are in xyz convention.
    for sid in ["767022", "782149", "755252", "788406", "767018", "790322"]:
        s = load_subject(sid)
        cz_zyx, cz_ids = centroids_um(s, "cz")
        hcr_zyx, hcr_ids = centroids_um(s, "hcr_gfp")
        hcr_all_zyx, _ = centroids_um(s, "hcr_all")
        cz_xyz = cz_zyx[:, [2, 1, 0]]
        hcr_xyz = hcr_zyx[:, [2, 1, 0]]
        hcr_all_xyz = hcr_all_zyx[:, [2, 1, 0]]

        # Seeds (xyz): always include HCR-all, HCR-gfp centroid, and z-shifts of hcr-gfp.
        gfp_c_xyz = hcr_xyz.mean(0)
        all_c_xyz = hcr_all_xyz.mean(0)
        seeds = {
            "hcr_all":   all_c_xyz.copy(),
            "hcr_gfp":   gfp_c_xyz.copy(),
            "gfp_dz+100": gfp_c_xyz + np.array([0, 0, 100]),
            "gfp_dz-100": gfp_c_xyz + np.array([0, 0, -100]),
            "gfp_dz+200": gfp_c_xyz + np.array([0, 0, 200]),
            "gfp_q25":   np.quantile(hcr_xyz, 0.25, axis=0),
            "gfp_q75":   np.quantile(hcr_xyz, 0.75, axis=0),
        }

        # GT for sanity: per-GT distance under each converged warp.
        id_to_cz = {int(i): k for k, i in enumerate(cz_ids)}
        id_to_hcr = {int(i): k for k, i in enumerate(hcr_ids)}
        rows = [(id_to_cz[int(r.cz_id)], id_to_hcr[int(r.hcr_id)])
                for _, r in s.coreg_table.iterrows()
                if int(r.cz_id) in id_to_cz and int(r.hcr_id) in id_to_hcr]
        idx_cz = np.array([r[0] for r in rows])
        idx_hcr = np.array([r[1] for r in rows])

        print(f"\n=== {sid}  (n_cz={len(cz_xyz)}, n_hcr_gfp={len(hcr_xyz)}, n_gt={len(idx_cz)}) ===")
        results = []
        for name, t in seeds.items():
            res = run_icp_from_seed(cz_xyz, hcr_xyz, t)
            if res is None:
                print(f"  {name:12s}: ICP failed")
                continue
            inl30, med_d = score_converged(res["pred_xyz"], hcr_xyz, 30.0)
            inl50, _    = score_converged(res["pred_xyz"], hcr_xyz, 50.0)
            # GT per-pair distance
            gt_d = np.linalg.norm(res["pred_xyz"][idx_cz] - hcr_xyz[idx_hcr], axis=1)
            gt_med = float(np.median(gt_d))
            gt_lt20 = int((gt_d < 20).sum())
            gt_lt50 = int((gt_d < 50).sum())
            print(f"  {name:12s}: rms={res['rms']:6.1f} scales=({res['fit'].scales[0]:.2f},{res['fit'].scales[1]:.2f},{res['fit'].scales[2]:.2f}) "
                  f"inl30={inl30:4d} inl50={inl50:4d} med_d={med_d:6.1f} | "
                  f"GT: med={gt_med:6.1f} <20={gt_lt20:3d} <50={gt_lt50:3d} / {len(idx_cz)}")
            results.append((name, inl30, inl50, res))
        # best by inl30
        if results:
            best = max(results, key=lambda r: r[1])
            print(f"  -> best by inl30: {best[0]} (inl30={best[1]}, inl50={best[2]})")


if __name__ == "__main__":
    main()
