"""S46-d probe 1 — measure I2's per-cell residual to GT on 782149.

Validation-only diagnostic (does NOT use GT for fitting). Run I2 to get the
MI-affine, apply it to CZ centroids, compare to HCR partners via coreg_table.
If median residual < 30 µm → direct NN matching is viable. If not → need I3.

Also runs on 788406 / 755252 / 767022 for context.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

import bench.candidates  # noqa: F401
from bench.harness import run_candidate  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._i2_sitk_affine import run_i2  # noqa: E402
from bench.candidate_impls._i1_axial_ncc import _load_cz_fullstack, _load_hcr_fullstack  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from lib.sitk_wrapper import mi_affine  # noqa: E402


def _apply_sitk_inverse(pts_zyx: np.ndarray, A: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """SITK inverse (moving→fixed = CZ→HCR).

    Forward: p_moving = A @ (p_fixed - c) + c + t.
    Inverse: p_fixed = A^-1 @ (p_moving - c - t) + c.
    """
    A_inv = np.linalg.inv(A)
    return ((pts_zyx - c - t) @ A_inv.T) + c


def _apply_sitk_forward(pts_zyx: np.ndarray, A: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """SITK forward (fixed→moving = HCR→CZ)."""
    return (pts_zyx - c) @ A.T + c + t


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    for subj in subjects:
        s = load_subject(subj)
        cz_um, cz_ids = centroids_um(s, "cz")
        hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

        t0 = time.time()
        r_i2 = run_i2(s)
        wall = time.time() - t0
        if r_i2.transform is None:
            print(f"  {subj}: I2 FAILED {r_i2.diagnostics}")
            continue
        A = np.asarray(r_i2.diagnostics["sitk_matrix_zyx"])
        t_v = np.asarray(r_i2.diagnostics["sitk_translation_zyx"])
        c_v = np.asarray(r_i2.diagnostics["sitk_center_zyx"])
        cz_warped = _apply_sitk_inverse(cz_um, A, t_v, c_v)

        # GT from coreg_table
        gt = s.coreg_table
        gt_cz = gt["cz_id"].astype(int).values
        gt_hcr = gt["hcr_id"].astype(int).values

        cz_idx_of = {int(c): i for i, c in enumerate(cz_ids)}
        hcr_idx_of = {int(h): i for i, h in enumerate(hcr_ids)}
        hits_cz = np.array([cz_idx_of.get(int(c), -1) for c in gt_cz])
        hits_hcr = np.array([hcr_idx_of.get(int(h), -1) for h in gt_hcr])
        ok = (hits_cz >= 0) & (hits_hcr >= 0)
        # Anisotropic scale (CZ→HCR): singular values of A^-1
        A_inv = np.linalg.inv(A)
        sv = np.linalg.svd(A_inv, compute_uv=False)
        print(f"\n  {subj}: n_gt={len(gt)} n_gt_in_sets={ok.sum()} wall={wall:.1f}s "
              f"scales_inv={np.round(sv, 3).tolist()}")

        if ok.sum() == 0:
            continue

        cz_w = cz_warped[hits_cz[ok]]
        hcr_tru = hcr_um[hits_hcr[ok]]
        res_um = np.linalg.norm(cz_w - hcr_tru, axis=1)
        print(f"    residual_um: n={len(res_um)} "
              f"median={np.median(res_um):.1f} "
              f"p10={np.percentile(res_um, 10):.1f} "
              f"p90={np.percentile(res_um, 90):.1f} "
              f"max={res_um.max():.1f}")

        # Check how many GT partners are within various radii
        for R in (5, 10, 20, 30, 50, 100, 200):
            n_within = (res_um < R).sum()
            print(f"    n(res<{R:3d}µm)={n_within}/{len(res_um)} "
                  f"({100*n_within/len(res_um):.1f} %)")

        # Nearest-HCR-to-warped-CZ: for each CZ that has GT, find nearest HCR
        # centroid. Is it the GT partner?
        from scipy.spatial import cKDTree
        tree = cKDTree(hcr_um)
        d_nn, i_nn = tree.query(cz_warped[hits_cz[ok]], k=1)
        correct_nn = (i_nn == hits_hcr[ok])
        print(f"    nearest_is_GT: {correct_nn.sum()}/{len(correct_nn)} "
              f"({100*correct_nn.mean():.1f} %)")

        # Is GT in top-K?
        for K in (5, 10, 20, 50):
            d_K, i_K = tree.query(cz_warped[hits_cz[ok]], k=K)
            in_topK = (i_K == hits_hcr[ok][:, None]).any(axis=1)
            print(f"    GT_in_top{K:3d}: {in_topK.sum()}/{len(correct_nn)} "
                  f"({100*in_topK.mean():.1f} %)")


if __name__ == "__main__":
    main()
