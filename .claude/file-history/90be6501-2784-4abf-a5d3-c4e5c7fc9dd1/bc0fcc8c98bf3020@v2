"""Compare two forward formulas applied to the ICP fit on 788406."""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

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
    res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)
    fit = res.fit
    # canonical pred = ((src - src_mean_src) @ R) * scales + dst_mean
    # But fit.translation = dst_mean - R @ (src_mean * scales) — we need src_mean.
    # Reconstruct: src_mean_xyz must be close to cz_xyz.mean(0).
    src_mean_used = cz_xyz.mean(0)
    # Recover dst_mean from translation + R @ (src_mean * scales)
    dst_mean_reconstructed = fit.translation + fit.R @ (src_mean_used * fit.scales)
    print(f"ICP src_mean used: {src_mean_used}")
    print(f"ICP dst_mean reconstructed from fit: {dst_mean_reconstructed}")
    print(f"ICP stored translation: {fit.translation}")
    print(f"ICP R.det = {np.linalg.det(fit.R):.6f}")

    # Apply both forms
    # (a) canonical (Procrustes): ((cz_xyz - src_mean) @ R) * scales + dst_mean
    pred_canon = ((cz_xyz - src_mean_used) @ fit.R) * fit.scales + dst_mean_reconstructed
    # (b) apply_aniso_fit in helper: (cz_xyz * scales) @ R.T + translation
    pred_helper = (cz_xyz * fit.scales) @ fit.R.T + fit.translation

    diff = pred_canon - pred_helper
    print(f"Pred-per-form diff norm: median={np.median(np.linalg.norm(diff,axis=1)):.2f}, "
          f"max={np.linalg.norm(diff,axis=1).max():.2f}")

    # GT pairs
    id_to_cz = {int(i): k for k, i in enumerate(cz_ids)}
    id_to_hcr = {int(i): k for k, i in enumerate(hcr_ids)}
    rows = []
    for _, r in s.coreg_table.iterrows():
        if int(r.cz_id) in id_to_cz and int(r.hcr_id) in id_to_hcr:
            rows.append((id_to_cz[int(r.cz_id)], id_to_hcr[int(r.hcr_id)]))
    idx_cz = np.array([r[0] for r in rows]); idx_hcr = np.array([r[1] for r in rows])
    print(f"GT pairs: {len(rows)}")

    d_canon = np.linalg.norm(pred_canon[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    d_helper = np.linalg.norm(pred_helper[idx_cz] - hcr_xyz[idx_hcr], axis=1)
    print(f"Per-GT distance CANONICAL: median={np.median(d_canon):.1f}, "
          f"p90={np.quantile(d_canon,0.9):.1f}, <50um={(d_canon<50).sum()}/{len(d_canon)}")
    print(f"Per-GT distance HELPER: median={np.median(d_helper):.1f}, "
          f"p90={np.quantile(d_helper,0.9):.1f}, <50um={(d_helper<50).sum()}/{len(d_helper)}")


if __name__ == "__main__":
    main()
