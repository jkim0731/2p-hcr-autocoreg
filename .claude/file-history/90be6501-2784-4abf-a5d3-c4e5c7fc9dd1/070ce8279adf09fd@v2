"""Diagnose 782149: where does ICP's best seed land vs where should it land?"""
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from benchmark_analysis import fit_anisotropic_similarity  # noqa
from anisotropic_icp import estimate_scales_icp_multi_start  # noqa
from lib.centroid_helpers import centroids_um  # noqa

from dataclasses import dataclass


@dataclass
class _Fit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


def gt_pairs(s):
    ct = s.coreg_table
    cz = s.cz_centroids.set_index("cz_id")
    hc = s.hcr_centroids.set_index("hcr_id")
    mask = ct["cz_id"].isin(cz.index) & ct["hcr_id"].isin(hc.index)
    ct = ct[mask]
    cz_rows = cz.loc[ct["cz_id"].values]
    hc_rows = hc.loc[ct["hcr_id"].values]
    cz_um = cz_px_to_um(cz_rows[["z_px", "y_px", "x_px"]].values, s)
    hc_um = hcr_px_to_um(hc_rows[["z_px", "y_px", "x_px"]].values, s)
    return cz_um[:, [2, 1, 0]], hc_um[:, [2, 1, 0]]


s = load_subject("782149")
cz_pts, _ = centroids_um(s, "cz")
hcr_pts, _ = centroids_um(s, "hcr_gfp")
cz_xyz = cz_pts[:, [2, 1, 0]]
hcr_xyz = hcr_pts[:, [2, 1, 0]]
cz_gt, hcr_gt = gt_pairs(s)

# Landmark-fit
fit_gt = fit_anisotropic_similarity(cz_gt, hcr_gt)
pred_gt = ((cz_xyz * fit_gt.scales) @ fit_gt.R.T) + fit_gt.translation
print(f"GT landmark fit: sxy={fit_gt.scales[0]:.2f}, sz={fit_gt.scales[2]:.2f}")
print(f"GT pred CZ centroid (xyz): {pred_gt.mean(0)}")
print(f"HCR centroid (xyz): {hcr_xyz.mean(0)}")
print(f"HCR extent (xyz): lo={hcr_xyz.min(0)}, hi={hcr_xyz.max(0)}")
print(f"CZ extent (xyz): lo={cz_xyz.min(0)}, hi={cz_xyz.max(0)}")
print(f"GT CZ region centroid: {hcr_gt.mean(0)}")

# Run ICP with default seeds
R_xyz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
seed_fit = _Fit(R=R_xyz, src_mean=cz_xyz.mean(0),
                translation=hcr_xyz.mean(0), scales=np.ones(3))
icp = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, seed_fit)
print(f"\nICP result: sxy={icp.fit.scales[0]:.2f}, sz={icp.fit.scales[2]:.2f}")
pred_icp = ((cz_xyz * icp.fit.scales) @ icp.fit.R.T) + icp.fit.translation
print(f"ICP pred CZ centroid (xyz): {pred_icp.mean(0)}")
print(f"Dist GT→ICP predicted centroids: {np.linalg.norm(pred_gt.mean(0) - pred_icp.mean(0)):.1f} µm")

# Plot X-Y projection
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.scatter(hcr_xyz[:, 0], hcr_xyz[:, 1], s=0.5, c="#bbb", label="HCR GFP+")
ax.scatter(hcr_gt[:, 0], hcr_gt[:, 1], s=4, c="red", label="HCR GT pairs", alpha=0.7)
pred_all_gt = ((cz_xyz * fit_gt.scales) @ fit_gt.R.T) + fit_gt.translation
ax.scatter(pred_all_gt[:, 0], pred_all_gt[:, 1], s=2, c="green", label="CZ @ GT fit", alpha=0.5)
ax.scatter(pred_icp[:, 0], pred_icp[:, 1], s=2, c="blue", label="CZ @ ICP fit", alpha=0.5)
ax.set_title("782149 X-Y: HCR + GT-CZ vs ICP-CZ")
ax.legend(fontsize=7)
ax.set_aspect("equal")
ax.set_xlabel("X µm"); ax.set_ylabel("Y µm")

ax = axes[1]
ax.scatter(hcr_xyz[:, 0], hcr_xyz[:, 2], s=0.5, c="#bbb", label="HCR GFP+")
ax.scatter(hcr_gt[:, 0], hcr_gt[:, 2], s=4, c="red", label="HCR GT pairs", alpha=0.7)
ax.scatter(pred_all_gt[:, 0], pred_all_gt[:, 2], s=2, c="green", label="CZ @ GT fit", alpha=0.5)
ax.scatter(pred_icp[:, 0], pred_icp[:, 2], s=2, c="blue", label="CZ @ ICP fit", alpha=0.5)
ax.set_title("782149 X-Z")
ax.legend(fontsize=7)
ax.set_aspect("equal")
ax.set_xlabel("X µm"); ax.set_ylabel("Z µm")

ax = axes[2]
ax.scatter(hcr_xyz[:, 1], hcr_xyz[:, 2], s=0.5, c="#bbb", label="HCR GFP+")
ax.scatter(hcr_gt[:, 1], hcr_gt[:, 2], s=4, c="red", label="HCR GT pairs", alpha=0.7)
ax.scatter(pred_all_gt[:, 1], pred_all_gt[:, 2], s=2, c="green", label="CZ @ GT fit", alpha=0.5)
ax.scatter(pred_icp[:, 1], pred_icp[:, 2], s=2, c="blue", label="CZ @ ICP fit", alpha=0.5)
ax.set_title("782149 Y-Z")
ax.legend(fontsize=7)
ax.set_aspect("equal")
ax.set_xlabel("Y µm"); ax.set_ylabel("Z µm")

plt.suptitle("782149: where does ICP converge vs where should it?")
plt.tight_layout()
plt.savefig("diag_782149.png", dpi=110)
print("\nSaved diag_782149.png")
