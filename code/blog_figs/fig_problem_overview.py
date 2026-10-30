"""fig_00 — The problem at a glance.

A ~400 µm in-vivo 2p (CZ) sub-volume must be localized inside a multi-mm
ex-vivo HCR light-sheet volume, under a ~180° rotation, anisotropic
expansion, nonrigid warp, and only partial cell correspondence. This figure
sets the scale: the located CZ footprint drawn to scale on the HCR 405 MIP,
with the CZ volume shown inset.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, "/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions/15_geom_features")
sys.path.insert(0, "/root/capsule/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions/16_intermediate_summary_materials")
from _common import COLORS, SID, load_s, mip, norm01, placed_cz_um

import benchmark_data_loader as _bdl
def _find_hcr_dir_processed(subject_id):          # exclude the *_HCR-ROI-label_* variant assets
    m = sorted(x for x in _bdl.DATA_DIR.glob(f"HCR_{subject_id}_*_processed_*") if "HCR-ROI-label" not in x.name)
    return m[-1]
_bdl._find_hcr_dir = _find_hcr_dir_processed
s = load_s()

from benchmark_analysis import load_hcr_volume
from cz_volume import load_cz_volume

hcr_vol, hcr_xy, hcr_z = load_hcr_volume(s, channel="405", level=4)
hcr_mip = norm01(mip(hcr_vol, 0))          # top-down xy
cz_vol = load_cz_volume(s)
cz_mip = norm01(mip(cz_vol, 0))

# located CZ footprint in HCR µm -> level-4 MIP pixels
p = placed_cz_um(s)
x_px = p[:, 2] / hcr_xy
y_px = p[:, 1] / hcr_xy
x0, x1 = x_px.min(), x_px.max()
y0, y1 = y_px.min(), y_px.max()

fig = plt.figure(figsize=(13, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.12)

# --- CZ inset (the source sub-volume) -----------------------------------
axc = fig.add_subplot(gs[0, 0])
axc.imshow(cz_mip, cmap="gray", aspect="equal")
axc.set_title("Two-photon z-stack", fontsize=17, fontweight="bold", pad=10)
axc.set_axis_off(); axc.set_anchor("N")
# scale bar 100 µm
sb = 100 / s.cz_xy_um
axc.plot([20, 20 + sb], [cz_mip.shape[0] - 25] * 2, color="w", lw=4)
axc.text(20 + sb / 2, cz_mip.shape[0] - 35, "100 µm", color="w", ha="center", fontsize=12)

# --- HCR MIP with located footprint -------------------------------------
axh = fig.add_subplot(gs[0, 1])
axh.imshow(hcr_mip, cmap="gray", aspect="equal")
axh.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                         edgecolor=COLORS["cz"], lw=2.5))
H, W = hcr_mip.shape
axh.set_title("HCR light-sheet volume (405 Rn28S)", fontsize=17, fontweight="bold", pad=10)
axh.set_axis_off(); axh.set_anchor("N")
sb = 500 / hcr_xy
axh.plot([20, 20 + sb], [H - 22] * 2, color="w", lw=4)
axh.text(20 + sb / 2, H - 32, "500 µm", color="w", ha="center", fontsize=12)

OUT = "/scratch/tmp/site/posts/2026-10-auto-coregistration-1-pan-inhibitory/figures/fig_00_problem_overview.jpg"
fig.savefig("/scratch/tmp/diagram/fig_00_problem_overview.png", dpi=200, bbox_inches="tight", facecolor="white")
from PIL import Image
im = Image.open("/scratch/tmp/diagram/fig_00_problem_overview.png").convert("RGB"); im.thumbnail((1800, 1800)); im.save(OUT, quality=88)
print("saved", OUT)
frac = (x1 - x0) * (y1 - y0) / (W * H)
print(f"CZ footprint covers {frac*100:.1f}% of HCR MIP area")
