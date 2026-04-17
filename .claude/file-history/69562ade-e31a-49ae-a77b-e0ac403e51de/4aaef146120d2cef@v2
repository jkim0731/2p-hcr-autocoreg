"""Diagnose why grid-smoothed approaches (N17/N18) regress on 782149.

Show per-column top-of-signal raw points on y-slab MIP, overlaid with
the quadratic fit (N11) and grid/residual surfaces (N17/N18).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_analysis import load_hcr_combined, load_hcr_y_slab, list_hcr_channels
from benchmark_data_loader import hcr_px_to_um, load_subject

import importlib.util as _iu
_expl = _iu.spec_from_file_location(
    "explore_790322",
    Path(__file__).parent / "03_surface_790322_explore.py",
)
_mod = _iu.module_from_spec(_expl)
sys.modules["explore_790322"] = _mod
_expl.loader.exec_module(_mod)
v_n11 = _mod.v_n11
v_n17_grid_smooth = _mod.v_n17_grid_smooth
v_n18_quad_plus_grid_residual = _mod.v_n18_quad_plus_grid_residual
_surf_z = _mod._surf_z
_per_column_top_in_band = _mod._per_column_top_in_band
_load_multichannel = _mod._load_multichannel

import importlib.util as _iu2
_sspec = _iu2.spec_from_file_location(
    "surface_iteration_v2",
    Path(__file__).parent / "03_surface_iteration_v2.py",
)
_sim = _iu2.module_from_spec(_sspec)
sys.modules["surface_iteration_v2"] = _sim
_sspec.loader.exec_module(_sim)
roi_quadratic_ceiling = _sim.roi_quadratic_ceiling
image_with_roi_ceiling = _sim.image_with_roi_ceiling

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"

SID = "782149"


def main():
    s = load_subject(SID)
    hcr_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    hcr_xyz = hcr_um[:, [2, 1, 0]]

    vols_sub, xx, yy, xy_um_sub, z_um, channels = _load_multichannel(s)

    # Anchors
    n6_anchor = roi_quadratic_ceiling(
        hcr_xyz, safety_offset_um=3.0, max_residual_quantile=1.0)
    vol, xy_um, z_um_c, _ = load_hcr_combined(s, level=4)

    # Per-column tops around the N6 anchor ±150 (matches N17/N18 band)
    anchor_zz = _surf_z(n6_anchor, xx, yy)
    z_low = anchor_zz - 150.0; z_high = anchor_zz + 150.0
    tz_band150 = _per_column_top_in_band(
        vols_sub, xy_um_sub, z_um, xx, yy, z_low, z_high,
        relative_margin=0.25, min_signal_abs_frac=0.20, min_thick_um=10.0)

    # Surfaces
    surf_n11 = v_n11(s, hcr_xyz)
    surf_n17 = v_n17_grid_smooth(
        s, hcr_xyz, vol, xy_um, z_um_c,
        smooth_sigma_um=120.0, anchor_source="n4")
    surf_n18 = v_n18_quad_plus_grid_residual(
        s, hcr_xyz, vol, xy_um, z_um_c,
        residual_sigma_um=120.0)

    # y-slab MIP
    y0 = float(np.median(hcr_xyz[:, 1]))
    half = 40.0
    mips = []
    xy_mip = z_mip = None
    for ch in channels:
        try:
            m, _, z_mip, xy_mip = load_hcr_y_slab(
                s, channel=ch, y_center_um=y0, half_width_um=half, level=4)
        except FileNotFoundError:
            continue
        mf = m.astype(np.float32)
        lo, hi = np.percentile(mf, [1, 99.5])
        mips.append(np.clip((mf - lo) / max(hi - lo, 1.0), 0.0, 1.0))
    max_mip = np.maximum.reduce(mips)
    extent = (0, max_mip.shape[1] * xy_mip, max_mip.shape[0] * z_mip, 0)

    y_slab_mask = np.abs(yy - y0) <= 50.0
    x_col = xx[y_slab_mask]
    z_col = tz_band150[y_slab_mask]
    valid = np.isfinite(z_col)

    xs_line = np.linspace(0, max_mip.shape[1] * xy_mip, 500)
    ys_line = np.full_like(xs_line, y0)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for ax, title in zip(
            axes, ["full view", "zoom z=0-400"]):
        ax.imshow(max_mip, extent=extent, cmap="gray",
                  aspect="auto", interpolation="nearest")
        ax.scatter(x_col[valid], z_col[valid], s=4, c="yellow",
                   alpha=0.7, label="per-col top-of-signal (band 150)")
        ax.plot(xs_line, _surf_z(surf_n11, xs_line, ys_line),
                lw=1.8, color="red", label="N11 (quadratic)")
        ax.plot(xs_line, _surf_z(surf_n17, xs_line, ys_line),
                lw=1.8, color="magenta", label="N17 (grid sig120)")
        ax.plot(xs_line, _surf_z(surf_n18, xs_line, ys_line),
                lw=1.8, color="lime", label="N18 (quad+resid sig120)")
        ax.set_xlim(0, max_mip.shape[1] * xy_mip)
        if title == "zoom z=0-400":
            ax.set_ylim(400, 0)
        else:
            ax.set_ylim(max_mip.shape[0] * z_mip, 0)
        ax.set_title(f"{SID} — {title}")
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"{SID} (12° tilt) — N11 vs N17/N18 vs per-column tops "
        f"(y={y0:.0f} µm slab)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / f"diagnose_{SID}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
