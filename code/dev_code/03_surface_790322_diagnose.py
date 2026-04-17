"""Diagnose 790322's dip: show per-column top-of-signal raw points
on top of the MIP, with the quadratic fits overlaid.

If the column tops follow the dip but the quadratic flattens it, the
problem is the polynomial order. If the column tops miss the dip, the
image evidence (or the search band) is the limiting factor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_analysis import list_hcr_channels, load_hcr_volume, load_hcr_y_slab
from benchmark_data_loader import hcr_px_to_um, load_subject

import importlib.util as _iu
_expl = _iu.spec_from_file_location(
    "explore_790322",
    Path(__file__).parent / "03_surface_790322_explore.py",
)
_expl_mod = _iu.module_from_spec(_expl)
sys.modules["explore_790322"] = _expl_mod
_expl.loader.exec_module(_expl_mod)
v_n11 = _expl_mod.v_n11
v_n13_n4_anchor = _expl_mod.v_n13_n4_anchor
v_n14_iterative = _expl_mod.v_n14_iterative
_surf_z = _expl_mod._surf_z
_per_column_top_in_band = _expl_mod._per_column_top_in_band
_load_multichannel = _expl_mod._load_multichannel
_clamp_to_roi_envelope = _expl_mod._clamp_to_roi_envelope
_robust_quadratic_fit = _expl_mod._robust_quadratic_fit

import importlib.util as _iu2
_mod_spec = _iu2.spec_from_file_location(
    "surface_iteration_v2",
    Path(__file__).parent / "03_surface_iteration_v2.py",
)
_sim = _iu2.module_from_spec(_mod_spec)
sys.modules["surface_iteration_v2"] = _sim
_mod_spec.loader.exec_module(_sim)
image_with_roi_ceiling = _sim.image_with_roi_ceiling
roi_quadratic_ceiling = _sim.roi_quadratic_ceiling

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"

SID = "790322"


def main():
    s = load_subject(SID)
    hcr_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    hcr_xyz = hcr_um[:, [2, 1, 0]]

    # Load all channels subsampled
    vols_sub, xx, yy, xy_um_sub, z_um, channels = _load_multichannel(s)
    Zs = vols_sub[0].shape[0]

    # Anchors
    n6_anchor = roi_quadratic_ceiling(
        hcr_xyz, safety_offset_um=3.0, max_residual_quantile=1.0)
    # Need a vol to build N4 anchor; grab first channel full volume at level 4
    from benchmark_analysis import load_hcr_combined
    vol, xy_um, z_um_c, _ = load_hcr_combined(s, level=4)
    n4_anchor = image_with_roi_ceiling(
        hcr_xyz, vol, z_um_c, xy_um,
        margin=0.005, safety_offset_um=0.0)

    # Three band setups + their top-of-signal
    setups = {
        "N6_anchor_b150": (n6_anchor, 150.0, 150.0),
        "N4_anchor_b150": (n4_anchor, 150.0, 150.0),
        "N6_anchor_b300below": (n6_anchor, 300.0, 50.0),
    }
    tops = {}
    for name, (anchor, below, above) in setups.items():
        anchor_zz = _surf_z(anchor, xx, yy)
        z_low = anchor_zz - below; z_high = anchor_zz + above
        tz = _per_column_top_in_band(
            vols_sub, xy_um_sub, z_um, xx, yy, z_low, z_high,
            relative_margin=0.25, min_signal_abs_frac=0.20,
            min_thick_um=10.0)
        tops[name] = tz

    # Also get each fit's surface (quadratic)
    surfs = {
        "N11_band150": v_n11(s, hcr_xyz),
        "N13_n4anchor_b150": v_n13_n4_anchor(
            s, hcr_xyz, vol, xy_um, z_um_c, band_um=150),
        "N14_iter_150_80": v_n14_iterative(s, hcr_xyz, band1=150, band2=80),
    }

    # Build MIP at median y
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

    # For each (xx, yy) column in the subsampled grid near y=y0, gather
    # (x, top_z) pairs to overlay as scatter
    y_slab_mask = np.abs(yy - y0) <= 50.0

    n_setups = len(setups)
    fig, axes = plt.subplots(n_setups, 2, figsize=(16, 3.5 * n_setups))
    for row, (name, (anchor, below, above)) in enumerate(setups.items()):
        tz = tops[name]
        x_col = xx[y_slab_mask]
        z_col = tz[y_slab_mask]
        valid = np.isfinite(z_col)
        # Anchor at y=y0
        xs_line = np.linspace(0, max_mip.shape[1] * xy_mip, 400)
        ys_line = np.full_like(xs_line, y0)
        anchor_line = _surf_z(anchor, xs_line, ys_line)

        # Full view
        ax = axes[row, 0]
        ax.imshow(max_mip, extent=extent, cmap="gray",
                  aspect="auto", interpolation="nearest")
        ax.scatter(x_col[valid], z_col[valid], s=3, c="yellow",
                   alpha=0.7, label="per-col top-of-signal")
        ax.plot(xs_line, anchor_line, "r-", lw=1.2, label="anchor")
        ax.fill_between(xs_line, anchor_line - below, anchor_line + above,
                        color="red", alpha=0.12)
        # Show the surfaces
        for sname, surf in surfs.items():
            if surf is None:
                continue
            ax.plot(xs_line, _surf_z(surf, xs_line, ys_line),
                    lw=1.4, label=sname)
        ax.set_xlim(0, max_mip.shape[1] * xy_mip)
        ax.set_ylim(400, 0)
        ax.set_title(f"{name}  ({len(np.where(valid)[0])} cols) — full view",
                     fontsize=9)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        ax.legend(fontsize=7, loc="upper left")

        # Zoom on dip region
        ax = axes[row, 1]
        ax.imshow(max_mip, extent=extent, cmap="gray",
                  aspect="auto", interpolation="nearest")
        ax.scatter(x_col[valid], z_col[valid], s=6, c="yellow",
                   alpha=0.8, label="per-col top")
        ax.plot(xs_line, anchor_line, "r-", lw=1.2, label="anchor")
        ax.fill_between(xs_line, anchor_line - below, anchor_line + above,
                        color="red", alpha=0.12)
        for sname, surf in surfs.items():
            if surf is None:
                continue
            ax.plot(xs_line, _surf_z(surf, xs_line, ys_line),
                    lw=1.5, label=sname)
        ax.set_xlim(1500, 2400)
        ax.set_ylim(500, 0)
        ax.set_title(f"{name} — ZOOM x=1500-2400", fontsize=9)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"{SID} — per-column top-of-signal vs quadratic fits "
        f"(y={y0:.0f} µm slab)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "diagnose_790322.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
