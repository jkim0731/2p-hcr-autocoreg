"""Step-by-step walkthrough of the winning HCR and CZ protocols, each
illustrated on one representative subject.

HCR protocol (N11 - multi-channel image in ROI-quadratic band):
  1. ROI centroids density-filtered + per-tile min-z envelope.
  2. Robust quadratic anchor surface through that envelope.
  3. Load every HCR channel; pointwise max-intensity combination.
  4. Inside the +/- band (150 um) around the anchor, per-column relative
     threshold finds the shallowest above-threshold z.
  5. Robust quadratic fit through the image-derived per-column top-z.
  6. ROI envelope safety clamp + small offset - final surface on MIP.

Subject: 790322 (has the localized curvature that broke earlier methods).

CZ protocol (unchanged):
  1. CZ z-stack MIP.
  2. Per-column top-of-signal plane.
  3. Per-tile ROI min-z.
  4. Clamp plane above ROI tiles + safety.
Subject: 767022.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_analysis import (
    filter_in_tissue,
    estimate_pia_surface_from_image,
    list_hcr_channels,
    load_hcr_volume,
    load_hcr_y_slab,
    load_cz_y_slab,
    plot_pia_overlay,
)
from benchmark_data_loader import cz_px_to_um, hcr_px_to_um, load_subject

import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "surface_iteration_v2",
    Path(__file__).parent / "03_surface_iteration_v2.py",
)
_mod = _iu.module_from_spec(_spec)
sys.modules["surface_iteration_v2"] = _mod
_spec.loader.exec_module(_mod)

roi_quadratic_ceiling = _mod.roi_quadratic_ceiling
multi_channel_image_in_band = _mod.multi_channel_image_in_band
cz_image_with_roi_ceiling = _mod.cz_image_with_roi_ceiling
load_cz_image = _mod.load_cz_image
_robust_quadratic_fit = _mod._robust_quadratic_fit

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# HCR walkthrough - 790322
# ---------------------------------------------------------------
HCR_SID = "790322"
BAND_UM = 150.0
REL_MARGIN = 0.25


def _surf_z(s, x, y):
    return (s["a"] * x + s["b"] * y + s["c"]
            + s.get("p", 0) * x * x + s.get("q", 0) * x * y
            + s.get("r", 0) * y * y)


def walkthrough_hcr():
    s = load_subject(HCR_SID)
    hcr_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    xyz = hcr_um[:, [2, 1, 0]]

    # Step 1-2: ROI envelope + anchor quadratic
    anchor = roi_quadratic_ceiling(
        xyz, safety_offset_um=3.0, max_residual_quantile=1.0)
    keep = filter_in_tissue(xyz, radius_um=30.0, min_neighbors=3)
    pts_f = xyz[keep]
    xs_, ys_, zs_ = pts_f[:, 0], pts_f[:, 1], pts_f[:, 2]
    xb = (xs_ // 120.0).astype(int); yb = (ys_ // 120.0).astype(int)
    key = yb * 100000 + xb
    agg = pd.DataFrame({"k": key, "x": xs_, "y": ys_, "z": zs_}).groupby("k").agg(
        x=("x", "median"), y=("y", "median"),
        z=("z", lambda v: float(np.quantile(v, 0.02))),
        n=("z", "size"))
    agg = agg[agg["n"] >= 5].reset_index(drop=True)
    tx, ty, tz = agg["x"].values, agg["y"].values, agg["z"].values

    # Step 3-5: final N11 surface
    final = multi_channel_image_in_band(
        s, band_um=BAND_UM, relative_margin=REL_MARGIN)

    # Visualization at one y-slab
    y0 = float(np.median(xyz[:, 1]))
    half = 40.0
    in_slab = np.abs(xyz[:, 1] - y0) <= half
    tile_in_slab = np.abs(ty - y0) <= half

    channels = list_hcr_channels(s)
    # Build per-channel MIPs at the slab location
    channel_mips = {}
    for ch in channels:
        try:
            mip, _, z_um, xy_um = load_hcr_y_slab(
                s, channel=ch, y_center_um=y0, half_width_um=half, level=4)
            channel_mips[ch] = (mip, z_um, xy_um)
        except FileNotFoundError:
            continue
    if not channel_mips:
        print("No MIPs; skipping")
        return
    # Use first channel's shape for combined evidence MIP
    ref_ch = channels[0]
    ref_mip, z_um, xy_um = channel_mips[ref_ch]
    # Normalized-max across channels for a single evidence MIP
    norm_mips = []
    for ch, (mip, _, _) in channel_mips.items():
        m = mip.astype(np.float32)
        lo, hi = np.percentile(m, [1, 99.5])
        norm = np.clip((m - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        norm_mips.append(norm)
    max_mip = np.maximum.reduce(norm_mips)

    xs_line = np.linspace(0, ref_mip.shape[1] * xy_um, 400)
    ys_line = np.full_like(xs_line, y0)
    anchor_line = _surf_z(anchor, xs_line, ys_line)
    final_line = _surf_z(final, xs_line, ys_line)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    extent = (0, ref_mip.shape[1] * xy_um, ref_mip.shape[0] * z_um, 0)

    # 1. ROI envelope (tile min-z) + density-filtered ROIs
    ax = axes[0, 0]
    ax.scatter(pts_f[np.abs(pts_f[:, 1] - y0) <= half, 0],
               pts_f[np.abs(pts_f[:, 1] - y0) <= half, 2],
               s=1, c="lightgray", alpha=0.5)
    ax.scatter(tx[tile_in_slab], tz[tile_in_slab],
               s=18, c="red", edgecolor="k", zorder=3)
    ax.invert_yaxis()
    ax.set_title(f"1. ROI tile min-z envelope\n({len(agg)} tiles, q=0.02)")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")

    # 2. Quadratic anchor + search band
    ax = axes[0, 1]
    ax.scatter(tx[tile_in_slab], tz[tile_in_slab],
               s=18, c="red", edgecolor="k", zorder=3, label="tile min-z")
    ax.plot(xs_line, anchor_line, "b-", lw=1.4,
            label=f"anchor quad (tilt={anchor['tilt_deg']:.1f}°)")
    ax.fill_between(xs_line, anchor_line - BAND_UM, anchor_line + BAND_UM,
                    color="blue", alpha=0.1, label=f"+/-{BAND_UM:.0f} um band")
    ax.invert_yaxis()
    ax.set_title("2. ROI quadratic anchor + search band")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    # 3. Multi-channel max-intensity MIP
    ax = axes[0, 2]
    ax.imshow(max_mip, extent=extent, cmap="gray",
              aspect="auto", interpolation="nearest")
    ax.set_title(f"3. Multi-channel max MIP\n"
                 f"({len(channel_mips)} channels combined)")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")

    # 4. Per-column top-of-signal inside band (show anchor band over max MIP)
    ax = axes[1, 0]
    ax.imshow(max_mip, extent=extent, cmap="gray",
              aspect="auto", interpolation="nearest")
    ax.plot(xs_line, anchor_line, "b-", lw=1.0, alpha=0.7, label="anchor")
    ax.fill_between(xs_line, anchor_line - BAND_UM, anchor_line + BAND_UM,
                    color="blue", alpha=0.12)
    ax.set_title("4. Band-constrained per-column\ntop-of-signal search")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    # 5. Robust quadratic fit through image top-z + final line
    ax = axes[1, 1]
    ax.imshow(max_mip, extent=extent, cmap="gray",
              aspect="auto", interpolation="nearest")
    ax.plot(xs_line, anchor_line, "b-", lw=1.0, alpha=0.5, label="anchor")
    ax.plot(xs_line, final_line, "red", lw=1.8,
            label=f"N11 final (tilt={final['tilt_deg']:.1f}°)")
    ax.set_title("5. Robust quadratic fit through\n"
                 "image top-z + safety clamp")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    # 6. Final surface with ROI overlay
    ax = axes[1, 2]
    plot_pia_overlay(ax, max_mip, z_um=z_um, x_um=xy_um, surface=final,
                     y_slab_um=y0, title="6. Final surface on MIP")
    ax.scatter(xyz[in_slab, 0], xyz[in_slab, 2],
               s=0.7, c="cyan", alpha=0.4)
    leg = ax.get_legend()
    if leg: leg.remove()

    fig.suptitle(
        f"HCR N11 protocol - subject {HCR_SID} (y = {y0:.0f} um slab)\n"
        f"channels combined = {sorted(channel_mips)}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "hcr_walkthrough.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------
# CZ walkthrough - 767022 (unchanged)
# ---------------------------------------------------------------
CZ_SID = "767022"
CZ_TILE_UM = 80.0


def walkthrough_cz():
    s = load_subject(CZ_SID)
    cz_um = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].values, s)
    xyz = cz_um[:, [2, 1, 0]]
    img = load_cz_image(s)

    base = estimate_pia_surface_from_image(
        img, s.cz_z_um, s.cz_xy_um,
        min_signal_abs=50.0, relative_margin=0.02,
    )
    surf = cz_image_with_roi_ceiling(
        xyz, img, s.cz_z_um, s.cz_xy_um,
        margin=0.02, safety_offset_um=3.0,
    )

    xs_, ys_, zs_ = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xb = (xs_ // CZ_TILE_UM).astype(int)
    yb = (ys_ // CZ_TILE_UM).astype(int)
    key = yb * 100000 + xb
    agg = pd.DataFrame({"k": key, "x": xs_, "y": ys_, "z": zs_}).groupby("k").agg(
        x=("x", "median"), y=("y", "median"),
        z=("z", "min"), n=("z", "size"))
    agg = agg[agg["n"] >= 3].reset_index(drop=True)
    tx, ty, tz = agg["x"].values, agg["y"].values, agg["z"].values

    y_center_px = int(xyz[:, 1].mean() / s.cz_xy_um)
    mip, y_px, z_um, xy_um = load_cz_y_slab(
        s, y_center_px=y_center_px, half_width_px=20)
    y0 = y_px * xy_um
    in_slab = np.abs(xyz[:, 1] - y0) <= 15 * s.cz_xy_um
    tile_in_slab = np.abs(ty - y0) <= 15 * s.cz_xy_um

    xs_line = np.linspace(0, mip.shape[1] * xy_um, 300)
    base_line = base["a"] * xs_line + base["b"] * y0 + base["c"]
    final_line = surf["a"] * xs_line + surf["b"] * y0 + surf["c"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    extent = (0, mip.shape[1] * xy_um, mip.shape[0] * z_um, 0)
    vmin, vmax = np.percentile(mip, [1, 99.5])

    ax = axes[0, 0]
    ax.imshow(mip, extent=extent, cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.set_title("1. CZ z-stack MIP")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")

    ax = axes[0, 1]
    ax.imshow(mip, extent=extent, cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.plot(xs_line, base_line, "orange", lw=1.5,
            label=f"image plane (2% margin, tilt={base['tilt_deg']:.1f} deg)")
    ax.set_title("2. Image-based plane fit")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.imshow(mip, extent=extent, cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.scatter(xyz[in_slab, 0], xyz[in_slab, 2], s=1, c="cyan", alpha=0.4)
    ax.scatter(tx[tile_in_slab], tz[tile_in_slab],
               s=18, c="red", edgecolor="k", zorder=3, label="tile min-z")
    ax.plot(xs_line, base_line, "orange", lw=1.2, label="image plane")
    ax.set_title(f"3. Per-tile ROI min-z (80 um tiles)")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.imshow(mip, extent=extent, cmap="gray", vmin=vmin, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.scatter(xyz[in_slab, 0], xyz[in_slab, 2], s=1, c="cyan", alpha=0.4)
    ax.plot(xs_line, base_line, "orange", lw=1.0, alpha=0.7,
            label="image plane (before clamp)")
    ax.plot(xs_line, final_line, "red", lw=1.8,
            label=f"final (lift={surf['lift_um']:.1f} um, off=3 um)")
    ax.set_title("4. Ceiling clamp + safety - final surface")
    ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
    ax.legend(fontsize=7)

    fig.suptitle(f"CZ protocol walkthrough - subject {CZ_SID}"
                 f" (y = {y0:.0f} um slab)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "cz_walkthrough.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    walkthrough_hcr()
    walkthrough_cz()
