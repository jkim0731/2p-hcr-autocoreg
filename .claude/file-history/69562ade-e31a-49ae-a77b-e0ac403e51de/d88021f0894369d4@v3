"""Compare top surface-estimation variants on 790322 and cross-validate
them on all benchmark subjects.

After the exploration in 03_surface_790322_explore.py, the best variants
on 790322 (by onset_depth / x_spread) are:

  A. N11_band150            (existing, onset 77.5, x_spread 50)
  B. N13_n4anchor_b150      (N4 image plane as anchor, onset 77.5, x_spread 15)
  C. N14_iter_150_80        (N11 then re-search ±80 around first fit,
                              onset 67.5, x_spread 25)
  D. N14_iter_n4_200_80     (N4 anchor → iterate, onset 77.5, x_spread 15)

This script produces:
  1. per-subject detail figure for 790322 with:
     - per-variant y-slab MIP + surface
     - zoom on x=1800-2300 (the dip region)
     - per-variant depth density profile
  2. all-subject summary: 6-subject depth profile grid
     (one panel per subject; one curve per variant)
  3. stats table across all subjects
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
    depth_profile, list_hcr_channels,
    load_hcr_combined, load_hcr_y_slab,
)
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS, hcr_px_to_um, load_subject,
)

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
v_n21_n4_quantile = _expl_mod.v_n21_n4_quantile
v_n22_n4_quantile_loose_clamp = _expl_mod.v_n22_n4_quantile_loose_clamp
_surf_z = _expl_mod._surf_z
quality_metrics = _expl_mod.quality_metrics

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    ("N11_roi_quad",
        lambda s, xyz, vol, xy, z: v_n11(s, xyz)),
    ("N13_n4anchor_b150",
        lambda s, xyz, vol, xy, z:
            v_n13_n4_anchor(s, xyz, vol, xy, z, band_um=150.0)),
    ("N21_quantile_q70",
        lambda s, xyz, vol, xy, z:
            v_n21_n4_quantile(s, xyz, vol, xy, z,
                              band_um=150.0, target_quantile=0.70)),
    ("N22_quantile_q70_wtq10",
        lambda s, xyz, vol, xy, z:
            v_n22_n4_quantile_loose_clamp(s, xyz, vol, xy, z,
                                          band_um=150.0,
                                          target_quantile=0.70,
                                          within_tile_q=0.10)),
]
COLORS = {
    "N11_roi_quad":             "#1f77b4",
    "N13_n4anchor_b150":        "#2ca02c",
    "N21_quantile_q70":         "#ff7f0e",
    "N22_quantile_q70_wtq10":   "#d62728",
}


def _fit_all(s, hcr_xyz, vol, xy_um, z_um):
    out = {}
    for name, fn in VARIANTS:
        try:
            surf = fn(s, hcr_xyz, vol, xy_um, z_um)
        except Exception as e:
            print(f"    {name}: FAILED {e!r}")
            surf = None
        out[name] = surf
    return out


def _combined_mip(s, y0, half=40.0):
    channels = list_hcr_channels(s)
    mips = []
    xy_mip = z_mip = None
    for ch in channels:
        try:
            mip, _, z_mip, xy_mip = load_hcr_y_slab(
                s, channel=ch, y_center_um=y0, half_width_um=half, level=4)
        except FileNotFoundError:
            continue
        m = mip.astype(np.float32)
        lo, hi = np.percentile(m, [1, 99.5])
        norm = np.clip((m - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        mips.append(norm)
    return np.maximum.reduce(mips) if mips else None, z_mip, xy_mip


def figure_790322_detail():
    sid = "790322"
    s = load_subject(sid)
    hcr_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    hcr_xyz = hcr_um[:, [2, 1, 0]]
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)

    surfs = _fit_all(s, hcr_xyz, vol, xy_um, z_um)

    # Pick 3 y positions: shallow, median, dip-y
    y_all = hcr_xyz[:, 1]
    ys = [float(np.quantile(y_all, 0.25)),
          float(np.quantile(y_all, 0.50)),
          float(np.quantile(y_all, 0.75))]

    # Build MIPs
    mips = []
    for y in ys:
        mip, zm, xym = _combined_mip(s, y, half=40.0)
        mips.append((y, mip, zm, xym))

    n_var = len(VARIANTS)
    n_y = len(ys)

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(n_y + 2, n_var,
                          height_ratios=[2.5] * n_y + [2.5, 2.0],
                          hspace=0.45, wspace=0.15)

    # Rows 0..n_y-1: each row = one y-slab, each col = one variant's
    # surface overlay on the same slab.
    for i, (y, mip, zm, xym) in enumerate(mips):
        extent = (0, mip.shape[1] * xym, mip.shape[0] * zm, 0)
        in_slab = np.abs(hcr_xyz[:, 1] - y) <= 40.0
        for j, (name, _) in enumerate(VARIANTS):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(mip, extent=extent, cmap="gray",
                      aspect="auto", interpolation="nearest")
            ax.scatter(hcr_xyz[in_slab, 0], hcr_xyz[in_slab, 2],
                       s=0.8, c="cyan", alpha=0.35)
            surf = surfs.get(name)
            if surf is not None:
                xs_line = np.linspace(0, mip.shape[1] * xym, 500)
                ys_line = np.full_like(xs_line, y)
                z_line = _surf_z(surf, xs_line, ys_line)
                ax.plot(xs_line, z_line, lw=1.6, color=COLORS[name])
            # Zoom out — show only the top 400 µm to make the surface clear
            ax.set_ylim(400, 0)
            if i == 0:
                ax.set_title(name, fontsize=9, color=COLORS[name])
            ax.text(0.02, 0.02, f"y={y:.0f}", transform=ax.transAxes,
                    fontsize=7, color="white",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1.5))
            ax.set_xlabel("x (µm)", fontsize=7)
            if j == 0:
                ax.set_ylabel("z (µm)", fontsize=7)
            ax.tick_params(labelsize=6)

    # Zoom row: x=1500-2400 (the dip region) at median y
    y_mid = ys[1]
    mip_mid, zm, xym = _combined_mip(s, y_mid, half=40.0)
    extent = (0, mip_mid.shape[1] * xym, mip_mid.shape[0] * zm, 0)
    for j, (name, _) in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[n_y, j])
        ax.imshow(mip_mid, extent=extent, cmap="gray",
                  aspect="auto", interpolation="nearest")
        in_slab = np.abs(hcr_xyz[:, 1] - y_mid) <= 40.0
        ax.scatter(hcr_xyz[in_slab, 0], hcr_xyz[in_slab, 2],
                   s=0.8, c="cyan", alpha=0.35)
        surf = surfs.get(name)
        if surf is not None:
            xs_line = np.linspace(1500, 2400, 300)
            ys_line = np.full_like(xs_line, y_mid)
            z_line = _surf_z(surf, xs_line, ys_line)
            ax.plot(xs_line, z_line, lw=2.0, color=COLORS[name])
        ax.set_xlim(1500, 2400)
        ax.set_ylim(500, 0)
        ax.set_title(f"ZOOM: dip region x=1500-2400  ({name})",
                     fontsize=8, color=COLORS[name])
        ax.set_xlabel("x (µm)", fontsize=7)
        if j == 0:
            ax.set_ylabel("z (µm)", fontsize=7)
        ax.tick_params(labelsize=6)

    # Depth profile row (spans full width)
    ax = fig.add_subplot(gs[n_y + 1, :])
    for name, _ in VARIANTS:
        surf = surfs.get(name)
        if surf is None:
            continue
        c, dens = depth_profile(
            hcr_xyz, surf, bin_um=10, depth_range=(-100, 400))
        if dens.max() > 0:
            ax.plot(c, dens / dens.max(), color=COLORS[name], lw=1.6,
                    label=name)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-100, 400)
    ax.set_xlabel("depth from pia (µm)")
    ax.set_ylabel("normalized ROI density")
    ax.set_title("ROI depth density (zoom: −100 to 400 µm)",
                 fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper right")

    fig.suptitle(f"790322 — surface variants comparison "
                 f"(rows: y-slabs; row {n_y}: x=1500-2400 zoom; "
                 f"bottom: density profiles)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = FIG / "compare_790322.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")

    # Stats for 790322
    rows = []
    for name, _ in VARIANTS:
        surf = surfs.get(name)
        if surf is None:
            continue
        q = quality_metrics(hcr_xyz, surf)
        rows.append({"subject": sid, "method": name,
                     "c": surf.get("c"), "tilt": surf.get("tilt_deg"),
                     **q})
    return pd.DataFrame(rows)


def figure_all_subjects():
    subjects = BENCHMARK_SUBJECTS
    stats_rows = []

    n = len(subjects)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for k, sid in enumerate(subjects):
        print(f"\n=== {sid} ===")
        ax = axes[k]
        s = load_subject(sid)
        hcr_um = hcr_px_to_um(
            s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
        hcr_xyz = hcr_um[:, [2, 1, 0]]
        vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
        surfs = _fit_all(s, hcr_xyz, vol, xy_um, z_um)

        for name, _ in VARIANTS:
            surf = surfs.get(name)
            if surf is None:
                continue
            q = quality_metrics(hcr_xyz, surf)
            stats_rows.append({"subject": sid, "method": name,
                               "c": surf.get("c"),
                               "tilt": surf.get("tilt_deg"),
                               **q})
            print(f"  {name}: c={surf.get('c'):.1f}  "
                  f"tilt={surf.get('tilt_deg',0):.2f}  "
                  f"above%={q['above_frac']*100:.3f}  "
                  f"onset={q['onset_depth_um']}  "
                  f"x_spread={q['onset_x_spread']}")
            c, dens = depth_profile(
                hcr_xyz, surf, bin_um=10, depth_range=(-100, 500))
            if dens.max() > 0:
                ax.plot(c, dens / dens.max(),
                        color=COLORS[name], lw=1.4, label=name)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(sid, fontsize=10)
        ax.set_xlabel("depth (µm)", fontsize=8)
        ax.set_ylabel("norm. density", fontsize=8)
        ax.set_xlim(-100, 500)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    for k in range(len(subjects), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Across-subject ROI density profiles per variant",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "compare_all_density.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")

    df = pd.DataFrame(stats_rows)
    df.to_csv(OUT / "compare_variants_stats.csv", index=False)
    print(f"Wrote {OUT / 'compare_variants_stats.csv'}")
    summary = df[["subject", "method", "c", "tilt", "above_frac",
                  "onset_depth_um", "onset_x_spread", "onset_y_spread"]]
    print("\n" + summary.to_string(index=False))
    return df


if __name__ == "__main__":
    print("\n### 790322 detail figure ###")
    det = figure_790322_detail()
    print(det.to_string(index=False))

    print("\n### All-subject comparison ###")
    figure_all_subjects()
