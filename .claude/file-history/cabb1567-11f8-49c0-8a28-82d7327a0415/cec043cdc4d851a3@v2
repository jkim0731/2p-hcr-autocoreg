"""Iteration 5 compute — render fitted surfaces overlaid on combined HCR
volume slices.  Audit for out-of-tissue (below / side) regions that
could perturb the Q / Qcon_n column-median scoring.

For every subject produces one figure with 2 × 2 panels:

    top-left   : XZ mid-Y slice, combined normalised volume
    top-right  : XZ mid-Y slice, 488 channel raw
    bot-left   : YZ mid-X slice, combined normalised volume
    bot-right  : YZ mid-X slice, 488 channel raw

All panels overlay (i) N22 centroid-based surface (green), (ii) v2.1
default surface (orange), (iii) auto-selected surface (blue).  We also
overlay the sampling grid used by score_surface_quality so we can see
whether any grid columns fall in OOT.

Outputs
-------
- figures/iter05_surfaces_<sid>.png  — one per subject
- data/iter05_audit.csv              — per-subject OOT stats
  (tissue thickness distribution, OOT column fraction under current grid)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

_spec = importlib.util.spec_from_file_location(
    "img_surface_v21",
    str(ROOT / "code" / "dev_code" / "03_image_based_surface.py"),
)
_img = importlib.util.module_from_spec(_spec)
sys.modules["img_surface_v21"] = _img
_spec.loader.exec_module(_img)

from benchmark_analysis import (
    estimate_pia_surface_image_autoselect,
    estimate_pia_surface_image_ceiling,
    estimate_pia_surface_quantile_ceiling,
    list_hcr_channels,
    load_hcr_combined,
    load_hcr_volume,
)
from benchmark_data_loader import hcr_px_to_um, load_subject

OUT_DATA = ROOT / "code" / "sessions" / "03c_onset_features" / "data"
OUT_FIG  = ROOT / "code" / "sessions" / "03c_onset_features" / "figures"

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]

N_SIDE = 20
EDGE_FRAC = 0.15


def surface_z(surface, x, y):
    return _img._surface_z(surface, x, y)


def get_n22(s, vol, xy_um, z_um, hcr_xyz):
    anchor = estimate_pia_surface_image_ceiling(
        hcr_xyz, vol, z_um, xy_um,
        relative_margin=0.005, min_signal_abs=0.05, safety_offset_um=0.0)
    if anchor is None:
        return None
    channels = list_hcr_channels(s)
    per_ch = []
    for ch in channels:
        try:
            v_ch, _, _ = load_hcr_volume(s, channel=ch, level=4)
            per_ch.append(v_ch.astype(np.float32, copy=False))
        except FileNotFoundError:
            continue
    sxy = max(1, int(round(10.0 / xy_um)))
    vols_sub = [v[:, ::sxy, ::sxy] for v in per_ch]
    _, Ys, Xs = vols_sub[0].shape
    xs_sub = np.arange(Xs) * sxy * xy_um
    ys_sub = np.arange(Ys) * sxy * xy_um
    xx_sub, yy_sub = np.meshgrid(xs_sub, ys_sub)
    return estimate_pia_surface_quantile_ceiling(
        hcr_xyz, vols_sub, z_um, sxy * xy_um, xx_sub, yy_sub, anchor)


def sampling_grid(vol_shape, xy_um):
    Z, Y, X = vol_shape
    x_pad = int(EDGE_FRAC * X); y_pad = int(EDGE_FRAC * Y)
    xs_i = np.linspace(x_pad, X - 1 - x_pad, N_SIDE).astype(int)
    ys_i = np.linspace(y_pad, Y - 1 - y_pad, N_SIDE).astype(int)
    xi, yi = np.meshgrid(xs_i, ys_i)
    return xi.ravel(), yi.ravel()


def tissue_support(vol, thr_frac=0.1, ref_percentile=99.5):
    """Per-column max; columns below thr_frac of the global ref are OOT.

    Reference is ``np.percentile(colmax, ref_percentile)`` (robust to
    single-pixel saturation, samples a typical bright *column*).  Using
    ``colmax.max() = vol.max()`` directly is wrong on HCR — saturated
    voxels inflate it 30–250 × above a typical bright column; see iter 6
    log entry for the full reasoning.
    """
    colmax = vol.max(axis=0)           # (Y, X)
    glob = float(np.percentile(colmax, ref_percentile))
    thr = thr_frac * glob
    return colmax >= thr, colmax, glob, thr


def render_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol_comb, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    vol_488, xy_um_488, z_um_488 = load_hcr_volume(s, channel="488", level=4)
    hcr_xyz = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]

    n22 = get_n22(s, vol_comb, xy_um, z_um, hcr_xyz)
    v21 = _img.estimate_surface_and_l2_image_based(
        vol_comb, z_um, xy_um, target_quantile=0.85).surface
    auto_surf, info, scores = estimate_pia_surface_image_autoselect(s, level=4)

    Z, Y, X = vol_comb.shape
    y_mid = Y // 2
    x_mid = X // 2
    x_um = np.arange(X) * xy_um
    y_um = np.arange(Y) * xy_um
    z_um_axis = np.arange(Z) * z_um

    xz_comb = vol_comb[:, y_mid, :]
    xz_488  = vol_488 [:, y_mid, :]
    yz_comb = vol_comb[:, :, x_mid]
    yz_488  = vol_488 [:, :, x_mid]

    # Surface traces along the mid-slice
    def z_along_x(surface, y_const_um):
        return surface_z(surface, x_um, np.full_like(x_um, y_const_um))

    def z_along_y(surface, x_const_um):
        return surface_z(surface, np.full_like(y_um, x_const_um), y_um)

    y_const = y_mid * xy_um
    x_const = x_mid * xy_um

    # --- Grid / OOT stats ---
    xi, yi = sampling_grid(vol_comb.shape, xy_um)
    mask_tis, colmax, glob, thr = tissue_support(vol_comb, thr_frac=0.05)
    mask_tis_grid = mask_tis[yi, xi]
    oot_frac = float(1.0 - mask_tis_grid.mean())

    # Per-column tissue thickness estimate (as z-range where intensity
    # exceeds 10 % of column max), gives a sense of how deep the tissue
    # goes underneath the pia.
    col_thr = 0.10 * colmax
    above_thr = vol_comb > col_thr[None]
    any_above = above_thr.any(axis=0)
    first_z = np.argmax(above_thr, axis=0)         # first z above thr
    last_z  = (Z - 1) - np.argmax(above_thr[::-1], axis=0)  # last z above thr
    thick_um = np.where(any_above, (last_z - first_z) * z_um, 0.0)
    top_z_um  = np.where(any_above, first_z * z_um, np.nan)
    bot_z_um  = np.where(any_above, last_z * z_um,  np.nan)

    tis = mask_tis & any_above
    if tis.any():
        p5, p50, p95 = np.percentile(thick_um[tis], [5, 50, 95])
        top_p50 = float(np.nanmedian(top_z_um[tis]))
        bot_p50 = float(np.nanmedian(bot_z_um[tis]))
    else:
        p5 = p50 = p95 = top_p50 = bot_p50 = np.nan

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def draw_panel(ax, img, title, xlabel, x_axis_um, y_axis_um, overlays):
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect='auto', cmap='gray', origin='upper',
                  extent=[x_axis_um[0], x_axis_um[-1], y_axis_um[-1], y_axis_um[0]],
                  vmin=0, vmax=max(vmax, 1e-6))
        for label, arr, color in overlays:
            if arr is None:
                continue
            ax.plot(x_axis_um, arr, color=color, lw=1.3, label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel('z (µm)')
        ax.set_title(title); ax.legend(loc='lower right', fontsize=8)

    ov_xz = [
        ('N22 (truth)',   z_along_x(n22, y_const) if n22 else None, 'tab:green'),
        ('v2.1 default',  z_along_x(v21, y_const) if v21 else None, 'tab:orange'),
        (f"auto: {info['selected']}", z_along_x(auto_surf, y_const) if auto_surf else None, 'tab:cyan'),
    ]
    ov_yz = [
        ('N22 (truth)',   z_along_y(n22, x_const) if n22 else None, 'tab:green'),
        ('v2.1 default',  z_along_y(v21, x_const) if v21 else None, 'tab:orange'),
        (f"auto: {info['selected']}", z_along_y(auto_surf, x_const) if auto_surf else None, 'tab:cyan'),
    ]
    draw_panel(axes[0, 0], xz_comb, f"{sid}  combined — XZ @ y={y_const:.0f} µm",
               'x (µm)', x_um, z_um_axis, ov_xz)
    draw_panel(axes[0, 1], xz_488,  f"{sid}  488ch raw — XZ @ y={y_const:.0f} µm",
               'x (µm)', x_um, z_um_axis, ov_xz)
    draw_panel(axes[1, 0], yz_comb, f"{sid}  combined — YZ @ x={x_const:.0f} µm",
               'y (µm)', y_um, z_um_axis, ov_yz)
    draw_panel(axes[1, 1], yz_488,  f"{sid}  488ch raw — YZ @ x={x_const:.0f} µm",
               'y (µm)', y_um, z_um_axis, ov_yz)

    # Tissue-thickness annotation
    fig.suptitle(
        f"{sid} — tissue thickness p5/p50/p95 = {p5:.0f} / {p50:.0f} / {p95:.0f} µm  "
        f"| grid OOT frac = {oot_frac*100:.1f} %  | tis top p50 = {top_p50:.0f} µm,"
        f"  tis bottom p50 = {bot_p50:.0f} µm",
        fontsize=11, y=1.00)
    plt.tight_layout()
    out_png = OUT_FIG / f"iter05_surfaces_{sid}.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}")

    return dict(
        subject=sid,
        oot_frac_grid=oot_frac,
        thick_p5=float(p5), thick_p50=float(p50), thick_p95=float(p95),
        top_z_p50=top_p50, bot_z_p50=bot_p50,
        selected=info['selected'],
    )


def main():
    rows = [render_subject(sid) for sid in HCR]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "iter05_audit.csv", index=False)
    print("\n=== audit summary ===")
    print(df.to_string(index=False, float_format=lambda x: f'{x:7.1f}'))
    print(f"\nwrote {OUT_DATA/'iter05_audit.csv'}")


if __name__ == "__main__":
    main()
