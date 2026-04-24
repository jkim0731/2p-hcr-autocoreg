"""Iter 08 — per-subject 4-column summary figures.

Each subject gets a single figure with 5 rows (y positions) and
4 columns:

    col 0: HCR combined log-slice @ y, with top + bottom boundaries
    col 1: HCR combined log-slice @ y, plain (no overlay)
    col 2: CZ log-slice @ y, with iter08 surface + 50 µm prior
    col 3: CZ log-slice @ y, plain

The five y-positions are evenly spaced over the interior (15% edge pad)
of each modality's volume.  Boundaries on the HCR panels are the
iter07 top (red) and iter08 bottom (magenta) surfaces evaluated at
each row's y.  The CZ overlay is the iter08 prior-selected surface
(red) with the 50 µm prior plane (gold dashed) for reference.

Outputs:
  figures/iter08_summary_<sid>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))
sys.path.insert(0, str(ROOT / "code" / "sessions" / "03c_onset_features" / "iterations"))

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import load_subject
from iter07_compute import (
    eval_polysurf,
    fit_polysurf,
    HUBER_K,
    POLY_DEGREE,
    EPS,
)
from iter08_cz_prior import (
    CZ_TARGET_Z_UM,
    _patch_log_columns as _cz_patch_cols,
    load_cz_volume,
    sampling_grid as _cz_grid,
    select_trans_frac,
    fit_gated_surface,
)
from iter08_hcr_bottom import (
    detect_bottom_transitions,
    detect_top_transitions,
    mad_gate,
)
from iter07_compute import sampling_grid as _hcr_grid

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
N_Y = 5
Y_INTERIOR_FRAC = 0.15


def _fit_hcr(vol, z_um, xy_um):
    xi, yi = _hcr_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs_top, _ = detect_top_transitions(vol, z_um, xi, yi)
    zs_bot, _ = detect_bottom_transitions(vol, z_um, xi, yi)
    top_fit = fit_polysurf(xs_um, ys_um, zs_top,
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    gate = mad_gate(zs_bot)
    bot_fit = fit_polysurf(xs_um[gate], ys_um[gate], zs_bot[gate],
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    return top_fit, bot_fit, (xs_um, ys_um, zs_top, zs_bot)


def _fit_cz(vol, z_um, xy_um):
    xi, yi = _cz_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    log_vol = np.log(vol + EPS)
    log_cols = _cz_patch_cols(log_vol, xi, yi)
    (sel_tf, sel_zs, _, sel_med), _ = select_trans_frac(
        log_cols, z_um, CZ_TARGET_Z_UM)
    polyfit, _, _ = fit_gated_surface(xs_um, ys_um, sel_zs)
    return polyfit, sel_tf, sel_med, (xs_um, ys_um, sel_zs)


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)

    print("  loading HCR combined …", flush=True)
    hcr_vol, hcr_xy, hcr_z, _ = load_hcr_combined(s, level=4)
    print("  fitting HCR top + bottom …", flush=True)
    top_fit, bot_fit, _ = _fit_hcr(hcr_vol, hcr_z, hcr_xy)

    print("  loading CZ volume …", flush=True)
    cz_vol = load_cz_volume(s)
    cz_z, cz_xy = s.cz_z_um, s.cz_xy_um
    print("  fitting CZ (prior-selected) …", flush=True)
    cz_fit, cz_tf, cz_med, _ = _fit_cz(cz_vol, cz_z, cz_xy)

    Zh, Yh, Xh = hcr_vol.shape
    Zc, Yc, Xc = cz_vol.shape
    x_um_h = np.arange(Xh) * hcr_xy
    z_um_h = np.arange(Zh) * hcr_z
    x_um_c = np.arange(Xc) * cz_xy
    z_um_c = np.arange(Zc) * cz_z

    y_lo_h = int(Y_INTERIOR_FRAC * Yh); y_hi_h = Yh - 1 - y_lo_h
    y_idx_h = np.linspace(y_lo_h, y_hi_h, N_Y).astype(int)
    y_lo_c = int(Y_INTERIOR_FRAC * Yc); y_hi_c = Yc - 1 - y_lo_c
    y_idx_c = np.linspace(y_lo_c, y_hi_c, N_Y).astype(int)

    fig, axes = plt.subplots(N_Y, 4, figsize=(20, 3.7 * N_Y), sharey=False)
    for row in range(N_Y):
        iyh = y_idx_h[row]
        iyc = y_idx_c[row]
        y_const_h = iyh * hcr_xy
        y_const_c = iyc * cz_xy

        img_h = np.log(hcr_vol[:, iyh, :] + EPS)
        img_c = np.log(cz_vol[:, iyc, :] + EPS)
        vmin_h = float(np.percentile(img_h, 5))
        vmax_h = float(np.percentile(img_h, 99.5))
        vmin_c = float(np.percentile(img_c, 5))
        vmax_c = float(np.percentile(img_c, 99.5))

        # Col 0: HCR with boundaries
        ax = axes[row, 0]
        ax.imshow(img_h, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um_h[0], x_um_h[-1], z_um_h[-1], z_um_h[0]],
                  vmin=vmin_h, vmax=vmax_h)
        if top_fit is not None:
            ax.plot(x_um_h, eval_polysurf(top_fit, x_um_h,
                                          np.full_like(x_um_h, y_const_h)),
                    color="tab:red", lw=1.6, label="top (iter07)")
        if bot_fit is not None:
            ax.plot(x_um_h, eval_polysurf(bot_fit, x_um_h,
                                          np.full_like(x_um_h, y_const_h)),
                    color="magenta", lw=1.6, label="bottom (iter08)")
        ax.set_ylabel("z (µm)")
        if row == 0:
            ax.set_title(f"{sid}  HCR (with boundaries)", fontsize=11)
        ax.text(0.01, 0.98, f"y = {y_const_h:.0f} µm",
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=2, lw=0))
        if row == 0:
            ax.legend(loc="lower right", fontsize=8)

        # Col 1: HCR plain
        ax = axes[row, 1]
        ax.imshow(img_h, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um_h[0], x_um_h[-1], z_um_h[-1], z_um_h[0]],
                  vmin=vmin_h, vmax=vmax_h)
        if row == 0:
            ax.set_title(f"{sid}  HCR (plain)", fontsize=11)
        ax.text(0.01, 0.98, f"y = {y_const_h:.0f} µm",
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=2, lw=0))

        # Col 2: CZ with boundary + prior
        ax = axes[row, 2]
        ax.imshow(img_c, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um_c[0], x_um_c[-1], z_um_c[-1], z_um_c[0]],
                  vmin=vmin_c, vmax=vmax_c)
        ax.axhline(CZ_TARGET_Z_UM, color="gold", lw=1.0, ls="--",
                   label=f"prior ({CZ_TARGET_Z_UM:.0f} µm)")
        if cz_fit is not None:
            ax.plot(x_um_c,
                    eval_polysurf(cz_fit, x_um_c,
                                  np.full_like(x_um_c, y_const_c)),
                    color="tab:red", lw=1.8,
                    label=f"iter08 CZ (tf={cz_tf})")
        if row == 0:
            ax.set_title(f"{sid}  CZ (with boundary)", fontsize=11)
        ax.text(0.01, 0.98, f"y = {y_const_c:.0f} µm",
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=2, lw=0))
        if row == 0:
            ax.legend(loc="lower right", fontsize=8)

        # Col 3: CZ plain
        ax = axes[row, 3]
        ax.imshow(img_c, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um_c[0], x_um_c[-1], z_um_c[-1], z_um_c[0]],
                  vmin=vmin_c, vmax=vmax_c)
        if row == 0:
            ax.set_title(f"{sid}  CZ (plain)", fontsize=11)
        ax.text(0.01, 0.98, f"y = {y_const_c:.0f} µm",
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=9, bbox=dict(facecolor="black", alpha=0.5, pad=2, lw=0))

        if row == N_Y - 1:
            axes[row, 0].set_xlabel("x (µm)")
            axes[row, 1].set_xlabel("x (µm)")
            axes[row, 2].set_xlabel("x (µm)")
            axes[row, 3].set_xlabel("x (µm)")

    xi_c, yi_c = _cz_grid(cz_vol.shape, cz_xy)
    z_cz_grid = eval_polysurf(cz_fit, xi_c * cz_xy, yi_c * cz_xy)
    cz_surf_med = float(np.nanmedian(z_cz_grid))
    fig.suptitle(
        f"Subject {sid}  —  HCR (top + bottom)  vs  CZ (prior-guided surface).  "
        f"CZ median surface z = {cz_surf_med:.1f} µm (target 50).",
        fontsize=13, y=0.995)
    plt.tight_layout()
    out = OUT_FIG / f"iter08_summary_{sid}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    for sid in SUBJECTS:
        render_subject(sid)


if __name__ == "__main__":
    main()
