"""CZ iter07 with noise-anchored column threshold.

Same pipeline as ``iter07_cz_proto.py`` (patch-MAX detector + IRLS-Huber
poly-deg2 surface), but the per-column threshold is

    thr = max(THR_FLOOR, col_p10 + NOISE_K * 1.4826 * MAD_lower)

where ``MAD_lower`` is the MAD of values ≤ column-median.  This fires at
"first sustained rise above the column's own OOT noise", which is what
HCR's range-relative rule does by accident (HCR's p10 ≡ OOT noise
because combined is bg-subtracted).  CZ's p10 is OOT-floor too, but the
range-relative rule's dependence on p90 pushed the firing point mid-
tissue on the ramp-shaped CZ onset.

Outputs:
- ``figures/iter07_cz_noise_<sid>.png`` — 4y × 2-col raw vs fit side-by-side
- ``data/iter07_cz_noise_transitions_<sid>.npz``
- ``data/iter07_cz_noise_summary.csv``
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

from benchmark_data_loader import load_subject
from iter07_compute import (
    fit_polysurf, eval_polysurf, sampling_grid,
    col_detect_transition,
    SMOOTH_Z_UM, SUSTAIN_Z_UM, PATCH_W, POLY_DEGREE, HUBER_K, NOISE_K,
)

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
OUT_DATA = SESSION / "data"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
EPS = 1e-3
CZ_THR_FLOOR = 4.5
N_Y = 4
Y_INTERIOR_FRAC = 0.15


def load_cz_volume(s):
    files = (list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
             or list(s.coreg_dir.glob("*zstack.tif")))
    if not files:
        raise FileNotFoundError(f"no CZ TIFF for {s.subject_id}")
    arr = tifffile.imread(str(files[0]))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"CZ TIFF shape={arr.shape} not ZYX")
    return arr.astype(np.float32, copy=False)


def detect_transitions(vol, z_um, grid_ix, grid_iy,
                       patch_w=PATCH_W, thr_floor=CZ_THR_FLOOR,
                       noise_k=NOISE_K):
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)
    zs = np.empty(len(grid_ix))
    thrs = np.empty(len(grid_ix))
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        z_vox, _, thr_col = col_detect_transition(
            log_col, z_um, smooth_z_um=SMOOTH_Z_UM,
            sustain_z_um=SUSTAIN_Z_UM, thr_floor=thr_floor,
            mode="noise_anchored", noise_k=noise_k)
        zs[i]   = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr_col
    return zs, thrs


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape

    log_vol = np.log(vol + EPS)
    p10, p50, p90, p99 = np.percentile(log_vol, [10, 50, 90, 99])
    print(f"  log percentiles p10={p10:.2f} p50={p50:.2f} "
          f"p90={p90:.2f} p99={p99:.2f}")

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs_um, thrs = detect_transitions(vol, z_um, xi, yi)
    valid = np.isfinite(zs_um)
    print(f"  transitions: {valid.sum()}/{len(zs_um)} valid, "
          f"median z = {np.nanmedian(zs_um):.1f} µm, "
          f"median thr = {np.nanmedian(thrs):.2f}")

    polyfit = fit_polysurf(xs_um, ys_um, zs_um, degree=POLY_DEGREE,
                           huber_k=HUBER_K)
    if polyfit is None:
        print("  polyfit failed")
        return None

    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_lo = int(Y_INTERIOR_FRAC * Y); y_hi = Y - 1 - y_lo
    y_idx = np.linspace(y_lo, y_hi, N_Y).astype(int)

    fig, axes = plt.subplots(N_Y, 2, figsize=(12, 3.5 * N_Y), sharey=True)
    for row, iy in enumerate(y_idx):
        y_const = iy * xy_um
        img = np.log(vol[:, iy, :] + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        z_fit = eval_polysurf(polyfit, x_um, np.full_like(x_um, y_const))
        for col, (title, overlays) in enumerate([
            ("raw (no fit)", []),
            (f"iter07 noise-anchored poly-deg{POLY_DEGREE}",
             [('poly', z_fit, 'tab:red')]),
        ]):
            ax = axes[row, col]
            ax.imshow(img, aspect='auto', cmap='gray', origin='upper',
                      extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                      vmin=vmin, vmax=vmax)
            for label, arr, color in overlays:
                ax.plot(x_um, arr, color=color, lw=1.8, label=label)
            if col == 0:
                ax.set_ylabel('z (µm)')
            ax.set_xlabel('x (µm)')
            if row == 0:
                ax.set_title(title)
            if overlays:
                ax.legend(loc='lower right', fontsize=8)
            ax.text(0.01, 0.98, f"y = {y_const:.0f} µm",
                    transform=ax.transAxes, va='top', ha='left',
                    color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2, lw=0))

    fig.suptitle(
        f"{sid} — CZ iter07 noise-anchored (NOISE_K={NOISE_K}, "
        f"poly-deg{POLY_DEGREE})",
        fontsize=12, y=1.00)
    plt.tight_layout()
    out = OUT_FIG / f"iter07_cz_noise_{sid}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    np.savez(OUT_DATA / f"iter07_cz_noise_transitions_{sid}.npz",
             xs_um=xs_um, ys_um=ys_um, zs_um=zs_um, thrs=thrs)

    return dict(
        subject=sid,
        log_p10=float(p10), log_p50=float(p50),
        log_p90=float(p90), log_p99=float(p99),
        thr_floor=CZ_THR_FLOOR,
        noise_k=NOISE_K,
        median_thr=float(np.nanmedian(thrs)),
        n_valid_trans=int(valid.sum()),
        median_trans_z_um=float(np.nanmedian(zs_um)) if valid.any() else np.nan,
    )


def main():
    rows = []
    for sid in SUBJECTS:
        r = render_subject(sid)
        if r is not None:
            rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = OUT_DATA / "iter07_cz_noise_summary.csv"
    df.to_csv(out, index=False)
    print("\n=== CZ noise-anchored summary ===")
    print(df.to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
