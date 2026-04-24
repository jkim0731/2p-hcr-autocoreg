"""CZ iter07 with explicit bg-subtraction before log.

Root cause diagnosis: HCR iter07 works because `load_hcr_combined`
subtracts background so OOT ≡ 0, making log(OOT+ε) = −6.9 a true floor.
CZ raw uint16 has a camera offset (~30–100 DN) that we hadn't removed,
so log(OOT) ≈ 5.5 and the p10→p90 range is compressed relative to HCR.
Applying the same range-relative threshold on that compressed range
fires mid-ramp instead of at onset.

Fix: estimate camera offset from the volume itself (global low quantile),
subtract it, clip at 0, add ε, log.  Then reuse the HCR detector
(`mode='range_relative'`) unchanged.

Outputs:
- ``figures/iter07_cz_bgsub_<sid>.png`` — 4y × 2-col raw vs fit
- ``data/iter07_cz_bgsub_transitions_<sid>.npz``
- ``data/iter07_cz_bgsub_summary.csv``
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
    SMOOTH_Z_UM, SUSTAIN_Z_UM, TRANS_FRAC, PATCH_W, POLY_DEGREE, HUBER_K,
)

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
OUT_DATA = SESSION / "data"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
EPS = 1e-3
OFFSET_BINS = 1024           # histogram bins for camera-offset mode estimator
CZ_THR_FLOOR = -6.3          # same as HCR now that we're bg-subtracted
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


def estimate_camera_offset(vol, bins=OFFSET_BINS):
    """Camera offset = mode of the intensity histogram.

    In a typical CZ stack, OOT voxels vastly outnumber tissue voxels and
    cluster tightly around the camera offset (≈ 100 DN).  The histogram
    has a single sharp peak there with a long tail toward brighter tissue.
    p1 is not a reliable estimator because rare zero voxels (edge
    clipping) drag it to 0; the mode is robust.
    """
    lo, hi = float(vol.min()), float(np.percentile(vol, 99))
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(vol, bins=bins, range=(lo, hi))
    i = int(np.argmax(hist))
    return 0.5 * (float(edges[i]) + float(edges[i + 1]))


def bg_subtract(vol):
    offset = estimate_camera_offset(vol)
    sub = np.clip(vol - offset, 0.0, None)
    return sub, offset


def detect_transitions(log_vol, z_um, grid_ix, grid_iy,
                       patch_w=PATCH_W, thr_floor=CZ_THR_FLOOR):
    Z, Y, X = log_vol.shape
    zs = np.empty(len(grid_ix))
    thrs = np.empty(len(grid_ix))
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        z_vox, _, thr_col = col_detect_transition(
            log_col, z_um, smooth_z_um=SMOOTH_Z_UM,
            sustain_z_um=SUSTAIN_Z_UM, trans_frac=TRANS_FRAC,
            thr_floor=thr_floor, mode="range_relative")
        zs[i]   = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr_col
    return zs, thrs


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape

    sub, offset = bg_subtract(vol)
    log_vol = np.log(sub + EPS)
    p10, p50, p90, p99 = np.percentile(log_vol, [10, 50, 90, 99])
    print(f"  offset (hist mode) = {offset:.1f} DN; "
          f"log percentiles p10={p10:.2f} p50={p50:.2f} "
          f"p90={p90:.2f} p99={p99:.2f}")

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs_um, thrs = detect_transitions(log_vol, z_um, xi, yi)
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
        img = np.log(np.clip(vol[:, iy, :] - offset, 0.0, None) + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        z_fit = eval_polysurf(polyfit, x_um, np.full_like(x_um, y_const))
        for col, (title, overlays) in enumerate([
            ("raw bg-sub log (no fit)", []),
            (f"iter07 bg-sub poly-deg{POLY_DEGREE}",
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
        f"{sid} — CZ iter07 bg-sub (offset={offset:.0f} DN, "
        f"poly-deg{POLY_DEGREE})",
        fontsize=12, y=1.00)
    plt.tight_layout()
    out = OUT_FIG / f"iter07_cz_bgsub_{sid}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)

    np.savez(OUT_DATA / f"iter07_cz_bgsub_transitions_{sid}.npz",
             xs_um=xs_um, ys_um=ys_um, zs_um=zs_um, thrs=thrs,
             offset=offset)

    return dict(
        subject=sid,
        offset=float(offset),
        log_p10=float(p10), log_p50=float(p50),
        log_p90=float(p90), log_p99=float(p99),
        thr_floor=CZ_THR_FLOOR,
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
    out = OUT_DATA / "iter07_cz_bgsub_summary.csv"
    df.to_csv(out, index=False)
    print("\n=== CZ bg-sub summary ===")
    print(df.to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
