"""CZ side-by-side: raw log(CZ) vs iter07 poly-deg2 overlay.

For each CZ subject, render 4 y-positions x 2 columns:
  col 0: raw log(vol+EPS) YZ slice, no overlay (to assess where the
         tissue boundary actually is by eye)
  col 1: same slice with iter07 poly-deg2 surface in red

Uses cached transitions in ``data/iter07_cz_transitions_<sid>.npz``
so the detector does not have to re-run. Output:
``figures/iter07_cz_sidebyside_<sid>.png``.
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
from iter07_compute import fit_polysurf, eval_polysurf, POLY_DEGREE, HUBER_K

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
DATA = SESSION / "data"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
N_Y = 4
Y_INTERIOR_FRAC = 0.15
EPS = 1e-3


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


def render(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape

    cache = np.load(DATA / f"iter07_cz_transitions_{sid}.npz")
    xs = cache["xs_um"]; ys = cache["ys_um"]; zs = cache["zs_um"]
    fit2 = fit_polysurf(xs, ys, zs, degree=POLY_DEGREE, huber_k=HUBER_K)

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
        z_fit = eval_polysurf(fit2, x_um, np.full_like(x_um, y_const))
        for col, (title, overlays) in enumerate([
            ("raw (no fit)", []),
            (f"iter07 poly-deg{POLY_DEGREE}",
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
        f"{sid} — CZ raw vs iter07 poly-deg{POLY_DEGREE}  "
        f"(log(CZ+ε); red = fitted pia surface)",
        fontsize=12, y=1.00)
    plt.tight_layout()
    out = OUT_FIG / f"iter07_cz_sidebyside_{sid}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    for sid in SUBJECTS:
        render(sid)


if __name__ == "__main__":
    main()
