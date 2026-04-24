"""Compare poly-deg2 vs poly-deg3 surfaces side by side with raw slice.

For each HCR subject, 4 y positions x 3 columns (raw log(combined) /
deg2 overlay / deg3 overlay). All three columns share the same
intensity scale, so the only visible difference across a row is the
overlay. Output: ``figures/iter07_degcompare_<sid>.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))
sys.path.insert(0, str(ROOT / "code" / "sessions" / "03c_onset_features" / "iterations"))

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import load_subject
from iter07_compute import fit_polysurf, eval_polysurf

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
DATA = SESSION / "data"

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]
N_Y = 4
Y_INTERIOR_FRAC = 0.15
EPS = 1e-3


def render(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    Z, Y, X = vol.shape

    cache = np.load(DATA / f"iter07_transitions_{sid}.npz")
    xs = cache["xs_um"]; ys = cache["ys_um"]; zs = cache["zs_um"]
    fit2 = fit_polysurf(xs, ys, zs, degree=2)
    fit3 = fit_polysurf(xs, ys, zs, degree=3)

    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_lo = int(Y_INTERIOR_FRAC * Y); y_hi = Y - 1 - y_lo
    y_idx = np.linspace(y_lo, y_hi, N_Y).astype(int)

    fig, axes = plt.subplots(N_Y, 3, figsize=(18, 3.5 * N_Y), sharey=True)
    for row, iy in enumerate(y_idx):
        y_const = iy * xy_um
        img = np.log(vol[:, iy, :] + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        z2 = eval_polysurf(fit2, x_um, np.full_like(x_um, y_const))
        z3 = eval_polysurf(fit3, x_um, np.full_like(x_um, y_const))
        for col, (title, overlays) in enumerate([
            ("raw (no fit)", []),
            ("poly-deg2",   [('deg2', z2, 'tab:orange')]),
            ("poly-deg3",   [('deg3', z3, 'tab:red')]),
        ]):
            ax = axes[row, col]
            ax.imshow(img, aspect='auto', cmap='gray', origin='upper',
                      extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                      vmin=vmin, vmax=vmax)
            for label, arr, color in overlays:
                ax.plot(x_um, arr, color=color, lw=1.6, label=label)
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
        f"{sid} — poly-deg2 vs poly-deg3 vs raw  "
        f"(log(combined+ε); orange=deg2, red=deg3)",
        fontsize=12, y=1.00)
    plt.tight_layout()
    out = OUT_FIG / f"iter07_degcompare_{sid}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    for sid in HCR:
        render(sid)


if __name__ == "__main__":
    main()
