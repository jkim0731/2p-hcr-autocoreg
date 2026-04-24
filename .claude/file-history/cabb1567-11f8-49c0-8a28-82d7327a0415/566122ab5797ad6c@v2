"""Sweep TPS smoothing values on cached iter07 transitions.

No volume load needed; just refits TPS with several smoothing values
and renders a per-subject figure showing the resulting XZ surface
(mid-y) overlaid on the log(combined) slice.  Use to pick a less-wiggly
TPS_SMOOTHING for iter07.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import load_subject

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
DATA = SESSION / "data"

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]
SMOOTHINGS = [200, 2000, 10000, 50000]
EPS = 1e-3


def fit(xs, ys, zs, smoothing):
    ok = np.isfinite(zs)
    if ok.sum() < 4:
        return None
    pts = np.column_stack([xs[ok], ys[ok]])
    return RBFInterpolator(pts, zs[ok], kernel='thin_plate_spline',
                           smoothing=smoothing)


def ev(rbf, x, y):
    if rbf is None:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    pts = np.column_stack([x.ravel(), y.ravel()])
    return rbf(pts).reshape(np.asarray(x).shape)


def render(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    Z, Y, X = vol.shape

    cache = np.load(DATA / f"iter07_transitions_{sid}.npz")
    xs = cache["xs_um"]; ys = cache["ys_um"]; zs = cache["zs_um"]

    fig, axes = plt.subplots(1, len(SMOOTHINGS), figsize=(5*len(SMOOTHINGS), 4.5),
                             sharey=True)
    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_mid = Y // 2; y_const = y_mid * xy_um
    img = np.log(vol[:, y_mid, :] + EPS)
    vmax = float(np.percentile(img, 99.5))
    vmin = float(np.percentile(img, 5))

    for ax, sm in zip(axes, SMOOTHINGS):
        rbf = fit(xs, ys, zs, sm)
        ax.imshow(img, aspect='auto', cmap='gray', origin='upper',
                  extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                  vmin=vmin, vmax=vmax)
        z_tps = ev(rbf, x_um, np.full_like(x_um, y_const))
        ax.plot(x_um, z_tps, color='tab:red', lw=1.5)
        # evaluate TPS curvature magnitude for diagnostic: stddev of (z - plane-fit)
        xx, yy = np.meshgrid(np.linspace(xs.min(), xs.max(), 40),
                             np.linspace(ys.min(), ys.max(), 40))
        zz = ev(rbf, xx, yy)
        # fit plane
        ok = np.isfinite(zz)
        A = np.column_stack([xx[ok], yy[ok], np.ones(ok.sum())])
        coef, *_ = np.linalg.lstsq(A, zz[ok], rcond=None)
        zp = coef[0]*xx + coef[1]*yy + coef[2]
        resid = np.nanstd(zz - zp)
        ax.set_title(f"smoothing={sm}\nresid-over-plane stddev = {resid:.1f} µm")
        ax.set_xlabel('x (µm)')
    axes[0].set_ylabel('z (µm)')
    fig.suptitle(f"{sid} — TPS smoothing sweep @ y={y_const:.0f} µm", fontsize=12)
    plt.tight_layout()
    out = OUT_FIG / f"iter07_smsweep_{sid}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    for sid in HCR:
        render(sid)


if __name__ == "__main__":
    main()
