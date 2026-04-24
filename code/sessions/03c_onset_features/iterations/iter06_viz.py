"""Iteration 6 visualisation — top-down OOT mask + sampling grid.

For every HCR subject produces a figure showing, in XY view, the column
max-intensity projection and the 20 × 20 sampling grid used by
``score_surface_quality``.  Grid columns are coloured by whether they
are in tissue (green, kept) or OOT (red, masked) under the new default
``min_col_max_frac = 0.05``.

Outputs
-------
- figures/iter06_oot_mask_<sid>.png  — one per subject
- data/iter06_oot_mask_summary.csv   — n_in_tissue / n_oot per subject
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import load_subject

OUT_DATA = ROOT / "code" / "sessions" / "03c_onset_features" / "data"
OUT_FIG  = ROOT / "code" / "sessions" / "03c_onset_features" / "figures"

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]
N_SIDE = 20
EDGE_FRAC = 0.15
MIN_COL_MAX_FRAC = 0.05
GLOBAL_MAX_PERCENTILE = 99.5


def render_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    Z, Y, X = vol.shape
    colmax = vol.max(axis=0)              # (Y, X)
    gmax = float(np.percentile(colmax, GLOBAL_MAX_PERCENTILE))
    col_thr = MIN_COL_MAX_FRAC * gmax

    x_pad = int(EDGE_FRAC * X); y_pad = int(EDGE_FRAC * Y)
    xs_i = np.linspace(x_pad, X - 1 - x_pad, N_SIDE).astype(int)
    ys_i = np.linspace(y_pad, Y - 1 - y_pad, N_SIDE).astype(int)
    xi, yi = np.meshgrid(xs_i, ys_i)
    xi = xi.ravel(); yi = yi.ravel()
    grid_col = colmax[yi, xi]
    mask_tis = grid_col >= col_thr
    n_in = int(mask_tis.sum()); n_oot = int((~mask_tis).sum())

    # µm coordinates for plotting
    x_um = np.arange(X) * xy_um
    y_um = np.arange(Y) * xy_um
    xu = xi * xy_um; yu = yi * xy_um

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = float(np.percentile(colmax, 99.5))
    ax.imshow(colmax, aspect="equal", cmap="viridis",
              extent=[x_um[0], x_um[-1], y_um[-1], y_um[0]],
              vmin=0, vmax=max(vmax, 1e-6))
    # contour at col_thr to show the OOT / tissue boundary
    ax.contour(x_um, y_um, colmax, levels=[col_thr],
               colors="white", linewidths=0.8, linestyles="--")
    ax.scatter(xu[mask_tis], yu[mask_tis], c="lime",
               s=22, edgecolor="black", linewidth=0.5,
               label=f"in tissue (n={n_in})")
    ax.scatter(xu[~mask_tis], yu[~mask_tis], c="red",
               s=22, edgecolor="black", linewidth=0.5,
               label=f"OOT (n={n_oot})")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    frac = n_oot / (n_in + n_oot) * 100 if (n_in + n_oot) else 0.0
    ax.set_title(
        f"{sid} — column-max (top-down)  |  OOT frac = {frac:.1f} %  "
        f"(thr = {MIN_COL_MAX_FRAC*100:.0f} % of p{GLOBAL_MAX_PERCENTILE} = {gmax:.4g})"
    )
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out_png = OUT_FIG / f"iter06_oot_mask_{sid}.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}  (in_tissue={n_in}, oot={n_oot})")

    return dict(subject=sid, n_in_tissue=n_in, n_oot=n_oot,
                oot_frac=frac / 100.0,
                global_max=gmax, col_thr=col_thr)


def main():
    rows = [render_subject(sid) for sid in HCR]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "iter06_oot_mask_summary.csv", index=False)
    print("\n=== OOT mask summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {OUT_DATA/'iter06_oot_mask_summary.csv'}")


if __name__ == "__main__":
    main()
