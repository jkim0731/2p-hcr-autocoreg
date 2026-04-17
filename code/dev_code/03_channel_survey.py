"""HCR channel survey.

For each subject:
  1. Build the N6 quadratic target surface from ROI centroids.
  2. Load every HCR channel at level 4.
  3. Inside a target band of +/- BAND_UM around the quadratic surface,
     find per-column top-of-signal (shallowest z in the band whose
     normalized intensity crosses THRESHOLDS).
  4. Report per-channel statistics: fraction of columns with valid
     hit, median z offset (top-of-signal minus quadratic), std of that
     offset, and the resulting robust-fit tilt.
  5. Emit a per-subject figure: one row per channel with (a) y-slab MIP
     overlaid with the quadratic anchor, (b) per-column top-of-signal
     z-offset histogram.

Output: sessions/03_surface_estimation_v2/channel_survey.csv
        figures/channel_survey_<subject>.png
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
    hcr_level_resolution,
    list_hcr_channels,
    load_hcr_volume,
    load_hcr_y_slab,
)
from benchmark_data_loader import BENCHMARK_SUBJECTS, hcr_px_to_um, load_subject

import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "surface_iteration_v2",
    Path(__file__).parent / "03_surface_iteration_v2.py",
)
_mod = _iu.module_from_spec(_spec)
sys.modules["surface_iteration_v2"] = _mod
_spec.loader.exec_module(_mod)
roi_quadratic_ceiling = _mod.roi_quadratic_ceiling

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BAND_UM = 100.0   # +/- search band around the ROI-quadratic surface
LEVEL = 4
THRESHOLDS = (0.02, 0.05, 0.10, 0.20)  # fractions of 99th pct inside band


def _surf_z(surf, x, y):
    """Evaluate the quadratic surface at (x, y)."""
    return (surf["a"] * x + surf["b"] * y + surf["c"]
            + surf["p"] * x * x + surf["q"] * x * y
            + surf["r"] * y * y)


def _per_column_top(vol, xy_um, z_um, surf, thresholds):
    """For each (y, x) column, find shallowest z inside the band where the
    intensity crosses each threshold (fractions of the band-wide P99).

    Returns dict[threshold -> (top_z_um array shape (Y, X))] with NaN
    for columns with no signal. Also returns (xx_um, yy_um) meshgrids and
    `band_mask` (Z, Y, X) boolean of the search band."""
    Z, Y, X = vol.shape
    zs = np.arange(Z, dtype=np.float32) * z_um
    ys = np.arange(Y, dtype=np.float32) * xy_um
    xs = np.arange(X, dtype=np.float32) * xy_um
    xx, yy = np.meshgrid(xs, ys)
    surf_zz = _surf_z(surf, xx, yy)  # (Y, X)

    band_mask = (zs[:, None, None] >= (surf_zz - BAND_UM)[None, :, :]) & \
                (zs[:, None, None] <= (surf_zz + BAND_UM)[None, :, :])
    band_vol = np.where(band_mask, vol, 0.0).astype(np.float32)
    p99 = float(np.percentile(band_vol[band_vol > 0], 99)) if np.any(band_vol > 0) else 1.0
    out = {}
    for t in thresholds:
        thr = t * p99
        # For each column, index of first z where value >= thr
        above = band_vol >= thr
        # Argmax gives first True index (zero if none, so mask by any)
        any_col = above.any(axis=0)
        first_idx = np.argmax(above, axis=0)
        top_z = np.where(any_col, zs[first_idx], np.nan)
        out[t] = top_z
    return out, xx, yy, surf_zz, band_vol, p99


def _robust_stats(top_z, surf_zz):
    mask = np.isfinite(top_z)
    if mask.sum() < 50:
        return None
    off = top_z[mask] - surf_zz[mask]
    # Robust stats: median-based
    return {
        "hit_frac": float(mask.mean()),
        "offset_median_um": float(np.median(off)),
        "offset_mad_um": float(1.4826 * np.median(np.abs(off - np.median(off)))),
        "offset_p05_um": float(np.quantile(off, 0.05)),
        "offset_p95_um": float(np.quantile(off, 0.95)),
    }


def _survey_subject(sid):
    s = load_subject(sid)
    channels = list_hcr_channels(s)
    print(f"\n=== {sid}: channels {channels} ===")

    hcr_um = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    xyz = hcr_um[:, [2, 1, 0]]
    surf = roi_quadratic_ceiling(
        xyz, safety_offset_um=3.0, max_residual_quantile=1.0)
    if surf is None:
        return [], {}

    results = []
    channel_tops = {}  # channel -> dict of threshold -> top_z
    channel_meta = {}

    for ch in channels:
        try:
            vol, xy_um, z_um = load_hcr_volume(s, channel=ch, level=LEVEL)
        except FileNotFoundError:
            continue
        vol = vol.astype(np.float32, copy=False)
        tops, xx, yy, surf_zz, band_vol, p99 = _per_column_top(
            vol, xy_um, z_um, surf, THRESHOLDS)
        channel_tops[ch] = tops
        channel_meta[ch] = dict(xy_um=xy_um, z_um=z_um, p99=p99,
                                surf_zz=surf_zz, shape=vol.shape)
        for t in THRESHOLDS:
            stats = _robust_stats(tops[t], surf_zz)
            row = {"subject": sid, "channel": ch, "threshold": t,
                   "p99_in_band": p99}
            if stats is None:
                row.update(hit_frac=0.0)
            else:
                row.update(stats)
            results.append(row)
            print(f"  ch={ch} thr={t:.2f}  hit={row.get('hit_frac',0):.2f}"
                  f"  med_off={row.get('offset_median_um',float('nan')):.1f}"
                  f"  mad={row.get('offset_mad_um',float('nan')):.1f}")
    return results, (s, surf, channel_tops, channel_meta, channels)


def _render_subject(sid, bundle):
    s, surf, channel_tops, channel_meta, channels = bundle
    if not channels:
        return
    thr_show = 0.05  # median threshold for the figure

    # Y-slab for MIP background
    hcr_um = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    xyz = hcr_um[:, [2, 1, 0]]
    y0 = float(np.median(xyz[:, 1]))
    half = 40.0

    n = len(channels)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.0 * n), squeeze=False)
    for i, ch in enumerate(channels):
        meta = channel_meta[ch]
        top_z = channel_tops[ch][thr_show]
        surf_zz = meta["surf_zz"]
        xy_um = meta["xy_um"]
        z_um = meta["z_um"]

        # MIP slab
        try:
            mip, yret, z_um_mip, xy_um_mip = load_hcr_y_slab(
                s, channel=ch, y_center_um=y0,
                half_width_um=half, level=LEVEL)
        except FileNotFoundError:
            for a in axes[i, :]:
                a.axis("off")
            continue

        ax = axes[i, 0]
        vmin, vmax = np.percentile(mip, [1, 99.5])
        extent = (0, mip.shape[1] * xy_um_mip, mip.shape[0] * z_um_mip, 0)
        ax.imshow(mip, extent=extent, cmap="gray", vmin=vmin, vmax=vmax,
                  aspect="auto", interpolation="nearest")
        # Quadratic anchor line at y=y0
        xs = np.linspace(0, mip.shape[1] * xy_um_mip, 300)
        ys0 = np.full_like(xs, y0)
        qz = _surf_z(surf, xs, ys0)
        ax.plot(xs, qz, "r-", lw=1.2, label="quad anchor")
        ax.fill_between(xs, qz - BAND_UM, qz + BAND_UM,
                        color="red", alpha=0.12, label="+/-100 um band")
        ax.set_title(f"{sid}  ch={ch}  MIP (y={y0:.0f} um)")
        ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
        ax.legend(fontsize=7)

        # Column top-of-signal map (z offset from quad)
        ax = axes[i, 1]
        off = top_z - surf_zz
        im = ax.imshow(off, cmap="coolwarm", vmin=-80, vmax=80,
                       aspect="auto")
        ax.set_title(f"top-of-signal z - quad (thr={thr_show})")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Histogram of offsets
        ax = axes[i, 2]
        valid = off[np.isfinite(off)]
        if len(valid) > 0:
            ax.hist(valid, bins=40, color="steelblue", alpha=0.8)
        ax.axvline(0, color="red", lw=1.2)
        ax.set_title(f"offset hist  hit={np.isfinite(off).mean():.2f}  "
                     f"p99={meta['p99']:.1f}")
        ax.set_xlabel("z - quad (um)")

    fig.suptitle(f"HCR channel survey - {sid}  (band = +/-{BAND_UM:.0f} um)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = FIG / f"channel_survey_{sid}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def main(subjects):
    all_rows = []
    for sid in subjects:
        rows, bundle = _survey_subject(sid)
        all_rows.extend(rows)
        if bundle:
            _render_subject(sid, bundle)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "channel_survey.csv", index=False)
    print(f"\nWrote {OUT / 'channel_survey.csv'}")
    print("\nSummary per (subject, channel) at threshold=0.05:")
    thr = df[df["threshold"] == 0.05]
    print(thr[["subject", "channel", "hit_frac",
               "offset_median_um", "offset_mad_um", "p99_in_band"]]
          .to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    main(args.subjects or BENCHMARK_SUBJECTS)
