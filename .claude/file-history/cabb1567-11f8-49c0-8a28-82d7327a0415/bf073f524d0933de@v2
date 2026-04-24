"""Iteration 2 compute — per-column top analysis.

We now replicate v2.1 Stage 2 (anchor) and Stage 3 (per-column tops in
the ±150 µm band) for each subject.  Each returned top is labelled
``pia`` or ``AF`` by comparing to the N22 surface (|z_top - N22(x,y)|
< AF_DIST_UM → pia; else AF).  For every top we also compute
``below_near`` (mean intensity in a 40 µm slab just below the top) in
the subsampled combined volume.

Goal: show per-top distributions of ``below_near`` for pia vs AF tops,
and find a threshold (or soft weighting) usable by the Stage-3
filtering step.

Outputs
-------
- data/iter02_tops.csv       — one row per column top (all 6 subjects)
- data/iter02_vol_stats.csv  — per-subject volume stats for context
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

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
    estimate_pia_surface_image_ceiling,
    estimate_pia_surface_quantile_ceiling,
    list_hcr_channels,
    load_hcr_combined,
    load_hcr_volume,
)
from benchmark_data_loader import hcr_px_to_um, load_subject

OUT_DATA = ROOT / "code" / "sessions" / "03c_onset_features" / "data"
OUT_DATA.mkdir(parents=True, exist_ok=True)

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]

AF_DIST_UM = 25.0  # label threshold — tops within this z of N22 are pia


def surface_z(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (surface["a"] * x + surface["b"] * y + surface["c"]
            + surface.get("p", 0.0) * x * x
            + surface.get("q", 0.0) * x * y
            + surface.get("r", 0.0) * y * y)


def get_n22(s, vol, xy_um, z_um, hcr_xyz):
    anchor = estimate_pia_surface_image_ceiling(
        hcr_xyz, vol, z_um, xy_um,
        relative_margin=0.005, min_signal_abs=0.05, safety_offset_um=0.0)
    if anchor is None:
        return None
    channels = list_hcr_channels(s)
    per_ch_vols = []
    for ch in channels:
        try:
            v_ch, _, _ = load_hcr_volume(s, channel=ch, level=4)
            per_ch_vols.append(v_ch.astype(np.float32, copy=False))
        except FileNotFoundError:
            continue
    sxy = max(1, int(round(10.0 / xy_um)))
    vols_sub = [v[:, ::sxy, ::sxy] for v in per_ch_vols]
    _, Ys, Xs = vols_sub[0].shape
    xs_sub = np.arange(Xs) * sxy * xy_um
    ys_sub = np.arange(Ys) * sxy * xy_um
    xx_sub, yy_sub = np.meshgrid(xs_sub, ys_sub)
    return estimate_pia_surface_quantile_ceiling(
        hcr_xyz, vols_sub, z_um, sxy * xy_um, xx_sub, yy_sub, anchor)


def process_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    hcr_xyz = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]

    n22 = get_n22(s, vol, xy_um, z_um, hcr_xyz)
    if n22 is None:
        print(f"  {sid}: N22 not available"); return None

    # Replicate v2.1 Stage 2 + 3 on subsampled combined volume
    sx = max(1, int(round(10.0 / xy_um)))
    vol_sub = vol[:, ::sx, ::sx].astype(np.float32, copy=False)
    Zs, Ys, Xs = vol_sub.shape
    xs_col = np.arange(Xs, dtype=np.float32) * sx * xy_um
    ys_col = np.arange(Ys, dtype=np.float32) * sx * xy_um
    xx, yy = np.meshgrid(xs_col, ys_col)
    z_axis = np.arange(Zs, dtype=np.float32) * z_um

    # Anchor (stage 2)
    a_top, _, _ = _img._per_column_top_in_band(
        vol_sub, z_um, None, None,
        relative_margin=0.05, min_signal_abs_frac=0.05, min_thick_um=15.0)
    mask_a = np.isfinite(a_top)
    xa = xx[mask_a].ravel().astype(np.float64)
    ya = yy[mask_a].ravel().astype(np.float64)
    za = a_top[mask_a].ravel().astype(np.float64)
    med = float(np.median(za)); mad = float(np.median(np.abs(za - med))) + 1e-6
    keep = np.abs(za - med) <= 6.0 * 1.4826 * mad
    anchor = _img._robust_plane_fit(xa[keep], ya[keep], za[keep])

    # Surface band (stage 3)
    anchor_zz = _img._surface_z(anchor, xx, yy).astype(np.float32)
    z_low = anchor_zz - 150.0
    z_high = anchor_zz + 150.0
    top_z, _, _ = _img._per_column_top_in_band(
        vol_sub, z_um, z_low, z_high,
        relative_margin=0.25, min_signal_abs_frac=0.20, min_thick_um=10.0)

    mask = np.isfinite(top_z)
    ys_idx, xs_idx = np.where(mask)
    xs_um = xs_idx * sx * xy_um
    ys_um = ys_idx * sx * xy_um
    zs_top = top_z[ys_idx, xs_idx]

    # N22 z at these (x, y) positions -> label as pia or AF
    z_n22 = surface_z(n22, xs_um, ys_um)
    dist_to_n22 = zs_top - z_n22
    label = np.where(np.abs(dist_to_n22) <= AF_DIST_UM, "pia", "AF")

    # Compute below_near per top on the subsampled volume column
    #   below_near = mean(I(z)) for z in [z_top+10, z_top+50]
    #   also measure below_mid, below_deep, and a ratio
    rows = []
    for k in range(len(xs_idx)):
        col = vol_sub[:, ys_idx[k], xs_idx[k]]  # (Z,)
        zt = zs_top[k]
        def wm(lo, hi):
            m = (z_axis >= zt + lo) & (z_axis < zt + hi)
            return float(col[m].mean()) if m.any() else np.nan
        rows.append({
            "subject": sid,
            "x_um": float(xs_um[k]), "y_um": float(ys_um[k]),
            "z_top": float(zt),
            "z_n22": float(z_n22[k]),
            "dist_to_n22": float(dist_to_n22[k]),
            "label": label[k],
            "below_near": wm(10.0, 50.0),
            "below_mid":  wm(50.0, 150.0),
            "below_deep": wm(150.0, 250.0),
            "above_near": wm(-50.0, -10.0),
        })
    df = pd.DataFrame(rows)
    print(f"  {sid}: {len(df)} tops, pia={ (df.label=='pia').sum() } AF={(df.label=='AF').sum()}")
    return df


def main():
    all_df = [process_subject(sid) for sid in HCR]
    all_df = [d for d in all_df if d is not None]
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(OUT_DATA / "iter02_tops.csv", index=False)
    print(f"wrote {OUT_DATA/'iter02_tops.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
