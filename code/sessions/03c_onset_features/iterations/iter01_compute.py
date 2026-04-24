"""Iteration 1 compute — intensity features at N22 vs v2.1 surface.

For each of the 6 HCR subjects, load the combined HCR volume, the N22
surface (quantile_ceiling), and the v2.1 image-based surface.  Sample a
grid of (x, y) columns in the image interior.  At each column, compute
the z-column intensity profile `I(z)` and extract a set of
surface-anchored features at *both* z_N22 and z_v21.

Outputs
-------
- data/iter01_features.parquet  (long-form table, one row per (subject, x, y))
- data/iter01_profiles.npz       (a handful of sample I(z) profiles for plots)

Run from repo root:
    python code/sessions/03c_onset_features/iterations/iter01_compute.py
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

# Load the v2.1 image-based surface module (filename starts with a digit)
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

# Feature windows (um)
WIN_ABOVE_NEAR = (-50.0, -10.0)
WIN_BELOW_NEAR = (10.0, 50.0)
WIN_BELOW_MID = (50.0, 150.0)
WIN_BELOW_DEEP = (150.0, 250.0)
# dip detection windows (relative to z)
WIN_DIP_PEAK = (0.0, 30.0)
WIN_DIP_TROUGH = (30.0, 150.0)


def surface_z(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (surface["a"] * x + surface["b"] * y + surface["c"]
            + surface.get("p", 0.0) * x * x
            + surface.get("q", 0.0) * x * y
            + surface.get("r", 0.0) * y * y)


def column_stats(prof: np.ndarray, z_um_axis: np.ndarray, z_center: float):
    """Return dict of features evaluated relative to z_center (in µm).

    prof : (Z,) 1-D intensity column
    z_um_axis : (Z,) z in µm
    """
    z_lo = z_um_axis.min(); z_hi = z_um_axis.max()

    def _window_mean(lo: float, hi: float) -> float:
        m = (z_um_axis >= z_center + lo) & (z_um_axis < z_center + hi)
        return float(prof[m].mean()) if m.any() else np.nan

    def _window_max(lo: float, hi: float) -> float:
        m = (z_um_axis >= z_center + lo) & (z_um_axis < z_center + hi)
        return float(prof[m].max()) if m.any() else np.nan

    def _window_min(lo: float, hi: float) -> float:
        m = (z_um_axis >= z_center + lo) & (z_um_axis < z_center + hi)
        return float(prof[m].min()) if m.any() else np.nan

    above_near = _window_mean(*WIN_ABOVE_NEAR)
    below_near = _window_mean(*WIN_BELOW_NEAR)
    below_mid = _window_mean(*WIN_BELOW_MID)
    below_deep = _window_mean(*WIN_BELOW_DEEP)
    dip_peak = _window_max(*WIN_DIP_PEAK)
    dip_trough = _window_min(*WIN_DIP_TROUGH)

    # gradient at z (central, over ±10 µm)
    m_grad_lo = (z_um_axis >= z_center - 15) & (z_um_axis < z_center - 5)
    m_grad_hi = (z_um_axis >= z_center + 5) & (z_um_axis < z_center + 15)
    if m_grad_lo.any() and m_grad_hi.any():
        grad = float(prof[m_grad_hi].mean() - prof[m_grad_lo].mean()) / 20.0
    else:
        grad = np.nan

    # "dip magnitude" — the key AF detector
    dip_mag = dip_peak - dip_trough if np.isfinite(dip_peak) and np.isfinite(dip_trough) else np.nan

    # Relative dip (scale-free, handy across subjects/channels)
    below_max = _window_max(10.0, 150.0)
    dip_rel = (dip_mag / below_max) if np.isfinite(dip_mag) and np.isfinite(below_max) and below_max > 0 else np.nan

    return {
        "above_near": above_near,
        "below_near": below_near,
        "below_mid": below_mid,
        "below_deep": below_deep,
        "grad": grad,
        "dip_peak": dip_peak,
        "dip_trough": dip_trough,
        "dip_mag": dip_mag,
        "dip_rel": dip_rel,
        "z_lo": z_lo,
        "z_hi": z_hi,
    }


def get_n22_surface(s, vol, xy_um, z_um, hcr_xyz):
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


def interior_grid(vol, xy_um, n_side=20, edge_frac=0.15):
    _, Y, X = vol.shape
    x_pad = int(edge_frac * X); y_pad = int(edge_frac * Y)
    xs_i = np.linspace(x_pad, X - 1 - x_pad, n_side).astype(int)
    ys_i = np.linspace(y_pad, Y - 1 - y_pad, n_side).astype(int)
    xi, yi = np.meshgrid(xs_i, ys_i)
    xi = xi.ravel(); yi = yi.ravel()
    xu = xi * xy_um; yu = yi * xy_um
    return xi, yi, xu, yu


def process_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol, xy_um, z_um, channels_used = load_hcr_combined(s, level=4)
    Z, Y, X = vol.shape
    z_axis = np.arange(Z, dtype=np.float32) * z_um

    hcr_um = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    hcr_xyz = hcr_um[:, [2, 1, 0]]

    n22 = get_n22_surface(s, vol, xy_um, z_um, hcr_xyz)
    v21 = _img.estimate_surface_and_l2_image_based(
        vol, z_um, xy_um, target_quantile=0.85)
    if n22 is None or v21.surface is None:
        print(f"  {sid}: skipping (missing surface)")
        return None, None

    xi, yi, xu, yu = interior_grid(vol, xy_um, n_side=20, edge_frac=0.15)
    z_n22 = surface_z(n22, xu, yu)
    z_v21 = surface_z(v21.surface, xu, yu)

    rows = []
    for k in range(xi.size):
        col = vol[:, yi[k], xi[k]]
        # Only keep rows where both surfaces are sampleable in the volume
        if not (0 < z_n22[k] < Z * z_um and 0 < z_v21[k] < Z * z_um):
            continue
        f_n = column_stats(col, z_axis, float(z_n22[k]))
        f_v = column_stats(col, z_axis, float(z_v21[k]))
        row = dict(subject=sid, xu=float(xu[k]), yu=float(yu[k]),
                   z_n22=float(z_n22[k]), z_v21=float(z_v21[k]))
        for key, val in f_n.items():
            row[f"n22_{key}"] = val
        for key, val in f_v.items():
            row[f"v21_{key}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)

    # Save a couple of sample column profiles per subject for plots
    samples = {}
    center = xi.size // 2
    for idx, label in [(0, "corner"), (center, "center"), (xi.size - 1, "far")]:
        col = vol[:, yi[idx], xi[idx]]
        samples[label] = {
            "prof": col.astype(np.float32),
            "z_axis": z_axis.astype(np.float32),
            "z_n22": float(z_n22[idx]),
            "z_v21": float(z_v21[idx]),
            "xu": float(xu[idx]),
            "yu": float(yu[idx]),
        }
    print(f"  {sid}: {len(df)} grid columns usable; N22 c={n22['c']:.1f}, v21 c={v21.surface['c']:.1f}")
    return df, samples


def main():
    all_df = []
    all_samples = {}
    for sid in HCR:
        df, samples = process_subject(sid)
        if df is not None:
            all_df.append(df)
            all_samples[sid] = samples
    out_df = pd.concat(all_df, ignore_index=True)
    out_df.to_csv(OUT_DATA / "iter01_features.csv", index=False)
    print(f"wrote {OUT_DATA/'iter01_features.csv'} ({len(out_df)} rows)")

    # Save samples
    flat = {}
    for sid, subs in all_samples.items():
        for label, d in subs.items():
            flat[f"{sid}_{label}_prof"] = d["prof"]
            flat[f"{sid}_{label}_zaxis"] = d["z_axis"]
            flat[f"{sid}_{label}_meta"] = np.array(
                [d["z_n22"], d["z_v21"], d["xu"], d["yu"]], dtype=np.float32)
    np.savez_compressed(OUT_DATA / "iter01_profiles.npz", **flat)
    print(f"wrote {OUT_DATA/'iter01_profiles.npz'}")


if __name__ == "__main__":
    main()
