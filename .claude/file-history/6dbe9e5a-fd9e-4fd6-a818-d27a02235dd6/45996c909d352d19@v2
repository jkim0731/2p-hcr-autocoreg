"""Iteration 3 compute — surface quality score + candidate-bank auto-selection.

Defines
  Q(surface, vol)  =  median over grid columns of
                      mean(I[z_surf + 10 : z_surf + 50])

Q is evaluated in the *reference* combined HCR volume (normalised as
`load_hcr_combined` does).  A good surface sits at pia → tissue starts
immediately below → high Q.  A surface in the AF-to-pia gap → low Q.

Experiments
-----------
1. Compute Q(N22) and Q(v21_default) per subject.  N22 should score
   at least as high as v21 on every subject and substantially higher
   on 755252.
2. For every subject build a candidate bank of v2.1 surfaces fit on
   different channel subsets and target_quantile values; score each
   on the reference combined volume; check that the highest-Q
   candidate is the one closest to N22 (minimal `|onset|` /
   `above_frac` when evaluated against centroids).

Outputs
-------
- data/iter03_scores.csv           — one row per (subject, candidate)
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

N_SIDE = 20  # 20x20 = 400 grid columns
EDGE_FRAC = 0.15
Z_BELOW = (10.0, 50.0)
Z_DEEP = (150.0, 250.0)
Z_ABOVE = (-50.0, -10.0)


def surface_z(surface: dict, x, y):
    return (surface["a"] * x + surface["b"] * y + surface["c"]
            + surface.get("p", 0.0) * x * x
            + surface.get("q", 0.0) * x * y
            + surface.get("r", 0.0) * y * y)


def q_score(surface, vol, xy_um, z_um):
    """Return several quality statistics.

    Q       = median below_near                     (raw brightness just below)
    Qabove  = median above_near                     (raw brightness just above)
    Qcon    = median (below_near - above_near)      (contrast; pia > 0, body ≈ 0, gap ≈ 0)
    Qcon_n  = median ((b - a) / (b + a + eps))      (normalised contrast in [-1,1])
    """
    if surface is None:
        return dict(Q=np.nan, Qabove=np.nan, Qcon=np.nan, Qcon_n=np.nan, n=0)
    Z, Y, X = vol.shape
    x_pad = int(EDGE_FRAC * X); y_pad = int(EDGE_FRAC * Y)
    xs_i = np.linspace(x_pad, X - 1 - x_pad, N_SIDE).astype(int)
    ys_i = np.linspace(y_pad, Y - 1 - y_pad, N_SIDE).astype(int)
    xi, yi = np.meshgrid(xs_i, ys_i)
    xi = xi.ravel(); yi = yi.ravel()
    xu = xi * xy_um; yu = yi * xy_um
    z_s = surface_z(surface, xu, yu)

    z_axis = np.arange(Z, dtype=np.float32) * z_um
    above = []; below = []
    for k in range(len(xi)):
        col = vol[:, yi[k], xi[k]]
        if not (0 < z_s[k] < Z * z_um):
            continue
        m_a = (z_axis >= z_s[k] + Z_ABOVE[0]) & (z_axis < z_s[k] + Z_ABOVE[1])
        m_b = (z_axis >= z_s[k] + Z_BELOW[0]) & (z_axis < z_s[k] + Z_BELOW[1])
        a = float(col[m_a].mean()) if m_a.any() else np.nan
        b = float(col[m_b].mean()) if m_b.any() else np.nan
        above.append(a); below.append(b)
    above = np.array(above); below = np.array(below)
    con = below - above
    con_n = con / (below + above + 1e-6)
    return dict(
        Q=float(np.nanmedian(below)),
        Qabove=float(np.nanmedian(above)),
        Qcon=float(np.nanmedian(con)),
        Qcon_n=float(np.nanmedian(con_n)),
        n=len(below))


def eval_against_centroids(surface, hcr_xyz):
    """Return (above_frac, onset_depth, c) against HCR centroids."""
    if surface is None:
        return (np.nan, np.nan, np.nan)
    zs = surface_z(surface, hcr_xyz[:, 0], hcr_xyz[:, 1])
    depth = hcr_xyz[:, 2] - zs
    n = len(depth)
    above = float((depth < 0).sum() / n)
    edges = np.arange(-200, 404, 5.0)
    h, _ = np.histogram(depth, bins=edges)
    density = h / (5.0 * max(n, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bulk_mask = (centers >= 50) & (centers <= 200)
    bulk = density[bulk_mask].mean()
    onset = np.nan
    if bulk > 0:
        for c, d in zip(centers, density):
            if c >= 0 and d >= 0.5 * bulk:
                onset = float(c); break
    return (above, onset, float(surface["c"]))


def get_n22(s, vol_combined, xy_um, z_um, hcr_xyz):
    anchor = estimate_pia_surface_image_ceiling(
        hcr_xyz, vol_combined, z_um, xy_um,
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


def mid_channel(sid: str) -> str:
    return "514" if sid in ("755252", "767022") else "561"


def build_candidates(s, sid, ref_vol, xy_um, z_um):
    """Fit v21 on several channel subsets and tq values.

    Returns list of (name, surface) pairs."""
    mid = mid_channel(sid)
    subsets = [
        ("all_tq0.85",   None,                        0.85),
        ("all_tq0.70",   None,                        0.70),
        ("all_tq0.95",   None,                        0.95),
        ("no405_tq0.85", ["488", mid, "594"],          0.85),
        ("mid488_tq0.85",["488", mid],                 0.85),
        ("only488_tq0.85",["488"],                     0.85),
        ("only594_tq0.85",["594"],                     0.85),
    ]
    cands = []
    for name, chs, tq in subsets:
        try:
            vol, xy, zu, used = load_hcr_combined(s, channels=chs, level=4)
            res = _img.estimate_surface_and_l2_image_based(
                vol, zu, xy, target_quantile=tq)
            cands.append((name, res.surface, used))
        except Exception as exc:
            print(f"  {sid} {name}: FAILED ({exc})")
            cands.append((name, None, []))
    return cands


def process_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol_ref, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    hcr_xyz = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]

    n22 = get_n22(s, vol_ref, xy_um, z_um, hcr_xyz)
    v21_default = _img.estimate_surface_and_l2_image_based(
        vol_ref, z_um, xy_um, target_quantile=0.85).surface

    def _mkrow(cand, surf, used):
        Q = q_score(surf, vol_ref, xy_um, z_um)
        above, onset, c_c = eval_against_centroids(surf, hcr_xyz)
        return dict(subject=sid, cand=cand, used=used,
                    Q=Q["Q"], Qabove=Q["Qabove"], Qcon=Q["Qcon"], Qcon_n=Q["Qcon_n"],
                    above=above, onset=onset, c=c_c)

    rows = []
    rows.append(_mkrow("N22", n22, "n/a"))
    rows.append(_mkrow("v21_default", v21_default, "all"))
    for name, surf, used in build_candidates(s, sid, vol_ref, xy_um, z_um):
        rows.append(_mkrow(name, surf, ",".join(used) if used else ""))

    for r in rows:
        onset_s = f"{r['onset']:.1f}" if np.isfinite(r['onset']) else "nan"
        print(f"  {r['cand']:<18} Q={r['Q']:.4f}  Qab={r['Qabove']:.4f}  "
              f"Qcon={r['Qcon']:.4f}  Qcon_n={r['Qcon_n']:.3f}  "
              f"above={r['above']:.4f}  onset={onset_s}  c={r['c']:.1f}")
    return rows


def main():
    all_rows = []
    for sid in HCR:
        all_rows.extend(process_subject(sid))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DATA / "iter03_scores.csv", index=False)
    print(f"\nwrote {OUT_DATA/'iter03_scores.csv'}")


if __name__ == "__main__":
    main()
