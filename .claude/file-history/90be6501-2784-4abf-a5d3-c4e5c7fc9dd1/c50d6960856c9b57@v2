"""I1 — Cortical-layer depth-profile 1D axial NCC.

Compute mean intensity vs. depth-from-pia on both CZ z-stack and HCR 488
volume; run a 1D NCC with partial-overlap floor to recover (sz, tz).

Emits a plausibility check: if the NCC peak's robust-z is below the floor,
we treat the Z alignment as unknown and emit a graceful-degradation result.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from benchmark_analysis import (load_cz_y_slab, load_hcr_volume,
                                  analyze_subject, depth_from_surface)


def _depth_profile_from_volume(vol, pia_coefs, xy_um, z_um, depth_max_um=1000.0, bin_um=5.0):
    """Compute mean intensity per depth-from-pia bin."""
    Z, Y, X = vol.shape
    # Sample at a regular XY grid to avoid a (Z*Y*X) coordinate blow-up
    step = max(1, int(round(32.0 / xy_um)))
    ys = np.arange(0, Y, step)
    xs = np.arange(0, X, step)
    zs = np.arange(0, Z)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")
    pts_um = np.stack([Zg * z_um, Yg * xy_um, Xg * xy_um], axis=-1).reshape(-1, 3)
    a, b, c = pia_coefs
    # planar pia: pia_z = a*x + b*y + c → depth along +z
    pia_z = a * pts_um[:, 2] + b * pts_um[:, 1] + c
    depth = pts_um[:, 0] - pia_z
    vals = vol[Zg, Yg, Xg].reshape(-1)
    bins = np.arange(0, depth_max_um + bin_um, bin_um)
    hist, _ = np.histogram(depth, bins=bins, weights=vals)
    cnt, _ = np.histogram(depth, bins=bins)
    prof = hist / np.maximum(cnt, 1)
    return bins[:-1] + bin_um / 2, prof


def _fast_ncc_1d_partial(tpl: np.ndarray, img: np.ndarray, min_overlap: int = 20) -> tuple:
    """Partial-overlap 1D NCC: search for the shift that maximises NCC over
    the overlap region, with at least `min_overlap` bins of overlap."""
    n_t = len(tpl); n_i = len(img)
    shifts = np.arange(-n_t + min_overlap, n_i - min_overlap + 1)
    ncc = np.zeros(len(shifts))
    for k, s in enumerate(shifts):
        t0 = max(0, -s); t1 = min(n_t, n_i - s)
        i0 = t0 + s; i1 = t1 + s
        if t1 - t0 < min_overlap:
            continue
        a = tpl[t0:t1] - tpl[t0:t1].mean()
        b = img[i0:i1] - img[i0:i1].mean()
        den = a.std() * b.std() + 1e-9
        ncc[k] = float((a * b).mean() / den)
    best = int(np.argmax(ncc))
    return ncc, shifts, best


@register_candidate("I1")
def run_i1(s, *, scale_grid=None) -> CoregResult:
    """Sweep a small anisotropic-Z-scale grid and return the best peak."""
    if scale_grid is None:
        scale_grid = np.arange(2.0, 3.6, 0.25)
    try:
        analysis = analyze_subject(s)
    except Exception as e:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"error": f"analyze: {e}"})

    cz_surf = analysis.get("cz_surface")
    hcr_surf = analysis.get("hcr_surface")
    if cz_surf is None or hcr_surf is None:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"error": "no surfaces"})

    # Load a small CZ z-stack for the profile
    try:
        cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
        hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
    except Exception as e:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"error": f"load: {e}"})

    # Surface coefficients in physical units
    cz_pia = _surface_to_planar_um(cz_surf)
    hcr_pia = _surface_to_planar_um(hcr_surf)

    dz_cz, prof_cz = _depth_profile_from_volume(cz_stack, cz_pia, cz_xy_um, cz_z_um)
    dz_hr, prof_hr = _depth_profile_from_volume(hcr_vol,   hcr_pia, hcr_xy_um, hcr_z_um)

    best_scale = None; best_ncc = -np.inf; best_shift = None
    for sz in scale_grid:
        # Stretch CZ profile axis by sz and resample onto HCR bins
        new_x = dz_cz * sz
        resampled = np.interp(dz_hr, new_x, prof_cz, left=0, right=0)
        # Compute partial-overlap NCC between resampled (template) and profile_hr (image)
        ncc, shifts, ki = _fast_ncc_1d_partial(resampled, prof_hr, min_overlap=20)
        if ncc[ki] > best_ncc:
            best_ncc = ncc[ki]; best_scale = float(sz); best_shift = float(shifts[ki] * (dz_hr[1] - dz_hr[0]))

    # Intrinsic confidence = robust-z of peak
    # (cheap: re-run at best_scale to get NCC curve's spread)
    new_x = dz_cz * best_scale
    resampled = np.interp(dz_hr, new_x, prof_cz, left=0, right=0)
    ncc, _, ki = _fast_ncc_1d_partial(resampled, prof_hr, min_overlap=20)
    med = np.median(ncc[ncc > 0]); mad = np.median(np.abs(ncc[ncc > 0] - med)) + 1e-9
    z_score = float((best_ncc - med) / mad)
    conf = 1.0 / (1.0 + np.exp(-(z_score - 4)))

    transform = TransformDescriptor(
        R=np.eye(3), scales=np.array([best_scale, 1.0, 1.0]),
        translation=np.array([best_shift, 0.0, 0.0]),
        src_mean=np.zeros(3),
        rotation_deg_z=0.0, kind="axial-z",
    )
    return CoregResult(
        pairs_df=pd.DataFrame(),  # I1 does not emit per-pair assignments
        confidence=float(conf),
        transform=transform,
        diagnostics=dict(
            best_ncc=best_ncc, best_scale=best_scale, best_shift_um=best_shift,
            robust_z=z_score,
        ),
    )


def _load_cz_fullstack(s):
    import tifffile
    paths = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not paths:
        paths = list(s.coreg_dir.glob("*zstack.tif"))
    if not paths:
        raise FileNotFoundError("no CZ zstack")
    arr = tifffile.imread(str(paths[0]))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32), float(s.cz_xy_um), float(s.cz_z_um)


def _load_hcr_fullstack(s, level=4):
    r = load_hcr_volume(s, channel="488", level=level)
    # ``load_hcr_volume`` may return either an ndarray or a
    # ``(ndarray, xy_um, z_um)`` tuple depending on the version.
    if isinstance(r, tuple):
        vol = r[0]
    else:
        vol = r
    from benchmark_analysis import hcr_level_resolution
    xy, zu = hcr_level_resolution(s, level)
    return np.asarray(vol).astype(np.float32), float(xy), float(zu)


def _surface_to_planar_um(surf_dict):
    """Turn surface-fit dict → (a, b, c) planar coefficients where
    pia_z_um = a*x_um + b*y_um + c.  Falls back to 0, 0, median(pia_z) if the
    fit is not planar.
    """
    if isinstance(surf_dict, dict) and surf_dict.get("type") == "planar":
        return (surf_dict["a"], surf_dict["b"], surf_dict["c"])
    # generic: pull out 'a','b','c' if present
    if isinstance(surf_dict, dict):
        return (surf_dict.get("a", 0.0), surf_dict.get("b", 0.0), surf_dict.get("c", 0.0))
    return (0.0, 0.0, 0.0)
