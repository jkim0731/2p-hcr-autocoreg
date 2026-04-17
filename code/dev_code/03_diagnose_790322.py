"""Diagnostic — why does N6 leave a gap on 790322 at high x?

Dumps, for each benchmark subject:
  - per-tile (x, y, min_z, n_roi) envelope used by N6
  - fitted quadratic surface + residuals
  - tiles with the largest positive deficit (surface below envelope)
  - x-binned and y-binned density-vs-depth curves under N6

Also runs a few candidate fixes (N7 = tile-count-weighted quadratic,
N8 = curvature-regularised quadratic) and prints summary metrics.
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
    depth_from_surface,
    filter_in_tissue,
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
_robust_quadratic_fit = _mod._robust_quadratic_fit
roi_quadratic_ceiling = _mod.roi_quadratic_ceiling

OUT = Path("/root/capsule/code/sessions/03_surface_estimation_v2")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _tile_envelope(xyz, *, tile_um=120.0, q_frac=0.02,
                   density_radius=30.0, density_min=3):
    """Return per-tile (x, y, z_q, n) for density-filtered ROIs."""
    keep = filter_in_tissue(xyz, radius_um=density_radius,
                            min_neighbors=density_min)
    pts = xyz[keep]
    if len(pts) < 50:
        return None
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    xb = (xs // tile_um).astype(int)
    yb = (ys // tile_um).astype(int)
    key = yb * 100000 + xb
    df = pd.DataFrame({"k": key, "x": xs, "y": ys, "z": zs})
    g = df.groupby("k")
    agg = g.agg(x=("x", "median"), y=("y", "median"),
                z=("z", lambda v: float(np.quantile(v, q_frac))),
                n=("z", "size"))
    return agg[agg["n"] >= 5].reset_index(drop=True)


def weighted_quadratic_fit(xs, ys, zs, w, max_iter=8):
    """_robust_quadratic_fit but with pre-specified sample weights `w`."""
    import math
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float); w = np.asarray(w, dtype=float)
    x0, y0 = float(xs.mean()), float(ys.mean())
    u, v = xs - x0, ys - y0
    X = np.column_stack([u, v, u * u, u * v, v * v, np.ones_like(u)])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(sw[:, None] * X, sw * zs, rcond=None)[0]
    for _ in range(max_iter):
        resid = zs - X @ beta
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma < 1e-6:
            break
        h = np.where(np.abs(resid) <= 1.345 * sigma, 1.0,
                     1.345 * sigma / np.abs(resid))
        wh = np.sqrt(w * h)
        beta = np.linalg.lstsq(wh[:, None] * X, wh * zs, rcond=None)[0]
    a_u, b_v, p, q, r, c_uv = beta
    A = float(a_u - 2 * p * x0 - q * y0)
    B = float(b_v - 2 * r * y0 - q * x0)
    C = float(c_uv - a_u * x0 - b_v * y0 + p * x0 * x0
              + q * x0 * y0 + r * y0 * y0)
    tilt = math.degrees(math.atan(math.hypot(a_u, b_v)))
    return {"a": A, "b": B, "c": C, "p": float(p), "q": float(q),
            "r": float(r), "tilt_deg": tilt,
            "residual_std_um": float(np.std(zs - X @ beta))}


def ridge_quadratic_fit(xs, ys, zs, lam=1e-3, max_iter=8):
    """Ridge-regularised quadratic: penalise p, q, r (curvature) only."""
    import math
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    x0, y0 = float(xs.mean()), float(ys.mean())
    u, v = xs - x0, ys - y0
    X = np.column_stack([u, v, u * u, u * v, v * v, np.ones_like(u)])
    # Pre-scale the quadratic columns so the penalty is in reasonable units.
    scale = np.array([1.0, 1.0,
                      np.std(u * u) + 1e-6,
                      np.std(u * v) + 1e-6,
                      np.std(v * v) + 1e-6,
                      1.0])
    Xs = X / scale
    P = np.diag([0, 0, lam, lam, lam, 0])
    A_norm = Xs.T @ Xs + P
    b_norm = Xs.T @ zs
    beta_s = np.linalg.solve(A_norm, b_norm)
    # IRLS with Huber on top of the ridge
    for _ in range(max_iter):
        resid = zs - Xs @ beta_s
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma < 1e-6:
            break
        w = np.where(np.abs(resid) <= 1.345 * sigma, 1.0,
                     1.345 * sigma / np.abs(resid))
        W = w[:, None]
        beta_s = np.linalg.solve(Xs.T @ (W * Xs) + P, Xs.T @ (w * zs))
    beta = beta_s / scale
    a_u, b_v, p, q, r, c_uv = beta
    A = float(a_u - 2 * p * x0 - q * y0)
    B = float(b_v - 2 * r * y0 - q * x0)
    C = float(c_uv - a_u * x0 - b_v * y0 + p * x0 * x0
              + q * x0 * y0 + r * y0 * y0)
    tilt = math.degrees(math.atan(math.hypot(a_u, b_v)))
    return {"a": A, "b": B, "c": C, "p": float(p), "q": float(q),
            "r": float(r), "tilt_deg": tilt,
            "residual_std_um": float(np.std(zs - Xs @ beta_s))}


def _clamp(fit, agg, quantile=1.0, safety=3.0):
    tx, ty, tz = agg["x"].values, agg["y"].values, agg["z"].values
    plane_z = (fit["a"] * tx + fit["b"] * ty + fit["c"]
               + fit["p"] * tx * tx + fit["q"] * tx * ty
               + fit["r"] * ty * ty)
    deficit = plane_z - tz
    pos = deficit[deficit > 0]
    lift = (float(np.quantile(pos, quantile)) if len(pos) else 0.0)
    out = dict(fit)
    out["c"] = float(out["c"] - lift - safety)
    out["lift_um"] = lift
    return out


def _val_metrics(xyz, surf):
    """Standard metrics + per-x-bin and per-y-bin onset/above_frac spread."""
    d = depth_from_surface(xyz, surf)
    above_frac = float((d < 0).mean())
    bin_um = 5.0
    edges = np.arange(-200.0, 800.0 + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(d, bins=edges)
    dens = h.astype(float) / bin_um
    bulk = dens[(centers >= 50) & (centers <= 200)].mean() or 1.0
    ksz = 6
    smooth = np.convolve(dens, np.ones(ksz) / ksz, mode="same")
    pos = centers >= 0
    over = pos & (smooth >= 0.5 * bulk)
    onset = float(centers[over][0]) if over.any() else float("nan")

    # Per-x-bin and per-y-bin onset
    def per_axis_onset(coord):
        edges_a = np.quantile(coord, np.linspace(0, 1, 7))
        onsets = []
        abs_ = []
        for i in range(len(edges_a) - 1):
            m = (coord >= edges_a[i]) & (coord <= edges_a[i + 1])
            if m.sum() < 200:
                onsets.append(float("nan")); abs_.append(float("nan")); continue
            db = d[m]
            abs_.append(float((db < 0).mean()))
            h2, _ = np.histogram(db, bins=edges)
            dn = h2.astype(float) / bin_um
            bulk2 = dn[(centers >= 50) & (centers <= 200)].mean() or 1.0
            sm = np.convolve(dn, np.ones(ksz) / ksz, mode="same")
            ov = pos & (sm >= 0.5 * bulk2)
            onsets.append(float(centers[ov][0]) if ov.any() else float("nan"))
        return np.array(onsets), np.array(abs_)

    ox, ax_ = per_axis_onset(xyz[:, 0])
    oy, ay_ = per_axis_onset(xyz[:, 1])
    return {
        "above_frac": above_frac,
        "onset_depth_um": onset,
        "onset_x_spread": float(np.nanmax(ox) - np.nanmin(ox)),
        "onset_y_spread": float(np.nanmax(oy) - np.nanmin(oy)),
        "above_x_max": float(np.nanmax(ax_)),
        "above_y_max": float(np.nanmax(ay_)),
    }


def diagnose(subjects):
    rows = []
    for sid in subjects:
        s = load_subject(sid)
        hcr_um = hcr_px_to_um(
            s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
        xyz = hcr_um[:, [2, 1, 0]]
        agg = _tile_envelope(xyz)
        tx, ty, tz, tn = (agg["x"].values, agg["y"].values,
                          agg["z"].values, agg["n"].values)

        # N6 (current winner, q95)
        n6 = roi_quadratic_ceiling(
            xyz, safety_offset_um=3.0, max_residual_quantile=0.95)
        # N7 — tile-count-weighted quadratic + 100%-max clamp
        w = np.log1p(tn)
        fit7 = weighted_quadratic_fit(tx, ty, tz, w)
        n7 = _clamp(fit7, agg, quantile=1.0, safety=3.0)
        # N8 — ridge-regularised quadratic + 100%-max clamp
        fit8 = ridge_quadratic_fit(tx, ty, tz, lam=5e-2)
        n8 = _clamp(fit8, agg, quantile=1.0, safety=3.0)
        # N7b — weighted + 100%-max but with *density filter strengthened*
        agg_tight = _tile_envelope(xyz, density_min=5)
        if agg_tight is not None and len(agg_tight) > 20:
            ttx, tty, ttz, ttn = (agg_tight["x"].values, agg_tight["y"].values,
                                  agg_tight["z"].values, agg_tight["n"].values)
            fit7b = weighted_quadratic_fit(ttx, tty, ttz, np.log1p(ttn))
            n7b = _clamp(fit7b, agg_tight, quantile=1.0, safety=3.0)
        else:
            n7b = None

        methods = {"N6_q95": n6, "N7_wquad_max": n7,
                   "N8_ridge_max": n8, "N7b_tight_max": n7b}
        for name, surf in methods.items():
            if surf is None:
                continue
            m = _val_metrics(xyz, surf)
            rows.append({"subject": sid, "method": name, **m})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hcr_validation.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    diagnose(BENCHMARK_SUBJECTS)
