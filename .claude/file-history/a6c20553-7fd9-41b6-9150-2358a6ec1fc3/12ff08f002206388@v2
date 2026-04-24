"""Iter 09 — CZ surface via baseline (OOT) statistics + bounded search.

Motivation
----------
iter08's CZ pipeline treats 50 µm as a *selection* signal: sweep
TRANS_FRAC values and keep the one whose median column z is closest to
50.  That bakes the prior in as an answer — if the experimenter's 50
µm target itself was off on a particular brain (which can happen — cf.
782149 and 767018), iter08 will still force the surface to 50 µm by
picking a TRANS_FRAC that does so.

iter09 uses the acquisition metadata differently: **the 50 µm target
is a weak constraint on the search region, not the answer**.  Pixel
values in the top ~30 µm of each column are almost always out-of-
tissue noise (the experimenter was trying to sit pia at 50 µm, so
z < 30 µm is below pia on every subject where the target held).  We
estimate OOT statistics directly from that window and place the
threshold a few standard deviations above the OOT mean.  The surface
detection is then an independent measurement; the prior appears only
as

 * the **baseline window** `z ∈ [0, 30 µm]` (derived from the target),
 * the **bounded search window** `z ∈ [0, 120 µm]` (no tissue onset
   expected past here if the target was even approximately right),
 * a **sanity diagnostic** — median surface z vs 50 µm — that can
   flag a subject whose acquisition deviated a lot, without biasing
   the detector.

Detector per column
-------------------
1. `log_col = log(patch-MAX(15×15) + ε)`, same as iter07/08.
2. Smooth over 5 µm in depth (1D Gaussian).
3. Baseline: take `smoothed[z ∈ [0, Z_BASELINE_UM]]`; keep only the
   **darkest half** of that window (`≤ median`) so any shallow
   tissue contamination drops out.  `μ_b = median`, `σ_b = 1.4826 ·
   MAD` of that darkest half.
4. Threshold: `thr = μ_b + max(K_SIGMA · σ_b, LOG_MARGIN)`.
   The `LOG_MARGIN` floor is the important piece: on clean dark
   air the per-voxel log-noise is Poisson-tight (σ_b ≈ 0.05, so
   3σ_b ≈ 0.15 corresponds to only a 16 % linear rise — any
   scattered-light ramp will cross it).  Requiring at least
   `LOG_MARGIN` above baseline forces a real fold-change.  No
   range-relative (p10→p90) term.
5. Sustained crossing within `z ∈ [0, Z_MAX_UM]` — first z where the
   smoothed profile stays above `thr` for `SUSTAIN_Z_UM`.  If no
   crossing within the window, return NaN for that column (don't
   stretch the detector past the physically plausible range).

Surface fit
-----------
IRLS-Huber bivariate degree-2 polynomial on the valid columns,
gated by a **3 × MAD cut around the per-subject median** (no
reference to 50 µm).  Same `fit_polysurf` as iter07/08.

Outputs
-------
* `figures/iter09_cz_<sid>.png` — 4-y-slice log(CZ) overlays, iter09
  surface in red, iter08 surface in cyan, 50 µm line in gold dashed.
* `data/iter09_cz_transitions_<sid>.npz` — per-column zs, thrs, μ_b,
  σ_b.
* `data/iter09_cz_summary.csv` — per-subject detection fraction,
  median z, |Δ prior|, iter08 z for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))
sys.path.insert(0, str(ROOT / "code" / "sessions" / "03c_onset_features" / "iterations"))

from benchmark_analysis import analyze_subject, estimate_pia_surface
from benchmark_data_loader import cz_px_to_um, load_subject
from iter07_compute import (
    eval_polysurf,
    fit_polysurf,
    sampling_grid,
    EPS,
    HUBER_K,
    PATCH_W,
    POLY_DEGREE,
    SMOOTH_Z_UM,
    SUSTAIN_Z_UM,
)
from iter08_cz_prior import (
    CZ_TARGET_Z_UM,
    load_cz_volume,
    select_trans_frac,
    fit_gated_surface as _iter08_fit,
    _patch_log_columns as _patch_cols,
)
from iter08_hcr_bottom import mad_gate

SESSION = ROOT / "code" / "sessions" / "03c_onset_features"
OUT_FIG = SESSION / "figures"
OUT_DATA = SESSION / "data"

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

# Detector parameters
Z_BASELINE_UM = 30.0         # "pia should be below this if target held"
Z_MAX_UM = 120.0             # don't search past here — 2.4× the 50 µm target
K_SIGMA = 3.0                # threshold = μ_baseline + K · σ_baseline
LOG_MARGIN = 0.7             # floor on the above-baseline gap (log units);
                             # 0.7 ≈ 2.0× linear fold-change — enforces a
                             # real rise over baseline when σ_b is small
BASELINE_LOWER_FRAC = 0.5    # darkest half of baseline window → robust to
                             # shallow-tissue contamination
SIGMA_FLOOR_LOG = 0.05       # minimum σ_b to avoid zero-spread degeneracy
                             # when the baseline is nearly constant
N_Y = 4
Y_INTERIOR_FRAC = 0.15


def col_detect_baseline(log_col, z_um, *,
                        z_baseline_um=Z_BASELINE_UM,
                        z_max_um=Z_MAX_UM,
                        k_sigma=K_SIGMA,
                        log_margin=LOG_MARGIN,
                        smooth_z_um=SMOOTH_Z_UM,
                        sustain_z_um=SUSTAIN_Z_UM,
                        baseline_lower_frac=BASELINE_LOWER_FRAC,
                        sigma_floor_log=SIGMA_FLOOR_LOG):
    """Baseline-anchored detector with a bounded search window.

    Returns ``(z_vox, thr, mu_b, sigma_b, smoothed)`` where ``z_vox``
    is the first-voxel-in-tissue index (or ``-1`` if no crossing
    within ``[0, z_max_um]``).
    """
    sigma_smooth_vox = max(1.0, smooth_z_um / z_um)
    smoothed = gaussian_filter1d(log_col, sigma=sigma_smooth_vox)
    sustain_vox = max(1, int(sustain_z_um / z_um))

    # Robust baseline over the darkest portion of the [0, z_baseline] window
    bi = max(1, int(z_baseline_um / z_um))
    bi = min(bi, len(smoothed))
    base_win = smoothed[:bi]
    cut = np.quantile(base_win, baseline_lower_frac)
    dark = base_win[base_win <= cut]
    if dark.size < 3:
        dark = base_win
    mu_b = float(np.median(dark))
    mad = float(np.median(np.abs(dark - mu_b)))
    sigma_b = max(sigma_floor_log, 1.4826 * mad)
    thr = mu_b + max(k_sigma * sigma_b, log_margin)

    zmax_vox = min(int(z_max_um / z_um), len(smoothed) - sustain_vox)
    for z in range(max(0, zmax_vox + 1)):
        if (smoothed[z:z + sustain_vox] > thr).all():
            return z, thr, mu_b, sigma_b, smoothed
    return -1, thr, mu_b, sigma_b, smoothed


def detect_on_grid(vol, z_um, xi, yi, patch_w=PATCH_W, **kwargs):
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)
    n = len(xi)
    zs = np.empty(n)
    thrs = np.empty(n)
    mus = np.empty(n)
    sigmas = np.empty(n)
    for i, (iy, ix) in enumerate(zip(yi, xi)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        z_vox, thr, mu_b, sigma_b, _ = col_detect_baseline(
            log_col, z_um, **kwargs)
        zs[i] = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr; mus[i] = mu_b; sigmas[i] = sigma_b
    return zs, thrs, mus, sigmas


def _iter08_surface(vol, z_um, xy_um):
    """Reproduce iter08's selected surface for side-by-side comparison."""
    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    log_vol = np.log(vol + EPS)
    log_cols = _patch_cols(log_vol, xi, yi)
    (tf, zs, _, med), _ = select_trans_frac(log_cols, z_um, CZ_TARGET_Z_UM)
    fit, _, _ = _iter08_fit(xs_um, ys_um, zs)
    return fit, tf, med


def _cz_existing_surface(s):
    try:
        return analyze_subject(s).get("cz_surface")
    except Exception:
        return None


def _cz_centroid_surface(s):
    try:
        cz_um = cz_px_to_um(
            s.cz_centroids[["z_px", "y_px", "x_px"]].values, s)
        return estimate_pia_surface(cz_um[:, [2, 1, 0]])
    except Exception:
        return None


def _surface_z(surface, xs, y0):
    if surface is None:
        return np.full_like(np.asarray(xs, dtype=float), np.nan)
    a, b, c = surface["a"], surface["b"], surface["c"]
    p = surface.get("p", 0.0); q = surface.get("q", 0.0)
    r = surface.get("r", 0.0)
    xs = np.asarray(xs, dtype=float)
    return (a * xs + b * y0 + c + p * xs * xs
            + q * xs * y0 + r * y0 * y0)


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs, thrs, mus, sigmas = detect_on_grid(vol, z_um, xi, yi)
    valid = np.isfinite(zs)
    print(f"  detection: {valid.sum()}/{len(zs)} columns valid in "
          f"[0, {Z_MAX_UM:.0f}] µm")
    print(f"  baseline μ: {np.nanmedian(mus):.2f}  "
          f"σ: {np.nanmedian(sigmas):.3f}  "
          f"thr: {np.nanmedian(thrs):.2f}")
    if valid.sum() == 0:
        print("  no valid columns — subject flagged for review")
        return None, zs

    med_col = float(np.nanmedian(zs))
    print(f"  median column z = {med_col:.1f} µm  "
          f"(|Δ prior| = {abs(med_col - CZ_TARGET_Z_UM):.1f})")

    # Surface fit, MAD-gated around the sample median (not around prior)
    gate = mad_gate(zs, k=3.0)
    polyfit = fit_polysurf(xs_um[gate], ys_um[gate], zs[gate],
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    if polyfit is None:
        print("  polyfit failed")
        return None, zs

    iter08_fit, iter08_tf, iter08_col_med = _iter08_surface(vol, z_um, xy_um)
    z_iter09_grid = eval_polysurf(polyfit, xs_um, ys_um)
    iter09_surf_med = float(np.nanmedian(z_iter09_grid))

    z_iter08_grid = eval_polysurf(iter08_fit, xs_um, ys_um)
    iter08_surf_med = float(np.nanmedian(z_iter08_grid))

    cz_existing = _cz_existing_surface(s)
    cz_centroid = _cz_centroid_surface(s)

    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_lo = int(Y_INTERIOR_FRAC * Y); y_hi = Y - 1 - y_lo
    y_idx = np.linspace(y_lo, y_hi, N_Y).astype(int)

    fig, axes = plt.subplots(1, N_Y, figsize=(5 * N_Y, 4.8), sharey=True)
    for ax, iy in zip(axes, y_idx):
        y_const = iy * xy_um
        img = np.log(vol[:, iy, :] + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                  vmin=vmin, vmax=vmax)
        ax.axhline(CZ_TARGET_Z_UM, color="gold", lw=1.0, ls="--",
                   label="50 µm target (reference)")
        ax.axhline(Z_BASELINE_UM, color="white", lw=0.6, ls=":",
                   label=f"baseline cap ({Z_BASELINE_UM:.0f} µm)")
        ax.axhline(Z_MAX_UM, color="white", lw=0.6, ls="-.",
                   label=f"search cap ({Z_MAX_UM:.0f} µm)")
        if cz_centroid is not None:
            ax.plot(x_um, _surface_z(cz_centroid, x_um, y_const),
                    color="tab:green", lw=1.1, label="CZ centroid")
        ax.plot(x_um,
                eval_polysurf(iter08_fit, x_um,
                              np.full_like(x_um, y_const)),
                color="tab:cyan", lw=1.2,
                label=f"iter08 (tf={iter08_tf})")
        ax.plot(x_um,
                eval_polysurf(polyfit, x_um,
                              np.full_like(x_um, y_const)),
                color="tab:red", lw=1.8, label="iter09 (baseline + search)")
        ax.set_xlabel("x (µm)")
        ax.set_title(f"y = {y_const:.0f} µm")
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("z (µm)")
    fig.suptitle(
        f"{sid} — iter09 CZ: baseline (z<{Z_BASELINE_UM:.0f} µm, "
        f"K={K_SIGMA}·σ), search in [0, {Z_MAX_UM:.0f}] µm. "
        f"Median surf z: iter09 = {iter09_surf_med:.1f} µm, "
        f"iter08 = {iter08_surf_med:.1f} µm, target = {CZ_TARGET_Z_UM:.0f} µm.",
        fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_FIG / f"iter09_cz_{sid}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    np.savez(
        OUT_DATA / f"iter09_cz_transitions_{sid}.npz",
        xs_um=xs_um, ys_um=ys_um, zs=zs, thrs=thrs,
        mus=mus, sigmas=sigmas, gate=gate,
        z_baseline_um=np.float32(Z_BASELINE_UM),
        z_max_um=np.float32(Z_MAX_UM),
        k_sigma=np.float32(K_SIGMA),
    )

    return dict(
        subject=sid,
        cz_shape=f"({Z}, {Y}, {X})",
        z_baseline_um=Z_BASELINE_UM,
        z_max_um=Z_MAX_UM,
        k_sigma=K_SIGMA,
        median_baseline_mu=float(np.nanmedian(mus)),
        median_baseline_sigma=float(np.nanmedian(sigmas)),
        median_thr=float(np.nanmedian(thrs)),
        n_valid_in_search_window=int(valid.sum()),
        median_col_z_iter09=med_col,
        median_surf_z_iter09=iter09_surf_med,
        median_surf_z_iter08=iter08_surf_med,
        abs_dev_iter09_from_prior=abs(iter09_surf_med - CZ_TARGET_Z_UM),
        abs_dev_iter08_from_prior=abs(iter08_surf_med - CZ_TARGET_Z_UM),
        iter08_tf=iter08_tf,
    ), zs


def main():
    rows = []
    for sid in SUBJECTS:
        r, _ = render_subject(sid)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows)
    out = OUT_DATA / "iter09_cz_summary.csv"
    df.to_csv(out, index=False)
    cols = [
        "subject", "n_valid_in_search_window",
        "median_baseline_mu", "median_baseline_sigma", "median_thr",
        "median_col_z_iter09", "median_surf_z_iter09",
        "median_surf_z_iter08",
        "abs_dev_iter09_from_prior", "abs_dev_iter08_from_prior",
        "iter08_tf",
    ]
    print("\n=== iter09 CZ summary ===")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
