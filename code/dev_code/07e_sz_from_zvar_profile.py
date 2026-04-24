"""Session 07e — sxy from xy-kNN + sz from σ_z(d) profile NCC, iterated.

Pipeline per subject:

0. Load subject; compute R1 (rigid rotation + tilt). Apply with
   scales = `s0 = [2, 2, 2]` (population prior, not GT).
1. xy window = AABB of CZ-post-R1. Depth band = matched
   [d_skin, min(s_k_z·d_CZ_bottom, d_HCR_bottom)] for sxy only.
   sxy from median xy-kNN(k=5) ratio on both clouds restricted
   to (xy window ∩ depth band).
2. σ_z(d) sliding-window on pia-relative depth. Asymmetric z range:
   CZ over [d_skin, d_CZ_bottom_in_HCR], HCR over full
   [d_skin, d_HCR_bottom].
3. sz from Pearson NCC of σ_z_HCR(d) vs stretched-CZ σ_z(d); both
   depth axis and σ_z magnitude scale by candidate c_z.
4. Iterate: s_{k+1} = s_k · c_{k+1}; converge when |c-1| < 0.01.
5. GT scoring via landmark-Procrustes; grep-gated — no leak into
   estimator path.

Outputs
-------
- `sessions/07e_sz_from_zvar_profile/results.json`.
- Per-subject figures in `figures/`.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import (  # noqa: E402
    analyze_subject,
    depth_from_surface,
    fit_anisotropic_similarity,
)
from benchmark_data_loader import (  # noqa: E402
    BENCHMARK_SUBJECTS,
    cz_px_to_um,
    hcr_px_to_um,
    landmark_pairs_um,
    load_subject,
)
from r1_revised import coarse_align_revised  # noqa: E402

SESSION_DIR = Path(__file__).resolve().parents[1] / "sessions" / "07e_sz_from_zvar_profile"
FIG_DIR = SESSION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Primary subjects for the probe
PRIMARY_SUBJECTS = ["788406", "790322"]

# Parameters
S0_DEFAULT = np.array([2.0, 2.0, 2.0])
D_SKIN_UM = 100.0
WINDOW_DEPTH_UM = 100.0
STRIDE_UM = 25.0
N_MIN_PER_BIN = 50
KNN_K = 5
CZ_GRID = np.arange(0.5, 2.0 + 1e-9, 0.01)
OVERLAP_MIN_UM = 400.0
MAX_ITER = 5
CONV_TOL = 0.01


# ----------------------------------------------------------------------
@dataclass
class IterationRecord:
    sxy: float
    sz: float
    c_xy: float
    c_z: float
    ncc_best: float
    overlap_um: float
    n_cz_knn: int
    n_hcr_knn: int


@dataclass
class SubjectResult:
    subject: str
    sxy: float
    sz: float
    sxy_gt: float
    sz_gt: float
    rel_err_sxy: float
    rel_err_sz: float
    pass5_sxy: bool
    pass5_sz: bool
    pass5_both: bool
    iterations: list[IterationRecord]
    converged: bool
    ncc_curve: list[float]
    sxy_trace: list[float]
    sz_trace: list[float]
    # profiles at final iterate (for plotting)
    cz_depth_grid: list[float]
    cz_sigma_z: list[float]
    cz_n: list[int]
    hcr_depth_grid: list[float]
    hcr_sigma_z: list[float]
    hcr_n: list[int]


# ----------------------------------------------------------------------
def _gt_scales(s) -> tuple[float, float]:
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    sxy_gt = float(np.sqrt(fit.scales[0] * fit.scales[1]))
    sz_gt = float(fit.scales[2])
    return sxy_gt, sz_gt


def _cz_xyz_um(s) -> np.ndarray:
    """All CZ centroids as (x, y, z) µm."""
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    um = cz_px_to_um(arr, s)
    return um[:, [2, 1, 0]]  # to (x, y, z)


def _hcr_gfp_xyz_um(s) -> np.ndarray:
    """HCR-GFP+ centroids as (x, y, z) µm, via hcr_id join."""
    if s.hcr_gfp_df.empty:
        return np.zeros((0, 3))
    gfp_ids = set(s.hcr_gfp_df["hcr_id"].astype(int).tolist())
    hcr_px = s.hcr_centroids.copy()
    hcr_px["_keep"] = hcr_px["hcr_id"].astype(int).isin(gfp_ids)
    arr = hcr_px[hcr_px["_keep"]][["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    if len(arr) == 0:
        return np.zeros((0, 3))
    um = hcr_px_to_um(arr, s)
    return um[:, [2, 1, 0]]


def _r1_minimal(cz_um: np.ndarray, hcr_gfp_um: np.ndarray,
                cz_surface: dict, hcr_surface: dict):
    """R1 rigid (R, t_centroid) only — scales replaced by caller."""
    fit = coarse_align_revised(
        cz_um, hcr_gfp_um, cz_surface, hcr_surface,
        aniso_refine=False,
    )
    return fit


def _apply_with_scales(src_xyz_um: np.ndarray, fit, scales_xyz: np.ndarray) -> np.ndarray:
    """Apply R1 rotation + given scales + R1 centroid translation."""
    src_c = src_xyz_um - fit.src_mean
    return (src_c @ fit.R) * scales_xyz + fit.minimal_translation


# ----------------------------------------------------------------------
def _median_knn_xy(points_xyz_um: np.ndarray, k: int = KNN_K) -> float:
    """Median k-NN distance on (x, y) only."""
    if len(points_xyz_um) < k + 1:
        return float("nan")
    xy = points_xyz_um[:, :2]
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=k + 1)
    return float(np.median(dists[:, k]))


def _sliding_sigma_z(
    points_xyz_um: np.ndarray,
    depth_um: np.ndarray,
    xy_window: tuple[float, float, float, float],
    d_lo: float,
    d_hi: float,
    window_depth_um: float = WINDOW_DEPTH_UM,
    stride_um: float = STRIDE_UM,
    n_min: int = N_MIN_PER_BIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """σ_z(d) and n(d) on sliding windows over depth.

    Only cells inside the xy window contribute. Returns the common
    grid (d_c) and (σ_z, n) arrays; bins with n < n_min are NaN.
    """
    x0, x1, y0, y1 = xy_window
    xy_mask = (
        (points_xyz_um[:, 0] >= x0)
        & (points_xyz_um[:, 0] <= x1)
        & (points_xyz_um[:, 1] >= y0)
        & (points_xyz_um[:, 1] <= y1)
    )
    z = points_xyz_um[xy_mask, 2]
    d = depth_um[xy_mask]

    half = window_depth_um / 2
    # Use stride-aligned grid
    d_grid = np.arange(d_lo, d_hi + 1e-9, stride_um)
    sigma = np.full(len(d_grid), np.nan)
    counts = np.zeros(len(d_grid), dtype=int)
    # Build sort index once for a linear sweep
    order = np.argsort(d)
    d_sorted = d[order]
    z_sorted = z[order]
    for i, d_c in enumerate(d_grid):
        lo, hi = d_c - half, d_c + half
        left = np.searchsorted(d_sorted, lo, side="left")
        right = np.searchsorted(d_sorted, hi, side="right")
        n = right - left
        counts[i] = n
        if n >= n_min:
            sigma[i] = float(np.std(z_sorted[left:right]))
    return d_grid, sigma, counts


def _pearson_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson NCC on aligned vectors (NaNs skipped)."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return float("nan")
    aa = a[m] - a[m].mean()
    bb = b[m] - b[m].mean()
    da = np.sqrt((aa ** 2).sum())
    db = np.sqrt((bb ** 2).sum())
    if da == 0 or db == 0:
        return float("nan")
    return float((aa * bb).sum() / (da * db))


def _fit_sz_via_ncc(
    cz_d_grid: np.ndarray,
    cz_sigma_z: np.ndarray,
    cz_n: np.ndarray,
    hcr_d_grid: np.ndarray,
    hcr_sigma_z: np.ndarray,
    hcr_n: np.ndarray,
    d_anchor: float,
    c_z_grid: np.ndarray = CZ_GRID,
    overlap_min_um: float = OVERLAP_MIN_UM,
) -> tuple[float, float, float, np.ndarray]:
    """Return (c_z_best, ncc_best, overlap_um, ncc_curve)."""
    # HCR profile on its native stride grid
    h_valid = np.isfinite(hcr_sigma_z)
    if h_valid.sum() < 4:
        return 1.0, float("nan"), 0.0, np.full_like(c_z_grid, np.nan, dtype=float)

    ncc_curve = np.full(len(c_z_grid), -np.inf)
    for i, c_z in enumerate(c_z_grid):
        # Stretched CZ: d' = c_z * (d - d_anchor) + d_anchor,  σ' = c_z * σ
        d_cz_stretched = c_z * (cz_d_grid - d_anchor) + d_anchor
        sig_cz_stretched = c_z * cz_sigma_z
        # Interpolate stretched CZ onto HCR grid (NaN outside range)
        valid = np.isfinite(sig_cz_stretched)
        if valid.sum() < 4:
            continue
        # sort to monotonic
        order = np.argsort(d_cz_stretched[valid])
        x_s = d_cz_stretched[valid][order]
        y_s = sig_cz_stretched[valid][order]
        # linear interpolation, NaN outside
        cz_on_hcr = np.interp(hcr_d_grid, x_s, y_s,
                              left=np.nan, right=np.nan)
        # overlap = where both stretched-CZ and HCR are finite
        mask = np.isfinite(cz_on_hcr) & np.isfinite(hcr_sigma_z)
        if mask.sum() < 4:
            continue
        overlap_um_i = (
            mask.sum() * float(hcr_d_grid[1] - hcr_d_grid[0])
            if len(hcr_d_grid) >= 2 else 0.0
        )
        if overlap_um_i < overlap_min_um:
            continue
        ncc_curve[i] = _pearson_ncc(cz_on_hcr[mask], hcr_sigma_z[mask])

    if not np.any(np.isfinite(ncc_curve)):
        return 1.0, float("nan"), 0.0, ncc_curve

    i_best = int(np.nanargmax(ncc_curve))
    c_z_best = float(c_z_grid[i_best])
    ncc_best = float(ncc_curve[i_best])

    # Recompute overlap at best for reporting
    d_cz_stretched = c_z_best * (cz_d_grid - d_anchor) + d_anchor
    sig_cz_stretched = c_z_best * cz_sigma_z
    valid = np.isfinite(sig_cz_stretched)
    x_s = d_cz_stretched[valid]
    order = np.argsort(x_s)
    x_s = x_s[order]
    y_s = sig_cz_stretched[valid][order]
    cz_on_hcr = np.interp(hcr_d_grid, x_s, y_s, left=np.nan, right=np.nan)
    mask = np.isfinite(cz_on_hcr) & np.isfinite(hcr_sigma_z)
    overlap_um = (
        mask.sum() * float(hcr_d_grid[1] - hcr_d_grid[0])
        if len(hcr_d_grid) >= 2 else 0.0
    )
    return c_z_best, ncc_best, float(overlap_um), ncc_curve


# ----------------------------------------------------------------------
def run_subject(sid: str, s0: np.ndarray = S0_DEFAULT) -> SubjectResult:
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf = info["cz_surface"]
    hcr_surf = info["hcr_surface"]
    hcr_bot = info.get("hcr_bottom_surface")
    if hcr_bot is None:
        # fall back to iter08 direct
        try:
            from surfaces_iter08 import get_hcr_bottom_surface_iter08
            hcr_bot = get_hcr_bottom_surface_iter08(s)
        except Exception as exc:
            raise RuntimeError(f"No HCR bottom surface for {sid}: {exc}")

    cz_um = _cz_xyz_um(s)
    hcr_gfp_um = _hcr_gfp_xyz_um(s)

    # R1 rigid (scales inside fit not used — we supply our own)
    r1_fit = _r1_minimal(cz_um, hcr_gfp_um, cz_surf, hcr_surf)

    # CZ native depth range (in CZ frame)
    cz_depth_native = depth_from_surface(cz_um, cz_surf)
    d_CZ_native_bottom = float(np.nanpercentile(cz_depth_native, 99))

    # HCR tissue range: compute from HCR centroid cloud depth-under-pia,
    # and under-HCR-bottom is where tissue ends.
    hcr_um_all = np.column_stack([
        *[s.hcr_centroids[k].to_numpy(dtype=float) for k in ("x_px", "y_px", "z_px")][-3:]
    ])
    # Rebuild in correct order: x, y, z
    hcr_all_px = s.hcr_centroids[["z_px", "y_px", "x_px"]].to_numpy(dtype=float)
    hcr_all_um = hcr_px_to_um(hcr_all_px, s)[:, [2, 1, 0]]
    # HCR bottom depth as "depth of bottom surface under pia"
    # Sample a grid of xy points inside the HCR cloud to measure pia->bottom
    xs_probe = hcr_all_um[:, 0]
    ys_probe = hcr_all_um[:, 1]
    # pia and bottom z at those xy (use depth_from_surface with 0 z to get surface z)
    zero = np.zeros_like(xs_probe)
    probe_xyz = np.column_stack([xs_probe, ys_probe, zero])
    pia_z = zero - depth_from_surface(probe_xyz, hcr_surf)  # since depth = z - surf_z
    bot_z = zero - depth_from_surface(probe_xyz, hcr_bot)
    d_HCR_bottom = float(np.nanpercentile(bot_z - pia_z, 90))

    # Iteration
    s_k = s0.astype(float).copy()
    iterations: list[IterationRecord] = []
    converged = False
    last_ncc_curve = np.full(len(CZ_GRID), np.nan)
    last_sxy_trace = [float(s_k[0])]
    last_sz_trace = [float(s_k[2])]
    last_cz_d = np.array([])
    last_cz_sigma = np.array([])
    last_cz_n = np.array([])
    last_hcr_d = np.array([])
    last_hcr_sigma = np.array([])
    last_hcr_n = np.array([])

    for k in range(MAX_ITER):
        cz_in_hcr = _apply_with_scales(cz_um, r1_fit, s_k)
        cz_depth_hcr = depth_from_surface(cz_in_hcr, hcr_surf)
        hcr_gfp_depth = depth_from_surface(hcr_gfp_um, hcr_surf)

        # xy window = AABB of CZ-in-HCR
        x0, x1 = float(cz_in_hcr[:, 0].min()), float(cz_in_hcr[:, 0].max())
        y0, y1 = float(cz_in_hcr[:, 1].min()), float(cz_in_hcr[:, 1].max())
        xy_window = (x0, x1, y0, y1)

        # --- Step 1: sxy from xy kNN on matched-depth crop ---
        d_CZ_bottom_in_HCR = s_k[2] * d_CZ_native_bottom
        d_sxy_hi = min(d_CZ_bottom_in_HCR, d_HCR_bottom)
        d_sxy_lo = D_SKIN_UM

        def _xy_depth_crop(pts, depths):
            m = (
                (depths >= d_sxy_lo)
                & (depths <= d_sxy_hi)
                & (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            )
            return pts[m]

        cz_xykn = _xy_depth_crop(cz_in_hcr, cz_depth_hcr)
        hcr_xykn = _xy_depth_crop(hcr_gfp_um, hcr_gfp_depth)
        knn_cz = _median_knn_xy(cz_xykn)
        knn_hcr = _median_knn_xy(hcr_xykn)
        if not (np.isfinite(knn_cz) and np.isfinite(knn_hcr)) or knn_cz <= 0:
            raise RuntimeError(f"kNN degenerate iter {k} sid {sid} "
                               f"n_cz={len(cz_xykn)} n_hcr={len(hcr_xykn)}")
        c_xy = knn_hcr / knn_cz
        sxy_new = float(s_k[0] * c_xy)

        # --- Step 2: σ_z profiles, asymmetric z range ---
        cz_d_grid, cz_sigma, cz_n = _sliding_sigma_z(
            cz_in_hcr, cz_depth_hcr, xy_window,
            D_SKIN_UM, d_CZ_bottom_in_HCR,
        )
        hcr_d_grid, hcr_sigma, hcr_n = _sliding_sigma_z(
            hcr_gfp_um, hcr_gfp_depth, xy_window,
            D_SKIN_UM, d_HCR_bottom,
        )

        # --- Step 3: sz from σ_z NCC ---
        # Anchor at R1 translation's pia-depth
        t_xyz = r1_fit.minimal_translation.reshape(1, 3)
        d_anchor = float(depth_from_surface(t_xyz, hcr_surf)[0])
        c_z, ncc_best, overlap_um, ncc_curve = _fit_sz_via_ncc(
            cz_d_grid, cz_sigma, cz_n,
            hcr_d_grid, hcr_sigma, hcr_n,
            d_anchor=d_anchor,
        )
        sz_new = float(s_k[2] * c_z)

        iterations.append(IterationRecord(
            sxy=sxy_new, sz=sz_new, c_xy=float(c_xy), c_z=float(c_z),
            ncc_best=ncc_best, overlap_um=overlap_um,
            n_cz_knn=int(len(cz_xykn)), n_hcr_knn=int(len(hcr_xykn)),
        ))
        last_ncc_curve = ncc_curve
        last_sxy_trace.append(sxy_new)
        last_sz_trace.append(sz_new)
        last_cz_d = cz_d_grid
        last_cz_sigma = cz_sigma
        last_cz_n = cz_n
        last_hcr_d = hcr_d_grid
        last_hcr_sigma = hcr_sigma
        last_hcr_n = hcr_n

        if abs(c_xy - 1.0) < CONV_TOL and abs(c_z - 1.0) < CONV_TOL:
            converged = True
            s_k = np.array([sxy_new, sxy_new, sz_new])
            break
        # damping: half-step if we're moving a lot
        update = np.array([sxy_new, sxy_new, sz_new])
        s_k = 0.5 * s_k + 0.5 * update

    sxy_gt, sz_gt = _gt_scales(s)
    err_sxy = (float(s_k[0]) - sxy_gt) / sxy_gt
    err_sz = (float(s_k[2]) - sz_gt) / sz_gt

    return SubjectResult(
        subject=sid,
        sxy=float(s_k[0]), sz=float(s_k[2]),
        sxy_gt=sxy_gt, sz_gt=sz_gt,
        rel_err_sxy=float(err_sxy), rel_err_sz=float(err_sz),
        pass5_sxy=bool(abs(err_sxy) <= 0.05),
        pass5_sz=bool(abs(err_sz) <= 0.05),
        pass5_both=bool(abs(err_sxy) <= 0.05 and abs(err_sz) <= 0.05),
        iterations=iterations, converged=converged,
        ncc_curve=[float(x) for x in last_ncc_curve],
        sxy_trace=[float(x) for x in last_sxy_trace],
        sz_trace=[float(x) for x in last_sz_trace],
        cz_depth_grid=[float(x) for x in last_cz_d],
        cz_sigma_z=[float(x) if np.isfinite(x) else None for x in last_cz_sigma],
        cz_n=[int(x) for x in last_cz_n],
        hcr_depth_grid=[float(x) for x in last_hcr_d],
        hcr_sigma_z=[float(x) if np.isfinite(x) else None for x in last_hcr_sigma],
        hcr_n=[int(x) for x in last_hcr_n],
    )


# ----------------------------------------------------------------------
def plot_subject(res: SubjectResult) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # σ_z profiles (final iterate)
    ax = axes[0]
    cz_d = np.array(res.cz_depth_grid)
    cz_sig = np.array([x if x is not None else np.nan for x in res.cz_sigma_z])
    hcr_d = np.array(res.hcr_depth_grid)
    hcr_sig = np.array([x if x is not None else np.nan for x in res.hcr_sigma_z])
    ax.plot(cz_d, cz_sig, 'o-', color='#268bd2', label='CZ-in-HCR', ms=3, lw=1.2)
    ax.plot(hcr_d, hcr_sig, 's-', color='#cb4b16', label='HCR-GFP+', ms=3, lw=1.2)
    ax.set_xlabel("pia depth (µm)")
    ax.set_ylabel("σ_z (µm)")
    ax.set_title(f"{res.subject} σ_z(d) final iterate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # NCC curve
    ax = axes[1]
    ncc = np.array(res.ncc_curve)
    cz_grid = CZ_GRID
    # x axis: absolute sz = s_prev_z * c_z ; we plot c_z directly
    ax.plot(cz_grid, ncc, color='#268bd2')
    ax.axvline(1.0, ls=':', color='grey', alpha=0.6, label='c_z=1 (fixed pt)')
    if res.iterations:
        c_z_last = res.iterations[-1].c_z
        ax.plot(c_z_last, res.iterations[-1].ncc_best, 'o',
                color='#859900', label=f'argmax {c_z_last:.2f}')
    ax.set_xlabel("c_z (σ_z stretch)")
    ax.set_ylabel("NCC")
    ax.set_title(f"NCC vs c_z (last iter)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Iteration trace
    ax = axes[2]
    ax.plot(res.sxy_trace, 'o-', color='#268bd2', label='sxy')
    ax.plot(res.sz_trace, 's-', color='#cb4b16', label='sz')
    ax.axhline(res.sxy_gt, ls='--', color='#268bd2', alpha=0.4, label=f'GT sxy={res.sxy_gt:.2f}')
    ax.axhline(res.sz_gt, ls='--', color='#cb4b16', alpha=0.4, label=f'GT sz={res.sz_gt:.2f}')
    ax.set_xlabel("iteration")
    ax.set_ylabel("scale")
    converged_str = "converged" if res.converged else "max iter"
    pass_str = "PASS" if res.pass5_both else "fail"
    ax.set_title(
        f"{res.subject} — {converged_str}, {pass_str}\n"
        f"sxy={res.sxy:.3f} ({res.rel_err_sxy*100:+.1f}%) "
        f"sz={res.sz:.3f} ({res.rel_err_sz*100:+.1f}%)"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / f"zvar_profile_{res.subject}.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  fig → {out}")


# ----------------------------------------------------------------------
def _result_to_dict(res: SubjectResult) -> dict:
    d = {
        "subject": res.subject,
        "sxy": res.sxy, "sz": res.sz,
        "sxy_gt": res.sxy_gt, "sz_gt": res.sz_gt,
        "rel_err_sxy_pct": res.rel_err_sxy * 100.0,
        "rel_err_sz_pct": res.rel_err_sz * 100.0,
        "pass5_sxy": res.pass5_sxy, "pass5_sz": res.pass5_sz,
        "pass5_both": res.pass5_both,
        "converged": res.converged,
        "iterations": [
            {
                "sxy": it.sxy, "sz": it.sz,
                "c_xy": it.c_xy, "c_z": it.c_z,
                "ncc_best": it.ncc_best,
                "overlap_um": it.overlap_um,
                "n_cz_knn": it.n_cz_knn, "n_hcr_knn": it.n_hcr_knn,
            }
            for it in res.iterations
        ],
        "sxy_trace": res.sxy_trace,
        "sz_trace": res.sz_trace,
        "ncc_curve": res.ncc_curve,
        "cz_depth_grid": res.cz_depth_grid,
        "cz_sigma_z": res.cz_sigma_z,
        "cz_n": res.cz_n,
        "hcr_depth_grid": res.hcr_depth_grid,
        "hcr_sigma_z": res.hcr_sigma_z,
        "hcr_n": res.hcr_n,
    }
    return d


def main(subjects: list[str] | None = None) -> None:
    if subjects is None:
        subjects = PRIMARY_SUBJECTS
    print(f"Running 07e on {subjects}  s0={S0_DEFAULT.tolist()}")
    summary = {}
    for sid in subjects:
        print(f"\n── {sid} ──")
        try:
            res = run_subject(sid)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()
            summary[sid] = {"status": "error", "error": str(exc)}
            continue
        print(f"  sxy = {res.sxy:.3f}  (GT {res.sxy_gt:.3f}, "
              f"err {res.rel_err_sxy*100:+.2f}%)  "
              f"sz = {res.sz:.3f}  (GT {res.sz_gt:.3f}, "
              f"err {res.rel_err_sz*100:+.2f}%)  "
              f"iter={len(res.iterations)}  "
              f"{'CONVERGED' if res.converged else 'max-iter'}")
        print(f"  pass5 sxy={res.pass5_sxy}  sz={res.pass5_sz}  "
              f"both={res.pass5_both}")
        plot_subject(res)
        d = _result_to_dict(res)
        d["status"] = "ok"
        summary[sid] = d

    n_pass_both = sum(
        1 for r in summary.values()
        if isinstance(r, dict) and r.get("pass5_both")
    )
    print(f"\n{n_pass_both}/{len(subjects)} subjects pass ±5 % on both axes")

    out = SESSION_DIR / "results.json"
    with open(out, "w") as f:
        json.dump({
            "summary": {
                "n_pass_both_5pct": n_pass_both,
                "n_attempted": len(subjects),
                "subjects": subjects,
                "s0": S0_DEFAULT.tolist(),
                "window_depth_um": WINDOW_DEPTH_UM,
                "stride_um": STRIDE_UM,
                "d_skin_um": D_SKIN_UM,
                "overlap_min_um": OVERLAP_MIN_UM,
            },
            "subjects": summary,
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].upper() == "ALL":
        main(BENCHMARK_SUBJECTS)
    elif args:
        main(args)
    else:
        main()
