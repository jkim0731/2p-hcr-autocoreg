"""Anisotropic ICP scale estimator (session 07).

Estimates per-axis scale ``(sxy, sz)`` between CZ GCaMP+ and HCR GFP+
centroids by iterating:

1. **Apply** the current affine ``(R, [sxy, sxy, sz], t)`` to CZ
   centroids (rotation `R` and translation `t` come from R1 minimal
   (R, t); scales start at a midpoint init and are re-estimated each
   iteration).
2. **Reciprocal-nearest-neighbour matching** against HCR GFP+ with an
   adaptive search radius ``r_it`` that shrinks each iteration.
3. **Inlier gating** by residual quantile.
4. **Procrustes refit** (``fit_anisotropic_similarity``) on the
   matched pairs ``(cz_i, hcr_j)`` — yields new ``(R', scales', t')``.
5. **Update** scales and iterate until the per-axis change falls
   below ``rel_tol`` or ``max_iter`` is reached.

Rationale (see `sessions/07_scale_failure_diagnosis/log.md` Part B):
the session-06 k-NN estimator was biased by a subject-specific
density disparity ``f = N_hcr_in_overlap / N_cz ∈ [0.66, 7.99]``.
Reciprocal-NN matching automatically excludes HCR extras (that have
no CZ partner) and CZ cells with no HCR counterpart — so the
Procrustes fit at step 4 uses approximately-matched populations, and
the `f` bias is removed by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from benchmark_analysis import ProcrustesFit, fit_anisotropic_similarity


# ----------------------------------------------------------------------
# Output dataclass
# ----------------------------------------------------------------------
@dataclass
class AnisotropicICPResult:
    sxy: Optional[float]
    sz: Optional[float]
    n_matched: int
    n_cz: int
    n_hcr: int
    iterations: int
    converged: bool
    fit: Optional[ProcrustesFit]
    history: list = field(default_factory=list)
    reason_unknown: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Apply the current affine to CZ
# ----------------------------------------------------------------------
def _apply_affine_row(cz_xyz: np.ndarray, R: np.ndarray,
                      src_mean: np.ndarray, scales: np.ndarray,
                      dst_mean: np.ndarray) -> np.ndarray:
    """pred = (cz - src_mean) @ R * scales + dst_mean  (row-vec convention)."""
    return (np.asarray(cz_xyz, dtype=float) - src_mean) @ R * scales + dst_mean


# ----------------------------------------------------------------------
# Reciprocal NN matching
# ----------------------------------------------------------------------
def _reciprocal_nn(mapped_cz: np.ndarray, hcr: np.ndarray,
                   r_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cz_indices, hcr_indices, distances) of mutual NN pairs.

    A pair (i, j) is kept iff:
      - j is the nearest hcr neighbour of mapped_cz_i (within r_um), AND
      - i is the nearest mapped_cz neighbour of hcr_j (within r_um).
    """
    tree_h = cKDTree(hcr)
    d_f, j_f = tree_h.query(mapped_cz, k=1, distance_upper_bound=r_um)
    tree_c = cKDTree(mapped_cz)
    d_r, i_r = tree_c.query(hcr, k=1, distance_upper_bound=r_um)

    n_cz = mapped_cz.shape[0]
    n_hcr = hcr.shape[0]
    cz_idx: list[int] = []
    hcr_idx: list[int] = []
    dists: list[float] = []
    for i in range(n_cz):
        j = j_f[i]
        if j >= n_hcr:  # cKDTree returns n (out-of-range) if no neighbour within r
            continue
        if i_r[j] == i:
            cz_idx.append(i)
            hcr_idx.append(int(j))
            dists.append(float(d_f[i]))
    return np.array(cz_idx, dtype=int), np.array(hcr_idx, dtype=int), np.array(dists, dtype=float)


# ----------------------------------------------------------------------
# R1 rotation + centroid translation adapter
# ----------------------------------------------------------------------
def _extract_r1_rt(coarse_fit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (R, src_mean, t_when_scales_are_1) out of an R1 CoarseAffineV2.

    R1's apply is: pred = (cz - src_mean) @ R * scales + translation.
    When scales = 1, pred = (cz - src_mean) @ R + translation.  For our
    ICP we keep R and src_mean fixed (from R1) and only update scales.
    translation is the HCR-side anchor; we treat it as dst_mean so
    pred = (cz - src_mean) @ R * scales + dst_mean when scales are
    re-applied.  Equivalent to R1's formula for any scales, by
    construction (R1 wrote translation as dst_mean under the row-vec
    convention).
    """
    R = np.asarray(coarse_fit.R, dtype=float)
    src_mean = np.asarray(coarse_fit.src_mean, dtype=float)
    dst_mean = np.asarray(coarse_fit.translation, dtype=float)
    return R, src_mean, dst_mean


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def estimate_scales_icp(
    cz_xyz_um: np.ndarray,
    hcr_gfp_xyz_um: np.ndarray,
    coarse_fit,
    *,
    sxy_init: float = 1.75,
    sz_init: float = 2.75,
    sxy_bounds: tuple[float, float] = (1.4, 2.0),
    sz_bounds: tuple[float, float] = (1.9, 4.0),
    max_iter: int = 20,
    rel_tol: float = 0.01,
    r_init_um: float = 150.0,
    r_min_um: float = 30.0,
    r_decay: float = 0.80,
    inlier_residual_quantile: float = 0.9,
    n_min_matched: int = 50,
) -> AnisotropicICPResult:
    """Iteratively refine ``(sxy, sz)`` via reciprocal-NN + Procrustes.

    Keeps R and the HCR-side anchor from R1; updates only the scales
    each iteration (preserves R1's rotation which grand-plan documents
    are trustworthy).

    Default init is the feasibility midpoint ``(sxy=1.75, sz=2.75)``
    (no per-subject priors; within the expansion-microscopy range
    documented in ``01 Data Description.md``).  Both initial and
    final values are clipped to the bounds.
    """
    cz_xyz = np.asarray(cz_xyz_um, dtype=float)
    hcr_xyz = np.asarray(hcr_gfp_xyz_um, dtype=float)
    n_cz = cz_xyz.shape[0]
    n_hcr = hcr_xyz.shape[0]

    R, src_mean, dst_mean = _extract_r1_rt(coarse_fit)

    sxy = float(np.clip(sxy_init, *sxy_bounds))
    sz = float(np.clip(sz_init, *sz_bounds))
    r = float(r_init_um)

    history: list = []
    last_fit: Optional[ProcrustesFit] = None
    converged = False

    for it in range(max_iter):
        scales = np.array([sxy, sxy, sz], dtype=float)
        mapped_cz = _apply_affine_row(cz_xyz, R, src_mean, scales, dst_mean)

        cz_idx, hcr_idx, d = _reciprocal_nn(mapped_cz, hcr_xyz, r_um=r)
        n_m = len(cz_idx)
        if n_m < n_min_matched:
            history.append({
                "it": it, "sxy": sxy, "sz": sz, "r_um": r,
                "n_matched": n_m, "status": "too_few_matched",
            })
            break

        # Inlier gate
        if inlier_residual_quantile is not None and inlier_residual_quantile < 1.0:
            r_cut = float(np.quantile(d, inlier_residual_quantile))
            keep = d <= r_cut
            cz_idx = cz_idx[keep]
            hcr_idx = hcr_idx[keep]
            d = d[keep]

        # Refit Procrustes on matched subset — anisotropic scale + rotation
        matched_cz = cz_xyz[cz_idx]
        matched_hcr = hcr_xyz[hcr_idx]
        try:
            fit = fit_anisotropic_similarity(matched_cz, matched_hcr)
        except ValueError:
            history.append({
                "it": it, "sxy": sxy, "sz": sz, "r_um": r,
                "n_matched": len(cz_idx), "status": "procrustes_failed",
            })
            break
        last_fit = fit

        new_sxy = float(np.clip(np.sqrt(fit.scales[0] * fit.scales[1]), *sxy_bounds))
        new_sz = float(np.clip(fit.scales[2], *sz_bounds))

        d_sxy = abs(new_sxy - sxy) / max(sxy, 1e-9)
        d_sz = abs(new_sz - sz) / max(sz, 1e-9)

        history.append({
            "it": it, "sxy": sxy, "sz": sz, "r_um": r,
            "n_matched_raw": n_m, "n_matched_inlier": int(len(cz_idx)),
            "median_residual_um": float(np.median(d)),
            "procrustes_rms_um": float(fit.rms_um),
            "procrustes_scales": [float(x) for x in fit.scales],
            "new_sxy": new_sxy, "new_sz": new_sz,
            "d_sxy": d_sxy, "d_sz": d_sz,
        })

        sxy, sz = new_sxy, new_sz
        r = max(r_min_um, r * r_decay)

        if d_sxy < rel_tol and d_sz < rel_tol:
            converged = True
            break

    n_matched_final = history[-1].get("n_matched_inlier", 0) if history else 0

    diagnostics = {
        "sxy_init": float(sxy_init),
        "sz_init": float(sz_init),
        "r_init_um": float(r_init_um),
        "r_min_um": float(r_min_um),
        "r_decay": float(r_decay),
        "inlier_residual_quantile": float(inlier_residual_quantile)
            if inlier_residual_quantile is not None else None,
        "n_cz": int(n_cz),
        "n_hcr": int(n_hcr),
    }

    if last_fit is None:
        return AnisotropicICPResult(
            sxy=None, sz=None,
            n_matched=int(n_matched_final), n_cz=int(n_cz), n_hcr=int(n_hcr),
            iterations=len(history), converged=False,
            fit=None, history=history,
            reason_unknown="no_valid_fit",
            diagnostics=diagnostics,
        )

    return AnisotropicICPResult(
        sxy=float(sxy), sz=float(sz),
        n_matched=int(n_matched_final),
        n_cz=int(n_cz), n_hcr=int(n_hcr),
        iterations=len(history), converged=converged,
        fit=last_fit, history=history,
        reason_unknown=None,
        diagnostics=diagnostics,
    )


# ----------------------------------------------------------------------
# Multi-start wrapper — sweeps sz_init to escape stationary basins
# ----------------------------------------------------------------------
def _evaluate_fit(cz_xyz: np.ndarray, hcr_xyz: np.ndarray,
                  R: np.ndarray, src_mean: np.ndarray, dst_mean: np.ndarray,
                  sxy: float, sz: float,
                  r_tight_um: float = 15.0,
                  r_wide_um: float = 50.0,
                  sxy_bounds: Optional[tuple] = None,
                  sz_bounds: Optional[tuple] = None) -> dict:
    """Score a final (sxy, sz) by tight-radius match count + sz-prior.

    Tight-radius count alone fails to discriminate between the correct
    scale and ICP's self-consistent *squeezed* stationary points — the
    squeezed basin (all CZ fitting into HCR range) often yields *more*
    reciprocal-NN pairs than the true basin, especially on subjects
    where HCR GFP+ has partial depth coverage (e.g. 782149 at
    HCR.z/CZ_mapped.z ≈ 0.4).

    To break this symmetry we add a weak linear sz prior ``+ 10 * sz``:
    expansion microscopy physics is always ``sz > 2``, and the squeezed
    stationary basin is typically at sz ∈ [1.9, 2.3], i.e. at the
    bottom of the feasibility range.  The +10*sz bias flips the
    ordering when the tight counts for two basins differ by < ~6,
    without affecting correctness on subjects where the tight counts
    are decisively higher at the correct basin.

    Boundary hits (sxy / sz clipped exactly to a feasibility bound by
    np.clip inside ICP) are flagged ``at_bound`` and penalised by
    ``-1e6`` — they are effectively rejected unless every candidate is
    clipped.  Rationale: a clipped basin indicates ICP wanted to drift
    outside the physical feasibility range; the clip is masking a
    spurious stationary attractor from the Procrustes step and the
    estimate sits at an arbitrary fence rather than a true local min.
    """
    scales = np.array([sxy, sxy, sz], dtype=float)
    mapped_cz = _apply_affine_row(cz_xyz, R, src_mean, scales, dst_mean)

    _, _, d_tight = _reciprocal_nn(mapped_cz, hcr_xyz, r_um=r_tight_um)
    n_tight = int(len(d_tight))

    cz_idx, hcr_idx, d_wide = _reciprocal_nn(mapped_cz, hcr_xyz, r_um=r_wide_um)
    n_wide = int(len(d_wide))
    if n_wide >= 5:
        mapped_matched = mapped_cz[cz_idx]
        hcr_matched = hcr_xyz[hcr_idx]
        delta = hcr_matched - mapped_matched
        xy_res = np.linalg.norm(delta[:, :2], axis=1)
        med_xy = float(np.median(xy_res))
        med_3d = float(np.median(d_wide))
    else:
        med_xy = None
        med_3d = None

    # Detect np.clip boundary hits (|x - bound| < eps → clipped exactly).
    eps = 1e-4
    at_bound = False
    if sxy_bounds is not None:
        lo, hi = sxy_bounds
        if sxy <= lo + eps or sxy >= hi - eps:
            at_bound = True
    if sz_bounds is not None:
        lo, hi = sz_bounds
        if sz <= lo + eps or sz >= hi - eps:
            at_bound = True

    # score = n_tight + sz prior + tie-break(1/med_xy) - boundary penalty
    score = float(n_tight) + 10.0 * float(sz)
    if med_xy is not None:
        score += 1.0 / (med_xy + 1.0)
    if at_bound:
        score -= 1e6

    return {"n_tight": n_tight, "n_wide": n_wide,
            "median_xy_um_wide": med_xy,
            "median_3d_um_wide": med_3d,
            "at_bound": at_bound,
            "score": score}


def estimate_scales_icp_multi_start(
    cz_xyz_um: np.ndarray,
    hcr_gfp_xyz_um: np.ndarray,
    coarse_fit,
    *,
    sxy_inits: tuple[float, ...] = (1.5, 1.75, 2.0),
    sz_inits: tuple[float, ...] = (2.25, 3.0, 3.75),
    r_tight_um: float = 15.0,
    r_wide_um: float = 50.0,
    **icp_kwargs,
) -> AnisotropicICPResult:
    """Run ICP from a grid of ``(sxy_init, sz_init)`` starts; pick the
    basin with the most tight-radius reciprocal-NN matches.

    Scoring: ``score = n_matched(r_tight) + 1/(median_xy(r_wide)+1)``.
    The tight count dominates; the wide-radius xy median breaks ties.

    Returns the winning ``AnisotropicICPResult`` with a ``multi_start``
    block appended to ``diagnostics`` recording every attempted start.
    """
    cz_xyz = np.asarray(cz_xyz_um, dtype=float)
    hcr_xyz = np.asarray(hcr_gfp_xyz_um, dtype=float)
    R, src_mean, dst_mean = _extract_r1_rt(coarse_fit)

    # Pull bounds from icp_kwargs or defaults so _evaluate_fit can flag
    # boundary-clipped basins.
    sxy_bounds = icp_kwargs.get("sxy_bounds", (1.4, 2.0))
    sz_bounds = icp_kwargs.get("sz_bounds", (1.9, 4.0))

    starts: list[dict] = []
    best_result: Optional[AnisotropicICPResult] = None
    best_score: float = -float("inf")
    best_ev: dict = {}
    best_init: tuple[float, float] = (float("nan"), float("nan"))

    for sxy_init in sxy_inits:
        for sz_init in sz_inits:
            r = estimate_scales_icp(
                cz_xyz_um=cz_xyz, hcr_gfp_xyz_um=hcr_xyz, coarse_fit=coarse_fit,
                sxy_init=sxy_init, sz_init=sz_init, **icp_kwargs,
            )
            if r.sxy is None or r.sz is None:
                starts.append({
                    "sxy_init": float(sxy_init), "sz_init": float(sz_init),
                    "status": "icp_failed",
                })
                continue

            ev = _evaluate_fit(cz_xyz, hcr_xyz, R, src_mean, dst_mean,
                               r.sxy, r.sz, r_tight_um=r_tight_um,
                               r_wide_um=r_wide_um,
                               sxy_bounds=sxy_bounds, sz_bounds=sz_bounds)
            starts.append({
                "sxy_init": float(sxy_init), "sz_init": float(sz_init),
                "sxy_final": float(r.sxy), "sz_final": float(r.sz),
                "iterations": int(r.iterations),
                "converged": bool(r.converged),
                **ev,
            })

            score = ev["score"]
            if score > best_score:
                best_score = score
                best_result = r
                best_ev = ev
                best_init = (float(sxy_init), float(sz_init))

    if best_result is None:
        return AnisotropicICPResult(
            sxy=None, sz=None,
            n_matched=0, n_cz=int(cz_xyz.shape[0]), n_hcr=int(hcr_xyz.shape[0]),
            iterations=0, converged=False, fit=None, history=[],
            reason_unknown="all_starts_failed",
            diagnostics={"multi_start": {"starts": starts}},
        )

    best_result.diagnostics = {
        **best_result.diagnostics,
        "multi_start": {
            "sxy_inits": [float(x) for x in sxy_inits],
            "sz_inits": [float(x) for x in sz_inits],
            "r_tight_um": float(r_tight_um),
            "r_wide_um": float(r_wide_um),
            "picked_init": {"sxy": best_init[0], "sz": best_init[1]},
            "picked_score": float(best_score),
            "picked_eval": best_ev,
            "starts": starts,
        },
    }
    return best_result


# ----------------------------------------------------------------------
# Synthetic sanity
# ----------------------------------------------------------------------
def _synthetic_sanity(seed: int = 0, n: int = 900,
                      sxy_true: float = 1.77, sz_true: float = 2.82,
                      extras_factor: float = 1.0,
                      noise_um: float = 5.0) -> dict:
    rng = np.random.default_rng(seed)
    cz = rng.uniform(0.0, 400.0, size=(n, 3))
    hcr_matched = cz.copy()
    hcr_matched[:, 0] *= sxy_true
    hcr_matched[:, 1] *= sxy_true
    hcr_matched[:, 2] *= sz_true
    hcr_matched += rng.normal(0, noise_um, size=hcr_matched.shape)

    if extras_factor > 0:
        n_extra = int(extras_factor * n)
        extras = np.column_stack([
            rng.uniform(-200, 400 * sxy_true + 200, size=n_extra),
            rng.uniform(-200, 400 * sxy_true + 200, size=n_extra),
            rng.uniform(-200, 400 * sz_true + 200, size=n_extra),
        ])
        hcr = np.vstack([hcr_matched, extras])
    else:
        hcr = hcr_matched

    class _IdentityFit:
        R = np.eye(3)
        scales = np.array([1.0, 1.0, 1.0])
        src_mean = cz.mean(axis=0)
        translation = hcr_matched.mean(axis=0)  # HCR anchor at matched centroid

    out = estimate_scales_icp(
        cz_xyz_um=cz, hcr_gfp_xyz_um=hcr,
        coarse_fit=_IdentityFit(),
    )
    return {
        "sxy_true": sxy_true, "sz_true": sz_true,
        "extras_factor": extras_factor, "noise_um": noise_um,
        "sxy_est": out.sxy, "sz_est": out.sz,
        "n_matched": out.n_matched, "n_cz": out.n_cz, "n_hcr": out.n_hcr,
        "iterations": out.iterations, "converged": out.converged,
    }


if __name__ == "__main__":
    import json
    print("Synthetic sanity (matched + extras + match noise):")
    for ef in (0.0, 1.0, 3.0, 7.0):
        r = _synthetic_sanity(extras_factor=ef)
        print(json.dumps(r, indent=2))
