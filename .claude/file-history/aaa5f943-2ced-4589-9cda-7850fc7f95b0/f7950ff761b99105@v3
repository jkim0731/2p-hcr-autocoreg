"""F9 candidates that consume the Stage A locked-prior warm-start.

P4, P6, and C5 don't accept a `cz_init` argument — they always call
`lib.centroid_helpers.default_warmstart_zyx` internally. We monkey-patch
that function for the duration of each LP candidate's invocation so the
v1 candidate code is unchanged.

Registers candidate IDs:
  P1_LP   — P1 + locked-prior warm-start
  P4_LP   — P4 + locked-prior warm-start
  P6_LP   — P6 + locked-prior warm-start
  C5_LP   — C5 dispatch with all three legs warm-started by the prior
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
_V2 = _THIS.parent
_CODE = _V2.parent
_V1 = _CODE / "full_automatic_execution_01"
for p in (_V1, _V1 / "lib", _CODE / "dev_code", _V2):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from bench.harness import register_candidate, CoregResult  # noqa: E402
from bench.candidate_impls._p1_teaser import run_p1  # noqa: E402
from bench.candidate_impls._p4_spectral import run_p4  # noqa: E402
from bench.candidate_impls._p6_bcpd import run_p6  # noqa: E402
from bench.candidate_impls._c5_p1_p4_p6_ensemble import run_c5  # noqa: E402
import lib.centroid_helpers as _ch  # noqa: E402

from locked_prior_warm import (  # noqa: E402
    LockedPriorWarmStart,
    apply_to_cz_um,
    compute_locked_prior_warm_start,
)


def _refine_from_lp(cz_um_zyx, hcr_um_zyx, lp: LockedPriorWarmStart):
    """LP cloud → tight translation refinement → iterative local refit.

    The LP already locks the global basin (warped CZ mean within ~50 µm
    of HCR mean for 5/6 subjects). We deliberately *skip* v1's
    ``estimate_scales_icp_multi_start`` and the ±300 µm coarse grid
    because both pulled the cloud out of the LP basin into a deeper
    spurious-denser basin (e.g. on 788406 the multistart-ICP scales fell
    to sxy=1.72 / sz=1.95 and the cloud shifted +760 µm in z, dropping
    r@30 from 757/932 → 91/932). Instead we keep a tight ±60 µm fine
    grid plus an inlier-Procrustes local refit guarded by an
    inlier-count monotone gate.
    """
    from benchmark_analysis import fit_anisotropic_similarity as _fas
    from scipy.spatial import cKDTree

    cz_um_zyx = np.asarray(cz_um_zyx, dtype=float)
    hcr_um_zyx = np.asarray(hcr_um_zyx, dtype=float)

    cz_init_zyx = apply_to_cz_um(lp, cz_um_zyx)
    cz_xyz = cz_um_zyx[:, [2, 1, 0]]
    hcr_xyz = hcr_um_zyx[:, [2, 1, 0]]
    pred_xyz = cz_init_zyx[:, [2, 1, 0]].copy()

    info: dict = {
        "warmstart": "locked_prior_v2_lp_only",
        "subject_id": lp.subject_id,
        "lp_translation_zyx": lp.translation.tolist(),
        "lp_scales_zyx": lp.scales.tolist(),
        "lp_pwr_ncc": float(lp.pwr_ncc),
    }

    tree = cKDTree(hcr_xyz)
    info["lp_inlier30"] = int((tree.query(pred_xyz, k=1)[0] < 30.0).sum())

    # Tight ±60 µm fine grid (10 µm steps) — absorbs sub-grid slip
    # without leaving the LP basin.
    def _score(delta, radius):
        d, _ = tree.query(pred_xyz + delta, k=1)
        return int((d < radius).sum())

    fine = np.zeros(3)
    fine_sc = _score(fine, 30.0)
    for dz in range(-60, 61, 10):
        for dy in range(-40, 41, 10):
            for dx in range(-40, 41, 10):
                cand = np.array([dx, dy, dz], dtype=float)
                sc = _score(cand, 30.0)
                if sc > fine_sc:
                    fine_sc = sc
                    fine = cand
    pred_xyz = pred_xyz + fine

    # Iterative local refit on reciprocal-NN inliers, monotone-gated on
    # the r@30 inlier count.
    best_score = int((tree.query(pred_xyz, k=1)[0] < 30.0).sum())
    for _ in range(5):
        d_c, idx_c = tree.query(pred_xyz, k=1)
        tree_src = cKDTree(pred_xyz)
        _, idx_h = tree_src.query(hcr_xyz, k=1)
        recip = idx_h[idx_c] == np.arange(len(pred_xyz))
        rd = d_c[recip]
        if rd.size < 8:
            break
        thr = max(float(np.quantile(rd, 0.7)), 20.0)
        sel = recip & (d_c < thr)
        if sel.sum() < 8:
            break
        src_s = cz_xyz[sel]
        dst_s = hcr_xyz[idx_c[sel]]
        try:
            refit = _fas(src_s, dst_s)
        except Exception:
            break
        pred_new = ((cz_xyz - src_s.mean(0)) @ refit.R) * refit.scales + dst_s.mean(0)
        sc_new = int((tree.query(pred_new, k=1)[0] < 30.0).sum())
        if sc_new <= best_score:
            break
        pred_xyz = pred_new
        best_score = sc_new

    cz_init_refined_zyx = pred_xyz[:, [2, 1, 0]]
    info.update(
        translation_grid_offset_xyz=fine.tolist(),
        final_inlier30=int(best_score),
    )
    return cz_init_refined_zyx, info


@contextlib.contextmanager
def _patched_warmstart(s, lp: LockedPriorWarmStart):
    """Replace `default_warmstart_zyx` with the LP→ICP→refine pipeline."""
    orig = _ch.default_warmstart_zyx

    def _patched(cz_um_zyx, hcr_um_zyx, *args, **kwargs):
        return _refine_from_lp(cz_um_zyx, hcr_um_zyx, lp)

    _ch.default_warmstart_zyx = _patched
    try:
        yield
    finally:
        _ch.default_warmstart_zyx = orig


def _attach_lp(result: CoregResult, lp: LockedPriorWarmStart) -> CoregResult:
    diag = dict(result.diagnostics) if result.diagnostics else {}
    diag["locked_prior_warm_start"] = {
        "subject_id": lp.subject_id,
        "sxy": lp.sxy_value,
        "sxy_source": lp.sxy_source,
        "rotation_deg_z": lp.rotation_deg_z,
        "translation": lp.translation.tolist(),
        "scales": lp.scales.tolist(),
        "src_mean": lp.src_mean.tolist(),
        "pwr_ncc": lp.pwr_ncc,
        "pwr_method": lp.pwr_method,
    }
    return CoregResult(
        pairs_df=result.pairs_df,
        confidence=result.confidence,
        transform=result.transform,
        diagnostics=diag,
    )


@register_candidate("P1_LP")
def run_p1_lp(s) -> CoregResult:
    lp = compute_locked_prior_warm_start(s)
    with _patched_warmstart(s, lp):
        out = run_p1(s)
    return _attach_lp(out, lp)


@register_candidate("P4_LP")
def run_p4_lp(s) -> CoregResult:
    lp = compute_locked_prior_warm_start(s)
    with _patched_warmstart(s, lp):
        out = run_p4(s)
    return _attach_lp(out, lp)


@register_candidate("P6_LP")
def run_p6_lp(s) -> CoregResult:
    lp = compute_locked_prior_warm_start(s)
    with _patched_warmstart(s, lp):
        out = run_p6(s)
    return _attach_lp(out, lp)


@register_candidate("C5_LP")
def run_c5_lp(s) -> CoregResult:
    lp = compute_locked_prior_warm_start(s)
    with _patched_warmstart(s, lp):
        out = run_c5(s)
    return _attach_lp(out, lp)
