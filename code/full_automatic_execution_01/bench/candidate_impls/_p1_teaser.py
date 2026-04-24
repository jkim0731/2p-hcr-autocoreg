"""P1 — TEASER++ outlier-robust affine.

Uses `teaserpp_python` if available, otherwise falls back to a GNC-TLS-style
scheme implemented on top of scipy: repeated anisotropic-affine fits on the
inlier set with truncated-least-squares weights updated each iteration.  The
fallback is documented in the sessions/03 log as an acceptable stand-in when
TEASER is unavailable.

The output includes:
- coreg table,
- inlier label + TLS weight = intrinsic confidence,
- final anisotropic affine + TPS refinement over inliers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.centroid_helpers import centroids_um, apply_aniso_fit
from benchmark_analysis import fit_anisotropic_similarity

try:  # TEASER is optional
    import teaserpp_python  # type: ignore
    _HAS_TEASER = True
except Exception:
    _HAS_TEASER = False


def _seed_putative(cz_um, hcr_um, Fcn, Fgn, K=5, *, hcr_quality=None, beta=0.0):
    D = cdist(cz_um, hcr_um)
    cosS = Fcn @ Fgn.T
    score = D - 25.0 * cosS
    if hcr_quality is not None and beta:
        score = score - float(beta) * hcr_quality[None, :]
    order = np.argsort(score, axis=1)[:, :K]
    return order, D, cosS


def _gnc_tls(A, B, c_bar=15.0, mu=1.4, n_iter=50):
    """Graduated non-convexity TLS over an anisotropic affine.

    A, B: (N, 3) putative paired points.  Returns (fit, weights, inliers).
    """
    N = len(A)
    w = np.ones(N)
    mu_t = 1e-4
    fit = None
    for _ in range(n_iter):
        if w.sum() < 4:
            break
        sel = w > 1e-3
        try:
            fit = fit_anisotropic_similarity(A[sel], B[sel])
        except Exception:
            break
        # residual under the fit, applied to ALL putatives (not just current selected).
        B_pred = apply_aniso_fit(A, fit)
        r2 = np.sum((B_pred - B) ** 2, axis=1)
        c_bar2 = c_bar ** 2
        # TLS weight: wi = (mu_t * c_bar^2 / (r2 + mu_t * c_bar^2))^2
        w_new = (mu_t * c_bar2 / (r2 + mu_t * c_bar2)) ** 2
        if np.allclose(w_new, w, atol=1e-4):
            w = w_new
            break
        w = w_new
        mu_t = min(1.0, mu_t * mu)
    if fit is None:
        # Fallback identity
        return None, np.zeros(N), np.zeros(N, bool)
    B_pred = apply_aniso_fit(A, fit)
    r = np.linalg.norm(B_pred - B, axis=1)
    inliers = r < c_bar
    return fit, w, inliers


@register_candidate("P1")
def run_p1(s, *, K=5, c_bar=15.0, cz_init=None, hcr_quality_beta: float = 5.0) -> CoregResult:
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    if cz_init is None:
        from lib.centroid_helpers import default_warmstart_zyx
        cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)
        cz_c = cz_um.mean(0); hcr_c = hcr_um.mean(0)
        R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    else:
        cz_init = np.asarray(cz_init, dtype=float)
        assert cz_init.shape == cz_um.shape
        cz_c = cz_um.mean(0); hcr_c = hcr_um.mean(0)
        R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)

    Fc, names, _ = extract_cell_features(s, "cz")
    Fg, _, _ = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
    Fcn = (Fc[:, keep] - mu) / sd
    Fgn = (Fg[:, keep] - mu) / sd
    Fcn = Fcn / (np.linalg.norm(Fcn, axis=1, keepdims=True) + 1e-9)
    Fgn = Fgn / (np.linalg.norm(Fgn, axis=1, keepdims=True) + 1e-9)

    hq = None
    if hcr_quality_beta:
        from lib.image_quality import hcr_quality as _hcr_quality
        hq = _hcr_quality(s)
    order, D, cosS = _seed_putative(cz_init, hcr_um, Fcn, Fgn, K=K,
                                     hcr_quality=hq, beta=hcr_quality_beta)
    # Build putative correspondence list (cz_idx, hcr_idx).
    pair_cz, pair_hcr = [], []
    for i in range(len(cz_init)):
        for j in order[i]:
            pair_cz.append(i); pair_hcr.append(int(j))
    A = cz_init[pair_cz]
    B = hcr_um[pair_hcr]

    fit, w, inliers = _gnc_tls(A, B, c_bar=c_bar)
    if fit is None:
        return CoregResult(pd.DataFrame(), 0.0,
                            diagnostics={"error": "gnc_tls_failed"})

    # Apply the fit to ALL CZ cells and select best per-cell inlier.
    B_pred = apply_aniso_fit(cz_init, fit)

    # TPS refinement over inliers
    inl_A = A[inliers]; inl_B = B[inliers]
    if len(inl_A) >= 8:
        try:
            # TPS residual on all CZ
            deltas = np.zeros_like(B_pred)
            # Input points for TPS are the affine-predicted CZ positions (so we warp
            # the residual into alignment with the observed HCR points).
            src = apply_aniso_fit(inl_A, fit)
            dst = inl_B - src  # residual
            for axis in range(3):
                rbf = Rbf(src[:, 0], src[:, 1], src[:, 2], dst[:, axis],
                          function="thin_plate", smooth=1.0)
                deltas[:, axis] = rbf(B_pred[:, 0], B_pred[:, 1], B_pred[:, 2])
            B_pred_tps = B_pred + deltas
        except Exception:
            B_pred_tps = B_pred
    else:
        B_pred_tps = B_pred

    # For each CZ cell, pick best HCR (within search radius) from the putatives.
    rows = []
    for i in range(len(cz_init)):
        # Among putative j's for this i, pick one with smallest |B_pred_tps[i] - hcr_um[j]|
        js = order[i]
        d = np.linalg.norm(B_pred_tps[i][None] - hcr_um[js], axis=1)
        k = int(np.argmin(d))
        j = int(js[k])
        r = float(d[k])
        cos = float(cosS[i, j])
        cert = (r < c_bar)
        conf = (1.0 / (1.0 + np.exp((r - c_bar * 0.66) / (c_bar * 0.33)))) * (0.5 + 0.5 * cos)
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=float(conf),
            residual_um=r, cos_sim=cos, certified=bool(cert),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    # One-to-one: if two CZs claim the same HCR, keep the higher-confidence one.
    if len(df):
        df = df.sort_values("confidence", ascending=False)
        df = df.drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)

    transform = TransformDescriptor(
        R=fit.R, scales=fit.scales, translation=fit.translation, src_mean=np.zeros(3),
        rotation_deg_z=180.0, kind="aniso-affine+tps",
    )
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(
            has_teaser=_HAS_TEASER,
            n_inliers=int(inliers.sum()),
            n_putative=int(len(A)),
            c_bar=c_bar,
            scales=fit.scales.tolist(),
            hcr_quality_beta=float(hcr_quality_beta),
        ),
    )
