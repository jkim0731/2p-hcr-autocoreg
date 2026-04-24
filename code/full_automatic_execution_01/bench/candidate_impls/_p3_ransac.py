"""P3 — RANSAC + anisotropic affine.

Sample 4 random putative correspondences, fit `fit_anisotropic_similarity`,
score by the number of *other* putative pairs within residual < τ µm; repeat
N times; return the consensus set.

Putative set: for each CZ cell i, top-K HCR cells by combined distance +
feature cosine.
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
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.centroid_helpers import centroids_um, apply_aniso_fit
from benchmark_analysis import fit_anisotropic_similarity


def _putative(cz_um, hcr_um, Fcn, Fgn, K=10):
    """Top-K HCR per CZ by weighted combined score."""
    from scipy.spatial.distance import cdist
    D = cdist(cz_um, hcr_um)
    cosS = Fcn @ Fgn.T  # (Nc, Nh), high is similar
    score = D - 20.0 * cosS
    order = np.argsort(score, axis=1)[:, :K]
    putative = []
    for i, row in enumerate(order):
        for j in row:
            putative.append((i, int(j), float(D[i, j]), float(cosS[i, j])))
    return putative


@register_candidate("P3")
def run_p3(s, *, K=10, n_iter=5000, tau_um=15.0, rng_seed=0) -> CoregResult:
    rng = np.random.default_rng(rng_seed)

    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    from lib.centroid_helpers import default_warmstart_zyx
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)
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

    putative = _putative(cz_init, hcr_um, Fcn, Fgn, K=K)
    if len(putative) < 4:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"error": "no putative"})

    best_inliers = []
    best_fit = None
    pt_arr = np.asarray(putative)
    n_put = len(putative)
    for _ in range(n_iter):
        idx = rng.choice(n_put, size=4, replace=False)
        quad = [putative[k] for k in idx]
        A = np.array([cz_init[i] for (i, j, _, _) in quad])
        B = np.array([hcr_um[j] for (i, j, _, _) in quad])
        try:
            fit = fit_anisotropic_similarity(A, B)
        except Exception:
            continue
        # Apply fit to all putative CZ points
        A_all = cz_init
        B_pred = apply_aniso_fit(A_all, fit)
        # Check residual per putative pair
        inliers = []
        for k, (i, j, _, _) in enumerate(putative):
            r = float(np.linalg.norm(B_pred[i] - hcr_um[j]))
            if r < tau_um:
                inliers.append((i, j, r))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_fit = fit

    if best_fit is None or not best_inliers:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"n_putative": n_put})

    # Refit on inliers
    A = np.array([cz_init[i] for (i, _, _) in best_inliers])
    B = np.array([hcr_um[j] for (_, j, _) in best_inliers])
    try:
        refit = fit_anisotropic_similarity(A, B)
    except Exception:
        refit = best_fit

    # Final assignment: for each CZ, pick the inlier with lowest residual under refit
    B_pred = apply_aniso_fit(cz_init, refit)
    best_for_cz = {}
    for i, j, _ in best_inliers:
        r = float(np.linalg.norm(B_pred[i] - hcr_um[j]))
        prev = best_for_cz.get(i)
        if prev is None or r < prev[1]:
            best_for_cz[i] = (j, r)

    rows = []
    for i, (j, r) in best_for_cz.items():
        conf = 1.0 / (1.0 + np.exp((r - tau_um * 0.66) / (tau_um * 0.33)))
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    transform = TransformDescriptor(
        R=refit.R, scales=refit.scales,
        translation=refit.translation, src_mean=np.zeros(3),
        rotation_deg_z=180.0, kind="aniso-affine",
    )
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(
            n_pairs=len(df), n_inliers=len(best_inliers),
            n_putative=n_put, tau_um=tau_um,
        ),
    )
