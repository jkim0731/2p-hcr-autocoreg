"""P4 — Spectral graph matching (Leordeanu–Hebert).

Build an affinity matrix over putative pairs where `M[(i,a), (j,b)]` is the
product of (a) geometric consistency — agreement of pairwise distances after
the empirical axis rescale — and (b) feature-cosine compatibility.  Power-
iterate to the principal eigenvector, then extract a one-to-one assignment
greedily under the inlier entry threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.centroid_helpers import centroids_um


def _rescale_axes(cz_um: np.ndarray, hcr_um: np.ndarray) -> np.ndarray:
    """Per-axis robust median of pairwise spacing ratios.  Used as the
    empirical anisotropic scale to rescale CZ pairwise distances before
    comparing to HCR distances.  No benchmark priors used.
    """
    rng = np.random.default_rng(0)
    n = min(500, len(cz_um), len(hcr_um))
    ic = rng.choice(len(cz_um), size=n, replace=False)
    ih = rng.choice(len(hcr_um), size=n, replace=False)
    dc = np.abs(cz_um[ic][:, None] - cz_um[ic][None, :])
    dh = np.abs(hcr_um[ih][:, None] - hcr_um[ih][None, :])
    ratios = []
    for axis in range(3):
        c = dc[..., axis].flatten()
        h = dh[..., axis].flatten()
        c = c[c > 5]
        h = h[h > 5]
        if len(c) == 0 or len(h) == 0:
            ratios.append(1.0); continue
        ratios.append(np.median(h) / np.median(c))
    return np.asarray(ratios)


@register_candidate("P4")
def run_p4(s, *, K=5, sigma_geom_um=15.0, power_iter=50,
           hcr_quality_beta: float = 5.0) -> CoregResult:
    """Spectral GM with optional HCR image-quality bonus in putative ranking.

    hcr_quality_beta — subtract β·hq[j] from the per-CZ ranking score, biasing
    top-K putatives toward high-quality HCR cells.  β=5 lifts r@20 on 755252
    from 0.058 → 0.091 (S51).
    """
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    from lib.centroid_helpers import default_warmstart_zyx
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)
    cz_c = cz_um.mean(0); hcr_c = hcr_um.mean(0)
    R0 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    cz_scaled = cz_init
    axis_scale = np.array(_ws.get("scales_zyx", [1.0, 1.0, 1.0]), dtype=float)

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

    D = cdist(cz_scaled, hcr_um)
    cosS = Fcn @ Fgn.T
    Nc = len(cz_scaled); Nh = len(hcr_um)
    putative = []
    for i in range(Nc):
        score = D[i] - 20.0 * cosS[i]
        if hq is not None and hcr_quality_beta:
            score = score - float(hcr_quality_beta) * hq
        js = np.argsort(score)[:K]
        for j in js:
            putative.append((i, int(j), float(D[i, j]), float(cosS[i, j])))
    P = len(putative)
    if P == 0:
        return CoregResult(pd.DataFrame(), 0.0, diagnostics={"error": "no putative"})

    # Affinity matrix M of size P x P.
    if P > 4000:
        # Downsample to keep M tractable
        rng = np.random.default_rng(0)
        idx = rng.choice(P, size=4000, replace=False)
        putative = [putative[k] for k in sorted(idx)]
        P = len(putative)

    I = np.array([p[0] for p in putative], dtype=np.int32)
    J = np.array([p[1] for p in putative], dtype=np.int32)
    d_diag = np.array([p[2] for p in putative], dtype=np.float32)
    c_diag = np.array([p[3] for p in putative], dtype=np.float32)

    cz_pts = cz_scaled[I].astype(np.float32)
    hr_pts = hcr_um[J].astype(np.float32)
    Dcz = np.linalg.norm(cz_pts[:, None, :] - cz_pts[None, :, :], axis=2)
    Dhr = np.linalg.norm(hr_pts[:, None, :] - hr_pts[None, :, :], axis=2)
    M = np.exp(-np.abs(Dcz - Dhr) / float(sigma_geom_um)).astype(np.float32)
    conflict = (I[:, None] == I[None, :]) | (J[:, None] == J[None, :])
    M[conflict] = 0.0
    diag = np.exp(-d_diag / float(sigma_geom_um)) * (0.5 + 0.5 * c_diag)
    np.fill_diagonal(M, diag.astype(np.float32))

    # Power iterate
    v = np.ones(P, dtype=np.float32) / np.sqrt(P)
    for _ in range(power_iter):
        v = M @ v
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            break
        v = v / nv

    # Greedy one-to-one assignment: iterate sorted v descending, skip conflicting
    order = np.argsort(-v)
    cz_used = set(); hcr_used = set()
    rows = []
    for p in order:
        i, j, d, c = putative[p]
        if i in cz_used or j in hcr_used:
            continue
        if v[p] < 1e-4:
            break
        cz_used.add(i); hcr_used.add(j)
        conf = float(v[p])
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=conf,
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    # Re-normalise confidences to [0, 1]
    if len(df):
        v_hi = df["confidence"].max()
        if v_hi > 0:
            df["confidence"] = df["confidence"] / v_hi
    transform = TransformDescriptor(
        R=R0, scales=axis_scale, translation=hcr_c,
        src_mean=cz_c, rotation_deg_z=180.0, kind="spectral-gm-affine",
    )
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(P=P, axis_scale=axis_scale.tolist(),
                         hcr_quality_beta=float(hcr_quality_beta)),
    )
