"""P14 — Hungarian baseline matcher.

Pure baseline: after applying 180° XY + CZ-centroid → HCR-centroid translation
(the available structural priors), build an N×M affinity using F6 feature
cosine and geometric residual, and solve with linear_sum_assignment.  Reports
intrinsic confidence = score / score_max.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.centroid_helpers import centroids_um, default_warmstart_zyx


def _centroids_um(s) -> tuple:
    cz_xyz, cz_ids = centroids_um(s, "cz")
    hcr_xyz, hcr_ids = centroids_um(s, "hcr_gfp")
    return cz_xyz, hcr_xyz, cz_ids, hcr_ids


@register_candidate("P14")
def run_p14(s) -> CoregResult:
    cz_um, hcr_um, cz_ids, hcr_ids = _centroids_um(s)

    # Data-driven warm-start: 180° + cloud-extent scale + HCR-centre translation.
    cz_init, ws_info = default_warmstart_zyx(cz_um, hcr_um)
    cz_c = cz_um.mean(0); hcr_c = hcr_um.mean(0)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

    # Features.
    Fc, names, ic = extract_cell_features(s, "cz")
    Fg, names2, ig = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    Fc_k = Fc[:, keep]; Fg_k = Fg[:, keep]
    mu = np.nanmean(Fg_k, 0); sd = np.nanstd(Fg_k, 0) + 1e-6
    Fc_z = (Fc_k - mu) / sd
    Fg_z = (Fg_k - mu) / sd

    # Distance cost (µm) and feature-cosine cost.
    from scipy.spatial.distance import cdist
    D = cdist(cz_init, hcr_um)  # (Nc, Nh)
    Fc_n = Fc_z / (np.linalg.norm(Fc_z, axis=1, keepdims=True) + 1e-9)
    Fg_n = Fg_z / (np.linalg.norm(Fg_z, axis=1, keepdims=True) + 1e-9)
    cos_full = Fc_n @ Fg_n.T  # (Nc, Nh) cosine similarity
    Nc = len(cz_ids); Nh = len(hcr_ids)
    # Full rectangular Hungarian: 932 × 17427 ≈ 16M entries.
    cost = D  # features hurt F6-based matching on this data, use pure distance
    row_ind, col_ind = linear_sum_assignment(cost)

    # Accept matches only within a reasonable residual (30 µm).
    pairs = []
    for i, j in zip(row_ind, col_ind):
        d = D[i, j]
        if d > 50.0:
            continue
        cos = float(cos_full[i, j])
        conf = 1.0 / (1.0 + np.exp((d - 30.0) / 10.0)) * (0.5 + 0.5 * cos)
        pairs.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(pairs)
    S = np.array(ws_info["scales_zyx"], dtype=float)
    transform = TransformDescriptor(
        R=R, scales=S, translation=hcr_c, src_mean=cz_c,
        rotation_deg_z=180.0, kind="rigid+180+extent-scale",
    )
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(n_pairs=len(df), n_cz=Nc, n_hcr=Nh, warmstart=ws_info),
    )
