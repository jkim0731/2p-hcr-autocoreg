"""P6 — Bayesian Coherent Point Drift (probreg).

Nonrigid point-set registration with rigid+uniform-scale prior and a
GP-coherence displacement field. Takes an anisotropic-warmstarted CZ cloud
(already near HCR space) and refines alignment + extracts per-CZ matches
to HCR GFP+ via nearest-neighbour in the warped space.

Critical for CZ↔HCR: BCPD's uniform-scale head cannot recover anisotropic
expansion on its own, so we rely on `default_warmstart_zyx`'s ICP-derived
anisotropic pre-scale. BCPD then absorbs residual rigid + uniform-scale +
nonrigid deformation (the 16-43 µm RMS benchmark nonrigid residual).

Confidence per pair = 1 / (1 + dist_to_nn / σ_med), where σ_med is the
median post-warp neighbour distance on the inlier set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.centroid_helpers import centroids_um, default_warmstart_zyx

try:
    from probreg import bcpd as _bcpd
    from probreg import cpd as _cpd
    _HAS_BCPD = True
except Exception:
    _HAS_BCPD = False


def _crop_hcr_around_cz(cz_warp_zyx: np.ndarray, hcr_um_zyx: np.ndarray,
                        hcr_ids: np.ndarray, pad_um: float = 150.0
                        ) -> "tuple[np.ndarray, np.ndarray]":
    lo = cz_warp_zyx.min(0) - pad_um
    hi = cz_warp_zyx.max(0) + pad_um
    keep = np.all((hcr_um_zyx >= lo) & (hcr_um_zyx <= hi), axis=1)
    return hcr_um_zyx[keep], hcr_ids[keep]


@register_candidate("P6")
def run_p6(s, *, maxiter: int = 40, w_outlier: float = 0.3, lmd: float = 2.0,
           crop_pad_um: float = 150.0, match_radius_um: float = 60.0,
           method: str = "cpd_nonrigid",
           hcr_quality_beta: float = 0.0,
           verbose: bool = True) -> CoregResult:
    """method ∈ {bcpd, cpd_nonrigid, cpd_affine}.

    Probreg BCPD and CPD-rigid/affine collapse to zero-scale on this
    problem (CZ:HCR-crop ratio ~1:3 causes the solver to put all source
    mass on a single target cluster). CPD nonrigid preserves the source
    extent by anchoring each point individually. Default = cpd_nonrigid.
    """
    if not _HAS_BCPD:
        raise RuntimeError("probreg not installed; `pip install probreg`")

    t0 = time.time()
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, ws_info = default_warmstart_zyx(cz_um, hcr_um)

    hcr_crop_zyx, hcr_crop_ids = _crop_hcr_around_cz(
        cz_init, hcr_um, hcr_ids, pad_um=crop_pad_um)
    if verbose:
        print(f"  p6[{method}]: cz={len(cz_init)} hcr_gfp_total={len(hcr_um)} "
              f"hcr_crop={len(hcr_crop_zyx)} (pad={crop_pad_um}µm)", flush=True)

    src = cz_init.astype(np.float32)
    tgt = hcr_crop_zyx.astype(np.float32)

    t_bcpd = time.time()
    if method == "bcpd":
        result = _bcpd.registration_bcpd(src, tgt, maxiter=maxiter,
                                         w=w_outlier, lmd=lmd)
        rigid_scale = float(result.rigid_trans.scale)
    elif method == "cpd_nonrigid":
        result = _cpd.registration_cpd(src, tgt, tf_type_name="nonrigid",
                                       maxiter=maxiter, w=w_outlier)
        rigid_scale = 1.0
    elif method == "cpd_affine":
        result = _cpd.registration_cpd(src, tgt, tf_type_name="affine",
                                       maxiter=maxiter, w=w_outlier)
        rigid_scale = 1.0
    else:
        raise ValueError(f"unknown method {method}")
    bcpd_wall = time.time() - t_bcpd

    if hasattr(result, "transform"):
        cz_warped = np.asarray(result.transform(src))
    else:
        cz_warped = np.asarray(result.transformation.transform(src))
    if verbose:
        print(f"  p6[{method}]: solver {bcpd_wall:.1f}s "
              f"rigid_scale={rigid_scale:.3f} "
              f"warped_extent_zyx={np.ptp(cz_warped, axis=0).round(1).tolist()}",
              flush=True)

    tree = cKDTree(hcr_crop_zyx)
    k_neighbors = 5 if hcr_quality_beta else 1
    d_nn_k, idx_nn_k = tree.query(cz_warped, k=k_neighbors)
    if k_neighbors == 1:
        d_nn_k = d_nn_k[:, None]; idx_nn_k = idx_nn_k[:, None]

    hq_full = None
    if hcr_quality_beta:
        from lib.image_quality import hcr_quality as _hcr_quality
        hq_full = _hcr_quality(s)
        id_to_idx = {int(x): k for k, x in enumerate(hcr_ids)}
        hq_crop = np.array([hq_full[id_to_idx[int(hcr_crop_ids[i])]]
                             for i in range(len(hcr_crop_ids))])
    sigma_med = float(np.median(d_nn_k[:, 0])) + 1e-6

    rows = []
    for i in range(len(cz_warped)):
        if hcr_quality_beta:
            score = d_nn_k[i] - float(hcr_quality_beta) * hq_crop[idx_nn_k[i]]
            best_k = int(np.argmin(score))
        else:
            best_k = 0
        d = float(d_nn_k[i, best_k])
        if d > match_radius_um:
            continue
        j = int(idx_nn_k[i, best_k])
        conf = 1.0 / (1.0 + d / sigma_med)
        rows.append(dict(
            cz_id=int(cz_ids[i]),
            hcr_id=int(hcr_crop_ids[j]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]),
            cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_crop_zyx[j, 2]),
            hcr_y_um=float(hcr_crop_zyx[j, 1]),
            hcr_z_um=float(hcr_crop_zyx[j, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False)
        df = df.drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)

    if method == "bcpd":
        transform = TransformDescriptor(
            R=np.asarray(result.rigid_trans.rot),
            scales=np.array([result.rigid_trans.scale] * 3),
            translation=np.asarray(result.rigid_trans.t),
            src_mean=np.zeros(3),
            kind="tps",
        )
    else:
        transform = TransformDescriptor(kind="tps")

    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=transform,
        diagnostics=dict(
            n_matched=int(len(df)),
            sigma_med_um=sigma_med,
            bcpd_wall_s=bcpd_wall,
            total_wall_s=float(time.time() - t0),
            hcr_crop_n=int(len(hcr_crop_zyx)),
            rigid_scale=rigid_scale,
            method=method,
            warmstart_source=ws_info.get("source"),
            warmstart_scales_zyx=ws_info.get("scales_zyx"),
            maxiter=int(maxiter), w_outlier=float(w_outlier),
            lmd=float(lmd), crop_pad_um=float(crop_pad_um),
        ),
    )
