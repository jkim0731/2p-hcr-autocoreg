"""I3 — SimpleITK MI B-spline deformable refinement on top of I2's affine.

Pipeline:
  1. Run I2 to get the coarse (R, S, t) affine.
  2. Feed that affine into `mi_bspline` as `SetMovingInitialTransform`, so the
     B-spline only fits the nonrigid residual on top.
  3. Forward-map HCR GFP+ centroids through the composed transform into CZ
     space; for each CZ centroid, match the nearest predicted-HCR point and
     emit that as a pair.

Produces a `pairs_df` (cz_id, hcr_id, confidence, centroids) so it can stand
alone or feed downstream C-series candidates as a warmstart.
"""
from __future__ import annotations

import sys
import time
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
from lib.sitk_wrapper import mi_bspline
from lib.centroid_helpers import centroids_um


@register_candidate("I3")
def run_i3(s, *, target_um: float = 8.0,
           grid_spacing_um=(60.0, 60.0, 120.0),
           bspline_iterations: int = 20,
           bspline_sampling_fraction: float = 0.02,
           match_radius_um: float = 60.0,
           skip_bspline: bool = False) -> CoregResult:
    """Chain I2 affine → B-spline refinement → centroid NN matching."""
    from bench.candidate_impls._i2_sitk_affine import run_i2
    from bench.candidate_impls._i1_axial_ncc import (_load_cz_fullstack,
                                                     _load_hcr_fullstack)
    from scipy.ndimage import zoom

    t0 = time.time()
    r_i2 = run_i2(s, target_um=target_um)
    i2_wall = time.time() - t0
    if r_i2.transform is None:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"error": "i2_failed",
                                        "i2_diag": r_i2.diagnostics})

    A = np.asarray(r_i2.diagnostics["sitk_matrix_zyx"])
    t_v = np.asarray(r_i2.diagnostics["sitk_translation_zyx"])
    c_v = np.asarray(r_i2.diagnostics["sitk_center_zyx"])

    # Load volumes (same downsample as I2).
    try:
        cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
        hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)
    except Exception as e:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"error": f"load: {e}"})

    cz_zoom = (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um)
    hcr_zoom = (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um)
    cz_ds = zoom(cz_stack, cz_zoom, order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol, hcr_zoom, order=1).astype(np.float32)

    t1 = time.time()
    if skip_bspline:
        # Skip-bspline path: just use I2's affine for NN matching. Useful as a
        # sanity baseline to compare I3 lift vs I2 alone.
        r_b = None
        diag_bspline = {"skipped": True}
    else:
        try:
            r_b = mi_bspline(
                cz_ds, hcr_ds,
                cz_xy_um=target_um, cz_z_um=target_um,
                hcr_xy_um=target_um, hcr_z_um=target_um,
                initial_affine=A,
                initial_translation=t_v,
                initial_center=c_v,
                grid_spacing_um=grid_spacing_um,
                n_iterations=bspline_iterations,
                sampling_fraction=bspline_sampling_fraction,
            )
            diag_bspline = dict(metric=r_b.metric, iterations=r_b.iterations,
                                converged=r_b.converged,
                                grid_spacing_um=r_b.bspline_grid_spacing_um)
        except Exception as e:
            return CoregResult(pd.DataFrame(), 0.0,
                               diagnostics={"error": f"bspline: {e}",
                                            "i2_diag": r_i2.diagnostics})
    bspline_wall = time.time() - t1

    # Load centroids in µm (both in (z, y, x) layout).
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")

    # Forward-map HCR centroids through the composed (affine ∘ bspline) transform.
    if r_b is not None and r_b._sitk_composite is not None:
        hcr_pred_cz = r_b.apply_forward_hcr_to_cz(hcr_um)
    else:
        # Affine-only forward: p_cz = A @ (p_hcr - c) + c + t
        hcr_pred_cz = ((hcr_um - c_v) @ A.T) + c_v + t_v

    # For each CZ centroid, find nearest HCR (by its predicted-CZ position).
    from scipy.spatial import cKDTree
    tree = cKDTree(hcr_pred_cz)
    dists, idxs = tree.query(cz_um, k=1)

    keep = dists < match_radius_um
    if keep.sum() == 0:
        return CoregResult(pd.DataFrame(), 0.0,
                           diagnostics={"i2_wall_s": i2_wall,
                                        "bspline_wall_s": bspline_wall,
                                        "i2_diag": r_i2.diagnostics,
                                        "bspline_diag": diag_bspline,
                                        "nn_dists_p50": float(np.median(dists)),
                                        "nn_dists_p90": float(np.quantile(dists, 0.9)),
                                        "n_match": 0})

    # Per-pair confidence: sharper NN distance → higher confidence.
    med_nn = float(np.median(dists[keep])) or 1.0
    conf = 1.0 / (1.0 + dists / med_nn)

    pairs = pd.DataFrame({
        "cz_id": np.array(cz_ids)[keep].astype(int),
        "hcr_id": hcr_ids[idxs[keep]].astype(int),
        "confidence": conf[keep].astype(float),
        "cz_x_um": cz_um[keep, 2],  # (z, y, x) layout in centroids_um
        "cz_y_um": cz_um[keep, 1],
        "cz_z_um": cz_um[keep, 0],
        "hcr_x_um": hcr_um[idxs[keep], 2],
        "hcr_y_um": hcr_um[idxs[keep], 1],
        "hcr_z_um": hcr_um[idxs[keep], 0],
    })
    pairs = pairs.drop_duplicates("cz_id", keep="first")
    pairs = pairs.drop_duplicates("hcr_id", keep="first").reset_index(drop=True)

    return CoregResult(
        pairs_df=pairs,
        confidence=float(pairs["confidence"].mean()) if len(pairs) else 0.0,
        transform=TransformDescriptor(
            R=np.asarray(r_i2.transform.R),
            scales=np.asarray(r_i2.transform.scales),
            translation=np.asarray(r_i2.transform.translation),
            src_mean=np.asarray(r_i2.transform.src_mean),
            rotation_deg_z=180.0,
            kind="mi-affine-plus-bspline" if r_b is not None else "mi-affine",
        ),
        diagnostics=dict(
            i2_wall_s=i2_wall,
            bspline_wall_s=bspline_wall,
            i2_diag=r_i2.diagnostics,
            bspline_diag=diag_bspline,
            match_radius_um=match_radius_um,
            nn_dists_p50=float(np.median(dists)),
            nn_dists_p90=float(np.quantile(dists, 0.9)),
            nn_dists_p50_kept=float(np.median(dists[keep])),
            n_match=int(keep.sum()),
            n_cz=int(len(cz_um)),
            n_hcr_gfp=int(len(hcr_um)),
        ),
    )
