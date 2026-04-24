"""Per-cell ROI cross-sectional-area sxy estimator — promoted main-pipeline entry.

Under isotropic lateral tissue expansion `sxy`, the xy footprint of a cell
body scales as `sxy^2`:

    area_HCR ≈ sxy^2 · area_CZ  →  sxy = sqrt(median(area_HCR) / median(area_CZ))

Cell bodies are represented by their tight xy bounding-box area.
CZ areas come from `segmentation_masks.tif` (regionprops). HCR areas come
from `cell_body_segmentation/segmentation_mask.zarr`, masked by the
per-cell label id inside the `metrics.pickle` tile bbox, to recover the
true tight xy footprint.

Scope
-----
Spot subjects (788406, 790322, 767018, 782149) — they ship
`cell_body_segmentation/metrics.pickle`. Intensity subjects
(755252, 767022) do not and are unsupported here.

On-disk cache
-------------
Per-cell tight HCR bboxes are cached at
``/root/capsule/code/dev_code/cached_hcr_cell_tight_bbox/
  {sid}_hcr_cell_tight_bbox_v1.parquet``.
Additive: re-running with a new ``hcr_ids`` set extends the cache.

Scale note — this was a subtle bug before promotion
---------------------------------------------------
``segmentation_mask.zarr`` is at LEVEL-0 in xy (0.247 µm/vox on spot
subjects); the sibling ``segmentation_mask_orig_res.zarr`` is the
level-2 version — the name "orig_res" is misleading. The standard
``s.hcr_xy_um`` (~0.988 µm) is the level-2 scale used by
``hcr_centroids``; applying it to level-0 voxel extents inflates each
xy span 4× and area 16×, so sxy comes out 4× too high. This module
uses ``s.hcr_seg_xy_um`` (= ``s.hcr_xy_um / HCR_SEG_XY_DOWNSAMPLE``)
when converting bbox extents from ``segmentation_mask.zarr``, and
rescales voxel centroids to the level-2 pixel frame before feeding
them to ``hcr_px_to_um``.

Validation (2026-04-23)
-----------------------
Median-area sxy vs landmark-Procrustes GT:

    788406   est 1.749   GT 1.778   −1.6 %
    790322   est 1.785   GT 1.763   +1.2 %
    767018   est 1.812   GT 1.702   +6.5 %
    782149   est 1.631   GT 1.924   −15.2 %  (HCR_span=566 µm;
             anomalous surface / FOV truncation, not a scale bug)

Public API
----------
* :func:`compute_tight_hcr_bboxes` — cache-backed per-cell tight bboxes
  from ``segmentation_mask.zarr``.
* :func:`cz_cell_tight_bboxes` — per-CZ-cell bbox + depth, from
  ``segmentation_masks.tif``.
* :func:`hcr_cell_tight_bboxes` — per-HCR-cell bbox + depth, level-0
  correctly applied, for a caller-supplied hcr_id set.
* :func:`estimate_sxy_roi_area` — full driver for one subject.
"""
from __future__ import annotations

import glob
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.measure import regionprops

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from benchmark_analysis import (
    analyze_subject,
    depth_from_surface,
    fit_anisotropic_similarity,
)
from benchmark_data_loader import (
    DATA_DIR,
    HCR_SEG_XY_DOWNSAMPLE,
    SubjectData,
    cz_px_to_um,
    hcr_px_to_um,
    landmark_pairs_um,
    load_subject,
)
from r1_revised import coarse_align_revised

SPOT_SUBJECTS = frozenset({"788406", "790322", "767018", "782149"})
INTENSITY_SUBJECTS = frozenset({"755252", "767022"})

TIGHT_BBOX_CACHE = Path(
    "/root/capsule/code/dev_code/cached_hcr_cell_tight_bbox"
)
TIGHT_BBOX_CACHE.mkdir(parents=True, exist_ok=True)
TIGHT_BBOX_VERSION = "v1"

D_SKIN_UM = 100.0  # cortex surface → first cortical cells


# -----------------------------------------------------------
# path discovery
# -----------------------------------------------------------
def _cz_seg_mask_path(sid: str) -> Path:
    for d in DATA_DIR.glob(f"multiplane-ophys_{sid}_*cortical-zstack-segmentation*"):
        tif = d / "channel_0_ref_0" / "segmentation_masks.tif"
        if tif.exists():
            return tif
    raise FileNotFoundError(f"CZ segmentation_masks.tif for {sid}")


def _hcr_metrics_path(sid: str) -> Path:
    hits = sorted(glob.glob(
        f"/root/capsule/data/HCR_{sid}_*/cell_body_segmentation/metrics.pickle"
    ))
    if not hits:
        raise FileNotFoundError(f"HCR cell_body_segmentation/metrics.pickle for {sid}")
    return Path(hits[-1])


def _hcr_seg_zarr_path(sid: str) -> Path:
    hits = sorted(glob.glob(
        f"/root/capsule/data/HCR_{sid}_*/cell_body_segmentation/segmentation_mask.zarr"
    ))
    if not hits:
        raise FileNotFoundError(f"HCR cell_body_segmentation/segmentation_mask.zarr for {sid}")
    return Path(hits[-1])


def _tight_bbox_cache_path(sid: str) -> Path:
    return TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_{TIGHT_BBOX_VERSION}.parquet"


# -----------------------------------------------------------
# HCR tight-bbox cache
# -----------------------------------------------------------
def compute_tight_hcr_bboxes(
    sid: str,
    hcr_ids,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cache-backed per-cell tight HCR bboxes in level-0 voxel indices.

    Parameters
    ----------
    sid : str
        Subject id (must be a spot subject with `metrics.pickle`).
    hcr_ids : iterable of int
        Cell label ids of interest. Cached entries are reused; missing
        ids are computed by reading the zarr crop covering the pickle
        tile bbox and masking `segmentation_mask == hcr_id`.
    force : bool
        Re-compute every id even if cached.

    Returns
    -------
    pd.DataFrame
        Columns: ``hcr_id, zmin_vox, zmax_vox, ymin_vox, ymax_vox,
        xmin_vox, xmax_vox, volume_vox, zc_vox, yc_vox, xc_vox``
        restricted to the requested `hcr_ids` (see README in the cache
        directory for the schema).
    """
    want = {int(x) for x in hcr_ids}
    cache_path = _tight_bbox_cache_path(sid)
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=[
            "hcr_id", "zmin_vox", "ymin_vox", "xmin_vox",
            "zmax_vox", "ymax_vox", "xmax_vox",
            "volume_vox", "zc_vox", "yc_vox", "xc_vox",
        ])

    have = set(int(x) for x in cached["hcr_id"].values)
    missing = sorted(want - have)

    if missing:
        if verbose:
            print(f"  [{sid}] tight-bbox: {len(have & want)} cached, "
                  f"{len(missing)} to compute")
        with open(_hcr_metrics_path(sid), "rb") as f:
            metrics = pickle.load(f)
        z = zarr.open(str(_hcr_seg_zarr_path(sid)))["0"]
        new_rows = []
        t0 = time.time()
        for i, hid in enumerate(missing, 1):
            m = metrics.get(hid)
            if m is None:
                continue
            bb = np.asarray(m["global_bbox"], dtype=int)
            zlo, ylo, xlo, zhi, yhi, xhi = bb
            # pickle bbox endpoints are inclusive (verified with hid=10317);
            # read +1 to include the last slice
            crop = np.asarray(z[0, 0, zlo:zhi + 1, ylo:yhi + 1, xlo:xhi + 1])
            mask = (crop == hid)
            if not mask.any():
                continue
            zs, ys, xs = np.where(mask)
            new_rows.append({
                "hcr_id": int(hid),
                "zmin_vox": int(zs.min() + zlo),
                "zmax_vox": int(zs.max() + zlo + 1),
                "ymin_vox": int(ys.min() + ylo),
                "ymax_vox": int(ys.max() + ylo + 1),
                "xmin_vox": int(xs.min() + xlo),
                "xmax_vox": int(xs.max() + xlo + 1),
                "volume_vox": int(mask.sum()),
                "zc_vox": float(zs.mean() + zlo),
                "yc_vox": float(ys.mean() + ylo),
                "xc_vox": float(xs.mean() + xlo),
            })
            if verbose and (i % 200 == 0 or i == len(missing)):
                dt = time.time() - t0
                print(f"    {i}/{len(missing)}  ({i/max(dt,1e-3):.1f} cells/s)")
        new_df = pd.DataFrame(new_rows)
        if not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates("hcr_id", keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("hcr_id").reset_index(drop=True)
        combined.to_parquet(cache_path, index=False)
        cached = combined

    return cached[cached["hcr_id"].isin(want)].reset_index(drop=True)


# -----------------------------------------------------------
# per-cell bbox tables
# -----------------------------------------------------------
def cz_cell_tight_bboxes(sid: str, s: SubjectData, cz_surf: dict) -> pd.DataFrame:
    """Per-CZ-cell tight xy-bbox area and pia depth."""
    mask = tifffile.imread(_cz_seg_mask_path(sid))
    rp = regionprops(mask)
    dz_um, dy_um, dx_um = s.cz_z_um, s.cz_xy_um, s.cz_xy_um
    rows = []
    for r in rp:
        zlo, ylo, xlo, zhi, yhi, xhi = r.bbox
        dz = (zhi - zlo) * dz_um
        dy = (yhi - ylo) * dy_um
        dx = (xhi - xlo) * dx_um
        zc, yc, xc = r.centroid
        xyz_um = cz_px_to_um(np.array([[zc, yc, xc]]), s)[0]
        x_um, y_um, z_um_c = xyz_um[2], xyz_um[1], xyz_um[0]
        depth = float(depth_from_surface(np.array([[x_um, y_um, z_um_c]]), cz_surf)[0])
        rows.append({
            "cz_id": r.label,
            "xy_area_um2": dx * dy,
            "bbox_dx_um": dx, "bbox_dy_um": dy, "bbox_dz_um": dz,
            "volume_vox": r.area,
            "volume_um3": r.area * dz_um * dy_um * dx_um,
            "x_um": x_um, "y_um": y_um, "z_um": z_um_c,
            "depth_um": depth,
        })
    return pd.DataFrame(rows)


def hcr_cell_tight_bboxes(
    sid: str,
    s: SubjectData,
    hcr_surf: dict,
    hcr_ids,
) -> pd.DataFrame:
    """Per-HCR-cell tight xy-bbox area and pia depth for the given ids.

    Uses `segmentation_mask.zarr` (level-0) via the tight-bbox cache.
    Applies `s.hcr_seg_xy_um` for µm conversion and rescales voxel
    centroids to the level-2 frame before `hcr_px_to_um`.
    """
    with open(_hcr_metrics_path(sid), "rb") as f:
        metrics_keys = set(pickle.load(f).keys())
    want = {int(i) for i in hcr_ids if int(i) in metrics_keys}
    if not want:
        return pd.DataFrame()
    tb = compute_tight_hcr_bboxes(sid, want)
    if tb.empty:
        return pd.DataFrame()

    seg_xy_um = s.hcr_seg_xy_um
    seg_z_um = s.hcr_seg_z_um
    dx_vox = (tb["xmax_vox"] - tb["xmin_vox"]).to_numpy(float)
    dy_vox = (tb["ymax_vox"] - tb["ymin_vox"]).to_numpy(float)
    dz_vox = (tb["zmax_vox"] - tb["zmin_vox"]).to_numpy(float)
    xy_area_um2 = (dx_vox * seg_xy_um) * (dy_vox * seg_xy_um)

    zyx_px = tb[["zc_vox", "yc_vox", "xc_vox"]].to_numpy(float).copy()
    zyx_px[:, 1] /= HCR_SEG_XY_DOWNSAMPLE
    zyx_px[:, 2] /= HCR_SEG_XY_DOWNSAMPLE
    xyz_um = hcr_px_to_um(zyx_px, s)[:, [2, 1, 0]]
    depths = depth_from_surface(xyz_um, hcr_surf)
    return pd.DataFrame({
        "hcr_id": tb["hcr_id"].to_numpy(),
        "xy_area_um2": xy_area_um2,
        "bbox_dx_um": dx_vox * seg_xy_um,
        "bbox_dy_um": dy_vox * seg_xy_um,
        "bbox_dz_um": dz_vox * seg_z_um,
        "volume_vox": tb["volume_vox"].to_numpy(),
        "volume_um3": tb["volume_vox"].to_numpy() * seg_z_um * seg_xy_um * seg_xy_um,
        "x_um": xyz_um[:, 0], "y_um": xyz_um[:, 1], "z_um": xyz_um[:, 2],
        "depth_um": depths,
    })


# -----------------------------------------------------------
# driver
# -----------------------------------------------------------
def _prefilter_center_fov(
    sid: str, s: SubjectData, hcr_ids
) -> set[int]:
    """Keep only cells whose approximate centroid lies in the HCR center
    ¼ FOV (linear half-width), using the cheap pickle-tile-bbox center
    so we avoid paying zarr I/O for cells we'd drop anyway."""
    with open(_hcr_metrics_path(sid), "rb") as f:
        metrics = pickle.load(f)
    ids = np.array(sorted(int(i) for i in hcr_ids if int(i) in metrics))
    if ids.size == 0:
        return set()
    bb = np.stack([np.asarray(metrics[int(i)]["global_bbox"], dtype=float)
                   for i in ids])
    zc = 0.5 * (bb[:, 0] + bb[:, 3])
    yc = 0.5 * (bb[:, 1] + bb[:, 4])
    xc = 0.5 * (bb[:, 2] + bb[:, 5])
    zyx_px = np.column_stack([zc, yc, xc]).copy()
    zyx_px[:, 1] /= HCR_SEG_XY_DOWNSAMPLE
    zyx_px[:, 2] /= HCR_SEG_XY_DOWNSAMPLE
    xyz_um = hcr_px_to_um(zyx_px, s)[:, [2, 1, 0]]
    x_um, y_um = xyz_um[:, 0], xyz_um[:, 1]
    cx, cy = 0.5 * (x_um.min() + x_um.max()), 0.5 * (y_um.min() + y_um.max())
    qx = (x_um.max() - x_um.min()) / 4
    qy = (y_um.max() - y_um.min()) / 4
    mask = ((x_um >= cx - qx) & (x_um <= cx + qx)
            & (y_um >= cy - qy) & (y_um <= cy + qy))
    return set(int(i) for i in ids[mask])


def estimate_sxy_roi_area(
    sid: str,
    strict_hcr_ids=None,
    center_fov_quarter: bool = True,
) -> dict:
    """Estimate sxy from CZ↔HCR per-cell tight xy-bbox area ratio.

    Parameters
    ----------
    sid : str
        Spot-subject id (others raise `ValueError`).
    strict_hcr_ids : iterable of int, optional
        HCR cell ids to use. Defaults to the BIC-GMM strict-GFP+ set
        from ``07b_gfp_intersection_threshold.strict_gfp_df``.
    center_fov_quarter : bool
        If True (default), restrict HCR cells to the center ¼ FOV so
        the CZ and HCR footprints match in aspect.

    Returns
    -------
    dict
        Includes ``sxy_median``, ``sxy_mean``, ``sxy_gt``,
        per-side medians / means, cell counts, depth spans, and the
        strict-cutoff metadata.
    """
    if sid not in SPOT_SUBJECTS:
        raise ValueError(
            f"{sid}: intensity/unsupported subject — no HCR "
            "cell_body_segmentation/metrics.pickle"
        )
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    # default strict-GFP+ set via the BIC-GMM intersection module
    strict_cutoff = None
    n_components = None
    if strict_hcr_ids is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_gfp_thr", THIS / "07b_gfp_intersection_threshold.py"
        )
        _gfp_thr = importlib.util.module_from_spec(spec)
        sys.modules["_gfp_thr"] = _gfp_thr
        spec.loader.exec_module(_gfp_thr)  # type: ignore
        gi = _gfp_thr.analyze_subject(sid)
        strict_cutoff = float(gi.cutoff_linear)
        n_components = int(gi.n_components)
        strict_df = _gfp_thr.strict_gfp_df(sid, strict_cutoff)
        strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)
    else:
        strict_hcr_ids = set(int(x) for x in strict_hcr_ids)

    if center_fov_quarter:
        fov_ids = _prefilter_center_fov(sid, s, strict_hcr_ids)
    else:
        fov_ids = strict_hcr_ids

    cz_df_all = cz_cell_tight_bboxes(sid, s, cz_surf)
    hcr_df_all = hcr_cell_tight_bboxes(sid, s, hcr_surf, fov_ids)

    cz_span = float(np.nanpercentile(cz_df_all.depth_um, 99))
    hcr_span = float(np.nanpercentile(hcr_df_all.depth_um, 99))

    cz_mask = (cz_df_all["depth_um"] >= D_SKIN_UM) & (cz_df_all["depth_um"] <= cz_span)
    hcr_mask = (hcr_df_all["depth_um"] >= D_SKIN_UM) & (hcr_df_all["depth_um"] <= hcr_span)
    cz_df = cz_df_all[cz_mask].copy()
    hcr_df = hcr_df_all[hcr_mask].copy()

    med_cz = float(cz_df.xy_area_um2.median())
    med_hcr = float(hcr_df.xy_area_um2.median())
    mean_cz = float(cz_df.xy_area_um2.mean())
    mean_hcr = float(hcr_df.xy_area_um2.mean())
    sxy_med = float(np.sqrt(med_hcr / med_cz))
    sxy_mean = float(np.sqrt(mean_hcr / mean_cz))

    fit = fit_anisotropic_similarity(*landmark_pairs_um(s, active_only=True))
    sxy_gt = float(np.sqrt(fit.scales[0] * fit.scales[1]))

    return {
        "sid": sid,
        "sxy_median": sxy_med,
        "sxy_mean": sxy_mean,
        "sxy_gt": sxy_gt,
        "err_pct_median": 100 * (sxy_med - sxy_gt) / sxy_gt,
        "err_pct_mean": 100 * (sxy_mean - sxy_gt) / sxy_gt,
        "n_cz": int(len(cz_df)),
        "n_hcr_strict": int(len(hcr_df)),
        "cz_area_median": med_cz,
        "hcr_area_median": med_hcr,
        "cz_area_mean": mean_cz,
        "hcr_area_mean": mean_hcr,
        "cz_span_um": cz_span,
        "hcr_span_um": hcr_span,
        "center_fov_quarter": bool(center_fov_quarter),
        "strict_cutoff_linear": strict_cutoff,
        "n_components_bic": n_components,
        "n_hcr_input": len(strict_hcr_ids),
        "n_hcr_center_fov": len(fov_ids),
    }


def _cz_xyz_um(s: SubjectData) -> np.ndarray:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
    return cz_px_to_um(arr, s)[:, [2, 1, 0]]


def _hcr_gfp_xyz_um(s: SubjectData, strict_hcr_ids) -> np.ndarray:
    """Return HCR strict-GFP+ centroids as (x, y, z) µm via hcr_id join."""
    want = set(int(x) for x in strict_hcr_ids)
    if not want or s.hcr_centroids.empty:
        return np.zeros((0, 3))
    px = s.hcr_centroids.copy()
    keep = px["hcr_id"].astype(int).isin(want)
    arr = px.loc[keep, ["z_px", "y_px", "x_px"]].to_numpy(float)
    if arr.size == 0:
        return np.zeros((0, 3))
    return hcr_px_to_um(arr, s)[:, [2, 1, 0]]


def estimate_sz_roi_density(
    sid: str,
    sxy: float | None = None,
    strict_hcr_ids=None,
) -> dict:
    """Estimate sz by ROI-density conservation across matched CZ ↔ HCR boxes.

    Under isotropic xy stretch ``sxy`` and axial stretch ``sz``, the same
    physical tissue occupies a volume that is ``sxy² · sz`` larger in
    HCR than in CZ. With the same cell count N in both, the native
    densities satisfy ``ρ_CZ / ρ_HCR = sxy² · sz``.

    Recipe
    ------
    1. ``sxy`` from :func:`estimate_sxy_roi_area` (tight-bbox area ratio)
       unless supplied.
    2. R1 (``coarse_align_revised``) projects CZ centroid mean into the
       HCR µm frame — the HCR box centre. The box xy extent is
       ``sxy × (CZ xy extent)`` so that after undoing the stretch the
       box represents the same physical column as the CZ FOV.
    3. In CZ native: keep centroids with pia depth ∈ [d_skin, p99];
       ``V_CZ = A_CZ · (d_cz_span − d_skin)``, ``N_CZ`` = surviving count.
    4. In HCR native: strict-GFP+ centroids whose xy lies in the R1-
       centred matched box and whose pia depth ∈ [d_skin, p99];
       ``V_HCR = (sxy² · A_CZ) · (d_hcr_span − d_skin)``.
    5. ``sz = (ρ_CZ / ρ_HCR) / sxy² = (N_CZ · T_HCR) / (N_HCR · T_CZ)``.

    Notes
    -----
    - ``A_CZ`` is the xy AABB of CZ centroids that survive the cortex
      filter (not the full FOV, so skin/outside-FOV volume isn't
      over-counted).
    - Accuracy depends on (i) sxy (which propagates as 1/sxy²), and
      (ii) strict-GFP+ vs CZ-active cell-count matching — 07c showed
      integrated GFP+/truth ≈ 1.0, so bulk density should be unbiased.
    """
    if sid not in SPOT_SUBJECTS:
        raise ValueError(
            f"{sid}: intensity/unsupported subject — no HCR "
            "cell_body_segmentation/metrics.pickle"
        )
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    n_components = None
    strict_cutoff = None
    if sxy is None:
        roi = estimate_sxy_roi_area(sid)
        sxy = float(roi["sxy_median"])
        n_components = roi.get("n_components_bic")
        strict_cutoff = roi.get("strict_cutoff_linear")

    if strict_hcr_ids is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_gfp_thr", THIS / "07b_gfp_intersection_threshold.py"
        )
        _gfp_thr = importlib.util.module_from_spec(spec)
        sys.modules["_gfp_thr"] = _gfp_thr
        spec.loader.exec_module(_gfp_thr)  # type: ignore
        gi = _gfp_thr.analyze_subject(sid)
        strict_cutoff = float(gi.cutoff_linear)
        n_components = int(gi.n_components)
        strict_df = _gfp_thr.strict_gfp_df(sid, strict_cutoff)
        strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)

    cz_um = _cz_xyz_um(s)
    gfp_um = _hcr_gfp_xyz_um(s, strict_hcr_ids)
    if len(gfp_um) == 0:
        raise RuntimeError(f"{sid}: no strict GFP+ centroids")

    # CZ side — cortex band + AABB
    cz_depth = depth_from_surface(cz_um, cz_surf)
    cz_span = float(np.nanpercentile(cz_depth, 99))
    cz_mask = (cz_depth >= D_SKIN_UM) & (cz_depth <= cz_span)
    n_cz = int(cz_mask.sum())
    if n_cz == 0:
        raise RuntimeError(f"{sid}: no CZ cortex cells")
    cz_xy = cz_um[cz_mask, :2]
    cz_x_span = float(cz_xy[:, 0].max() - cz_xy[:, 0].min())
    cz_y_span = float(cz_xy[:, 1].max() - cz_xy[:, 1].min())
    cz_A = cz_x_span * cz_y_span
    cz_T = cz_span - D_SKIN_UM
    cz_V = cz_A * cz_T

    # R1 to locate CZ centroid mean in HCR frame (rigid only; ignore
    # R1's own scales — sxy is trusted from the area estimator).
    fit = coarse_align_revised(
        cz_um, gfp_um, cz_surf, hcr_surf, aniso_refine=False
    )
    hcr_box_center = fit.minimal_translation  # (x, y, z) µm in HCR

    # HCR matched box — xy at the R1-projected CZ centre, half-widths
    # = sxy × (CZ half-spans), so A_HCR = sxy² · A_CZ
    half_x = 0.5 * sxy * cz_x_span
    half_y = 0.5 * sxy * cz_y_span
    box_x0 = hcr_box_center[0] - half_x
    box_x1 = hcr_box_center[0] + half_x
    box_y0 = hcr_box_center[1] - half_y
    box_y1 = hcr_box_center[1] + half_y
    hcr_A = (box_x1 - box_x0) * (box_y1 - box_y0)  # = sxy² · cz_A

    # HCR side — depth band on GFP+ cells inside the matched xy box
    gfp_depth = depth_from_surface(gfp_um, hcr_surf)
    in_box_xy = (
        (gfp_um[:, 0] >= box_x0) & (gfp_um[:, 0] <= box_x1)
        & (gfp_um[:, 1] >= box_y0) & (gfp_um[:, 1] <= box_y1)
    )
    gfp_in_box_depth = gfp_depth[in_box_xy]
    if gfp_in_box_depth.size == 0:
        raise RuntimeError(f"{sid}: no HCR strict-GFP+ cells in matched box")
    hcr_span = float(np.nanpercentile(gfp_in_box_depth, 99))
    hcr_mask = in_box_xy & (gfp_depth >= D_SKIN_UM) & (gfp_depth <= hcr_span)
    n_hcr = int(hcr_mask.sum())
    hcr_T = hcr_span - D_SKIN_UM
    hcr_V = hcr_A * hcr_T

    rho_cz = n_cz / cz_V
    rho_hcr = n_hcr / hcr_V if hcr_V > 0 else float("nan")
    if rho_hcr == 0 or not np.isfinite(rho_hcr):
        raise RuntimeError(f"{sid}: degenerate HCR density")
    sz_est = (rho_cz / rho_hcr) / (sxy ** 2)

    fit_gt = fit_anisotropic_similarity(*landmark_pairs_um(s, active_only=True))
    sxy_gt = float(np.sqrt(fit_gt.scales[0] * fit_gt.scales[1]))
    sz_gt = float(fit_gt.scales[2])

    return {
        "sid": sid,
        "sz": sz_est,
        "sz_gt": sz_gt,
        "err_pct": 100 * (sz_est - sz_gt) / sz_gt,
        "sxy_used": float(sxy),
        "sxy_gt": sxy_gt,
        "n_cz": n_cz,
        "n_hcr": n_hcr,
        "cz_T_um": cz_T,
        "hcr_T_um": hcr_T,
        "cz_A_um2": cz_A,
        "hcr_A_um2": hcr_A,
        "rho_cz_per_um3": rho_cz,
        "rho_hcr_per_um3": rho_hcr,
        "hcr_box_xy": [box_x0, box_x1, box_y0, box_y1],
        "r1_translation": hcr_box_center.tolist(),
        "strict_cutoff_linear": strict_cutoff,
        "n_components_bic": n_components,
    }


if __name__ == "__main__":
    import json
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd == "sxy":
        sids, fn = sys.argv[2:] or sorted(SPOT_SUBJECTS), estimate_sxy_roi_area
    elif cmd == "sz":
        sids, fn = sys.argv[2:] or sorted(SPOT_SUBJECTS), estimate_sz_roi_density
    else:
        sids = sys.argv[1:] or sorted(SPOT_SUBJECTS)
        fn = None

    if fn is None:
        print(f"{'sid':<8} {'sxy':>6} {'sxy_GT':>7} {'sxy_err':>8}  "
              f"{'sz':>6} {'sz_GT':>7} {'sz_err':>8}")
        for sid in sids:
            try:
                rsxy = estimate_sxy_roi_area(sid)
                rsz = estimate_sz_roi_density(sid, sxy=rsxy["sxy_median"])
            except Exception as e:
                print(f"{sid}: ERROR {e}")
                continue
            print(f"{sid:<8} {rsxy['sxy_median']:>6.3f} "
                  f"{rsxy['sxy_gt']:>7.3f} {rsxy['err_pct_median']:>+7.1f}%  "
                  f"{rsz['sz']:>6.3f} {rsz['sz_gt']:>7.3f} "
                  f"{rsz['err_pct']:>+7.1f}%")
    else:
        out = [fn(sid) for sid in sids]
        print(json.dumps(out, indent=2, default=float))
