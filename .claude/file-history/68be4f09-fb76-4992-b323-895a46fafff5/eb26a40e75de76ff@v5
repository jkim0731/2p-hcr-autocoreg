"""Driver: run the image-based surface + L2 estimator on all benchmark
subjects, HCR (combined channels) and CZ (raw + segmentation-masked
background for L2 detection).

Outputs go to /tmp/03_image_based_surface/ (mirrored into
/root/capsule/code/sessions/03_image_based_surface_estimation/ as far
as permissions allow).  A CSV with per-subject metrics and PNG
walkthroughs per subject.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path("/root/capsule")
DEV = ROOT / "code" / "dev_code"
sys.path.insert(0, str(DEV))

# Load the estimator module (filename starts with a digit so we can't
# use plain import).
_spec = importlib.util.spec_from_file_location(
    "img_surface_l2", str(DEV / "03_image_based_surface.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["img_surface_l2"] = _mod
_spec.loader.exec_module(_mod)
estimate_surface_and_l2_image_based = _mod.estimate_surface_and_l2_image_based
depth_profile_stats = _mod.depth_profile_stats

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS,
    cz_px_to_um,
    hcr_px_to_um,
    load_subject,
)


OUT_TMP = Path("/tmp/03_image_based_surface")
OUT_SESSION = ROOT / "code" / "sessions" / "03_image_based_surface_estimation"
(OUT_TMP / "figures").mkdir(parents=True, exist_ok=True)
(OUT_SESSION / "figures").mkdir(parents=True, exist_ok=True)


def _find_cz_seg_mask(subject_id: str) -> Path | None:
    """Return the newest cz-segmentation folder's segmentation_masks.tif."""
    patterns = [
        f"data/multiplane-ophys_{subject_id}_*_cortical-zstack-segmentation_*",
        f"data/multiplane-ophys_{subject_id}_*_cortical-zstack-seg_*",
    ]
    for pat in patterns:
        dirs = sorted(glob.glob(str(ROOT / pat)))
        if not dirs:
            continue
        cand = Path(dirs[-1]) / "channel_0_ref_0" / "segmentation_masks.tif"
        if cand.exists():
            return cand
    return None


def _load_cz_image(s):
    cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not cz_tifs:
        cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
    if not cz_tifs:
        return None
    img = tifffile.imread(str(cz_tifs[0]))
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    return img.astype(np.float32, copy=False)


def run_one(subject_id: str) -> dict:
    out = {"subject": subject_id}
    t0 = time.time()
    s = load_subject(subject_id)
    print(f"[{subject_id}] loaded. cz_z={s.cz_z_um}, hcr_z={s.hcr_z_um}")

    cz_xyz = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].values, s
    )[:, [2, 1, 0]]
    hcr_xyz = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s
    )[:, [2, 1, 0]]

    # -----------------------------------------------------------------
    # HCR combined volume
    # -----------------------------------------------------------------
    try:
        vol, xy_um, z_um, channels = load_hcr_combined(s, level=4)
    except Exception as exc:
        print(f"[{subject_id}] HCR load failed: {exc}")
        vol = None
    hcr_result = None
    if vol is not None:
        hcr_result = estimate_surface_and_l2_image_based(
            vol, z_um, xy_um,
            anchor_xy_tile_um=150.0,
            column_stride_um=10.0,
            band_um=150.0,
            column_margin=0.25,
            min_signal_abs_frac=0.20,
            column_min_thick_um=10.0,
            target_quantile=0.85,
            clamp_tile_um=120.0,
            clamp_within_tile_q=0.10,
            safety_offset_um=3.0,
        )
        print(
            f"[{subject_id}] HCR  surface={hcr_result.surface is not None}",
            f"tops={hcr_result.n_column_tops}",
            f"anchor_hits={hcr_result.n_anchor_hits}/{hcr_result.n_anchor_tiles}",
            f"lift={hcr_result.clamp_lift_um:.1f} clamp_tiles={hcr_result.n_clamp_tiles}",
            f"bg={hcr_result.baseline_mean:.3f} σ={hcr_result.baseline_sigma:.3f}",
        )
        if hcr_result.surface is not None:
            out.update({
                "hcr_surf_c": hcr_result.surface["c"],
                "hcr_surf_a": hcr_result.surface["a"],
                "hcr_surf_b": hcr_result.surface["b"],
                "hcr_surf_p": hcr_result.surface.get("p", 0.0),
                "hcr_surf_q": hcr_result.surface.get("q", 0.0),
                "hcr_surf_r": hcr_result.surface.get("r", 0.0),
                "hcr_surf_tilt_deg": hcr_result.surface["tilt_deg"],
                "hcr_surf_resid_mad": hcr_result.surface["resid_mad_um"],
                "hcr_n_column_tops": hcr_result.n_column_tops,
                "hcr_n_anchor_hits": hcr_result.n_anchor_hits,
                "hcr_n_anchor_tiles": hcr_result.n_anchor_tiles,
                "hcr_n_clamp_tiles": hcr_result.n_clamp_tiles,
                "hcr_clamp_lift_um": hcr_result.clamp_lift_um,
                "hcr_baseline_mean": hcr_result.baseline_mean,
                "hcr_baseline_sigma": hcr_result.baseline_sigma,
                "hcr_surf_thr": hcr_result.surf_thr,
                "hcr_dyn_hint": hcr_result.dyn_hint,
            })
            stats = depth_profile_stats(hcr_xyz, hcr_result.surface)
            out.update({f"hcr_{k}": v for k, v in stats.items()})

    # -----------------------------------------------------------------
    # CZ volume + CZ segmentation mask
    # -----------------------------------------------------------------
    cz_img = _load_cz_image(s)
    cz_result = None
    if cz_img is not None:
        cz_mask_path = _find_cz_seg_mask(subject_id)
        bg_mask = None
        if cz_mask_path is not None:
            cz_seg = tifffile.imread(str(cz_mask_path))
            while cz_seg.ndim > 3 and cz_seg.shape[0] == 1:
                cz_seg = cz_seg[0]
            if cz_seg.shape == cz_img.shape:
                bg_mask = (cz_seg == 0)
            else:
                print(
                    f"[{subject_id}] CZ seg shape {cz_seg.shape}"
                    f" != img {cz_img.shape}; skipping L2 mask"
                )

        cz_result = estimate_surface_and_l2_image_based(
            cz_img, s.cz_z_um, s.cz_xy_um,
            anchor_xy_tile_um=80.0,
            column_stride_um=8.0,
            band_um=100.0,
            column_margin=0.25,
            min_signal_abs_frac=0.20,
            column_min_thick_um=10.0,
            target_quantile=0.70,
            clamp_tile_um=80.0,
            clamp_within_tile_q=0.10,
            safety_offset_um=3.0,
        )
        print(
            f"[{subject_id}] CZ   surface={cz_result.surface is not None}",
            f"tops={cz_result.n_column_tops}",
            f"anchor_hits={cz_result.n_anchor_hits}/{cz_result.n_anchor_tiles}",
            f"lift={cz_result.clamp_lift_um:.1f} clamp_tiles={cz_result.n_clamp_tiles}",
            f"bg={cz_result.baseline_mean:.1f} σ={cz_result.baseline_sigma:.1f}",
        )
        if cz_result.surface is not None:
            out.update({
                "cz_surf_c": cz_result.surface["c"],
                "cz_surf_a": cz_result.surface["a"],
                "cz_surf_b": cz_result.surface["b"],
                "cz_surf_p": cz_result.surface.get("p", 0.0),
                "cz_surf_q": cz_result.surface.get("q", 0.0),
                "cz_surf_r": cz_result.surface.get("r", 0.0),
                "cz_surf_tilt_deg": cz_result.surface["tilt_deg"],
                "cz_surf_resid_mad": cz_result.surface["resid_mad_um"],
                "cz_n_column_tops": cz_result.n_column_tops,
                "cz_n_anchor_hits": cz_result.n_anchor_hits,
                "cz_n_anchor_tiles": cz_result.n_anchor_tiles,
                "cz_n_clamp_tiles": cz_result.n_clamp_tiles,
                "cz_clamp_lift_um": cz_result.clamp_lift_um,
                "cz_baseline_mean": cz_result.baseline_mean,
                "cz_baseline_sigma": cz_result.baseline_sigma,
                "cz_surf_thr": cz_result.surf_thr,
                "cz_dyn_hint": cz_result.dyn_hint,
            })
            stats = depth_profile_stats(cz_xyz, cz_result.surface)
            out.update({f"cz_{k}": v for k, v in stats.items()})

    out["wall_time_s"] = round(time.time() - t0, 1)
    out["hcr_result"] = hcr_result
    out["cz_result"] = cz_result
    return out


def main():
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        out = run_one(sid)
        # Keep only scalars in the CSV, drop the full result objects
        scalar = {k: v for k, v in out.items()
                  if k not in {"hcr_result", "cz_result"}}
        rows.append(scalar)
        # Also dump tile records + save figure (separate module)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TMP / "results.csv", index=False)
    try:
        df.to_csv(OUT_SESSION / "results.csv", index=False)
    except PermissionError:
        pass
    print(df)
    return df


if __name__ == "__main__":
    main()
