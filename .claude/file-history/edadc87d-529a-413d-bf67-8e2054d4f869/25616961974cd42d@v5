"""Export cropped HCR (405, 488) and filtered segmentations, all resampled
into the CZ-stack voxel frame (CZ extent + margin), for BigWarp inspection
of pairwise-unmix missed cells.

Per subject we emit:
  cz_stack.tif              — CZ 488 zstack, zero-padded to match grid
  cz_seg_all.tif            — full CZ ROI mask, zero-padded to grid
  cz_seg_coreg.tif          — CZ ROI mask filtered to coreg_table.cz_id
  hcr_405_in_cz.tif         — HCR 405 (L2 source) warped into CZ frame
  hcr_488_in_cz.tif         — HCR 488 (L2 source) warped into CZ frame
  hcr_seg_coreg_in_cz.tif   — HCR seg (L0) filtered to coreg_table.hcr_id
  hcr_seg_missed_in_cz.tif  — HCR seg (L0) filtered to coreg.hcr_id NOT in
                              unmixed_all_cells.cell_id (the "missed" cells)
  metadata.json             — transform params, voxel sizes, ID set sizes
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev_code"))
# dev_code imports reference `/data/claude_data/sessions/...` but the archive
# was re-attached at `claude-data_ophys-mfish-autocoreg_260503/`. Prepend the
# two needed session paths so the top-level imports in
# surface_registration_v2 / surfaces_iter08 resolve.
_ARCHIVE_SESSIONS = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503/sessions")
for _sub in ("08_surface_vascular_match", "03c_onset_features/iterations"):
    _p = _ARCHIVE_SESSIONS / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_analysis import hcr_level_resolution
from benchmark_data_loader import load_subject
from cz_volume import load_cz_volume
from hcr_to_cz_warp import (
    build_cz_output_grid,
    cz_mean_depth_um,
    pad_cz_stack,
    warp_hcr_zarr_to_cz_grid,
    _effective_translation,
)
from locked_prior_warm import compute_locked_prior_warm_start
from sz_estimator import get_sz

SESSION_13_SUBJECTS = ["755252", "767022", "782149", "788406"]
OUT_ROOT = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp/outputs/bigwarp_export")


def _find_unmixed_csv(sid: str) -> Path | None:
    """Locate the cell×gene CSV. Priority:
      1. pairwise-unmixing nested layout:
         HCR_{sid}_pairwise-unmixing_*/pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv
         (5 subjects: 755252, 767022, 782149, 788406, 790322)
      2. pairwise-unmixing flat layout:
         HCR_{sid}_pairwise-unmixing_*/unmixed_cell_by_gene_all_rounds.csv
         (767018 / 2026-03-06 run)
      3. cell-typing fallback:
         HCR_{sid}_cell-typing_*/all_cells/mapped_data/basic_results.csv
    All three have a `cell_id` column that maps to HCR seg-zarr label values.
    """
    patterns = [
        f"/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv",
        f"/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/unmixed_cell_by_gene_all_rounds.csv",
        f"/root/capsule/data/HCR_{sid}_cell-typing_*/all_cells/mapped_data/basic_results.csv",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[-1])
    return None


def _load_unmixed_ids(sid: str) -> set[int]:
    p = _find_unmixed_csv(sid)
    if p is None:
        return set()
    # Both unmixed_all_cells.csv and basic_results.csv have a `cell_id`
    # column. basic_results.csv has comment lines starting with '#'.
    df = pd.read_csv(p, comment="#", usecols=lambda c: c == "cell_id")
    return set(int(x) for x in df["cell_id"].dropna().astype(int).unique())


def _find_cz_seg_tiff(sid: str) -> Path:
    pat = (f"/root/capsule/data/multiplane-ophys_{sid}_*-segmentation_*/"
           "channel_0_ref_0/segmentation_masks.tif")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No CZ seg TIFF for {sid}: {pat}")
    return Path(hits[-1])


def _save_ome_tiff(path: Path, vol: np.ndarray, xy_um: float, z_um: float,
                   compress: bool = False) -> None:
    """Write a 3D volume as an OME-TIFF with physical pixel sizes (µm).
    ``compress=True`` enables zlib (deflate) compression — useful for
    sparse label volumes where ~90% of voxels are zero.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": xy_um,
            "PhysicalSizeY": xy_um,
            "PhysicalSizeZ": z_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
        },
        bigtiff=vol.nbytes >= (1 << 31),
    )
    if compress:
        kwargs["compression"] = "zlib"
        kwargs["compressionargs"] = {"level": 6}
    tifffile.imwrite(str(path), vol, **kwargs)


def _pad_to_grid(vol: np.ndarray, grid_shape: tuple[int, int, int],
                 cz_vol_shape: tuple[int, int, int], margin_z: int,
                 margin_xy: int) -> np.ndarray:
    """Zero-pad a CZ-frame volume (3D, same xyz as cz_stack) to ``grid_shape``."""
    out = np.zeros(grid_shape, dtype=vol.dtype)
    out[margin_z : margin_z + cz_vol_shape[0],
        margin_xy : margin_xy + cz_vol_shape[1],
        margin_xy : margin_xy + cz_vol_shape[2]] = vol
    return out


def _filter_labels(vol: np.ndarray, keep_ids: set[int]) -> np.ndarray:
    """Zero out labels not in ``keep_ids``. Returns same dtype."""
    if vol.size == 0:
        return vol
    max_id = int(vol.max())
    if max_id == 0 or not keep_ids:
        return np.zeros_like(vol) if not keep_ids else vol
    keep_arr = np.array(sorted(int(i) for i in keep_ids
                               if 0 < int(i) <= max_id), dtype=vol.dtype)
    lut = np.zeros(max_id + 1, dtype=vol.dtype)
    lut[keep_arr] = keep_arr
    return lut[vol]


def export_subject(sid: str, *, margin_um: float = 20.0, dry_run: bool = False,
                   skip_existing_channels: bool = True) -> dict:
    """If ``skip_existing_channels``, do NOT regenerate cz_stack/405/488 when
    they already exist (they don't depend on filter sets); always (re)write
    the 4 seg outputs."""
    t0 = time.time()
    s = load_subject(sid)
    out_dir = OUT_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {sid} ===", flush=True)

    # --- Load registration + sz ---
    lp = compute_locked_prior_warm_start(s)
    sz_dict = get_sz(s)
    sz_best = float(sz_dict["sz_best"])
    cz_depth_um = cz_mean_depth_um(s, lp)
    T_eff = _effective_translation(lp, sz_best, cz_depth_um)
    print(f"  sz_lp={lp.scales[0]:.3f}  sz_best={sz_best:.3f}  "
          f"cz_mean_depth={cz_depth_um:.1f}µm", flush=True)

    # --- CZ volume + grid ---
    cz_vol = load_cz_volume(s)
    grid = build_cz_output_grid(s, margin_um=margin_um, cz_shape=cz_vol.shape)
    margin_z = int(round(grid.margin_um[0] / float(s.cz_z_um)))
    margin_xy = int(round(grid.margin_um[1] / float(s.cz_xy_um)))
    print(f"  CZ shape={cz_vol.shape}  output grid={grid.shape}  "
          f"margin={margin_um}µm  (cz_voxel xy={s.cz_xy_um}µm z={s.cz_z_um}µm)", flush=True)

    # --- ID sets ---
    coreg = s.coreg_table
    coreg_hcr_ids = set(int(x) for x in coreg["hcr_id"].dropna().astype(int).unique())
    coreg_cz_ids = set(int(x) for x in coreg["cz_id"].dropna().astype(int).unique())
    unmix_ids = _load_unmixed_ids(sid)
    missed_hcr_ids = coreg_hcr_ids - unmix_ids  # in coreg but NOT in unmixed
    print(f"  coreg_hcr={len(coreg_hcr_ids)}  coreg_cz={len(coreg_cz_ids)}  "
          f"unmixed={len(unmix_ids)}  missed={len(missed_hcr_ids)}", flush=True)

    # --- HCR zarr paths ---
    ch_root = s.hcr_dir / "image_tile_fusing" / "fused"
    path_405 = ch_root / "channel_405.zarr"
    path_488 = ch_root / "channel_488.zarr"
    path_seg = s.hcr_dir / "cell_body_segmentation" / "segmentation_mask.zarr"
    for p in (path_405, path_488, path_seg):
        if not p.exists():
            raise FileNotFoundError(f"Missing zarr: {p}")
    xy_l2, z_l2 = hcr_level_resolution(s, 2)
    xy_l0 = float(s.hcr_xy_um) / 4.0
    z_l0 = float(s.hcr_z_um)

    info_405 = info_488 = None

    # --- cz_stack.tif ---
    p = out_dir / "cz_stack.tif"
    if not skip_existing_channels or not p.exists():
        cz_padded = pad_cz_stack(cz_vol, grid, s).astype(np.float32)
        if not dry_run:
            _save_ome_tiff(p, cz_padded, xy_um=s.cz_xy_um, z_um=s.cz_z_um)
            print(f"  wrote {p.name}", flush=True)
        del cz_padded

    # --- HCR 405/488 (L2) ---
    p = out_dir / "hcr_405_in_cz.tif"
    if not skip_existing_channels or not p.exists():
        t = time.time()
        vol_405, info_405 = warp_hcr_zarr_to_cz_grid(
            str(path_405), level=2, hcr_xy_um=xy_l2, hcr_z_um=z_l2,
            lp=lp, sz=sz_best, T_eff=T_eff, grid=grid,
            order=1, cval=0.0, dtype_out=np.uint16, chunk_out_z=64,
        )
        print(f"  405 L2: warp={time.time()-t:.1f}s", flush=True)
        if not dry_run:
            _save_ome_tiff(p, vol_405, xy_um=s.cz_xy_um, z_um=s.cz_z_um)
        del vol_405

    p = out_dir / "hcr_488_in_cz.tif"
    if not skip_existing_channels or not p.exists():
        t = time.time()
        vol_488, info_488 = warp_hcr_zarr_to_cz_grid(
            str(path_488), level=2, hcr_xy_um=xy_l2, hcr_z_um=z_l2,
            lp=lp, sz=sz_best, T_eff=T_eff, grid=grid,
            order=1, cval=0.0, dtype_out=np.uint16, chunk_out_z=64,
        )
        print(f"  488 L2: warp={time.time()-t:.1f}s", flush=True)
        if not dry_run:
            _save_ome_tiff(p, vol_488, xy_um=s.cz_xy_um, z_um=s.cz_z_um)
        del vol_488

    # --- CZ ROI seg (all + coreg-filtered) ---
    cz_seg_tif = _find_cz_seg_tiff(sid)
    cz_seg = tifffile.imread(str(cz_seg_tif))
    if cz_seg.shape != cz_vol.shape:
        raise ValueError(f"CZ seg shape {cz_seg.shape} != cz_vol shape {cz_vol.shape}")
    cz_seg_padded = _pad_to_grid(cz_seg, grid.shape, cz_vol.shape, margin_z, margin_xy)
    cz_seg_coreg = _filter_labels(cz_seg_padded, coreg_cz_ids)
    cz_seg_all_ids = int((cz_seg_padded > 0).sum() > 0) and int(len(np.unique(cz_seg_padded)) - 1)
    cz_seg_coreg_ids = int(len(np.unique(cz_seg_coreg)) - 1)
    print(f"  cz_seg: all={cz_seg_all_ids} IDs, coreg-filtered={cz_seg_coreg_ids} IDs", flush=True)

    if not dry_run:
        _save_ome_tiff(out_dir / "cz_seg_all.tif", cz_seg_padded,
                       xy_um=s.cz_xy_um, z_um=s.cz_z_um, compress=True)
        _save_ome_tiff(out_dir / "cz_seg_coreg.tif", cz_seg_coreg,
                       xy_um=s.cz_xy_um, z_um=s.cz_z_um, compress=True)
    del cz_seg_padded, cz_seg_coreg, cz_seg

    # --- HCR seg (L0) — two filter sets in one zarr-read pass ---
    t = time.time()
    keep_sets = {"coreg": coreg_hcr_ids, "missed": missed_hcr_ids}
    seg_outs, info_seg = warp_hcr_zarr_to_cz_grid(
        str(path_seg), level=0, hcr_xy_um=xy_l0, hcr_z_um=z_l0,
        lp=lp, sz=sz_best, T_eff=T_eff, grid=grid,
        order=0, cval=0, label_keep_sets=keep_sets, dtype_out=np.uint32,
        chunk_out_z=16,
    )
    coreg_in_out = int(len(np.unique(seg_outs["coreg"][seg_outs["coreg"] > 0])))
    missed_in_out = int(len(np.unique(seg_outs["missed"][seg_outs["missed"] > 0])))
    print(f"  hcr_seg L0: {len(info_seg['chunks'])} chunks  warp={time.time()-t:.1f}s  "
          f"coreg_in_overlap={coreg_in_out}  missed_in_overlap={missed_in_out}", flush=True)

    if not dry_run:
        _save_ome_tiff(out_dir / "hcr_seg_coreg_in_cz.tif", seg_outs["coreg"],
                       xy_um=s.cz_xy_um, z_um=s.cz_z_um, compress=True)
        _save_ome_tiff(out_dir / "hcr_seg_missed_in_cz.tif", seg_outs["missed"],
                       xy_um=s.cz_xy_um, z_um=s.cz_z_um, compress=True)
    del seg_outs

    # --- Drop the obsolete union seg file from the previous version. ---
    legacy = out_dir / "hcr_seg_in_cz.tif"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}", flush=True)

    meta = {
        "subject_id": sid,
        "cz_dir": str(s.coreg_dir),
        "hcr_dir": str(s.hcr_dir),
        "cz_seg_path": str(cz_seg_tif),
        "margin_um": margin_um,
        "output_shape_zyx": list(grid.shape),
        "output_voxel_um": {"z": s.cz_z_um, "y": s.cz_xy_um, "x": s.cz_xy_um},
        "cz_stack_shape_zyx": list(cz_vol.shape),
        "hcr_level2_voxel_um": {"xy": xy_l2, "z": z_l2},
        "hcr_level0_voxel_um": {"xy": xy_l0, "z": z_l0},
        "locked_prior": {
            "R": lp.R.tolist(),
            "scales_sz_sy_sx": lp.scales.tolist(),
            "translation_zyx_um": lp.translation.tolist(),
            "src_mean_zyx_um": lp.src_mean.tolist(),
            "sxy_value": lp.sxy_value,
            "sxy_source": lp.sxy_source,
            "rotation_deg_z": lp.rotation_deg_z,
            "pwr_method": lp.pwr_method,
            "pwr_ncc": lp.pwr_ncc,
        },
        "sz_best": sz_best,
        "sz_lp": float(lp.scales[0]),
        "T_eff_zyx_um": T_eff.tolist(),
        "cz_mean_depth_um": cz_depth_um,
        "id_sets": {
            "coreg_hcr_id_count": len(coreg_hcr_ids),
            "coreg_cz_id_count": len(coreg_cz_ids),
            "unmixed_cell_id_count": len(unmix_ids),
            "missed_hcr_count_total": len(missed_hcr_ids),
            "cz_seg_unique_id_count": cz_seg_all_ids,
            "cz_seg_coreg_ids_in_output": cz_seg_coreg_ids,
            "hcr_seg_coreg_in_overlap": coreg_in_out,
            "hcr_seg_missed_in_overlap": missed_in_out,
        },
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote → {out_dir}  ({time.time()-t0:.1f}s total)", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="*", default=None,
                        help="Subset of session-13 subjects (default: all 4)")
    parser.add_argument("--margin-um", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    subjects = args.subjects or SESSION_13_SUBJECTS
    results = []
    for sid in subjects:
        try:
            results.append(export_subject(sid, margin_um=args.margin_um,
                                          dry_run=args.dry_run))
        except Exception as e:
            print(f"  !! {sid} FAILED: {e}", flush=True)
            results.append({"sid": sid, "error": str(e)})
    print("\nSummary:")
    for r in results:
        sid = r.get("sid") or r.get("subject_id", "?")
        if "error" in r:
            print(f"  {sid}: FAIL — {r['error']}")
        else:
            print(f"  {sid}: OK  ({r.get('elapsed_sec', 0):.1f}s)")


if __name__ == "__main__":
    main()
