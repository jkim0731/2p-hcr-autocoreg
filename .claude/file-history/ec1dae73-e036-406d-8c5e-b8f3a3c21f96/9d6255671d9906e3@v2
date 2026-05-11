"""S15 — Cellpose-SAM on HCR 405 inside the GT-matched volume.

For each subject:
  1. Compute a tight bounding box (level-2 voxel coords) around the HCR
     centroids of every cell in `coreg_table.csv` (those are the GT-matched
     HCR cells), expanded by `MARGIN_UM` on each axis.
  2. Slice the 405 channel (level-2) and the existing segmentation mask
     (`segmentation_mask_orig_res.zarr`, also level-2) to that box.
  3. Run Cellpose-SAM (cellpose 4 default model) in 3D on the 405 crop,
     once with `cellprob_threshold = -3` and once with `0`.
  4. Build per-cell tables for: existing-in-crop, cellpose@-3, cellpose@0.
  5. Match cellpose ROIs to existing ROIs with greedy assignment using
     centroid distance + IoU.
  6. Write per-subject outputs (npz + parquet + log).

Notes
-----
* HCR centroids and `segmentation_mask_orig_res.zarr` live in the level-2
  voxel frame (~0.988 µm/voxel xy, ~1 µm/voxel z). 405 zarr group level "2"
  matches that frame.
* `metrics.pickle` global_bbox is in level-0 voxels; we don't use it here —
  we use the level-2 mask directly.
* For each cellpose run we save the mask as int32 npz so downstream
  sessions can re-load it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, "/root/capsule/code/dev_code")
from benchmark_data_loader import load_subject  # type: ignore

OUT_DIR = Path("/root/capsule/code/sessions/v3_S15_cellpose_sam_eval/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARGIN_UM = 30.0          # padding on each side of the GT bbox
CELLPOSE_DIAMETER = 12    # voxels at level-2 (~12 µm at ~0.988 µm/voxel)
CELLPROBS = (-3.0, 0.0)


def hcr_405_level2(hcr_dir: Path) -> zarr.Array:
    z = zarr.open(str(hcr_dir / "image_tile_fusing/fused/channel_405.zarr"), mode="r")
    return z["2"]


def hcr_seg_level2(hcr_dir: Path) -> zarr.Array:
    return zarr.open(
        str(hcr_dir / "cell_body_segmentation/segmentation_mask_orig_res.zarr"),
        mode="r",
    )


def gt_matched_bbox_l2_vox(s, margin_um: float) -> dict:
    """Bounding box of `coreg_table` HCR centroids, padded by `margin_um`.

    Returns level-2 voxel slice indices: dict with keys z, y, x = (lo, hi).
    """
    matched_ids = set(int(i) for i in s.coreg_table["hcr_id"].values)
    c = s.hcr_centroids
    cm = c[c["hcr_id"].isin(matched_ids)]
    if cm.empty:
        raise RuntimeError(f"[{s.subject_id}] no GT-matched HCR centroids found")
    pad_z = margin_um / s.hcr_z_um
    pad_xy = margin_um / s.hcr_xy_um
    z_lo = int(np.floor(cm["z_px"].min() - pad_z))
    z_hi = int(np.ceil (cm["z_px"].max() + pad_z))
    y_lo = int(np.floor(cm["y_px"].min() - pad_xy))
    y_hi = int(np.ceil (cm["y_px"].max() + pad_xy))
    x_lo = int(np.floor(cm["x_px"].min() - pad_xy))
    x_hi = int(np.ceil (cm["x_px"].max() + pad_xy))
    return {"z": (z_lo, z_hi), "y": (y_lo, y_hi), "x": (x_lo, x_hi),
            "n_matched_centroids": int(len(cm))}


def clip_bbox(bbox: dict, shape: tuple) -> dict:
    z_lo = max(0, bbox["z"][0]);  z_hi = min(shape[-3], bbox["z"][1])
    y_lo = max(0, bbox["y"][0]);  y_hi = min(shape[-2], bbox["y"][1])
    x_lo = max(0, bbox["x"][0]);  x_hi = min(shape[-1], bbox["x"][1])
    out = dict(bbox)
    out.update(z=(z_lo, z_hi), y=(y_lo, y_hi), x=(x_lo, x_hi))
    return out


def compute_per_label_stats(label_vol: np.ndarray, intensity_vol: np.ndarray | None = None,
                            voxel_um=(1.0, 0.988, 0.988)) -> pd.DataFrame:
    """Compact per-label table.

    Columns: label, n_vox, vol_um3, eqd_um, cz, cy, cx (centroid in voxel),
             zlo, zhi, ylo, yhi, xlo, xhi, intens_mean, intens_p10, intens_p50,
             intens_p90, boundary_touch.
    """
    from scipy import ndimage as ndi
    if label_vol.size == 0:
        return pd.DataFrame()
    lbls = np.unique(label_vol)
    lbls = lbls[lbls != 0]
    if len(lbls) == 0:
        return pd.DataFrame()
    sums = ndi.sum_labels(np.ones_like(label_vol, dtype=np.float32), label_vol, lbls)
    cz   = ndi.sum_labels(np.indices(label_vol.shape, dtype=np.float32)[0], label_vol, lbls) / sums
    cy   = ndi.sum_labels(np.indices(label_vol.shape, dtype=np.float32)[1], label_vol, lbls) / sums
    cx   = ndi.sum_labels(np.indices(label_vol.shape, dtype=np.float32)[2], label_vol, lbls) / sums

    bbox = ndi.find_objects(label_vol)
    rows = []
    vz, vy, vx = voxel_um
    vox_um3 = vz * vy * vx
    Zsh, Ysh, Xsh = label_vol.shape
    for i, lab in enumerate(lbls):
        sl = bbox[lab - 1] if lab - 1 < len(bbox) and bbox[lab - 1] is not None else None
        if sl is None:
            continue
        zlo, zhi = sl[0].start, sl[0].stop
        ylo, yhi = sl[1].start, sl[1].stop
        xlo, xhi = sl[2].start, sl[2].stop
        n = float(sums[i])
        boundary = int(zlo == 0 or zhi == Zsh or ylo == 0 or yhi == Ysh or xlo == 0 or xhi == Xsh)
        row = dict(
            label=int(lab), n_vox=n, vol_um3=n * vox_um3,
            eqd_um=(6 * n * vox_um3 / np.pi) ** (1 / 3),
            cz=float(cz[i]), cy=float(cy[i]), cx=float(cx[i]),
            zlo=int(zlo), zhi=int(zhi),
            ylo=int(ylo), yhi=int(yhi),
            xlo=int(xlo), xhi=int(xhi),
            boundary_touch=boundary,
        )
        if intensity_vol is not None:
            sub_lab = label_vol[zlo:zhi, ylo:yhi, xlo:xhi]
            sub_int = intensity_vol[zlo:zhi, ylo:yhi, xlo:xhi]
            mask = sub_lab == lab
            vals = sub_int[mask].astype(np.float32)
            if vals.size:
                row["intens_mean"] = float(vals.mean())
                row["intens_std"]  = float(vals.std())
                row["intens_p10"]  = float(np.percentile(vals, 10))
                row["intens_p50"]  = float(np.percentile(vals, 50))
                row["intens_p90"]  = float(np.percentile(vals, 90))
                # outside (1-vox shell)
                # slow but small per-cell bbox
                from scipy.ndimage import binary_dilation
                shell = binary_dilation(mask, iterations=1) & (~mask)
                vals_out = sub_int[shell].astype(np.float32)
                if vals_out.size:
                    row["intens_outside_p50"] = float(np.percentile(vals_out, 50))
                    row["intens_inside_minus_outside_p50"] = (
                        row["intens_p50"] - row["intens_outside_p50"]
                    )
                # core (3-vox erosion) vs shell (1-vox dilation outside)
                from scipy.ndimage import binary_erosion
                core = binary_erosion(mask, iterations=2)
                vals_core = sub_int[core].astype(np.float32) if core.any() else None
                if vals_core is not None and vals_core.size:
                    row["intens_core_p50"] = float(np.percentile(vals_core, 50))
                    row["intens_shell_minus_core_p50"] = (
                        (row["intens_outside_p50"] if "intens_outside_p50" in row else row["intens_p50"])
                        - row["intens_core_p50"]
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def match_cellpose_to_existing(cp_df: pd.DataFrame, ex_df: pd.DataFrame,
                                cp_labels: np.ndarray, ex_labels: np.ndarray,
                                voxel_um=(1.0, 0.988, 0.988),
                                max_dist_um=10.0,
                                min_iou=0.10) -> pd.DataFrame:
    """Greedy match by centroid proximity then verify with IoU.

    For every cellpose ROI, find the nearest existing ROI within `max_dist_um`,
    then compute IoU; record both candidates and accept the pair if IoU >=
    `min_iou`.
    """
    if cp_df.empty or ex_df.empty:
        return pd.DataFrame()
    from scipy.spatial import cKDTree
    vz, vy, vx = voxel_um
    cp_xyz = cp_df[["cz", "cy", "cx"]].to_numpy() * np.array([vz, vy, vx])
    ex_xyz = ex_df[["cz", "cy", "cx"]].to_numpy() * np.array([vz, vy, vx])
    tree = cKDTree(ex_xyz)
    d, idx = tree.query(cp_xyz, k=1)

    rows = []
    for i, (di, j) in enumerate(zip(d, idx)):
        if di > max_dist_um:
            rows.append(dict(cp_label=int(cp_df.iloc[i]["label"]),
                             ex_label=-1, dist_um=float(di), iou=0.0,
                             matched=False))
            continue
        cp_lab = int(cp_df.iloc[i]["label"])
        ex_lab = int(ex_df.iloc[j]["label"])
        # IoU on the union bbox
        cp_row = cp_df.iloc[i]; ex_row = ex_df.iloc[j]
        zlo = int(min(cp_row["zlo"], ex_row["zlo"])); zhi = int(max(cp_row["zhi"], ex_row["zhi"]))
        ylo = int(min(cp_row["ylo"], ex_row["ylo"])); yhi = int(max(cp_row["yhi"], ex_row["yhi"]))
        xlo = int(min(cp_row["xlo"], ex_row["xlo"])); xhi = int(max(cp_row["xhi"], ex_row["xhi"]))
        a = cp_labels[zlo:zhi, ylo:yhi, xlo:xhi] == cp_lab
        b = ex_labels[zlo:zhi, ylo:yhi, xlo:xhi] == ex_lab
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        iou = inter / union if union else 0.0
        rows.append(dict(cp_label=cp_lab, ex_label=ex_lab,
                         dist_um=float(di), iou=float(iou),
                         matched=bool(iou >= min_iou)))
    return pd.DataFrame(rows)


def run_subject(sid: str, model, cellprobs=CELLPROBS, force=False) -> dict:
    s = load_subject(sid)
    print(f"[{sid}] hcr_xy_um={s.hcr_xy_um:.3f} hcr_z_um={s.hcr_z_um:.3f}")
    print(f"[{sid}] coreg matched: {len(s.coreg_table)}, hcr cells: {len(s.hcr_centroids)}")

    z405 = hcr_405_level2(s.hcr_dir)
    seg = hcr_seg_level2(s.hcr_dir)
    print(f"[{sid}] 405 lvl2: {z405.shape}; seg orig_res: {seg.shape}")

    bbox = gt_matched_bbox_l2_vox(s, MARGIN_UM)
    bbox = clip_bbox(bbox, z405.shape)
    print(f"[{sid}] bbox z={bbox['z']} y={bbox['y']} x={bbox['x']} "
          f"(span ~{bbox['z'][1]-bbox['z'][0]}×{bbox['y'][1]-bbox['y'][0]}×{bbox['x'][1]-bbox['x'][0]} vox)")

    # Slice
    z0,z1 = bbox["z"]; y0,y1 = bbox["y"]; x0,x1 = bbox["x"]
    t0 = time.time()
    c405 = np.asarray(z405[0, 0, z0:z1, y0:y1, x0:x1]).astype(np.float32)
    seg_l2 = np.asarray(seg[0, 0, z0:z1, y0:y1, x0:x1])
    print(f"[{sid}] loaded crops in {time.time()-t0:.1f}s; "
          f"c405 shape {c405.shape}, seg shape {seg_l2.shape}")

    # Existing per-label stats over the crop
    ex_df = compute_per_label_stats(seg_l2.astype(np.int32), c405,
                                    voxel_um=(s.hcr_z_um, s.hcr_xy_um, s.hcr_xy_um))
    print(f"[{sid}] existing labels in crop: {len(ex_df)}")

    out_subj = {"subject_id": sid, "bbox": bbox,
                "n_existing_in_crop": int(len(ex_df))}
    cp_results = {}

    for cprob in cellprobs:
        cache = OUT_DIR / f"{sid}_cellpose_cprob{cprob:+.0f}.npz"
        if cache.exists() and not force:
            print(f"[{sid}] {cprob:+.1f}: loading cache {cache.name}")
            d = np.load(cache, allow_pickle=False)
            masks = d["masks"]
        else:
            t0 = time.time()
            print(f"[{sid}] {cprob:+.1f}: running cellpose-SAM…")
            masks, *_ = model.eval(
                c405, do_3D=True,
                diameter=CELLPOSE_DIAMETER,
                cellprob_threshold=float(cprob),
                flow_threshold=0.4,
                z_axis=0,
                anisotropy=float(s.hcr_z_um / s.hcr_xy_um),
                batch_size=4,
            )
            dt = time.time() - t0
            print(f"[{sid}] {cprob:+.1f}: cellpose done in {dt:.1f}s; "
                  f"n_labels={int(masks.max())}")
            np.savez_compressed(cache, masks=masks.astype(np.int32))

        cp_df = compute_per_label_stats(masks.astype(np.int32), c405,
                                        voxel_um=(s.hcr_z_um, s.hcr_xy_um, s.hcr_xy_um))
        match_df = match_cellpose_to_existing(
            cp_df, ex_df, masks.astype(np.int32), seg_l2.astype(np.int32),
            voxel_um=(s.hcr_z_um, s.hcr_xy_um, s.hcr_xy_um),
        )
        cp_df.to_parquet(OUT_DIR / f"{sid}_cellpose_cprob{cprob:+.0f}_per_label.parquet", index=False)
        match_df.to_parquet(OUT_DIR / f"{sid}_match_cprob{cprob:+.0f}.parquet", index=False)

        n_match = int(match_df["matched"].sum()) if not match_df.empty else 0
        n_cp = len(cp_df)
        cp_results[f"cprob_{cprob:+.0f}"] = dict(
            n_cellpose=n_cp,
            n_matched=n_match,
            match_rate_cellpose=n_match / max(1, n_cp),
            match_rate_existing=n_match / max(1, len(ex_df)),
            mean_iou_matched=float(match_df.loc[match_df["matched"], "iou"].mean())
                              if n_match else 0.0,
        )
        print(f"[{sid}] {cprob:+.1f}: cp={n_cp} matched={n_match} "
              f"to-existing={n_match/max(1,len(ex_df)):.2f} "
              f"mean-IoU={cp_results[f'cprob_{cprob:+.0f}']['mean_iou_matched']:.2f}")

    out_subj["cellpose"] = cp_results
    ex_df.to_parquet(OUT_DIR / f"{sid}_existing_per_label.parquet", index=False)
    (OUT_DIR / f"{sid}_summary.json").write_text(json.dumps(out_subj, indent=2))
    return out_subj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", nargs="+", default=["788406", "790322"])
    p.add_argument("--cellprobs", nargs="+", type=float, default=list(CELLPROBS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    from cellpose import models
    model = models.CellposeModel(gpu=True)
    print(f"cellpose model loaded; gpu={getattr(model, 'gpu', None)}")

    summaries = []
    for sid in args.subjects:
        print(f"\n=== {sid} ===")
        s = run_subject(sid, model, cellprobs=tuple(args.cellprobs), force=args.force)
        summaries.append(s)
    (OUT_DIR / "all_subjects_summary.json").write_text(json.dumps(summaries, indent=2))
    print("\nDONE -> ", OUT_DIR)


if __name__ == "__main__":
    main()
