"""S46-b — per-centroid image-feature diagnostic.

S45 diagnosis: top-K ceiling shows centroid-only F6+dist is structurally
insufficient on stress subjects (782149 GT-in-top-500 = 0/303; 755252/
767022 p95 rank 5 034 / 11 998). This probe tests whether per-centroid
image features from the CZ z-stack (`reg-dim-swapped.ome.tif`) and HCR
channel-488 fused volume provide discriminating signal orthogonal to
F6.

Features per centroid, both modalities, computed in a small isotropic
µm bbox (~±3 µm XY, ±2 µm Z):
  - img_mean — mean intensity
  - img_std  — std intensity
  - img_p90  — 90th percentile intensity (robust peak)
  - img_lap  — mean |Laplacian| (texture/edge magnitude)

Per-pair AUC (correct > wrong? within same CZ putative group) and a
within-CZ z-scored logistic regression against a baseline of
(z_dist, z_cos). Tested on all 4 subjects; if signal is present on the
stress subjects, promote to F6 as an additional invariant block.

Run: `python probe_image_feat_signal.py [subject_id ...]`
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import laplace
from scipy.spatial.distance import cdist

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa: E402
from lib.cell_features import extract_cell_features, invariant_feature_mask  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# Per-centroid image-feature extraction
# ----------------------------------------------------------------------


def _cz_zstack_path(s) -> Path:
    files = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*zstack.tif"))
    if not files:
        raise FileNotFoundError(f"No CZ z-stack in {s.coreg_dir}")
    return files[0]


def _hcr_zarr(s, channel: str = "488"):
    zp = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    return zarr.open(str(zp), mode="r")


def _extract_bbox_features(vol: np.ndarray, pts_zyx: np.ndarray,
                           half_z: int, half_y: int, half_x: int) -> np.ndarray:
    """For each integer voxel center in pts_zyx, extract a bbox from vol and
    compute (mean, std, p90, lap_mean). Returns (N, 4).

    `vol` can be a numpy ndarray or a zarr Array (any array that supports
    slicing returning a numpy ndarray)."""
    n = len(pts_zyx)
    out = np.full((n, 4), np.nan, dtype=float)
    Z, Y, X = vol.shape
    for i, (z, y, x) in enumerate(pts_zyx):
        z0 = int(z) - half_z; z1 = int(z) + half_z + 1
        y0 = int(y) - half_y; y1 = int(y) + half_y + 1
        x0 = int(x) - half_x; x1 = int(x) + half_x + 1
        # Clip to volume bounds.
        if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z or y1 > Y or x1 > X:
            # Partial bbox: clip and continue.
            z0 = max(0, z0); y0 = max(0, y0); x0 = max(0, x0)
            z1 = min(Z, z1); y1 = min(Y, y1); x1 = min(X, x1)
            if z1 - z0 < 1 or y1 - y0 < 1 or x1 - x0 < 1:
                continue
        patch = np.asarray(vol[z0:z1, y0:y1, x0:x1]).astype(np.float32)
        if patch.size == 0:
            continue
        out[i, 0] = float(patch.mean())
        out[i, 1] = float(patch.std())
        out[i, 2] = float(np.percentile(patch, 90))
        # Laplacian magnitude (3D) in the patch.
        lap = laplace(patch)
        out[i, 3] = float(np.abs(lap).mean())
    return out


def cz_image_features(s, centroids_px: np.ndarray,
                      bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                      ) -> np.ndarray:
    """Per-centroid CZ image features in (z, y, x) pixel coords.
    bbox_um = (half_z, half_y, half_x) in µm; converted to voxels.
    """
    path = _cz_zstack_path(s)
    with tifffile.TiffFile(path) as tf:
        vol = tf.asarray()
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = vol[0]
    half_z = max(1, int(round(bbox_um[0] / s.cz_z_um)))
    half_y = max(1, int(round(bbox_um[1] / s.cz_xy_um)))
    half_x = max(1, int(round(bbox_um[2] / s.cz_xy_um)))
    pts = np.asarray(centroids_px, dtype=int)
    print(f"  CZ bbox voxels (hz,hy,hx)=({half_z},{half_y},{half_x}); n={len(pts)}", flush=True)
    return _extract_bbox_features(vol, pts, half_z, half_y, half_x)


def hcr_image_features(s, centroids_px: np.ndarray, *, channel: str = "488",
                       level: int = 2,
                       bbox_um: tuple[float, float, float] = (2.0, 3.0, 3.0)
                       ) -> np.ndarray:
    """Per-centroid HCR image features. Centroids are in native HCR pyramid
    level-0 pixel coords, as in `s.hcr_centroids[['z_px','y_px','x_px']]`;
    we divide by 2^(level-... ) as needed. Actually `hcr_centroids` is at
    level-2 pixel coords already (matches `cell_centroids.npy`), so for
    level=2 no conversion is needed.
    """
    arr = _hcr_zarr(s, channel=channel)
    vol = arr[str(level)][0, 0]  # zarr Array — chunked disk access
    # Work out pyramid factor between centroid coord system (level 2) and
    # requested level.
    factor_xy = 2 ** (level - 2)
    factor_z = 2 ** max(0, level - 2)
    xy_um = s.hcr_xy_um * factor_xy
    z_um = s.hcr_z_um * factor_z
    half_z = max(1, int(round(bbox_um[0] / z_um)))
    half_y = max(1, int(round(bbox_um[1] / xy_um)))
    half_x = max(1, int(round(bbox_um[2] / xy_um)))
    pts = np.asarray(centroids_px, dtype=int)
    if factor_xy != 1 or factor_z != 1:
        pts = np.stack([pts[:, 0] // factor_z, pts[:, 1] // factor_xy,
                        pts[:, 2] // factor_xy], axis=1)
    print(f"  HCR bbox voxels (hz,hy,hx)=({half_z},{half_y},{half_x}); n={len(pts)} "
          f"level={level} xy_um={xy_um:.3f}", flush=True)
    return _extract_bbox_features(vol, pts, half_z, half_y, half_x)


# ----------------------------------------------------------------------
# P1's putative generator (same scoring as _p1_teaser._seed_putative)
# ----------------------------------------------------------------------


def p1_putatives(s, K: int = 5):
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    Fc, names, _ = extract_cell_features(s, "cz")
    Fg, _, _ = extract_cell_features(s, "hcr_gfp")
    inv = invariant_feature_mask(names)
    keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
    mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
    Fcn = (Fc[:, keep] - mu) / sd; Fgn = (Fg[:, keep] - mu) / sd
    Fcn = Fcn / (np.linalg.norm(Fcn, axis=1, keepdims=True) + 1e-9)
    Fgn = Fgn / (np.linalg.norm(Fgn, axis=1, keepdims=True) + 1e-9)

    D = cdist(cz_init, hcr_um)
    cosS = Fcn @ Fgn.T
    score = D - 25.0 * cosS
    order = np.argsort(score, axis=1)[:, :K]
    return order, cz_ids, hcr_ids, D, cosS


# ----------------------------------------------------------------------
# Main probe
# ----------------------------------------------------------------------


def run_subject(subj: str, K: int = 5) -> dict:
    print(f"\n=== {subj} ===", flush=True)
    s = load_subject(subj)
    cz_px = s.cz_centroids[["z_px", "y_px", "x_px"]].values
    hcr_px = s.hcr_centroids[["z_px", "y_px", "x_px"]].values

    # Image features per centroid, full populations.
    t0 = time.time()
    cz_img = cz_image_features(s, cz_px)
    print(f"  CZ image features in {time.time()-t0:.1f}s  "
          f"nan rows={int(np.isnan(cz_img).any(1).sum())}/{len(cz_img)}", flush=True)

    # HCR: only need GFP+ subset for this probe. Filter to GFP+ IDs.
    gfp_ids = set(s.hcr_gfp_df["hcr_id"].astype(int).tolist())
    hcr_ids_all = s.hcr_centroids["hcr_id"].astype(int).values
    keep_mask = np.array([int(i) in gfp_ids for i in hcr_ids_all])
    hcr_px_gfp = hcr_px[keep_mask]
    hcr_ids_gfp = hcr_ids_all[keep_mask]

    t0 = time.time()
    hcr_img = hcr_image_features(s, hcr_px_gfp, channel="488", level=2)
    print(f"  HCR image features in {time.time()-t0:.1f}s  "
          f"nan rows={int(np.isnan(hcr_img).any(1).sum())}/{len(hcr_img)}", flush=True)

    # P1 putatives.
    t0 = time.time()
    order, cz_ids, hcr_ids, D, cosS = p1_putatives(s, K=K)
    print(f"  P1 putatives in {time.time()-t0:.1f}s  K={K}  "
          f"n_cz={len(cz_ids)} n_hcr={len(hcr_ids)}", flush=True)

    # Build a lookup: cz_id → cz_img_row_idx.
    cz_id_to_idx = {int(cid): i for i, cid in enumerate(
        s.cz_centroids["cz_id"].astype(int).values)}
    hcr_id_to_img_idx = {int(hid): i for i, hid in enumerate(hcr_ids_gfp)}
    # Note: `hcr_ids` from p1_putatives is the GFP+ id order used inside
    # centroids_um — align by id, not position.
    hcr_pos_to_id = {i: int(hid) for i, hid in enumerate(hcr_ids)}
    gt_map = dict(zip(s.coreg_table["cz_id"].astype(int),
                      s.coreg_table["hcr_id"].astype(int)))

    rows = []
    cz_in_topk = 0; cz_not_in_topk = 0
    for i_cz, cid in enumerate(cz_ids):
        gt_h = gt_map.get(int(cid))
        if gt_h is None:
            continue
        if int(gt_h) not in set(int(hcr_pos_to_id[j]) for j in order[i_cz]):
            cz_not_in_topk += 1
            continue
        cz_in_topk += 1
        cz_row_idx = cz_id_to_idx.get(int(cid))
        if cz_row_idx is None:
            continue
        cz_feat = cz_img[cz_row_idx]
        for rank, jpos in enumerate(order[i_cz]):
            jnat = int(hcr_pos_to_id[jpos])
            hcr_row_idx = hcr_id_to_img_idx.get(jnat)
            if hcr_row_idx is None:
                continue
            hcr_feat = hcr_img[hcr_row_idx]
            row = dict(
                subject=subj, cz_id=int(cid), hcr_id=jnat, rank=int(rank),
                dist_um=float(D[i_cz, jpos]), cos=float(cosS[i_cz, jpos]),
                cz_mean=cz_feat[0], cz_std=cz_feat[1],
                cz_p90=cz_feat[2], cz_lap=cz_feat[3],
                hcr_mean=hcr_feat[0], hcr_std=hcr_feat[1],
                hcr_p90=hcr_feat[2], hcr_lap=hcr_feat[3],
                y=int(jnat == int(gt_h)),
            )
            # Cross-modal differences (normalized within each CZ group in LR).
            row["abs_diff_mean"] = abs(cz_feat[0] - hcr_feat[0])
            row["abs_diff_std"] = abs(cz_feat[1] - hcr_feat[1])
            row["abs_diff_p90"] = abs(cz_feat[2] - hcr_feat[2])
            row["abs_diff_lap"] = abs(cz_feat[3] - hcr_feat[3])
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  GT in top-{K}: {cz_in_topk}/{cz_in_topk + cz_not_in_topk}", flush=True)

    # Per-feature within-CZ pairwise AUC: for each feature, compare
    # the correct putative to wrong putatives in the same CZ group.
    if len(df) == 0:
        print("  no pairs — skipping AUC.")
        return dict(subject=subj, n_pair=0)

    feats = ["dist_um", "cos",
             "cz_mean", "cz_std", "cz_p90", "cz_lap",
             "hcr_mean", "hcr_std", "hcr_p90", "hcr_lap",
             "abs_diff_mean", "abs_diff_std", "abs_diff_p90", "abs_diff_lap"]
    auc_rows = []
    by_cz = df.groupby("cz_id")
    for f in feats:
        num = 0; den = 0
        for cid_, g in by_cz:
            corr = g.loc[g["y"] == 1, f].astype(float).values
            wrng = g.loc[g["y"] == 0, f].astype(float).values
            corr = corr[np.isfinite(corr)]; wrng = wrng[np.isfinite(wrng)]
            for x in corr:
                for z in wrng:
                    num += (x > z) + 0.5 * (x == z)
                    den += 1
        auc = num / max(1, den)
        auc_rows.append({"subject": subj, "feature": f, "auc": auc})
    auc_df = pd.DataFrame(auc_rows)
    print(auc_df.to_string(index=False))

    # Within-CZ z-scored logistic regression.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = df.copy()
    for f in feats:
        g = X.groupby("cz_id")[f]
        mu = g.transform("mean"); sd = g.transform("std").replace(0, 1e-6)
        X[f"z_{f}"] = (X[f] - mu) / sd

    base = ["z_dist_um", "z_cos"]
    full = base + [f"z_{f}" for f in feats if f not in ("dist_um", "cos")]
    for name, cols in [("baseline (z_dist+z_cos)", base),
                       ("full (baseline + image feats)", full)]:
        A = X[cols].values.astype(float)
        y = X["y"].values.astype(int)
        mask = np.isfinite(A).all(1)
        A = A[mask]; y = y[mask]
        if len(np.unique(y)) < 2 or len(A) < 20:
            print(f"  {name}: skipped (insufficient data)")
            continue
        lr = LogisticRegression(max_iter=1000).fit(A, y)
        p = lr.predict_proba(A)[:, 1]
        auc = roc_auc_score(y, p)
        print(f"  {name:45s} AUC={auc:.3f}  n={len(A)}")

    return dict(subject=subj, n_pair=len(df), auc_df=auc_df, pairs=df)


def main():
    subjects = sys.argv[1:] or ["788406", "755252", "767022", "782149"]
    all_auc = []
    for sid in subjects:
        try:
            out = run_subject(sid, K=5)
            if "auc_df" in out:
                all_auc.append(out["auc_df"])
        except Exception as e:
            print(f"  !!! error on {sid}: {e}", flush=True)

    if all_auc:
        print("\n=== SUMMARY — within-CZ per-feature AUC ===")
        summary = pd.concat(all_auc, ignore_index=True)
        pivot = summary.pivot(index="feature", columns="subject", values="auc")
        print(pivot.to_string())
        Path(__file__).parent.joinpath("image_feat_auc.csv").write_text(pivot.to_csv())


if __name__ == "__main__":
    main()
