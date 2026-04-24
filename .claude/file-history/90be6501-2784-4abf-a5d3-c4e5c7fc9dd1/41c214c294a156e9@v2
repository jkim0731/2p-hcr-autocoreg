"""S48/S46-c — centroid-seeded watershed on CZ z-stack to produce per-cell labels.

Alternative to cellpose 3D on CPU (would be 30-60 min/subject). We already have
932 CZ centroids + the z-stack intensity. Seed watershed at each centroid; label
propagates until it hits a basin (low intensity). Optional cap by max-radius.

Probe on 788406 first — measure per-cell volume distribution, label coverage,
sanity-check against the (known-binary) seg-mask-outline as a weak basin prior.

Run: `python probe_watershed.py 788406`
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage.segmentation import watershed  # noqa: E402


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "788406"
    s = load_subject(subj)
    # load z-stack
    zs_path = next(Path(s.coreg_dir).glob("*reg-dim-swapped.ome.tif"))
    t0 = time.time()
    vol = tifffile.imread(zs_path).astype(np.float32)
    print(f"  load z-stack: {vol.shape} {vol.dtype} in {time.time()-t0:.1f}s  "
          f"range=[{vol.min():.0f},{vol.max():.0f}]")

    # smooth lightly (cell bodies ~ 6-10 µm; CZ voxel ~ 1x0.78x0.78 µm so sigma ~ 1-2 px)
    t0 = time.time()
    vol_smooth = gaussian_filter(vol, sigma=(1.0, 1.5, 1.5))
    print(f"  gaussian smooth (1, 1.5, 1.5): {time.time()-t0:.1f}s")

    # seed markers from centroids
    markers = np.zeros(vol.shape, dtype=np.int32)
    pts = s.cz_centroids[["z_px", "y_px", "x_px"]].astype(int).values
    ids = s.cz_centroids["cz_id"].astype(int).values
    Z, Y, X = vol.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, Z - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, Y - 1)
    pts[:, 2] = np.clip(pts[:, 2], 0, X - 1)
    for pt, cid in zip(pts, ids):
        markers[pt[0], pt[1], pt[2]] = int(cid)
    print(f"  seeded {len(pts)} markers")

    # watershed on negative intensity (bright = basin center)
    t0 = time.time()
    # use mask = foreground (intensity > low threshold) to prevent filling background
    p5 = np.percentile(vol_smooth, 5)
    bg_thresh = max(p5, 100.0)  # adjust per-subject
    mask = vol_smooth > bg_thresh
    print(f"  bg_thresh={bg_thresh:.0f}  mask voxels={mask.sum():,} "
          f"({mask.mean()*100:.1f}%)")
    labels = watershed(-vol_smooth, markers=markers, mask=mask)
    print(f"  watershed: {time.time()-t0:.1f}s  uniq_labels={len(np.unique(labels))}")

    # per-cell volume stats
    sizes = np.bincount(labels.ravel())
    nonzero = sizes[1:]  # skip background
    per_cell = nonzero[nonzero > 0]
    cz_um3 = float(s.cz_xy_um) ** 2 * float(s.cz_z_um)
    print(f"  cells_labeled: {len(per_cell)} / {len(pts)}")
    print(f"  vol_px: median={np.median(per_cell):.0f}  "
          f"p10={np.percentile(per_cell, 10):.0f}  "
          f"p90={np.percentile(per_cell, 90):.0f}")
    print(f"  vol_um3: median={np.median(per_cell)*cz_um3:.0f}  "
          f"p10={np.percentile(per_cell, 10)*cz_um3:.0f}  "
          f"p90={np.percentile(per_cell, 90)*cz_um3:.0f}")

    # save labels as tif
    out = Path(__file__).resolve().parent / f"cz_labels_{subj}.tif"
    tifffile.imwrite(str(out), labels.astype(np.uint32), compression="zlib")
    print(f"  wrote {out}")

    # sanity per-cell per-axis extent
    # for a sample of 10 cells, compute bbox
    sample_ids = np.random.RandomState(0).choice(ids, 10, replace=False)
    for cid in sample_ids:
        where = np.where(labels == int(cid))
        if len(where[0]) == 0:
            print(f"  cz_id={cid}: EMPTY")
            continue
        dz = (where[0].max() - where[0].min()) * float(s.cz_z_um)
        dy = (where[1].max() - where[1].min()) * float(s.cz_xy_um)
        dx = (where[2].max() - where[2].min()) * float(s.cz_xy_um)
        n = len(where[0])
        print(f"  cz_id={cid}: n_vox={n}  bbox_um=({dz:.1f}, {dy:.1f}, {dx:.1f})")


if __name__ == "__main__":
    main()
