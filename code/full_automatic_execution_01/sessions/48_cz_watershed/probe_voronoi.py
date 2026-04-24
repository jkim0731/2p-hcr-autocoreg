"""S48/S46-c — radius-bounded voronoi CZ labels.

Assign each voxel to its nearest CZ centroid within radius R µm, intersected
with an intensity threshold. Bounded in both size (R) and in signal presence.

Run: `python probe_voronoi.py 788406 [R_um=8]`
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
from scipy.ndimage import distance_transform_edt, gaussian_filter  # noqa: E402


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "788406"
    R_um = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    s = load_subject(subj)
    zs_path = next(Path(s.coreg_dir).glob("*reg-dim-swapped.ome.tif"))
    t0 = time.time()
    vol = tifffile.imread(zs_path).astype(np.float32)
    print(f"[{subj} R={R_um}µm] z-stack {vol.shape} in {time.time()-t0:.1f}s")

    vol_s = gaussian_filter(vol, sigma=(1.0, 1.5, 1.5))

    # seed mask: 1 at every centroid voxel
    markers = np.zeros(vol.shape, dtype=bool)
    pts = s.cz_centroids[["z_px", "y_px", "x_px"]].astype(int).values
    ids = s.cz_centroids["cz_id"].astype(int).values
    Z, Y, X = vol.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, Z - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, Y - 1)
    pts[:, 2] = np.clip(pts[:, 2], 0, X - 1)
    markers[pts[:, 0], pts[:, 1], pts[:, 2]] = True

    # anisotropic distance transform w/ index return (voronoi)
    # sampling in (z, y, x) µm
    t0 = time.time()
    sampling = (float(s.cz_z_um), float(s.cz_xy_um), float(s.cz_xy_um))
    # invert: we want distance FROM the seeds, so pass markers as a truthy seed-mask
    # and call distance_transform_edt on the inverse
    dist, idx = distance_transform_edt(
        ~markers, sampling=sampling, return_indices=True
    )
    print(f"  edt+indices: {time.time()-t0:.1f}s")

    # look up the centroid at each voxel
    # idx is shape (ndim, Z, Y, X) giving the nearest seed's coords
    # assignment: id at (idx[0], idx[1], idx[2])
    t0 = time.time()
    # build a lookup from (z, y, x) seed to cz_id via a sparse label volume
    seed_lookup = np.zeros(vol.shape, dtype=np.int32)
    seed_lookup[pts[:, 0], pts[:, 1], pts[:, 2]] = ids
    labels = seed_lookup[idx[0], idx[1], idx[2]]

    # apply radius cap
    labels[dist > R_um] = 0

    # intensity threshold: cells are brighter than background
    # use a per-volume percentile
    p50 = np.percentile(vol_s, 50)
    p90 = np.percentile(vol_s, 90)
    # adaptive: use midpoint of p50..p90
    thr = p50 + 0.3 * (p90 - p50)
    labels[vol_s < thr] = 0

    print(f"  radius+intensity cap: {time.time()-t0:.1f}s  thr={thr:.0f}")

    sizes = np.bincount(labels.ravel())
    per_cell = sizes[1:][sizes[1:] > 0]
    cz_um3 = float(s.cz_xy_um) ** 2 * float(s.cz_z_um)
    print(f"  cells_labeled: {len(per_cell)} / {len(ids)}")
    if len(per_cell):
        print(f"  vol_px: median={np.median(per_cell):.0f}  "
              f"p10={np.percentile(per_cell, 10):.0f}  "
              f"p90={np.percentile(per_cell, 90):.0f}")
        print(f"  vol_um3: median={np.median(per_cell)*cz_um3:.0f}  "
              f"p10={np.percentile(per_cell, 10)*cz_um3:.0f}  "
              f"p90={np.percentile(per_cell, 90)*cz_um3:.0f}")

    # save
    out = Path(__file__).resolve().parent / f"cz_labels_vor_{subj}_R{int(R_um)}.tif"
    tifffile.imwrite(str(out), labels.astype(np.uint32), compression="zlib")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
