# cached_hcr_cell_tight_bbox

Per-cell **tight** bounding boxes for HCR cell-body-segmentation labels.

## Why this cache exists

`cell_body_segmentation/metrics.pickle` stores a per-cell
`global_bbox`, but it is the bbox of the segmentation **tile/chunk**
that contains the cell — not the tight bbox of the cell itself.
Median occupancy (cell voxels / bbox voxels) is ~0.016 for
788406. Cross-section / diameter estimates computed from the
pickle bbox are therefore wrong by 10× or more.

The tight bbox has to be recovered from the raw
`segmentation_mask.zarr` volume by reading a crop and locating
the label voxels. This is the canonical place to store those
tight-bbox measurements so downstream sessions don't repeat
the I/O.

## Files

```
{sid}_hcr_cell_tight_bbox_v1.parquet   per-subject tight-bbox table
README.md                              this file
```

Subjects present depend on which ones have been populated
(any session can call `07e_sxy_from_roi_area.compute_tight_hcr_bboxes`
to extend the cache).

## Schema (v1)

| column       | dtype | units         | description                                  |
| ------------ | ----- | ------------- | -------------------------------------------- |
| `hcr_id`     | int64 | —             | HCR cellpose label id                         |
| `zmin_vox`   | int64 | zarr voxels   | tight bbox: first z-slice (inclusive)         |
| `zmax_vox`   | int64 | zarr voxels   | tight bbox: last z-slice + 1 (exclusive)      |
| `ymin_vox`   | int64 | zarr voxels   | tight bbox y-start (inclusive)                |
| `ymax_vox`   | int64 | zarr voxels   | tight bbox y-end (exclusive)                  |
| `xmin_vox`   | int64 | zarr voxels   | tight bbox x-start (inclusive)                |
| `xmax_vox`   | int64 | zarr voxels   | tight bbox x-end (exclusive)                  |
| `volume_vox` | int64 | voxels        | actual voxel count (matches pickle `volume`)  |
| `zc_vox`     | float | zarr voxels   | voxel-weighted centroid (z)                   |
| `yc_vox`     | float | zarr voxels   | voxel-weighted centroid (y)                   |
| `xc_vox`     | float | zarr voxels   | voxel-weighted centroid (x)                   |

Slice convention is NumPy-style (`[lo:hi]` half-open).
HCR zarr voxel size varies by subject (see
`benchmark_data_loader.load_subject(sid)` →
`s.hcr_xy_um`, `s.hcr_z_um`). As of 2026-04, all spot subjects
use xy=0.9865 µm, z=1.0 µm.

## Source

Generated from
`/root/capsule/data/HCR_{sid}_*/cell_body_segmentation/segmentation_mask.zarr`
(main array `"0"`, shape `(1, 1, Z, Y, X)`, uint32 / uint16).
For each hcr_id we read a crop covering the pickle tile bbox,
mask by `== hcr_id`, and record the tight (zmin..xmax) and
centroid of the masked voxels.

## Producer

`dev_code/07e_sxy_from_roi_area.py :: compute_tight_hcr_bboxes`.
Subsequent sessions may import it; the cache is additive
(new hcr_ids are merged into the existing parquet).

## Subjects covered

Currently only the four spot subjects
(788406, 790322, 767018, 782149) have HCR
`cell_body_segmentation/metrics.pickle`. 755252 and 767022
(intensity subjects) don't have the pickle so this cache
doesn't apply to them yet.
