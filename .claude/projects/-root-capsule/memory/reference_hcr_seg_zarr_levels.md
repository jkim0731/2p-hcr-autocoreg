---
name: HCR cell_body_segmentation zarr is LEVEL-0; orig_res is LEVEL-2
description: The array named `segmentation_mask.zarr` is at level-0 (0.247 µm/vox xy); `segmentation_mask_orig_res.zarr` is at level-2 (0.988 µm/vox xy). Misleading naming; verified 2026-04-23.
type: reference
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
Inside every `HCR_{sid}_*/cell_body_segmentation/`:

| file | xy resolution | shape (788406) | matches |
| ---- | ------------- | -------------- | ------- |
| `segmentation_mask.zarr`          `"0"` | **level-0, 0.2474 µm/vox** | (1,1,1518,9282,9269) | stitched level-0 tile grid (5×5 × 1910 with ~3 % overlap) |
| `segmentation_mask_orig_res.zarr` `"0"` | **level-2, 0.988 µm/vox**  | (1,1,1518,2320,2317) | exactly 4× downsampled in xy from seg_mask |
| `cell_centroids.npy`                    | level-2 pixel indices       | max (1338, 2315, 2309)       | matches orig_res shape |
| `metrics.pickle` `global_bbox`          | **level-0 voxel indices**   | values up to ~9000 in xy     | matches seg_mask (the big one) |

`benchmark_data_loader._read_hcr_resolution` returns the LEVEL-2
scale (`dims["x"][0] * 4e6`, so ~0.988 µm); that matches
`hcr_centroids` but NOT `segmentation_mask.zarr`. The name
"orig_res" is misleading — it is NOT the original resolution; the
un-suffixed `segmentation_mask.zarr` is.

Rule of thumb when reading from `cell_body_segmentation/`:
 - coords from `cell_centroids.npy` or the top-level centroid
   CSVs → use `s.hcr_xy_um` (level-2) as-is.
 - coords from `metrics.pickle` or voxels indexed inside
   `segmentation_mask.zarr` → divide by 4 to reach the level-2
   frame, or use `s.hcr_xy_um / 4` as level-0 µm/vox.

Proof: `fused_ng.json` declares level-0 as 2.4736e-7 m (0.2474 µm);
per-tile `image_radial_correction/Tile_*_ch_488.ome.zarr` level-0
shape is (Z, 1910, 1910) and confirms stitched layout.
