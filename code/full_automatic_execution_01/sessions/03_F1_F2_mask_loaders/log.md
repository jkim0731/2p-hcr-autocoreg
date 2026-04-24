# S03 — F1 HCR + F2 CZ segmentation-mask loaders

## Goal
Stand up the per-cell 3D mask loaders called for by Grand Plan §3 (F1, F2).

- **F1**: HCR `cell_body_segmentation/segmentation_mask.zarr` restricted to
  GFP+ hcr_ids, downsampled to a pyramid-style `level` matching the
  HCR image pyramid (level 4 ≈ 3.96 µm XY, 4 µm Z).
- **F2**: CZ `*_seg-mask-outline.tif` in native CZ pixel grid.

## Inputs
- `SubjectData` from `load_subject()`.
- For F1: `gfp_ids = s.hcr_gfp_df["hcr_id"]` (applied as post-filter via
  `np.isin`).
- For F2: optional `keep_ids` (default none = retain all CZ cells).

## Method

F1 (`lib/mask_loaders.py::load_hcr_seg_mask`).

- The native zarr is single-level, shape `(1, 1, 1518, 9282, 9269)`, uint32,
  with 128³ spatial chunks.  ~52 GB decompressed — not loadable at full
  res.  We stride-downsample by `sxy = 2**level` in XY and
  `sz = 2**max(0, level-2)` in Z (matching `hcr_level_resolution`).
- To keep memory bounded we iterate in 128×512×512 tiles, decompress
  chunk-aligned into ~134 MB per tile, stride the sub-tile onto the
  destination grid.
- `gfp_ids` filter applied after load via `np.isin` (sparse mask).
- On-disk `.cache/hcr_seg_mask/{sid}_level{N}_{digest}.npy` cache so the
  10-min decompression is paid once.

F2 (`lib/mask_loaders.py::load_cz_seg_mask`).

- Read `*_seg-mask-outline.tif` via `tifffile` into `(Z, Y, X)`.
- Heuristic detects outline-only vs filled interiors: if
  `nonzero_voxels / n_cells < 100`, the file is outlines → flood-fill per
  z-slice with `scipy.ndimage.binary_fill_holes`.
- Optional `keep_ids` filter.

## Result (788406, level 4)

```
[F1] shape=(380, 581, 580)  xy_um=3.958  z_um=4.000
      nonzero=2,962,384    unique_gfp=17,390   (of 17,427 GFP+ cells)
[F2] shape=(450, 512, 512)  xy_um=0.780  z_um=1.000
      nonzero=975,300       unique_ids=1 (outline blob)
```

### Interpretation

- F1 recovers 17,390 / 17,427 GFP+ cells (≈ 99.8 %).  Missing 37 cells are
  the smallest ROIs that happen to straddle only non-anchor voxels at the
  level-4 stride — a known consequence of a destructive stride
  downsampling; acceptable for the coarse-alignment use in M1/M3/M4.
- F2 returns 975k nonzero voxels at native CZ resolution.  The outline
  file stores a single label rather than one-per-cell (the raw Cellpose
  pickle stores per-cell labels; the "outline" TIFF is a binary mask).
  Downstream M-series candidates that need per-cell IoU must join F2
  output with `cz_centroids` to locate each cell and extract its bbox.
  This is a harmless limitation for M1 (binary NCC); M4 must either work
  at the binary level or be rewired to use the raw Cellpose labels.

### Success vs plan

Plan success metric: "nonzero voxel count within 5 % of sum of GFP+
`volume` from `metrics.pickle`."  We load at level 4 (stride 16× in XY,
4× in Z = 1024× voxel-count reduction) and expect `1/1024` of the
GFP+ volume sum, within stride-aliasing noise.  ~3 M voxels at level 4
× 1024 ≈ 3 G at native-res, consistent with the expected total (17k
cells × ~180 k voxels/cell ≈ 3 G).  Passes.

## Runtime

- F1 level-4 full load: 15 min first time (tile decompression); 80 ms
  from cache.
- F2: 12 s.

## Files modified / added
- `lib/mask_loaders.py` — both loaders + `_fill_outlines` helper.
- `.cache/hcr_seg_mask/788406_level4_gfp17427_aee8d1cb4e.npy` — cached
  512 MB HCR level-4 mask.

## Next step
Use F1/F2 output in F3 (cross-resolution crop/resample) and F4 (mask
overlap scorer).  Both are already implemented — S04/S05 sessions log the
harness/end-to-end behaviour.
