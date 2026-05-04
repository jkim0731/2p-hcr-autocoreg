---
name: HCR tile stitching boundaries
description: BigStitcher tile-grid metadata per HCR subject — XML location, tile size 1920², ~5-6% xy overlap, no z-tile.
type: reference
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
HCR data are acquired as a 2-D xy mosaic of `Tile_X_xxxx_Y_yyyy_Z_0000_ch_NNN` 1920×1920×Z₀ tiles (single z-tile per subject, Z₀ varies 594–1641 vox/µm). xy overlap ≈ 5–6 % (96–122 lvl0 vox = ~24–30 lvl2 vox = ~24–30 µm).

**XML locations (depend on processing pipeline version):**
- Newer subjects (782149/788406/790322): `image_tile_alignment/combined_stitching_cam_alignment_all_channels.xml` — has both refined "Stitching Transform" and "Translation to Nominal Grid".
- Older subjects (754803/755252/767018/767022): `image_tile_fusing/metadata/stitching/stitching_single_channel_updated.xml` — same schema.

**Composing the per-tile origin in fused-image voxels (lvl0):**
1. Sum translations: `final = "Translation to Nominal Grid" + "Stitching Transform"` (both are 3-element T's; XML order is X,Y,Z).
2. Subtract `image_tile_fusing/fused/channel_405.zarr/.zattrs → Bigstitcher-Spark.Boundingbox_min` (X,Y,Z).
3. Divide xy by 4 to get lvl2 (centroid-frame) coords; z stays.

**Tile grids (n_tiles, x×y):** 754803 5×5 (25), 755252 8×8 (59 active), 767018 5×5 (25), 767022 8×10 (74), 782149/788406/790322 5×5 (25).

**Pipeline outputs (refresh by re-running `code/sessions/v3_S11_roi_quality/06_log_tile_boundaries.py`):**
- `code/sessions/v3_S11_roi_quality/outputs/tile_boundaries.json` — full per-tile + per-overlap-zone in lvl0 and lvl2.
- `code/sessions/v3_S11_roi_quality/outputs/tile_boundaries.csv` — flat per-tile origin/end + stitch-refine.

Cross-check: 788406 has 8.8 % of HCR cells with centroid in an x-or-y overlap zone (4.2 % x, 4.9 % y), matching the ~5 % overlap fraction. Useful for ROI-quality QC (cells in seams may have stitching artefacts).
