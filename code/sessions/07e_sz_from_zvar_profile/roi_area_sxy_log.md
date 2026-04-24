# Session 07e — sxy from ROI xy-area (promoted to main pipeline)

**Status:** SUCCESS on 3/4 spot subjects, within ±7 %. 782149 fails at
−15 % because its HCR `hcr_span=566 µm` is ~half the other subjects —
a surface-span / FOV-truncation issue, not a scale bug. Intensity
subjects (755252, 767022) unsupported because their HCR package has
no `cell_body_segmentation/metrics.pickle`.

## Hypothesis

Under isotropic lateral tissue expansion `sxy`, the xy footprint of a
cell body scales by `sxy²`:

    area_HCR ≈ sxy² · area_CZ
    ⇒ sxy = sqrt( median(area_HCR) / median(area_CZ) )

CZ areas come from regionprops on `segmentation_masks.tif`; HCR
areas come from the per-cell tight xy bbox computed over
`cell_body_segmentation/segmentation_mask.zarr`, masked by the
cellpose label id (the `metrics.pickle` tile-bbox is the tile the
cell lives in, not the tight cell bbox — median occupancy ≈ 0.016).

## Method

1. Strict BIC-GMM GFP+ threshold on `log(density)` (per session 07b
   / 07c convention) → `strict_hcr_ids`.
2. Restrict to center ¼ FOV using pickle-bbox centers (cheap; avoids
   zarr I/O on cells we'd drop).
3. For each surviving id, read a zarr crop covering the pickle tile
   bbox, mask by `== hcr_id`, record tight (zmin..xmax) + voxel
   centroid. Cached per-subject to
   `dev_code/cached_hcr_cell_tight_bbox/{sid}_hcr_cell_tight_bbox_v1.parquet`.
4. Depth filter: keep cells within `[d_skin=100 µm, pia_99pct]`.
5. Scale: `sxy = sqrt(median area ratio)` (also report mean).

## Zarr level clarification (critical — this was the blocker)

`cell_body_segmentation/` contains **two** arrays whose relationship
was not documented:

| file | xy resolution | 788406 shape | used by |
| ---- | ------------- | ------------ | ------- |
| `segmentation_mask.zarr` `"0"` | **level-0, 0.2474 µm/vox** | (1,1,1518,9282,9269) | `metrics.pickle` bbox indices — and, confusingly, NOT tagged `orig_res` |
| `segmentation_mask_orig_res.zarr` `"0"` | **level-2, 0.988 µm/vox** | (1,1,1518,2320,2317) | `cell_centroids.npy` (shape matches) |

The name `orig_res` is misleading: it is NOT original resolution,
it's the 4×-downsampled xy version. The un-suffixed
`segmentation_mask.zarr` is at LEVEL-0 in xy.

Evidence for level-0 interpretation:

- `fused_ng.json` reports `dimensions.x[0] = 2.4736e-7 m = 0.2474 µm`
  (the native / level-0 value).
- Per-tile `image_radial_correction/Tile_*_ch_488.ome.zarr` level-0
  shape is `(Z, 1910, 1910)`. For 788406 the tile grid is 5×5
  (X_0000…X_0004 × Y_0000…Y_0004), so stitched level-0 xy is
  `5 × 1910 = 9550` before overlap. The observed `9282, 9269`
  matches at ~2.9 % overlap. ✓
- `segmentation_mask_orig_res.zarr` xy is exactly ¼ of `.zarr` xy
  (9282/2320 = 4.001); z is the same (1518), consistent with the
  standard "downsample xy only" pattern used here.

`benchmark_data_loader._read_hcr_resolution` returns `dims.x[0] *
4e6` ≈ 0.988 µm, i.e. the level-2 value — correct for
`hcr_centroids` / `hcr_gfp_df` (which are in level-2 pixel indices,
confirmed by `cell_centroids.npy` max ≈ (1338, 2315, 2309)), but
wrong for anything read out of `segmentation_mask.zarr`.

Before the fix, using `s.hcr_xy_um` on level-0 bbox extents
inflated each xy span 4×, area 16×, and `sqrt(16) = 4` dominated
the sxy estimate (values ~7 instead of ~1.75).

## Promotion

- **`dev_code/benchmark_data_loader.py`** now exposes:
  - `HCR_SEG_XY_DOWNSAMPLE = 4` (module constant).
  - `SubjectData.hcr_seg_xy_um = hcr_xy_um / 4` (level-0 µm/vox).
  - `SubjectData.hcr_seg_z_um = hcr_z_um`.
  - Docstring on `_read_hcr_resolution` clarifying level-2 vs level-0.
  Any future code reading `segmentation_mask.zarr` should use
  `s.hcr_seg_xy_um`, not `s.hcr_xy_um`.
- **`dev_code/roi_area_sxy.py`** (new, promoted module). Public API:
  - `compute_tight_hcr_bboxes(sid, hcr_ids)` — cache-backed per-cell
    tight HCR bboxes from `segmentation_mask.zarr`.
  - `cz_cell_tight_bboxes(sid, s, cz_surf)` — per-CZ-cell tight bboxes.
  - `hcr_cell_tight_bboxes(sid, s, hcr_surf, hcr_ids)` — per-HCR-cell
    tight bboxes, level-0 correctly applied.
  - `estimate_sxy_roi_area(sid)` — full driver with GT scoring.
- **`dev_code/07e_sxy_from_roi_area.py`** is now a thin session
  driver that imports from `roi_area_sxy` and owns only the
  per-subject figure + session JSON output.
- On-disk cache: `dev_code/cached_hcr_cell_tight_bbox/` (documented
  in its `README.md`), populated for the four spot subjects.

## Validation

```
sid      sxy_med   sxy_mean   GT      err_med   err_mean
788406    1.749     1.742    1.778    -1.6 %    -2.0 %
790322    1.785     1.757    1.763    +1.2 %    -0.4 %
767018    1.812     1.766    1.702    +6.5 %    +3.8 %
782149    1.631     1.606    1.924   -15.2 %   -16.5 %  (HCR_span=566 µm)
```

Median and mean give consistent answers (within ~3 % of each
other), so the estimator is not driven by tail outliers.

## Failure mode — 782149

`hcr_span_um ≈ 566 µm` vs ~1000 µm for the other three subjects.
The HCR cortex span used here is `p99` of the pia-surface depth on
the strict center-FOV cells; if the HCR top or bottom surface is
mis-placed, or if the tissue actually stops earlier in this subject,
the retained HCR cells are shallower than the CZ matched population
and their average xy footprint is smaller. That pulls the estimate
below GT. This is a surface / depth-filter issue, independent of
the scale conversion — out of scope for this promotion.

## Open work

1. 782149 HCR surface / span diagnosis.
2. Intensity-subject support (755252, 767022): their HCR package
   doesn't ship `cell_body_segmentation/metrics.pickle`; ROI-area
   sxy is blocked until an equivalent per-cell label store exists.
3. The estimator returns sxy only — sz is still unresolved
   (see `project_07e_zvar_profile.md`: the pia-normal σ_z profile
   family is exhausted across 06/07d/07e).

## Files touched

```
dev_code/benchmark_data_loader.py      # HCR_SEG_XY_DOWNSAMPLE + SubjectData
dev_code/roi_area_sxy.py               # NEW — promoted module
dev_code/07e_sxy_from_roi_area.py      # thin driver on top of roi_area_sxy
dev_code/cached_hcr_cell_tight_bbox/   # v1 parquet cache + README
sessions/07e_sz_from_zvar_profile/
    roi_area_sxy_log.md                # THIS file
    roi_area_sxy.json                  # per-subject numeric results
    figures_roi_area/roi_area_{sid}.png
```
