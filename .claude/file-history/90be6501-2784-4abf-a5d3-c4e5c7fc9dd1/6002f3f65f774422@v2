# S08 — F8 synthetic-warp sampler for self-supervised training

## Goal
Generate paired `(source, warped, correspondence)` samples from an HCR
centroid cloud, used by G-series and P8-P11 learned methods.

## API
`lib/synthetic_warps.py::sample_warped_pair(points_um, rng,
  cube_um=400.0, xy_scale_range=(1.5, 2.0), z_scale_range=(2.0, 3.5),
  rot_jitter_xy_deg=10.0, rot_jitter_z_deg=5.0, tps_n_cp=8,
  tps_jitter_um=25.0, drop_rate=0.2, always_180=True) → WarpSample`

## Method
1. Sample a 400 µm cube centred at a random valid position inside the
   HCR cloud (probability weighted by local density).
2. Apply: 180° XY rotation (structural prior), ± `rot_jitter_xy_deg`
   XY jitter, ± `rot_jitter_z_deg` Z jitter, anisotropic scale
   per axis from the stated ranges, per-axis TPS with `tps_n_cp` random
   control points jittered ± `tps_jitter_um`.
3. Drop `drop_rate` fraction of cells to simulate segmentation loss.
4. Return `(source_cube, warped_cube, index_correspondence,
   transform_metadata)`.

## Binding-rule compliance
The ranges above are used only as sampling bounds — no model parameter
is tuned against benchmark metrics.  The ranges are intentionally wider
than the benchmark-observed distribution so the learned models see more
than the specific test subjects.

## Files
- `lib/synthetic_warps.py`
