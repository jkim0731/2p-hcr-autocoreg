# S50 — True seg-mask NCC coarse alignment probe (abandoned)

**Status:** abandoned (negative result). Seg-mask-NCC fails on both 788406 and 782149 because CZ voronoi label density (~4.4 % of volume) is an order of magnitude denser than HCR GFP+ seg-mask (~0.2 % of volume); spatial profiles don't match so the NCC peak lands at wrong (sxy, sz).

## Motivation

S49 established 782149 as unreachable by any centroid-based coarse alignment (I2 residual 546 µm, 28× the 1-NN spacing). The Grand Plan lists M1 (volumetric mask NCC) as tier-1; prior M1 benchmarks used density-map-NCC (S31-abandoned). This probe tested whether **true** segmentation masks — F1 HCR seg-mask (GFP+ restricted) + S48 `cz_voronoi_labels` — would fare better.

## Approach

`probe.py` resamples both masks to a shared 20 µm isotropic grid, Gaussian-smooths at σ=40 µm, applies the 180° XY prior, sweeps `(sxy, sz)` in `{1.4, 1.6, 1.8, 2.0, 2.2} × {1.8, 2.2, 2.6, 3.0, 3.4}`, and picks the combo with the highest NCC z-score against the HCR grid via FFT-NCC (reuses `_m1_mask_ncc::_ncc3d_valid`).

## Results

| subject | best sxy | best sz | ncc peak | z-score | ORIGIN_ERR | GT sxy | GT sz |
|---------|---------:|--------:|---------:|--------:|-----------:|-------:|------:|
| 788406  | 2.0      | 1.8     | 0.528    | 2.9     | 627 µm     | 1.797  | 2.821 |
| 782149  | 1.6      | 1.8     | 0.261    | 3.0     | 1171 µm    | 1.965  | 2.907 |

Both subjects pick `sz=1.8` (grid lower bound), severely under-estimating the ~2.8× Z expansion. The CZ voronoi volume (bright dense ball of ~3 M nonzero voxels at 0.78 µm isotropic XY / 1 µm Z) is spatially nearly-homogeneous inside its bbox after Gaussian smoothing; the HCR GFP+ mask is a sparse scatter of isolated cell footprints (~0.5 M – 3 M voxels at ~4 µm iso). The two density profiles are spatially mismatched → NCC peak is dominated by bbox-overlap area rather than internal structure.

## Decision

Do not pursue seg-mask-NCC coarse alignment. The density mismatch is structural, not a parameter choice. M1 ships as superseded (per S44); no new M1 variant to register. 782149 remains deferred to post-F8/G1 learned methods. Pivoted to S51 (P4 + HCR image-quality β bonus — orthogonal lift to P1).

## Files

- `probe.py` — F1 HCR mask + S48 CZ voronoi → 20 µm isotropic grid → FFT-NCC over (sxy, sz) grid.
- `probe.log` — raw results.
