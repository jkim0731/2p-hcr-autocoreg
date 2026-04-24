# S15 — M3 mask+centroid hybrid (M1 warm-start → P1)

## Goal
Use M1's coarse (R, S, t) as a warm-start for P1 (TEASER-style
GNC-TLS).  The hypothesis is that a mask-level coarse aligns the
global origin and scale well enough that P1 can refine centroids
to a correct one-to-one correspondence.

## API
`bench/candidate_impls/_m3_mask_centroid.py::run_m3(s)`

## Benchmark (788406)
- n_pred = 229, recall = 0.000, runtime 55.1 s.
- M3 inherits M1's coarse-scale output (peak NCC 0.046 — weak); P1's
  GNC-TLS re-fit on the M1-warped centroids produces the same 229
  pairs P1 alone produces (same warm-start).

## Interpretation
M1's best-grid NCC peak of 0.046 is too low (expected ≥ 0.3 for a
real match) — the 3D NCC denominator approximation is under-
normalising, or the 20 µm isotropic downsample is too coarse, or the
binary-mask representation loses too much structure at that scale.
Investigating via the sanity priors: M1's best (sxy=1.5, sz=2.0) is at
the LOW end of the benchmark-observed expansion ranges, which
suggests the sweep grid was not wide enough in some direction.

## Follow-up
Widen the scale grid to `sxy ∈ {1.3, 1.5, 1.75, 2.0, 2.25}` and
`sz ∈ {1.8, 2.2, 2.6, 3.0, 3.4}`.  Also consider soft-label mask
(distance-transform smoothed) to avoid binary aliasing.

## Files
- `bench/candidate_impls/_m3_mask_centroid.py`
