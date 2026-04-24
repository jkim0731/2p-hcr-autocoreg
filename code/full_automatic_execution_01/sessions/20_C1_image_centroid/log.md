# S20 — C1 image→centroid hybrid (I2 + P1)

## Goal
Use I2's image-level MI affine as a warm-start for P1 on centroids —
the "combined" mode that stitches image-level coarse with centroid-
level refinement.

## API
`bench/candidate_impls/_c1_image_centroid.py::run_c1(s)`

## Dependency chain
I2 (MI affine, SimpleITK) → P1 (GNC-TLS) on CZ/HCR-GFP+ centroids
warm-started by I2's (R, S, t).

## Status
Blocked on I2's `'tuple' object has no attribute 'astype'` fix; post-
fix C1 should emit a candidate result.  See sessions/18 for the
underlying load fix.

## Files
- `bench/candidate_impls/_c1_image_centroid.py`
