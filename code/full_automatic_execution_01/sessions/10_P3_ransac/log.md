# S10 — P3 RANSAC + anisotropic affine baseline

## Goal
Classical RANSAC floor for P1 — samples 4 random putative correspondences,
fits `fit_anisotropic_similarity`, scores by inlier count within a µm
residual, retains the largest consensus set.

## API
`bench/candidate_impls/_p3_ransac.py::run_p3(s)` — registered under `"P3"`.

## Benchmark (788406)
- n_pred = 73, recall = 0.0013, precision = 0.014, runtime 158.9 s.
- Origin error 1304 µm — same coarse-scale failure mode as P1.

## Why recall is still near 0
RANSAC consensus sets are structurally consistent but locked to the
wrong coarse pose.  With the 180° prior only, RANSAC finds a correct-
looking affine that happens to be ~1.3 mm off the true HCR origin.
The top consensus set is a self-consistent but global-minimum-miss.

Addressed downstream by M3 (M1 warm-start → P1/P3).

## Files
- `bench/candidate_impls/_p3_ransac.py`
