# S09 — P1 TEASER++-style GNC-TLS on GFP+ centroids

## Goal
Certifiable outlier-robust anisotropic affine matcher between CZ centroids
and HCR-GFP+ centroids.  Since the reference `teaserpp_python` binding was
not available in this environment, we re-implemented the
**GNC-TLS (graduated non-convexity with truncated least squares)** loop
on top of `fit_anisotropic_similarity`.

## API
`bench/candidate_impls/_p1_teaser.py::run_p1(s)` — registered under `"P1"`.

The candidate:
1. Builds CZ and HCR-GFP+ centroids via `lib/centroid_helpers.centroids_um`.
2. Applies the 180° XY rotation prior (structural).
3. Generates top-K putative HCR neighbours for each CZ cell using F6
   features + Euclidean distance in the common space.
4. Runs GNC-TLS: repeated anisotropic Procrustes re-fits with TLS weights
   annealed over ~10 iterations.
5. Re-scores with TPS residual thresholding.

## Convention fix
`ProcrustesFit` does not carry `src_mean` — the `translation` field already
absorbs the source-mean offset (canonical formula
`dst = (src * S) @ R.T + t`).  Earlier versions of the candidate called
`(A - fit.src_mean) @ ...` which crashed with `AttributeError` and blew the
run away.  We introduced `lib/centroid_helpers.apply_aniso_fit(pts, fit)`
and rewrote every candidate file to call it.

## Benchmark (788406)
- n_pred = 229, recall = 0.000, median error = 220 µm, origin error = 1466 µm.
- Runtime 6.9 s (GNC-TLS converges quickly with the 180° prior).

## Why recall is still 0
The 180° + CZ-centroid-at-HCR-centroid warm-start lands ~1.5 mm away
from the true HCR ROI — because we do not yet recover the
anisotropic **scale** in this warm-start.  GNC-TLS latches onto a local
optimum that is geometrically self-consistent but wrong globally.
Fixing this requires either (a) a mask- or image-level coarse pre-alignment
(M1/I2) or (b) an explicit axis-scale search inside the GNC-TLS warm-start.

## Files
- `bench/candidate_impls/_p1_teaser.py`
- `lib/centroid_helpers.py::apply_aniso_fit` (shared)

## Next step
See `sessions/15_M3_mask_centroid/` for the M1-warm-start hybrid that
should address the coarse-scale failure.
