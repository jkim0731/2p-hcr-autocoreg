# S23 — B1 seed-constellation matcher

## Goal
Produce the first 4–6 CZ↔HCR landmarks automatically using an
anisotropy-invariant cluster descriptor (pairwise-distance histogram
+ inner-angle + hull volume).

## API
`bench/candidate_impls/_b1_b2_seed_tps.py::run_b1(s)` (registered "B1")

## Method
1. Enumerate candidate 4–6 CZ clusters ranked by convex-hull
   emptiness, tight intra-cluster spacing, near-surface proximity.
2. Compute axis-normalised pairwise-distance histograms per cluster.
3. kd-tree search matching descriptors in HCR-GFP+.
4. Fit anisotropic affine + TPS on top-K candidates, score by residual
   ratio (seed vs. next-best) + descriptor-distance margin.

## Benchmark (788406)
- n_pred = 5, confidence 0.898, recall = 0.000.
- Runtime 53.8 s.

## Observations
Seed constellations do emit 5 pairs with high intrinsic confidence,
but they are not in the GT set.  The ones that score highest on
descriptor match happen to lie outside the GFP+ region the reviewer
saw, pointing at an issue with the ranking criterion — "convex-hull
emptiness" may be biased toward CZ's boundary cells.

## Files
- `bench/candidate_impls/_b1_b2_seed_tps.py`
