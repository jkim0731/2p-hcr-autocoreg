# S12 — P5 Fused / Partial Gromov-Wasserstein

## Goal
Optimal transport that is invariant to isometric deformation of the
internal pairwise distance matrices (via the GW side) while using F6
features for the Wasserstein side.

## API
`bench/candidate_impls/_p5_fgw.py::run_p5(s, alpha=0.5, eps=0.05,
n_hcr_max=3000)` — uses POT (`ot`).

## Method
1. Warm-start 180° + CZ-centroid-at-HCR-centroid; AABB crop of HCR to
   3× CZ extent.  Random-subsample HCR to ≤ 3000 (O(n²m²) memory).
2. Build intra-distance matrices `C_cz`, `C_hcr` normalised by
   median pairwise distance (so anisotropic expansion is absorbed).
3. F6 invariant-feature z-score difference for the `M` Wasserstein cost.
4. `ot.gromov.entropic_fused_gromov_wasserstein(M, C_cz, C_hcr, a, b,
   alpha=0.5, epsilon=0.05)`.
5. argmax-per-row of transport plan T; per-pair confidence = T-mass
   relative to uniform.

## Benchmark (788406)
- n_pred = 787, recall = 0.000, runtime 148 s.
- The POT call converges, all CZ cells get a transport-plan argmax —
  they are just not GT matches.

## Why recall is 0
Same coarse-scale failure as P1/P4.  Even with median-normalised
distance matrices, FGW's coupling mass is dominated by within-cloud
structure near CZ's centroid — an arbitrary position in HCR that
happens to maximise transport economy given the warm-start.

The FGW *assignments* are self-consistent, but the warm-start is too
far from the true ROI for the transport plan to concentrate on the
right sub-region.  Needs M-warm-start (see sessions/15).

## Files
- `bench/candidate_impls/_p5_fgw.py`
