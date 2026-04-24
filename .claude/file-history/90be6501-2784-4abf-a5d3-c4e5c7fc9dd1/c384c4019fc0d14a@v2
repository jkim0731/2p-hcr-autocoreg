# S11 — P4 Spectral graph matching (Leordeanu–Hebert)

## Goal
Pairwise-consistent assignment via the principal eigenvector of an
affinity matrix over putative pairs.

## API
`bench/candidate_impls/_p4_spectral.py::run_p4(s)`

## Method
1. Build top-K = 5 putative pairs per CZ cell using F6 feature cosine +
   spatial distance (after 180° warm-start).
2. Affinity matrix M: diagonal = per-pair feature agreement; off-diag =
   geometric pairwise-distance compatibility (robust-median per-axis
   scale estimated from putatives).
3. Power-iterate to the dominant eigenvector; Sinkhorn normalisation
   for one-to-one extraction.

## Benchmark (788406)
- n_pred = 786, confidence 0.87, recall = 0.000, landmark origin err
  1734 µm, rotation 180° (correct rotation prior).
- Runtime 125 s.

## Interpretation
Spectral GM happily produces one-to-one assignments covering almost all
CZ cells, but again at the wrong global pose (origin off by 1.7 mm).
The pairwise-distance affinity is invariant to translation and to
anisotropic scale through the robust-median normalisation, so it can
find an internally consistent matching that is not the GT matching.

Addressing the coarse-scale failure is the main job of M-series
warm-starts.

## Files
- `bench/candidate_impls/_p4_spectral.py`
