# S13 — P14 Hungarian assignment baseline

## Goal
The simplest possible baseline — `scipy.optimize.linear_sum_assignment`
on an NxM affinity matrix combining F6 feature cosine similarity with
spatial distance after 180° + CZ-centroid-at-HCR-centroid.

## API
`bench/candidate_impls/_p14_hungarian.py::run_p14(s)`

## Benchmark (788406)
- n_pred = 723, recall = 0.017, precision = 0.018,
  recall@10µm = 0.018, recall@20µm = 0.020.
- Median error 191 µm, p95 error 348 µm, runtime 55 s.
- Landmark origin error 795.9 µm (best of all P-series so far).

## Observations
- P14 is the *only* P-series candidate with non-zero recall, despite
  being the simplest.  Why?  Because it consumes the 180°+centroid
  prior directly — no attempt at scale recovery — and the F6 cosine
  similarity + local 1-NN matching recovers the few CZ cells that happen
  to fall inside the true HCR ROI after that coarse placement.
- P1/P3/P4 produce *self-consistent* affines that are ~1.5 mm off the
  GT; they find the wrong global minimum.  P14 does not search for an
  affine — it just pairs cells — which is more robust when the coarse
  scale is missing.

## Binding-rule compliance
The only benchmark-derived number used is the 180° XY rotation prior,
which is a structural imaging-geometry constant, not a population
statistic.

## Files
- `bench/candidate_impls/_p14_hungarian.py`
