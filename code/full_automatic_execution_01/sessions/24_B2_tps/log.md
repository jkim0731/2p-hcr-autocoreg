# S24 — B2 TPS-expansion loop

## Goal
Grow the landmark set starting from B1's seed using per-axis
`Rbf(thin_plate)` warp + k-NN neighbour matching with an intrinsic
residual-ratio gate.

## API
`bench/candidate_impls/_b1_b2_seed_tps.py::run_b2(s)` (registered "B2")

## Method
1. Fit per-axis `Rbf(thin_plate)` on B1 seed landmarks.
2. For each unmatched CZ cell, warp to HCR and take top-5 GFP+
   neighbours by density.
3. Intrinsic gate: reject if TPS residual > 3σ of local landmark
   residuals OR if 1-NN-graph topology disagrees.
4. Accept → append, refit TPS, iterate.

## Benchmark (788406)
- n_pred = 5 (no expansion beyond the seed), recall = 0.000.
- Runtime 6.9 s.

## Observations
Because B1's seed was not on a GT pair, B2's TPS residual distribution
is noise-dominated and the 3σ gate rejects everything.  B2 is
fundamentally downstream of B1 quality — it cannot bootstrap from a
bad seed.  Needs either a better B1 seed ranking, or M-series /
image-based coarse as a seed-free alternative.

## Files
- `bench/candidate_impls/_b1_b2_seed_tps.py`
