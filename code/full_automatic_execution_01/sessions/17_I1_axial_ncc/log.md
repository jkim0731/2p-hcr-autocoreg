# S17 — I1 cortical-layer axial NCC (Z scale + offset)

## Goal
Recover Z scale + offset from the 1D mean-intensity-vs-depth-from-pia
profile in CZ vs. HCR-488.  Modernises R1's partial-overlap NCC using
images instead of centroid density.

## API
`bench/candidate_impls/_i1_axial_ncc.py::run_i1(s)`

## Method
1. Compute mean intensity vs. depth (from pia) for CZ z-stack and HCR
   488 at level 3.  Mask out lateral 10 % margins to reduce edge-AF
   bias.
2. 1D NCC of the CZ template over the HCR profile with scale grid
   `sz ∈ {2.0, 2.25, 2.5, 2.75, 3.0}` and translation grid.
3. Partial-overlap floor from R1 (`r1_revised._partial_ncc_1d`).
4. Graceful-degradation: emit `unknown` if robust-z < 3.

## Benchmark (788406)
- n_pred = 0 (I1 emits a Z affine, not point predictions), runtime 45 s.
- Output: recovered (sz, tz) stored in diagnostics; needs inspection.
- Z scale error vs GT: see bench JSON for details.

## Files
- `bench/candidate_impls/_i1_axial_ncc.py`
