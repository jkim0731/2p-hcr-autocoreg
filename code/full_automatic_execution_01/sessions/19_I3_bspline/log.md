# S19 — I3 MI + B-spline deformable

## Goal
Extend I2's affine with a B-spline deformable stage to absorb the
nonrigid residual (~16–43 µm per benchmark priors).

## API
`bench/candidate_impls/_i3_bspline.py::run_i3(s)`

## Method
1. Run I2 to get an affine initialiser.
2. Call `lib/sitk_wrapper.mi_bspline` with 30 µm XY × 60 µm Z control
   grid.
3. Compose affine + B-spline; re-score.

## Benchmark (788406)
- runtime 1.5 s, n_pred = 0.  I3 inherits I2's failure mode (same
  SimpleITK dependency gap).

## Files
- `bench/candidate_impls/_i3_bspline.py`
