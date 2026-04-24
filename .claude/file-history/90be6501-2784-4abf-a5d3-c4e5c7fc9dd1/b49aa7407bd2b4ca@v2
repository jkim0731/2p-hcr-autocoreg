# S18 — I2 SimpleITK MI affine

## Goal
End-to-end coarse affine from intensity volumes using SimpleITK's
Mattes MI metric on a multi-resolution pyramid.

## API
`bench/candidate_impls/_i2_sitk_affine.py::run_i2(s, target_um=4.0)`

## Method
1. Resample CZ z-stack and HCR-488 (level 3) to 4 µm isotropic via F3.
2. Initialise transform with 180° XY rotation, centroid translation,
   anisotropic scale `(sxy, sxy, sz)` = `(1.8, 1.8, 2.8)`.
3. Call `lib/sitk_wrapper.mi_affine`.
4. Emit a `TransformDescriptor` with the recovered `(R, S, t)`.

## Benchmark (788406)
- runtime 1.5 s, n_pred = 0 — I2 crashed or short-circuited (check JSON
  diagnostics).  Most likely cause: SimpleITK dependency missing in
  the conda env (`import SimpleITK` fails → graceful-degradation
  branch emits empty DataFrame).

## Follow-up
Install SimpleITK (`pip install SimpleITK`) and re-run; alternatively,
the wrapper could fall back to `scipy.ndimage.affine_transform` + MI
metric implemented ad-hoc.

## Files
- `bench/candidate_impls/_i2_sitk_affine.py`
