# S06 — F5 SimpleITK MI affine + B-spline wrapper

## Goal
Thin wrapper around SimpleITK's Mattes MI metric for anisotropic
voxel-spacing 3D arrays.  Exports `mi_affine` and `mi_bspline`.

## API
`lib/sitk_wrapper.py`:
- `mi_affine(cz_arr, hcr_arr, cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um,
    init_rotation_deg_z=180.0, init_scale=(1.8, 1.8, 2.8),
    n_iterations=200, pyramid_levels=(4, 2, 1)) → MIFitResult`
- `mi_bspline(...)` for a B-spline deformable stage on top of the affine.

## Method
- Wrap each ndarray as a `sitk.Image` with voxel spacing
  `(xy_um, xy_um, z_um)` (SimpleITK expects `(x, y, z)`).
- Permute `(z, y, x) → (x, y, z)` on entry / exit.
- Use `ImageRegistrationMethod.SetMetricAsMattesMutualInformation`
  with regular-step gradient-descent optimizer; multi-resolution pyramid.
- For B-spline: initialise a transform grid at 30 µm XY × 60 µm Z,
  compose on top of the affine.

## Self-test
Non-trivial: SimpleITK is noise-sensitive on binary masks.  A toy test
on paired 3D Gaussians with a known 180° + 2× scale converges to the
known transform within 5 µm and 1° after 2 pyramid levels.  Pass.

## Notes / gotchas
- SimpleITK CompositeTransform does not expose `GetMatrix()` directly; we
  extract the nested AffineTransform via `GetNthTransform(0)` and rewrap.
- The `MIFitResult` returns `metric` (raw Mattes MI, lower is "better"
  under SITK's sign convention → we return `-metric` as a pseudo-score).

## Files
- `lib/sitk_wrapper.py`
