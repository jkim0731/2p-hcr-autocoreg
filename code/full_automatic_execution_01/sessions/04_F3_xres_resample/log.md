# S04 — F3 cross-resolution crop + resample

## Goal
Given an affine `(R, t, S)` mapping CZ → HCR µm, resample CZ and HCR 3D
arrays (image or mask) onto a shared isotropic grid that covers both
extents at a user-specified voxel spacing.  Downstream M- and I-series
candidates call F3 before computing NCC / MI / Dice.

## API
`lib/xres_resample.py::resample_to_shared_grid(cz_arr, hcr_arr,
  cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um, R, t_um, S, src_mean_um,
  target_spacing_um=4.0, mode='image'|'mask', margin_um=40.0) →
  ResampleResult`

## Method
1. Build the CZ physical-coordinate bbox from the input `cz_arr` shape
   and its resolution.  Apply the affine to the 8 corners; take the AABB.
2. Allocate an isotropic target grid of size `(D, H, W)` covering both
   AABBs + `margin_um`.
3. For each modality, build the 4×4 affine that maps target voxel →
   native voxel and call `scipy.ndimage.affine_transform` with `order=1`
   for image and `order=0` for mask.
4. Return `(cz_resampled, hcr_resampled, target_spacing_um, origin_um)`.

## Self-test result
Synthetic known-affine test: produce a CZ by cropping/warping the HCR,
run F3, confirm CZ cropped back matches the synthetic CZ within
interpolation error (< 2.5 % max residual).  Pass.

## Files
- `lib/xres_resample.py`

## Next step
F4 mask-overlap scorer consumes F3 output.
