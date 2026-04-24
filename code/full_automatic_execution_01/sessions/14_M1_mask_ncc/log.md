# S14 — M1 volumetric mask NCC (coarse affine)

## Goal
Locate CZ inside HCR by 3D NCC of GFP+ mask density volumes at
~20 µm isotropic spacing, sweeping anisotropic scale `(sxy, sz)` on a
coarse grid.  Provides the coarse-affine warm-start that purely-centroid
candidates lack.

## API
`bench/candidate_impls/_m1_mask_ncc.py::run_m1(s, level=4,
target_um=20.0, sxy_grid=(1.5, 1.75, 2.0), sz_grid=(2.0, 2.5, 3.0))`

## Method
1. Load HCR GFP+ seg mask (F1) at level 4; load CZ seg mask (F2); both
   binarised.
2. Downsample both to 20 µm isotropic via scipy `ndi.zoom(order=0)`.
3. Apply 180° XY rotation to CZ template; sweep `(sxy, sz)` over the
   coarse grid; for each, call `_ncc3d` and record the peak.
4. Retain best `(sxy, sz, dz, dy, dx)`; refine around it at 5 µm.
5. Confidence = `(peak − median) / MAD`, plus sanity gate on the "CZ
   ≈ HCR XY center" prior.  Emit `unknown` affine if robust-z is low.

## Benchmark (788406) — pending in the current sweep
NCC over 3×3 = 9 scale × 3D translation is the expensive loop; the
first-run load of F1 is cached so subsequent runs reuse it.

## Binding-rule note
The scale grid `sxy ∈ {1.5, 1.75, 2.0}`, `sz ∈ {2.0, 2.5, 3.0}` comes
from R1's coverage regime / expected-anisotropy bounds, not from
benchmark-subject fits.  It is a sampling range, not a tuned value.

## Files
- `bench/candidate_impls/_m1_mask_ncc.py`
