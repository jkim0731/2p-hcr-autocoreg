---
name: 2-D surface registration v2 PROMOTED
description: CZ↔HCR top-slab 2-D registration shipped as code/dev_code/surface_registration_v2.py; cache-aware accessor + best-of-{rigid, affine, PWR 3×3, PWR 4×4} on raw 488 MIP target.
type: project
originSessionId: 047cc593-52e1-4c25-b60d-e60303d78a4c
---
`code/dev_code/surface_registration_v2.py` was promoted from session
`08_surface_vascular_match/` on 2026-04-26 as the main-pipeline 2-D
top-slab registration stage between CZ pia-flat ROIs and HCR top
surface.

**Why:** 2-D image-level surface registration is a pre-cell stage
that produces a tight shared cropped frame and a per-pixel
displacement field — downstream candidate matchers (P/M/I/C/G series)
can consume `reg["crop_bbox"]` to constrain search to the same
field-of-view. Manual landmarks were the previous baseline; the v2
pipeline is the automated branch (sketched in
`docs/04 Current protocol.md`).

**How to apply:**
- Entry point is `get_surface_registration(s)` (JSON cache at
  `dev_code/cached_surface_registration/<sid>.json`); use
  `apply_registration(s, reg)` to re-render the warp on demand
  (warps are NOT cached — only parameters).
- Default scoring metric: Pearson NCC of warped CZ binary vs **raw
  488 MIP** at HCR level-4 grid, computed on the M1-bbox + 15 %
  margin crop. Stage-1 rigid bootstrap uses the watershed-binarised
  488 MIP only because the rigid θ + tx/ty sweep is unstable when
  initial overlap is poor.
- PWR is seeded from the affine warp (not rigid). Per-grid overlap
  optima are 3×3 → 0.30, 4×4 → 0.0 (sweep:
  `sessions/08_surface_vascular_match/pwr_overlap.py`).
- Best method on 6-subject benchmark: M4b (PWR 4×4) on 5/6 subjects,
  M4a on 790322; mean Δ-NCC over rigid = +0.066. See
  `docs/08 Automated surface registration.md` for the full table.
- Cache-invalidation triggers: any change to CZ/HCR top surface
  (`dev_code/cached_surfaces/`), `sxy` re-estimate, or CZ outline
  TIFF.
- Alternate HCR target (binary GFP+ ROI MIP from
  `segmentation_mask_orig_res.zarr`) is supported as a sanity-check
  via `sessions/08_surface_vascular_match/hcr_gfp_seg_mip.py` but is
  NOT the default scoring target.
- Doc: `code/docs/08 Automated surface registration.md`.
- Grand Plan ledger row: "S08 surface-vascular registration /
  promoted" (Section 11); accessor noted in Section 1.2.

**Blocked-bug regression check:** previous PWR had a sign error in
the inverse warp (`coords = [yy − dy, xx − dx]` rather than
`[yy + dy, xx + dx]`). Regression test:
`pwr_overlap.synthetic_sign_test()`.
