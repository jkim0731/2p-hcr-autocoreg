---
name: sz_estimator promoted (iter-7 slab-side-view FFT)
description: 2026-05-02 dev_code/sz_estimator.py::get_sz(s) ships the iter-7 slab FFT NCC algorithm; 6/6 subjects exact-match GT; cache schema v2; consumed by overlap_crop.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
**Decision (2026-05-02)**: `code/dev_code/sz_estimator.py::get_sz(s)` is the
production slab-rigid sz estimator. Algorithm = **iter-7 slab-side-view FFT
rigid NCC** (binarized CZ ROI warped into the HCR crop via the locked-prior
frame; x-MIP slabs over the central 500 µm of x as 5×100 µm slabs; 2-D FFT
cross-correlation per slab; mean NCC across the 5 slabs scored on a sz grid).

**Why iter-7, not iter-8+ `spot_mask_3d`**: a prior promotion attempt shipped
the iter-8+ 3-D `spot_mask_3d` scoring algorithm. It failed on 788406
(returned sz=3.45 vs GT=2.82, fell back to sz_lp), which made
`get_overlap_crop(788406)` give the wrong z-extent. Iter-7 is what the GT
CSV at `data/claude_data/full_automatic_execution_02/sessions/v2_S02_sz_image_sweep/results_iter7_summary.csv`
was produced with, and it is what `docs/11 Work In Progress.md` documents.

**Per-subject sz_best vs iter-7 GT** (all err = 0.000, grid step 0.10):

| sid    | sz_best | GT iter-7 |
|--------|--------:|----------:|
| 755252 |    2.10 |      2.10 |
| 767018 |    3.60 |      3.60 |
| 767022 |    2.40 |      2.40 |
| 782149 |    2.90 |      2.90 |
| 788406 |    2.80 |      2.80 |
| 790322 |    3.10 |      3.10 |

**Public API (unchanged):** `from sz_estimator import get_sz; get_sz(s)`
returns dict with `sz_best` headline field and JSON cache at
`code/dev_code/cached_sz/<sid>.json` (cache version bumped to **v2** —
older v1 JSONs from the wrong algorithm are invalid; remove them on
suspicion of staleness).

**Private helpers retained (other modules import these):** `HCR_LEVEL`,
`_warp_cz_into_hcr_crop`, `_build_affine`. Removed: `estimate_sz_image_ncc`,
`SzSweepResult`, `_foreground_z_profile`, `_dog_spot_mask`,
`_smoothed_voxel_ncc`, `_spot_mask_3d_ncc`, `_dog_3d_ncc`, `_shifted_ncc_1d`.

**Crop sanity (788406 with margin_frac=0.10):** sz_used=2.80; raw CZ bbox
dz=1282 µm dy=824 µm dx=795 µm; final 1518×989×954 µm. (Earlier wrong
sz=3.45 inflated z to ~1552 µm.) These dimensions are geometrically
consistent with the CZ ROI volume (450×399×399 µm at sz=2.80 → ~1260 µm
z-extent before margin); a smaller "~600×500×300 µm" estimate I gave during
debugging was wrong.

**How to apply:** any session that needs sz must call `get_sz(s)`; do not
re-implement the sweep. The working-volume contract in
`docs/11 Work In Progress.md` depends on this estimator being correct,
because it feeds `overlap_crop.get_overlap_crop(s)`.
