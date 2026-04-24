# S05 — F4 mask-overlap scorer

## Goal
Compute Dice, Jaccard, 3D NCC, and per-cell Dice between a CZ mask and an
HCR mask at a specified affine.  Downstream: M1, M3, M4.

## API
`lib/mask_overlap.py`:
- `mask_dice_jaccard_ncc(cz_mask, hcr_mask, spacing_um,
    ncc_search_um=60.0, compute_sdf=False) → MaskOverlapScores`
- `per_cell_dice(cz_mask, hcr_mask, pairs) → per-pair Dice`.

NCC implementation uses a brute-force roll over ±r voxels (r =
`ncc_search_um / spacing_um`); returns peak NCC and its translation
offset.

## Self-test
Rolling a known binary volume by 5 voxels and feeding it back returned
Dice = 0.9 at the known translation and NCC peak = 1.0 at the correct
offset.  Pass.

## Files
- `lib/mask_overlap.py`

## Next step
M1 uses `mask_dice_jaccard_ncc` for coarse alignment; M4 uses
`per_cell_dice`.
