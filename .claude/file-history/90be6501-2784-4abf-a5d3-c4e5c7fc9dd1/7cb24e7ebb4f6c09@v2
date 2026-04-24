# S16 — M4 per-cell Dice / IoU refinement

## Goal
Once a coarse affine is in hand (from M1/M3/P1), compute per-cell IoU
between each CZ cell's mask and its candidate HCR cell's mask.
Provides a per-pair volumetric confidence that catches
segmentation-error pairs.

## API
`bench/candidate_impls/_m4_per_cell_dice.py::run_m4(s)`

## Method
1. Load CZ mask (F2) and HCR GFP+ mask (F1) at shared µm grid via F3.
2. Run M3 internally to get a candidate pair list.
3. For each pair, crop both masks to the CZ cell's bbox, compute
   Dice/Jaccard.
4. Return per-pair IoU; optional accept/reject threshold.

## Files
- `bench/candidate_impls/_m4_per_cell_dice.py`

## Status
Pending sweep result.
