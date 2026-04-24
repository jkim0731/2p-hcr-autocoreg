# Session 31 — M1 widened-scale grid + fine-scale refinement

## Goal

Push M1 NCC peak to a confidence level that provides a useful ICP
warm-start seed, especially on the subjects where ICP-alone fails
(782149 primarily; also 767022 and 755252).

## What was built

1. **Widened scale grid** in `bench/candidate_impls/_m1_mask_ncc.py`:
   `sxy ∈ [1.4..2.2]` step 0.1 × `sz ∈ [1.8..3.8]` step 0.2 (9 × 11 = 99
   combos, ~60 survive the "template fits in HCR" filter).
2. **Fine-scale refinement** — 5×5 refinement around coarse best
   (sxy step 0.025, sz step 0.05).
3. **Anisotropic sigma** for Gaussian density rendering via geometric
   mean `sigma = sqrt(sigma_xy × sigma_z)` defaults (40, 60) µm.
4. **Per-scale z-score ranker** inside `_sweep_ncc` — picks the scale
   pair whose NCC peak is most anomalous vs its own NCC grid's mean/std,
   not absolute NCC peak (addresses bias: smaller templates produce
   larger valid-correlation search-spaces and hence higher expected-max
   under the null).

## Result

Ran sweep on all 6 subjects (`sweep_m1.py`, see `sweep.csv`).

| Subject | M1 sxy | GT sxy | M1 sz | GT sz | M1 n_lt100 | ICP (from M1 seed) n_lt50 |
|---------|-------:|-------:|------:|------:|-----------:|---------------------------:|
| 788406  | 1.30 | 1.80 | 2.90 | 2.82 | 0 | 0 |
| 755252  | 1.30 | 1.53 | 2.50 | 2.14 | 1 | 0 |
| 767018  | 2.15 | 1.73 | 1.90 | 3.51 | 0 | 0 |
| 767022  | 1.30 | 1.67 | 2.50 | 2.49 | 3 | 0 |
| **782149** | **—** | **—** | **—** | **—** | **no transform** | **no transform** |
| 790322  | 1.30 | 1.80 | 1.90 | 3.03 | 0 | 0 |

**Every subject fails.** Z-score ranker systematically biases sxy → 1.3
(the lower bound of the widened grid) and sz → low values. Seeding ICP
from M1's recovered (S, t) gives 0 GT recall on every subject.

### Why M1 density NCC is fundamentally limited here

Point-density NCC with anisotropic scale is inherently ambiguous because:
- Smaller template → larger valid-correlation search space → higher
  expected max under the null (raw-NCC bias).
- Per-scale z-score addresses the null-bias but introduces a reverse
  bias: at small scales the NCC grid has higher variance (spikier
  point patterns at coarse binning), inflating z-scores for wrong
  scales.
- The HCR GFP+ cloud has large Z extent with high density throughout
  (many cells, not a crisp cortical slab), so there is no sharp
  density peak at the true (sxy, sz) — any reasonable scale produces
  a broad plateau of density matches.

## Partial-overlap — the real root cause of 782149 failure

Quantified HCR-coverage of each subject's CZ (after GT-warp):

| Subject | cz_gt/cz_all | cz_gt Z span | frac_CZ_inside_HCR_Z | ICP-alone result |
|---------|-------------:|-------------:|---------------------:|------------------|
| 788406  | 0.84 | 403 µm | 1.00 | works |
| 755252  | 0.77 | 365 µm | 1.00 | ~0.05 rec@5 |
| 767018  | 0.35 | 281 µm | 0.93 | modest |
| 767022  | 0.86 | 405 µm | 1.00 | fixed by S29 |
| **782149**  | **0.34** | **177 µm** | **0.55** | **0 rec@5** |
| 790322  | 0.77 | 330 µm | 0.92 | works |

**782149 is an extreme partial-overlap case**: only 55 % of CZ lands
inside HCR's Z range under the ground-truth warp, and only 34 % of CZ
cells have an HCR partner in the coreg table. The existing
`anisotropic_icp.py` `_evaluate_fit` docstring already calls this
out (lines 272-276): *"the squeezed basin ... often yields more
reciprocal-NN pairs than the true basin, especially on subjects where
HCR GFP+ has partial depth coverage (e.g. 782149 at
HCR.z/CZ_mapped.z ≈ 0.4)"*. The existing `+10*sz` prior penalty is
evidently insufficient.

`diag_782149.py` plots the X-Y / X-Z / Y-Z projections of the HCR
cloud, the GT-warped CZ, and the ICP-warped CZ (`diag_782149.png`).
ICP's solution shrinks CZ to fit inside HCR's Z range; the true
solution has 45 % of CZ outside HCR.

## Decision

**Abandon M1 as a warm-start generator.** Pivot to **S32 — trimmed
(partial-overlap) ICP**. Idea:
- Sweep `inlier_residual_quantile ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`
  inside the multi-start loop. Aggressive trim (quantile=0.4 drops the
  worst 60 %) lets ICP converge on a matching subset rather than
  forcing the full CZ cloud to fit inside HCR.
- Rank trim levels with the existing S29 self-supervised ranker
  (`recip × unique_frac`).
- 782149's expected trim level is ~0.4 (34 % match rate); well-covered
  subjects should stay near 0.9.

## Files

- `_m1_mask_ncc.py` — widened grid + fine refine + anisotropic sigma
  + per-scale z-score ranker. Left in place but deprecated as the
  sole warm-start — kept as a diagnostic tool.
- `sweep_m1.py` — 6-subject M1 sweep.
- `sweep.csv` — results table showing every subject fails.
- `diag_782149.py`, `diag_782149.png` — X-Y / X-Z / Y-Z projection
  showing ICP's squeezed-basin solution vs the correct GT-warped CZ.
