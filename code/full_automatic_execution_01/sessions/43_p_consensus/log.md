# S43 — P1 ∩ P14 cross-candidate consensus

**Status.** validated_negative — P1 and P14 make strongly correlated
errors; intersection is not a high-precision subset. Fallback variant
equals P1 exactly. No path to lift without a matcher with orthogonal
error modes.

## Setup

After S42 showed SS-ranker regresses P1 by −0.055, try an
identity-based consensus with no score:

- **intersect**   — emit only pairs where `P1(cz).hcr == P14(cz).hcr`.
- **fallback_p1** — keep all P1 pairs; mark consensus pairs with
  `confidence=1`, others keep P1's intrinsic score.

If agreement between two independent matchers concentrates on correct
pairs, intersect should show high precision even on stress subjects.

## Results

| Subject | n_gt | common_cz | agree | agree_frac | intersect r@20 | intersect prec | fallback r@20 | P1 r@20 |
|---------|-----:|----------:|------:|-----------:|---------------:|---------------:|--------------:|--------:|
| 788406  | 787  | 594       | 418   | 0.704      | 0.146          | **0.310**      | 0.210         | 0.210   |
| 755252  | 639  | 629       | 405   | 0.644      | 0.033          | 0.052          | 0.044         | 0.044   |
| 767022  | 793  | 483       | 190   | 0.393      | 0.039          | 0.123          | 0.108         | 0.108   |
| 782149  | 303  | 111       |  71   | 0.640      | 0.000          | 0.000          | 0.000         | 0.000   |

**Totals (r@20, sum across 4 subjects):** P1 = 0.362; fallback = 0.362
(+0.000); intersect = 0.218.

**Intersect median precision: 0.088.** Min 0.000 (782149).

## Diagnosis

**P1 and P14 share the same error modes.** Both use F6-cosine + Euclidean
distance for putative ranking and commit to TLS/Hungarian basins derived
from those same putatives. Agreement ≠ identity-correctness:

- 788406: agree 418/594 = 70 %; of those agreed, precision = 31 % — most
  agreements are wrong-pair coincidences in tight dense regions where
  both matchers converge to the same wrong partner.
- 755252: agree 64 %; intersect precision 5 % — wrong-basin lock-in
  correlates across matchers.
- 767022: agree only 39 % — lowest overlap — and intersect precision
  12 %. Even where they disagree, P1's predictions are more reliable
  than P14's (r@20 0.108 vs 0.053), so the intersection drops the half
  that P1 alone got right.
- 782149: agree 64 %, intersect precision 0 — neither matcher ever
  reaches the truth on this subject (consistent with S41 r@20=0 for
  every P-candidate).

**Fallback_p1 r@20 = P1 r@20 exactly.** By design, fallback only
changes `confidence`, not `hcr_id`. r@20 is distance-based and
confidence-independent → identical.

## Decision

- **Do NOT ship P1 ∩ P14 consensus.** Agreement between correlated
  matchers concentrates on shared errors, not shared truth.
- **P1 remains the centroid-only production baseline** at r@20 = 0.362
  across 4 subjects (0.210 + 0.044 + 0.108 + 0.000).
- **Centroid-only ensemble ceiling is 0.377** (S41 oracle). Getting
  there requires a matcher with orthogonal errors to P1/P14 — P4
  spectral GM (different putative space via pairwise consistency) is
  the only remaining P-series candidate that might decorrelate errors,
  but on 788406 and 767022 it regresses (0.159 and 0.048 vs P1 0.210
  and 0.108), so P4-inclusive consensus also caps at or below current.
- **Pivot to mask / image modalities for any further lift.** M1
  (mask-NCC via F1/F2), M3 (mask warm-start → P1), C1 (I2 affine → P1),
  I2 (SimpleITK MI affine) exist in the candidate registry but have
  only been run on 788406 in prior sessions. S44: cross-modality
  bakeoff — M1, M3, C1, I2 on the 4 subjects, same shape as S41.
  Primary goal: check whether any mask/image method reaches 782149
  r@20 > 0 OR lifts 755252/767022 above their centroid plateau.

## Artifacts

- `probe_consensus.py` — 2 candidates × 4 subjects + 2 consensus variants = 16 evaluations.
- `probe_consensus.log` — stdout with per-subject agreement + precision.
- `consensus_raw.csv` — machine-readable per-(subj, variant) rows.
- `consensus_summary.csv` — agreement rates + consensus r@20 per subject.
