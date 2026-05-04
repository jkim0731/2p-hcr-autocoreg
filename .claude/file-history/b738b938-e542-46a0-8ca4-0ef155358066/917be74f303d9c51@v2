---
name: S11 labelling priorities (v5d-driven)
description: 2026-05-01 priority list of 405 unlabelled cells for relabelling, prioritising 782149 bad_ok and good↔bad_ok confusion zones; CSV at outputs/labelling_priority_v5d.csv.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
Active labelling targets to lift v5d 4-class f1_macro 0.733 → ~0.78. Driven
by per-fold f1, label imbalance, and v5d OOF margin uncertainty.

**Tier 1 — biggest leverage:**
1. **782149 bad_ok**: only 30 labels, f1=0.28 (worst across all
   subject×class). Single-fold bottleneck for macro f1. Target +50.
2. **good ↔ bad_ok confusion globally**: confusion matrix shows
   true_good→pred_bad_ok 19% (81/420) and true_bad_ok→pred_good 15%
   (51/347). Largest systematic error.

**Tier 2:**
- 767018 bad_ok (n=50, f1=0.61): +30
- 755252 good (n=65, f1=0.68): +30 — over-called as bad_ok
- 767022 bad (n=19, f1=0.65): +20

**Tier 3:**
- 790322 bad (n=21, f1=0.67): +15
- 788406 good (n=58, f1=0.74): +20
- All others: top-up of small classes (10-15 each)

**Selection method:** v5d 4-class OOF probas → `margin = max_p - 2nd_max_p`
→ smallest margin = most confused cell. Per (subject × predicted_class) take
the K lowest-margin UNLABELLED cells (median margins are 0.001-0.010 — these
are genuinely confused).

**Output:**
`code/sessions/v3_S11_roi_quality/outputs/labelling_priority_v5d.csv`
(405 cells; cols: sid, hcr_id, predicted_class, max_p, margin, p_bad,
p_bad_ok, p_good, p_merged). Rebuilt with the inline script in this session
or any equivalent.

**Per-subject × predicted_class targets:**
| sid    | bad | bad_ok | good | merged | total |
|--------|----:|-------:|-----:|-------:|------:|
| 782149 |  15 |     50 |   15 |     15 |    95 |
| 767018 |  15 |     30 |   15 |     15 |    75 |
| 755252 |  10 |     25 |   30 |     10 |    75 |
| 767022 |  20 |     20 |   10 |     10 |    60 |
| 790322 |  15 |     10 |   10 |     15 |    50 |
| 788406 |  10 |     10 |   20 |     10 |    50 |
| total  |  85 |    145 |  100 |     75 |   405 |

**Why this allocation:** weighted by (1 − f1_class) × n_class_total / n_eval_class
on each fold — under-fitted classes get priority, with extra weight on the
classes where confusion fractions exceed 10%.

**How to apply:** load the priority CSV in the ROI viewer/QC GUI, walk the
list filtered per (sid, predicted_class), confirm or correct the label, write
through the existing `roi_qc_actions.jsonl` log. Then re-run
`05h_train_stage2_v5d.py`. Expected lift: f1_macro 0.733 → 0.77-0.79
(historical: ~+0.04 per +250 well-targeted labels in this codebase).
