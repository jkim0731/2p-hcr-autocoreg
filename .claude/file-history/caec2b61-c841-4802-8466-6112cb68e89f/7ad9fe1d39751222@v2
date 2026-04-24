---
name: Threshold methods must be distribution-driven
description: When picking GFP+ or related thresholds on benchmark data, never default to a fixed percentile/fraction — the coreg table is validation-only and won't exist for future subjects.
type: feedback
originSessionId: f7382aeb-5235-4094-a1cd-69297f8e005e
---
Never choose a fixed-percentile or fixed-fraction threshold as the default
for GFP+ (or any similar per-subject cutoff). The constraint is
generalisability: future subjects have no `coreg_table.csv` to tune against,
so any hand-picked number won't survive.

**Why:** The user rejected the v1 subgoal-01 winner (`fixed_frac @ 20 %`)
explicitly with "Never use a fixed percentile because it cannot be
generalized. reminder - There will be no coreg table for future data."

**How to apply:** When ranking threshold strategies:
- Exclude fixed-percentile / top-N % / count-min candidates from the default
  slot (they may remain as diagnostic options).
- Prefer distribution-driven methods (Yen / Otsu / GMM / Triangle / Isodata
  on `log(feature)`).
- Use the coreg table only to *validate* coverage, never to derive the
  threshold.
- Among passing methods, prefer the one producing the highest per-subject
  threshold (higher is more stringent).
