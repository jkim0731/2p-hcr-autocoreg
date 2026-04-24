# Session 30 — Rotation-augmented multi-start ICP for 782149

## Goal

Lift 782149 rec@5 above 0.05 on at least one tier-1 candidate. S29's
translation-only multi-start left 782149 at 0 because every translation
seed converges to the same wrong basin. Rotation error is the suspected
cause (pia tilt ≈ 12° on this subject; anisotropic Z expansion may also
interact with the 180° XY prior).

## Approach

Extend `default_warmstart_zyx` with optional ICP rotation-seed
perturbations:

1. For each of the 6 translation seeds from S29, additionally run ICP
   with the initial CZ rotated by `{0°, +5°, -5°, +10°, -10°}` about
   the Z axis, and also `{0°, ±5°}` about the X axis (cortical-depth
   tilt).
2. Total seeds: 6 × 5 × 3 = 90. Prune aggressively — skip any seed
   whose initial inlier@50 is below the global median * 0.5.
3. Score converged warps with the same `recip × unique_frac` ranker.
4. Return best; cascade to existing translation-refine + local-refit.

## Why this may work

782149 has 12° pia tilt. The 180° XY prior + no-rotation ICP init
means ICP starts at `R = diag(-1, -1, 1)` which is exactly wrong for
a 12°-tilted subject. ICP's local basin is small relative to 12°, so
no amount of translation search will fix it. A ±5° or ±10° rotation
seed should land inside the correct basin.

## Risk

- **Combinatorial blowup**: 90 seeds × (ICP + translation refine +
  local refit) ≈ 15× current warm-start cost ≈ 30–60 s per subject.
  Mitigation: aggressive early rejection based on initial inlier@50
  before running full ICP.
- **Overfitting to 782149**: pruning criteria must not be tuned on GT.
  The initial-inlier gate uses only raw inlier counts, which is the
  same signal the existing translation-refine step uses.

## Plan

1. Add `rotation_seeds=False` flag to `default_warmstart_zyx`. When
   True, enumerate rotation perturbations inside the multi-start loop.
2. Add early-rejection: compute initial inlier@50 for each seed before
   running ICP; skip if below a threshold.
3. Run the sweep on 782149 (primary target) and all 6 stress subjects
   to check for regressions on working subjects.
4. If 782149 lifts, fold `rotation_seeds=True` into the default.

## Findings (2026-04-19)

Ran a 6 × 9 (translation × rotation) seed probe on all 6 subjects
(`probe_rotations.py`, 54 ICP runs per subject) and compared to the
landmark-fit ceiling (`ceiling_check.py`).

### Negative result on 782149

Landmark-fit ceiling on 782149:
- rms = 29 µm, median = 35 µm, n_gt_lt50 = 243/303 (→ strong recall
  if ICP could find it).

Raw ICP across all 54 seeds on 782149:
- median GT error across seeds: 360–480 µm (every seed).
- n_gt_lt50 = 0 on every seed.
- Converged sxy ∈ [1.43, 2.00], sz ∈ [1.90, 3.04] — scales are in a
  reasonable range and close to the landmark-fit scales (sxy=1.96,
  sz=2.91), so the failure is **not a scale-convergence issue**.
- Translation is off by hundreds of µm; ICP lands in a wrong region
  of HCR that has matching density but wrong identity.

**Conclusion: rotation-augmented multi-start does not help 782149.**
The failure is ICP landing in a false-positive density match, not a
basin-size issue. Translation/rotation grids are the wrong lever.

### Collateral check on other subjects

| Subject | Ranker top r_seed | Ranker n_lt50 | Oracle top | Oracle n_lt50 |
|---------|-------------------|--------------:|------------|--------------:|
| 788406  | `gfp_dz-100 rx-10` | 256         | same       | 256           |
| 755252  | `hcr_gfp rx-5`    | 141           | `hcr_gfp rz-5` | 199       |
| 767018  | `hcr_gfp rz-5`    | 28            | `hcr_gfp none` | 41        |
| 767022  | `gfp_dz-100 rx+5` | 0             | (all 0)    | 0             |
| 782149  | `gfp_q25 rz-5`    | 0             | (all 0)    | 0             |
| 790322  | `hcr_gfp rx-5`    | 260           | same       | 260           |

- **790322** would benefit massively (ranker picks oracle-best,
  n_lt50: 58 → 260). Candidates like P14/P1 already got +2–4 pp in
  S29 from finer translation alone; rx seeds could push higher.
- **788406** ranker picks a rx-10 seed with n_lt50=256, matching oracle.
- **767018** ranker picks a worse seed than current default (rz-5 has
  n_lt50=28 vs none's 41). Rotation seeds would **regress** this subject.
- **755252** ranker picks a rx-5 seed which improves modestly, but the
  oracle-best (rz-5) is not picked — there's a ranker-oracle gap.

### Decision

**Do not fold rotation seeds into the default warm-start.** The ranker's
self-supervised score is not consistent enough to reliably pick good
rotation perturbations — it actively regresses 767018 and misses the
oracle-best on 755252. And the target subject (782149) is not fixable
by rotation at all.

Keep the probe as a reference (files: `probe_rotations.py`,
`probe_all.csv`, `ceiling_check.py`, `ceiling.csv`).

### Next step

Pivot to S31: **M1 widened-scale-grid + soft-mask NCC**. Mask-level
density matching should be robust to the false-positive local minimum
ICP falls into on 782149, because masks encode coarse cortical geometry
rather than per-cell constellation.

## Files

- `probe_rotations.py` — 6 × 9 seed probe.
- `probe_all.csv` (324 rows) — per-seed ranker score + GT metrics.
- `probe_788406.csv` ... `probe_790322.csv` — per-subject slices.
- `ceiling_check.py` — landmark-fit ceiling per subject.
- `ceiling.csv` — ceiling table showing 782149 is recoverable in principle.
