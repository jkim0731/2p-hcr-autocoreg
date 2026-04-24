# S41 — P-series tier-1 bakeoff on 4 subjects

**Status.** validated — matrix yields per-subject winners; P4 and P14 are
competitive with P1 on 755252; P14 ties P1 on 788406 at 1/3 the runtime;
P5 is broken; 782149 unreachable by every P candidate.

## Setup

Ran P1, P3, P4, P5, P14 on {788406, 755252, 767022, 782149} via F9
harness (same default settings as S39/S40). No hyperparameter tuning.

## Results — r@20 matrix

| Subject | P1 (TEASER) | P3 (RANSAC) | P4 (Spectral) | P5 (FGW) | P14 (Hungarian) |
|---------|------------:|------------:|--------------:|---------:|----------------:|
| 788406  | **0.210**   | 0.083       | 0.159         | 0.000    | **0.211**       |
| 755252  | 0.044       | 0.009       | **0.058**     | 0.000    | **0.058**       |
| 767022  | **0.108**   | 0.028       | 0.048         | 0.000    | 0.053           |
| 782149  | 0.000       | 0.000       | 0.000         | 0.000    | 0.000           |

## Results — recall_id matrix (exact hcr_id match)

| Subject | P1     | P3     | P4     | P5     | P14        |
|---------|-------:|-------:|-------:|-------:|-----------:|
| 788406  | 0.202  | 0.080  | 0.146  | 0.000  | **0.203**  |
| 755252  | 0.033  | 0.008  | 0.045  | 0.000  | **0.050**  |
| 767022  | **0.081** | 0.018 | 0.039 | 0.000  | 0.039      |
| 782149  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000      |

## Findings

1. **P14 (Hungarian baseline) matches or beats P1 on 3/4 subjects.** On 788406
   (0.203/0.211 vs 0.202/0.210) and 755252 (0.050/0.058 vs 0.033/0.044),
   P14's simple F6+distance + `linear_sum_assignment` ties or wins — at
   **15 s wall vs P1's 55 s** (≈ 4× faster).

2. **P4 (spectral GM) wins 755252 r@20 tied with P14.** Pairwise-consistency
   affinity + power-iteration separates GT from distractors slightly better
   than P1's TLS on this subject. But P4 is slower (80 s) and regresses
   788406 (0.146 vs 0.202).

3. **P1 still wins 767022** (0.108 vs P4 0.048 / P14 0.053). 767022's
   success for P1 is TEASER-specific — pairwise-consistency (P4) and
   Hungarian (P14) don't reproduce it.

4. **P5 FGW is broken** — recall_id = 0, r@20 = 0, median 590–754 µm on
   all 4 subjects. n_pred = n_gt means it returns a dense coupling but
   picks wrong cells. Known failure modes (entropy reg, partial mass,
   warm-start) have not been diagnosed. Do not include P5 in any
   ensemble until fixed.

5. **P3 RANSAC is strictly worst** for every subject — rejects too many
   putatives, leaves ≤ 126 pairs, and those that survive are not lift
   over P1. No reason to pursue further.

6. **782149: r@20 = 0 for every P-candidate.** Confirms S32/S34/S35/S40
   structural finding — centroid-based matching cannot reach truth on
   this subject regardless of matcher. Requires a different evidence
   source (mask, image, or feature embedding trained with asymmetric
   supervision).

## Per-subject winner

| Subject | Winner      | r@20  | Wall |
|---------|-------------|------:|-----:|
| 788406  | P14 ≈ P1    | 0.211 | 15 s |
| 755252  | P4 = P14    | 0.058 | 80/25 s |
| 767022  | P1          | 0.108 | 38 s |
| 782149  | none        | 0.000 | — |

**Per-subject oracle total (r@20):** 0.211 + 0.058 + 0.108 + 0.000 = 0.377
versus P1-only: 0.210 + 0.044 + 0.108 + 0.000 = 0.362. A perfect ensemble
selector would gain +0.015 r@20 (about 30 more GT pairs across the 4
subjects), concentrated on 755252.

## Decision

- **S42 candidate:** P1 + P4 + P14 ensemble with no-peek SS selection.
  Cheap to implement (wrap existing candidates; score each by sum of
  residuals < 30 µm from pairs_df; pick winner per subject). Expected
  gain: modest lift on 755252; no regression elsewhere.
- **Do NOT pursue P3, P5 further** without explicit upstream fixes.
- **Stop attacking 782149 via point-cloud methods.** Pivot required —
  mask/image registration (M1/I2/C3) or feature-based matching with
  cross-modal supervision.

## Artifacts

- `probe_bakeoff.py` — 5 candidates × 4 subjects = 20 runs via F9.
- `probe_bakeoff.log` — stdout with per-run recall + wall.
- `bakeoff.csv` — machine-readable matrix.
