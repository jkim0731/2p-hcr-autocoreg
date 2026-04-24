# S42 — P1+P4+P14 ensemble with no-peek SS selection

**Status.** validated_negative — SS ranker regresses P1-only by **−0.055 r@20**
on 4-subject benchmark. No-peek residual-fit basin score cannot discriminate
"geometrically self-consistent but wrong-pair" from "self-consistent and
correct" basins across candidates.

## Setup

Wrap P1, P4, P14 (same F9 defaults as S41). For each (subject, candidate):

- Run candidate → `pairs_df`.
- Fit `fit_anisotropic_similarity(CZ_xyz, HCR_xyz)` on its own pairs.
- SS = count of pairs with per-pair residual < 30 µm.

Per-subject pick the candidate with highest SS. Compare SS-pick to oracle
(r@20) and to P1-only baseline. No ground-truth data enters the ranker.

Generalises S40's SS criterion (which only scored variants of P1) to
cross-candidate selection.

## Results

| Subject | P1 SS | P4 SS | P14 SS | SS-pick r@20 | Oracle r@20 | P1 r@20 |
|---------|------:|------:|-------:|-------------:|------------:|--------:|
| 788406  | **263** | 207 | 262 | P1=0.210     | P14=0.211   | 0.210   |
| 755252  | **472** | 455 | 455 | P1=0.044     | P4=0.058    | 0.044   |
| 767022  | 230   | 130 | **246** | P14=0.053 | P1=0.108    | 0.108   |
| 782149  | **108** | 64 | 88  | P1=0.000     | P1=0.000    | 0.000   |

**Totals (sum over 4 subjects):**

- **SS-picker r@20: 0.306**
- Oracle r@20: 0.377
- **P1-only r@20: 0.362**
- **SS lift over P1: −0.055** (regression)
- Oracle lift over P1: +0.015

## Failure modes

1. **767022 — catastrophic (−0.055).** P14 SS=246 > P1 SS=230, but P1 r@20=0.108
   while P14 r@20=0.053. P14's output is geometrically tighter (smaller median
   residual from its own affine fit: 31.1 vs 39.1 µm) but half of its ID
   assignments are wrong-pair. SS has no way to see this.

2. **755252 — small (−0.014).** P1 SS=472 vs P4/P14 SS=455. P1 loses
   0.014 r@20 by picking the tightest-fit basin — which happens to be the
   wrong basin. Mirror of S40 finding: *"the basin's close in
   residuals-of-putative-pairs sense but the putatives themselves are wrong-pair
   by ID."*

3. **788406 — noise (−0.001).** P1 SS=263 vs P14 SS=262; ties within 1 pair.

**Cross-candidate bias.** SS favors P1-style outputs on 3/4 subjects because
P1 emits slightly fewer, tighter-residual pairs after GNC-TLS rejection,
while P4/P14 emit larger pair sets with higher per-pair residuals. SS count
conflates "tight fit" with "large inlier set × low per-pair residual" —
neither of which tracks ID-correctness.

## Diagnosis

SS (count residuals < 30 µm from same-sample affine fit) is a **self-consistency**
metric, not an **identity-correctness** metric. It succeeds when comparing
*variants* of the same matcher (S40: 3 seeds × P1 — all use same F6 putatives,
same TLS rejection → tight basin ≈ correct basin). It fails when comparing
*different matchers* (S42: P1 vs P4 vs P14 — different putative generators,
different rejection rules → tight basin ≠ correct basin).

The stress-subject bottleneck remains what S40 identified: **F6-cosine+distance
ranking plus TLS rejection commits to wrong-pair basins even when the coarse
region is right**. Changing the matcher (P14 in place of P1) does not change
the ranking problem — it just produces a differently-tight wrong basin.

## Decision

- **Do NOT ship SS-selected P1+P4+P14 ensemble.** Regression on 767022 is
  unacceptable (halves that subject's recall).
- **Stay on P1 at default** as production-recommended for 4-subject benchmark.
- **Explore cross-candidate consensus (S43)** — intersection of P1 and P14
  predictions (same CZ→HCR), not SS score. S41 showed P14 matches P1 on
  788406/755252; the intersection should be a higher-precision subset.
  Cheap to implement (zero new infra).
- **Confirm stress-subject pivot** — 782149 centroid methods exhausted
  (S32/S34/S35/S41); requires M1 mask-level registration, I2-image refinement,
  or cross-modal feature learning (C2/C3/G2-domain-adapted).

## Artifacts

- `probe_ensemble.py` — 3 candidates × 4 subjects = 12 F9 runs.
- `probe_ensemble.log` — stdout with per-run SS + GT metrics.
- `ensemble_raw.csv` — machine-readable per-(subj, cand) rows.
- `ensemble_decision.csv` — SS-pick vs oracle per subject.
