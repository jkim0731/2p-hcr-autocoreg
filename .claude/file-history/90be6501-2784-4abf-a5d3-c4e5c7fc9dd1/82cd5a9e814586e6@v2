# S62 — B1/B2 retrial with smart seeds (B3, B3b)

## Goal

S60 closed the consensus-vote path and identified C5 sum r@20 = 1.080 as
the centroid-only ceiling, but the B1/B2 family had only been tested once
(S23/S24 on 788406, recall = 0, unilateral abandonment). The user's
diagnostic questions prompted a retrial: would **smarter seeds** than
B1's hull-emptiness heuristic — specifically C5's highest-confidence
picks — unlock the TPS-expansion path?

S62 tests two variants:

- **B3**: C5's top-30 spatially-diverse (≥50 µm in CZ) high-confidence
  seeds → robust LOO outlier dropout → per-axis TPS → greedy one-to-one
  nearest-neighbour with τ=40 µm. **Replaces** C5 entirely.
- **B3b**: additive. Keep every C5 pair unchanged; use TPS only to
  propose matches for CZ cells C5 did not cover, with τ=25 µm.
  Cannot lower recall vs C5.

Validation roster (per 2026-04-21 scope revision):
788406 / 790322 / 767018.

## Results — both refuted

### B3 (replace-with-TPS)

| Subject | C5 r@20 | B3 r@20 | B3 med (µm) | Δ (B3 − C5) |
|---------|--------:|--------:|------------:|------------:|
| 788406  | 0.262   | 0.103   | 111.4       | **−0.159** |
| 790322  | 0.289   | 0.075   | 80.1        | **−0.215** |
| 767018  | 0.315   | 0.187   | 44.5        | **−0.128** |
| **SUM** | **0.866** | **0.364** | —         | **−0.502** |

### B3b (C5 ∪ TPS-for-C5-gaps, τ=25 µm)

| Subject | C5 r@20 | B3b r@20 | gaps | added | Δ (B3b − C5) |
|---------|--------:|---------:|-----:|------:|-------------:|
| 788406  | 0.262   | 0.262    | 16   | 0     | **+0.000** |
| 790322  | 0.289   | 0.290    | 123  | 1     | **+0.001** |
| 767018  | 0.315   | 0.315    | 41   | 0     | **+0.000** |
| **SUM** | **0.866** | **0.867** | 180 | 1     | **+0.001** |

Across **180 CZ cells C5 skipped**, the TPS fit correctly placed exactly
**one** within τ=25 µm. For the other 179, TPS-predicted HCR positions
were > 25 µm from any free HCR GFP+ cell — i.e. the 23-seed TPS is
unable to bridge C5's coverage gap either.

## Why smart seeds still fail

**TPS residuals are fundamentally too high.** B3's median residual is
44.5 µm (767018) to 111.4 µm (788406); the 20 µm recall threshold gates
out most predicted pairs. The 23-seed TPS (after LOO dropout from 30
C5-confidence picks ≥50 µm apart in CZ) **cannot** absorb the local
nonrigid warp in a form that would predict HCR positions within 20 µm
of truth.

Two contributing causes:

1. **C5's highest-confidence picks are not globally-optimal anchors.**
   C5's per-pair confidence is the sum of per-method (P1/P4/P6) inlier
   certificates. A "high confidence" pair may lie near the
   CZ-population centroid (where putative correspondences are
   concentrated) yet sit on a locally distorted patch. 23 such seeds
   do not span the nonrigid warp's spatial-frequency content.

2. **Local warp is finer-grained than 23-seed TPS can capture.** The
   benchmark's post-affine landmark RMS is 16–43 µm — i.e. the
   nonrigid component itself is at the same scale as the recall
   threshold. A TPS with only 23 control points, each potentially
   misplaced up to ~20 µm, compounds this residual instead of reducing
   it.

## Why B3b is also flat

**B3b's gap set is small** (16 on 788406; 123 on 790322). C5's per-
density dispatcher already emits a match for ≥88% of CZ cells on these
subjects. The CZ cells C5 leaves unmatched are precisely the ones
where no single method reached threshold — and TPS-based nearest-
neighbour at τ=25 µm is a *weaker* predictor than P1/P4/P6 individual
certificates for those cells (evidenced by only 0–1 of 123 gap-TPS
predictions landing within τ on 790322 verified as GT-correct within
the 20 µm recall bound).

## Implication — the **replace-or-augment** corner of B-series is dead

S23/S24's B1/B2 rejection (hull-emptiness seeds) could have been
explained as *"bad seeds → bad TPS"*. S62 now rules out the benign
reading: even **C5's own best picks** are insufficient to anchor a
sparse TPS that meets 20 µm recall. Any B-series variant based on
sparse-seed TPS is refuted:

- B1 (hull-emptiness seeds + TPS): already rejected S23/S24.
- B3 (C5-confidence seeds + TPS replacement): **rejected S62**.
- B3b (C5-confidence seeds + TPS for C5 gaps only): **≈ 0 delta; rejected**.

Dense-seed TPS (500+ seeds) could in principle reduce residual variance,
but sourcing 500+ accurate seeds without a correspondence oracle is
circular — if we had them, we'd already be at ≥90% recall. Rejected.

## Decision — B-series closed; proceed to S63 G1-LOSO

B-series is refuted under every seed-source tried. Next cheap experiment
per 2026-04-21 priority queue: G1-LOSO supervised retraining (bypass F8
synthetic-warp training entirely; use real `coreg_table` pairs from the
other 5 subjects, LOSO per validation subject).

## Files

- `bench/candidate_impls/_b3_c5_seed_tps.py` — B3 replace-with-TPS.
- `bench/candidate_impls/_b3b_c5_union_tps.py` — B3b additive TPS-gap-fill.
- `bench_b3.py` + `.log` + `.csv` — B3 3-subject bench.
- `bench_b3b.py` + `.log` + `.csv` — B3b 3-subject bench.

## Status: `abandoned` (B-series closed)
