# S51 — P4 spectral GM + HCR image-quality β bonus

**Status:** validated. β=5 adopted as P4 default. Orthogonal lift to P1: P4+β=5 beats P1+β=5 on 755252 by +0.047 r@20 (0.091 vs 0.044).

## Motivation

S47 shipped a β=5 `hcr_quality_beta` bonus in P1 that lifted cross-subject r@20 by +0.019 by biasing TEASER putative ranking toward high-quality HCR cells. P4 (spectral GM) shares P1's putative-ranking stage, so the same bonus should apply. Secondary goal: quantify whether P4+β is orthogonal to P1+β and could anchor a P1∪P4 consensus pipeline.

## Change

`bench/candidate_impls/_p4_spectral.py::run_p4` now accepts `hcr_quality_beta: float = 5.0`. When `β > 0`, putative ranking subtracts `β·hq[j]` from `(D[i] - 20·cos(F6))`, mirroring the S47 P1 bonus. Off-diagonal affinity and the eigenvector loop are unchanged.

Also inline: vectorised the P4 affinity-matrix build. Prior Python double-loop took ~12 min per (subject, β) on P=4000 putatives; now ~15–90 s.

## Results

All 4 subjects, K=5, c_bar=40 (default).

| subj    | β=0    | β=3    | β=5    | β=8    |
|---------|--------|--------|--------|--------|
| 788406  | 0.159  | 0.154  | **0.173** | 0.142  |
| 755252  | 0.058  | 0.070  | **0.091** | 0.092  |
| 767022  | 0.048  | 0.047  | 0.048  | 0.028  |
| 782149  | 0.000  | 0.000  | 0.000  | 0.000  |

(r@20 — fraction of GT CZ cells whose predicted HCR centroid is within 20 µm of GT.)

β=5 is the best single setting across the 4 subjects; β=8 regresses on 788406/767022. β=5 is the adopted default.

## Comparison to S47 P1+β=5 (same hcr_quality, same β)

| subj    | P1+β=5 r@20 | P4+β=5 r@20 | Δ (P4−P1) |
|---------|-------------|-------------|-----------|
| 788406  | 0.234       | 0.173       | −0.061    |
| 755252  | 0.044       | 0.091       | **+0.047** |
| 767022  | 0.103       | 0.048       | −0.055    |
| 782149  | 0.000       | 0.000       |   0.000   |

P4+β=5 wins on 755252 only — the sparse-GFP+ stress subject. This is consistent with the spectral-GM hypothesis that pairwise-distance consistency helps when per-cell feature affinity is noisy. For a consensus pipeline, P4 is a useful tie-breaker on 755252-style subjects.

P4 does **not** unlock 782149; the 0 r@20 matches P1, C1, B1/B2, M-family ceiling — the generator is structurally broken for that subject (S45 top-K=500 analysis).

## Shipped

- `bench/candidate_impls/_p4_spectral.py` — `hcr_quality_beta: float = 5.0` default. Vectorised affinity-matrix build. All P4 benchmarks pre-S51 used β=0 (equivalent to the new β=0 column above).

## Files

- `sessions/51_p4_hcr_quality_beta/sweep.py` — 4-subject × 4-β grid.
- `sessions/51_p4_hcr_quality_beta/sweep.log` — raw results.
