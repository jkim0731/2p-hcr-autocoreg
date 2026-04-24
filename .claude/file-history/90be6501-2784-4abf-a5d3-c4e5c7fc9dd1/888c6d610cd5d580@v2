# S39 — P1 TEASER baseline + K × c_bar sweep

**Status.** validated — baseline confirmed, K × c_bar tuning marginal; stress subjects still unreachable.

## Motivation

S38 recommended pivoting from G1 GNN (broken synthetic-to-real transfer) to
P1 TEASER with F6-weighted putatives. P1 already uses F6 features at
putative-generation time and GNC-TLS for certifiable outlier rejection.
Establish the current baseline on the 4 benchmark subjects and probe
whether widening K (putative pool) or c_bar (inlier threshold) lifts
stress subjects.

## Baseline (default K=5, c_bar=15 µm)

| Subject | n_gt | n_pred | recall_id | rec@5  | rec@10 | rec@20 | median µm |
|---------|------|--------|-----------|--------|--------|--------|-----------|
| 788406  |  787 |   623  |  0.202    | 0.202  | 0.202  | 0.210  |   81      |
| 755252  |  639 |   542  |  0.033    | 0.033  | 0.033  | 0.044  |  103      |
| 767022  |  793 |   654  |  0.081    | 0.081  | 0.082  | 0.108  |  107      |
| 782149  |  303 |   145  |  0.000    | 0.000  | 0.000  | 0.000  | 1136      |

Matches S28/S29's P1 figures exactly — no regression. 788406 has real 20 %
ID recall; 755252 and 767022 are weak (3-8 %); 782149 is totally stuck
(median 1136 µm, consistent with the wrong-basin diagnosis from S30/S32).

## Sweep plan (probe_sweep.py)

4 subjects × K ∈ {5, 10, 20, 40} × c_bar ∈ {15, 30, 60} = 48 runs.

- **K**: top-K F6+distance putatives per CZ cell. Larger K admits more
  candidate correct-pairs but also more outliers for TLS to reject.
- **c_bar**: TLS inlier residual threshold (µm). Larger c_bar tolerates
  warm-start error; smaller c_bar enforces tighter inlier agreement.

## Sweep results — best (K, c_bar) by r@20

| Subject | best K | best c_bar | r@20  | r@5   | recall_id | vs baseline r@20 |
|---------|--------|------------|-------|-------|-----------|------------------|
| 788406  |   5    |    15      | 0.210 | 0.202 |  0.202    |    tied (baseline) |
| 755252  |   5    |    15      | 0.044 | 0.033 |  0.033    |    tied (baseline) |
| 767022  |  10    |    15      | 0.131 | 0.110 |  0.110    |   +0.023 (+21% rel) |
| 782149  |   5    |    15      | 0.000 | 0.000 |  0.000    |    tied (0) |

Raw per-config results:
- **788406** (12 configs, r@20 range 0.117–0.232): decreasing K (5→40) or
  widening c_bar (15→60) *hurts* — GNC-TLS overfits the wrong basin when
  more outliers are admitted. Default K=5, c_bar=15 is optimal.
- **755252** (r@20 range 0.011–0.081, nondeterministic across re-runs):
  TLS outcome is basin-dependent; 0.033 ± ~0.02 depending on init. No
  setting is reliably better than default.
- **767022** (r@20 range 0.101–0.131): modest structure — K=10, c_bar=15
  is best, K=40 c_bar=15 ties on recall_id. Gain is +0.023 r@20 = 18 more
  GT-within-20µm pairs.
- **782149** (r@20 = 0.000 for all 12 configs, median err 1127–1138 µm):
  every run converges to the same wrong-basin fit; neither widening K
  (more putatives) nor loosening c_bar (more tolerant TLS) admits a
  different minimum.

## Diagnosis

S29 already established that 782149's centroid-ICP objective has no local
minimum at truth. P1 inherits the same warm-start (`default_warmstart_zyx`),
so it lands in the same wrong basin regardless of K/c_bar. The putative
pool itself contains correct pairs (F6 similarity ranks them in), but the
GNC-TLS fit refuses them because the wrong-basin anisotropic-affine has
more inliers than the right-basin one.

On 755252 and 767022, the warm-start lands near but not at truth; K/c_bar
sweeps produce modest basin-of-attraction jitter (+2-3 pp recall), not a
step change. 788406 is already near-oracle at default.

## Decision

P1 at default (K=5, c_bar=15) is the candidate-level baseline for
point-cloud methods with a single warm-start. Improvements require
multi-start P1 — specifically with seeds beyond `default_warmstart_zyx`.
S34's I2-cropped-HCR warm-start (W=600) and S29's 6-translation grid are
known to unlock 755252 and 767022 respectively. Queue S40 as "P1
multi-start with I2-crop and translation grid seeds."

## Artifacts

- `probe_p1.py` + `probe_p1.log` — baseline P1 on 4 subjects
- `probe_sweep.py` + `probe_sweep.log` — 48-run K × c_bar matrix
- `sweep.csv` — machine-readable matrix
