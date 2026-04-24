# Session 28 — Warm-start diagnosis + propagation

**Goal.** Diagnose why P14 reported `n_pred=0` after earlier ICP-warm-start fix, and close the gap between the ICP warm-start and the landmark-perfect warp (LM median 37.5 µm vs ICP 143.5 µm per GT pair on 788406).

## Findings

1. **`scales_zyx` missing from ICP path of `default_warmstart_zyx`.** The helper stored the scales under `icp_fit_scales_xyz` when `source=icp_fit`, but P14 (and every other candidate) read `scales_zyx`. Every candidate silently raised `KeyError: scales_zyx` and the harness swallowed the exception, yielding `n_pred=0` rows with an empty diagnostics JSON.
   - **Fix.** `default_warmstart_zyx` now always writes `scales_zyx` regardless of path (`lib/centroid_helpers.py`).

2. **ICP starts from HCR-all centroid, which is ~156 µm offset from the CZ sub-ROI's true centre in Z.** The CZ sub-slab covers only a portion of the cortex; using the HCR-wide GFP+ centroid as the translation seed pulls ICP's converged transform off-centre. ICP's matched set (256 pairs) is consistent with the wrong local minimum, so rms=24 µm but per-GT median=143.5 µm.
   - **Fix.** After ICP converges, do a coarse-then-fine grid search over translation offsets that maximises the count of warped CZ cells landing within 50/30 µm of any HCR cell. Brings per-GT median from 143.5 → 79 µm on 788406.

3. **Iterative Procrustes refit on reciprocal-NN within the current 70%-quantile distance.** Converges in 1–2 iterations from the translation-refined warp; marginal gain (~8 extra inlier@30 on the refit pass). Adopted-scales replace ICP scales when refit improves score.

4. **P14 feature-cosine weight swept.** With the improved warm-start, F6 features hurt Hungarian: `w=0` (pure distance) gives 195/784 correct vs 186/784 at `w=15` and 147/784 at `w=150`. F6 invariant features as implemented are not sufficiently modality-invariant for this data. Switched P14's cost to pure distance.

## Candidates updated

Every candidate that had its own ad-hoc `(cz_um - cz_c) @ R0.T + hcr_c` warm-start now calls `default_warmstart_zyx`:
- P1 (TEASER-fallback), P3 (RANSAC), P4 (spectral), P5 (FGW), B1/B2 (seed + TPS).
- G1 (GNN matcher) and G2 (contrastive embed) now build their k-NN graphs on `cz_init` (warped), not raw `cz_um` — otherwise the CZ graph spacing is 1.86× too small relative to HCR and graph features don't transfer.

## 788406 post-fix numbers

| Cand | rec@5µm | rec@10 | rec@20 | n_pred | med (µm) | t (s) |
|------|--------:|-------:|-------:|-------:|--------:|------:|
| P14  | 0.203   | 0.203  | 0.211  | 606    | 79.3    | 48    |
| P1   | 0.202   | 0.202  | 0.210  | 623    | 81.1    | 10    |
| C1   | 0.202   | 0.202  | 0.210  | 623    | 81.1    | 24    |
| P4   | 0.146   | 0.146  | 0.159  | 751    | 91.5    | 79    |
| P3   | 0.080   | 0.080  | 0.083  | 100    | 0.0     | 153   |
| P5   | 0.000   | 0.000  | 0.000  | 787    | 754.1   | 87    |
| B1/2 | 0.001   | 0.001  | 0.001  | 4      | 39.1    | 11    |

Pre-fix everything below P14 reported `recall=0`. P14 itself went from ~0.017 → 0.203.

## Ceiling analysis

Warm-start per-GT distance distribution (788406, 784 GT pairs in GFP+):
- <5 µm: 0
- <10 µm: 6
- <20 µm: 36
- <30 µm: 88
- <50 µm: 239

Landmark-perfect warp achieves:
- <10 µm: 24
- <20 µm: 116
- <50 µm: 544

So warm-start is ~2× off from the ceiling a landmark fit provides. Closing this gap would likely raise recall from ~20% to ~50%. Candidate next steps:
- Run `estimate_scales_icp_multi_start` with a *better* initial translation (e.g., fine-grid seeded by mask NCC from M1 when M1 works).
- Try multi-start ICP with translation seeds on a 3D grid, pick lowest rms ICP converged fit.
- Use P14-output to seed a B2 TPS-expansion pass.

## Stress-subject sweep (post-fix)

rec@5µm × subject × candidate:

| Cand | 788406 | 755252 | 767018 | 767022 | 782149 | 790322 |
|------|-------:|-------:|-------:|-------:|-------:|-------:|
| P14  | 0.203  | 0.050  | 0.267  | 0.000  | 0.000  | 0.208  |
| P1   | 0.202  | 0.033  | 0.143  | 0.000  | 0.000  | 0.226  |
| P4   | 0.146  | 0.045  | 0.304  | 0.000  | 0.000  | 0.180  |
| P3   | 0.080  | 0.008  | 0.088  | 0.000  | 0.000  | 0.087  |
| C1   | 0.202  | 0.033  | 0.143  | 0.000  | 0.000  | 0.226  |

**3/6 subjects recover meaningful recall; 3/6 fail (767022, 782149 at 0%; 755252 near 0%).** Warm-start converges to a wrong local minimum on those subjects — `default_warmstart_zyx`'s translation grid search is not exhaustive enough.  P4 wins on 767018 (0.304); P1 wins on 790322 (0.226); P14 wins on 788406 (0.203).  Mean recall across all 6 subjects: P4 0.129, P14 0.121, P1 0.101, C1 0.101, P3 0.044.

## Next step

Break past the warm-start ceiling on the failing 3 subjects by (a) seeding ICP from M1 mask-NCC peak (requires F1/F2 mask loaders — Section 4.2 of the plan), or (b) multi-start ICP with a denser 3-D translation grid. Session 29 should pick one.

## Files modified

- `lib/centroid_helpers.py` — scales_zyx unified, translation search, iterative local refit.
- `bench/candidate_impls/_p14_hungarian.py` — cost = D, no feature weight.
- `bench/candidate_impls/_p1_teaser.py` — warm-start via helper.
- `bench/candidate_impls/_p3_ransac.py` — warm-start via helper.
- `bench/candidate_impls/_p4_spectral.py` — warm-start via helper; drops its own `_rescale_axes`.
- `bench/candidate_impls/_p5_fgw.py` — warm-start via helper; crops HCR around warped centroid.
- `bench/candidate_impls/_b1_b2_seed_tps.py` — warm-start via helper.
- `bench/candidate_impls/_g1_gnn_matcher.py` — graph built on warped CZ.
- `bench/candidate_impls/_g2_contrastive.py` — graph built on warped CZ.
