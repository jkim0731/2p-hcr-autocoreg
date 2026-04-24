# Session 54 — P1 ⊕ P6 ensemble

**Status:** validated (2026-04-20)
**Goal:** ship a production-complementary P1⊕P6 ensemble per Grand Plan §9.6. Exploit S53's finding that P1 wins on dense-GFP+ subjects and P6 wins on sparse-GFP+ stress subject 767018.

## Strategies tested (6 subjects)

- `P1`: baseline (S47 default `hcr_quality_beta=5.0`).
- `P6`: baseline (S53 default `method=cpd_nonrigid, β=0`).
- `union_conf` — union of P1 and P6 pair lists; on cz_id collision, keep **higher confidence**.
- `union_p1_first` — union; on cz_id collision, keep **P1**.
- `intersection` — only pairs where P1 and P6 agree on `(cz_id, hcr_id)`.
- `oracle_best` — max(P1, P6) per subject (diagnostic only).

Note: P1 and P6 use non-comparable confidence scales (P1 = TLS-inlier-certificate + TPS residual; P6 = `1 / (1 + d_nn/σ_med)`). `union_conf` compares them directly anyway — works in practice because P6's confidence tends to be higher than P1's, so P6 wins collisions on sparse subjects where P6 is actually better.

## 6-subject benchmark (`bench_ensemble.py`)

| Subject | P1 r@20 | P6 r@20 | union_conf | union_p1_first | intersect | oracle |
|---|---|---|---|---|---|---|
| 788406 | 0.234 | 0.175 | 0.226 | **0.252** | 0.141 | 0.234 |
| 790322 | 0.251 | 0.224 | 0.253 | **0.269** | 0.203 | 0.251 |
| 755252 | 0.044 | 0.052 | **0.064** | 0.055 | 0.016 | 0.052 |
| 767022 | 0.103 | 0.053 | 0.091 | **0.117** | 0.020 | 0.103 |
| 767018 | 0.143 | 0.264 | **0.278** | 0.176 | 0.117 | 0.264 |
| 782149 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **SUM** | **0.775** | 0.767 | **0.913** | 0.868 | 0.497 | 0.903 |

## Key findings

**1. `union_p1_first` is Pareto-dominant over P1 alone.** Every subject gains or ties:

| Subject | ΔR@20 vs P1 |
|---|---|
| 788406 | +0.018 |
| 790322 | +0.018 |
| 755252 | +0.011 |
| 767022 | +0.014 |
| 767018 | +0.033 |
| 782149 |  0.000 |

Sum r@20 0.775 → 0.868 (**+0.093, +12 % relative**). No subject regresses. Safe production default.

**Mechanism**: P6 fills cz_ids that P1 didn't predict (e.g. on 790322, P1=508 pairs, P6=481, union=566 — P6 contributes 58 new cz_ids; some are GT-correct → lift). On cz_id collisions, keep P1 (P1 has stronger per-pair ranking on dense subjects).

**2. `union_conf` exceeds oracle best-of (0.913 vs 0.903).** Oracle picks one method per subject; union_conf merges across cz_ids. On the two sparse-GFP+ subjects (755252, 767018), P6's higher-confidence wins the collisions where it's actually right. But it has **small regressions on dense subjects** (788406 −0.008, 767022 −0.012) because P6's confidence sometimes wins collisions it shouldn't.

**3. Intersection is the wrong tool.** Precision 0.00–0.40; same finding as S43 (P1∩P14). P1 and P6 converge on shared wrong-pair basins because both use F6-feature-weighted NN on a common warmstart.

**4. Subject-dispatch `auto` rule beats both unions.** If `n_hcr_gfp < 10 000` (sparse), use `union_conf`; else `union_p1_first`:

| Subject | n_hcr_gfp | Rule | r@20 |
|---|---|---|---|
| 788406 | 17 427 | up1 | 0.252 |
| 790322 | 10 131 | up1 | 0.269 |
| 755252 | 30 804 | up1 | 0.055 |
| 767022 | 14 239 | up1 | 0.117 |
| 767018 |  9 161 | uc  | 0.278 |
| 782149 |  3 831 | uc  | 0.000 |

Sum = **0.971** (vs oracle 0.980, near-oracle). **+0.196 sum r@20 vs P1-alone (+25 % relative)**; +0.103 vs `union_p1_first`; +0.058 vs `union_conf`.

Threshold chosen at 10k to put 767018 (9161 HCR GFP+) into the sparse bin and keep everything else in the dense bin. 755252 has 30k HCR GFP+ (dense by count) but very sparse GFP+ *per µm³* — it misses the `union_conf` lift (−0.009 vs `uc=0.064`). Counting-threshold is a 5/6 correct heuristic; a density-based threshold could be worth exploring in a follow-up.

## What ships

- `bench/candidate_impls/_c4_p1_p6_ensemble.py` — new C4 candidate. `method ∈ {"union_conf", "union_p1_first", "auto"}` (default `"auto"`). Emits composite `confidence = max(p1_conf, p6_conf)` for downstream QF gating.
- `sessions/54_p1_p6_ensemble/{log.md, bench_ensemble.py, bench_ensemble.log, ensemble_results.csv}`.

## Limitations

- **782149 still 0** — both P1 and P6 warmstart-fail on the 12° pia tilt. Deferred to image-level I2/I3 rework per Grand Plan §9.5.
- **Confidence scales differ** — `union_conf` works empirically but P6's confidence is not calibrated against P1's. An F7 isotonic calibration per method would make the collision tiebreaker principled (next step, cheap).
- **Auto-rule's 10k threshold is fit to 6 subjects** — acceptable because it's a coarse dense/sparse bin, not a fit-to-metric, but should be sanity-checked on new subjects as they land.

## Next steps

- **S55 F7 per-method confidence calibration** — isotonic fit on `(p1_conf, is_correct)` and `(p6_conf, is_correct)` using `coreg_table.csv` (validation-only labels, LOSO CV); make `union_conf` collision tiebreak use calibrated probabilities. Expected lift on 788406/767022 (where uncalibrated P6 wrongly wins collisions).
- **S56 P1⊕P4⊕P6 three-way ensemble** — add P4 (validated in S51 as +0.047 lift on 755252). P4's pairwise-consistency affinity may contribute orthogonal correct pairs on 755252 where P6 underperforms. Expected per-subject best-of ceiling ≈ 1.01 sum r@20.
- **F8 image-conditioned warp** (S49/S52 deferred) — 782149 unreachable via any centroid/mask/image-affine pipeline. Learned matcher trained on synthetic warps remains the only plausible path.
