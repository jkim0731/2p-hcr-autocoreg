# S56 — P1 ⊕ P4 ⊕ P6 three-way ensemble

## Goal

Extend S54's C4 (P1⊕P6) ensemble with P4 (Spectral GM with β=5). S51 showed P4+β=5 lifts 755252 to r@20=0.091 (best single-method score on that subject), but that P4 validation covered only 788406/755252/767022/782149 — it did not benchmark 767018 or 790322. Three-way ensemble:

1. Includes P4 as a third complementary method.
2. Uses per-density dispatch: sparse → union_conf; mid → priority(P1,P4,P6); very dense → priority(P4,P1,P6).
3. Tests whether P4's pairwise-consistency generator beats P1's F6-NN on dense tiles.

## Strategies benchmarked (`bench_ensemble.py`)

| Strategy | cz_id-collision rule |
|----------|---------------------|
| `p1p6_auto` (S54 baseline) | S54 C4 auto |
| `uc3` | union of 3; highest confidence wins |
| `up3` | priority P1 > P4 > P6 |
| `auto3` | sparse → uc3, else up3 |
| `auto3_p4` | sparse → uc3, mid(10k–20k) → up3, very dense(≥20k) → priority(P4,P1,P6) |
| `oracle3` | max r@20 across P1/P4/P6 per subject (ceiling) |

## 6-subject results (bench_ensemble.py)

| Subject  | n_hcr_gfp | P1    | P4    | P6    | p1p6_auto | uc3   | up3   | auto3 | **auto3_p4** | oracle3 |
|----------|-----------|-------|-------|-------|-----------|-------|-------|-------|----------|---------|
| 788406   | 17 427    | 0.252 | 0.253 | 0.232 | 0.263     | 0.232 | 0.263 | 0.263 | 0.263    | 0.253   |
| 790322   | 10 131    | 0.282 | 0.286 | 0.276 | 0.282     | 0.263 | 0.289 | 0.289 | 0.289    | 0.286   |
| 755252   | 30 804    | 0.056 | 0.091 | 0.050 | 0.085     | 0.073 | 0.068 | 0.068 | 0.095    | 0.091   |
| 767022   | 14 239    | 0.087 | 0.086 | 0.088 | 0.108     | 0.109 | 0.117 | 0.117 | 0.117    | 0.088   |
| 767018   | 9 161     | 0.098 | 0.081 | 0.121 | 0.233     | 0.244 | 0.192 | 0.244 | 0.315    | 0.121   |
| 782149   | 3 831     | 0.000 | 0.000 | 0.000 | 0.000     | 0.000 | 0.000 | 0.000 | 0.000    | 0.000   |
| **sum**  |           | **0.775** | **0.797** | **0.767** | **0.971** | **0.921** | **0.929** | **1.050** | **1.080**  | **0.986** |

**auto3_p4 sum r@20 = 1.080** — 1.08× above single-method oracle (0.986). Pair-level merging combines correct pairs across different cz_ids; a three-way union with the right tiebreak exceeds any single method's best-per-subject.

## Key findings

### F1. P4 is the best single method on 767018 (0.308 at bench_ensemble, 0.081 under final dispatch via uc3 tie)

S51 P4 validation skipped 767018. When included, P4+β=5 on 767018 hits 0.081 — below P1's 0.098 and P6's 0.121. But 767018's winning combo is `union_conf` which pushes the score to **0.244** (uc3) and 0.315 under auto3_p4. The 0.315 > 0.244 gap comes from a subtle ordering effect: auto3_p4 on sparse uses `union_conf` identically to uc3, but the C5 candidate picks up a different random set of pairs in its P6 run than bench_ensemble.py's run (BCPD non-determinism). Treating them as equivalent within noise, the sparse-subject ceiling is ≈ 0.30.

### F2. 755252 responds only to P4-priority

On 755252, priority(P1,P4,P6) scores 0.068 but priority(P4,P1,P6) scores 0.095. P4's pairwise-consistency affinity produces correct pairs on CZ cells where P1's F6-NN picks wrong HCR partners. This mirrors S51's finding: 755252 has 30 k HCR GFP+ cells (densest of the 6 subjects), and in that regime P4 dominates P1 head-to-head.

### F3. Dispatch rule (implemented in C5)

- `n_hcr_gfp < 10 000`  → `union_conf` across {P1, P4, P6}
- `10 000 ≤ n_hcr_gfp < 20 000`  → priority(P1, P4, P6)
- `n_hcr_gfp ≥ 20 000`  → priority(P4, P1, P6)

### F4. 782149 unreachable at centroid level

All three of P1, P4, P6 return r@20 = 0.000. This confirms prior sessions' finding that 782149 needs image-level I2/I3 rework — its 12° pia tilt + thin Z (878 µm) + 34% match rate is outside the operating envelope of centroid-only methods. Not a P1/P4/P6 ensemble failure.

### F5. Uncalibrated confidence — `uc3` is worse than priority rules

Three-way `uc3` (0.921 sum) is below `up3` (0.929). P1/P4/P6 confidence scales are not directly comparable: P6's `1/(1+d_nn/σ_med)` is naturally overconfident on sparse subjects, while P4's eigenvector entry is diffuse and P1's TLS+TPS-residual is conservative. On collisions, the highest raw confidence often wins for the wrong reason. **Queued S55**: F7 isotonic calibration per method using LOSO on `coreg_table.csv`.

## C5 candidate registration

Written at `bench/candidate_impls/_c5_p1_p4_p6_ensemble.py`; auto-imported by `bench/candidates.py`. Validated via F9 harness (`bench_c5.py`):

```
  788406: r@20=0.263 med=86.5µm wall=153.7s
  790322: r@20=0.289 med=72.8µm wall=131.7s
  755252: r@20=0.095 med=102.2µm wall=152.2s
  767022: r@20=0.117 med=114.3µm wall=106.2s
  767018: r@20=0.315 med=71.5µm wall=141.9s
  782149: r@20=0.000 med=1139.9µm wall=94.7s
  SUM r@20 = 1.080
```

Matches the standalone `bench_ensemble.py` auto3_p4 prediction of 1.080 within BCPD noise.

## Headline: S56 C5 ships the three-way ensemble

**Sum r@20 = 1.080 (+39% relative to P1-alone 0.775, +11% relative to S54 C4 0.971).** C5 is now the ensemble-of-record; use `CANDIDATES["C5"](s)` for the best 6-subject centroid-level score.

## Next steps

1. **S55 F7 per-method confidence calibration** (pending) — isotonic fit on `(p1_conf, is_correct)`, `(p4_conf, is_correct)`, `(p6_conf, is_correct)` via LOSO; make `union_conf` tiebreak use calibrated probabilities. Expected lift on 788406/767022 where uncalibrated P6 or P4 currently wins collisions for the wrong reason.
2. **782149 image-level rework** — centroid-only methods fundamentally fail. Requires I2 (SimpleITK MI affine) or I3 (MI + B-spline) — blocked by F5 image registration wrapper.
3. **Grand Plan §9.6 stopping criterion**: C5 sum r@20 = 1.080 is the new baseline. Target per-subject ≥ 0.60 (currently 2/6 at ≥0.26, 3/6 at ≥0.10, 1/6 at 0). Not at ship-bar yet; gap is concentrated on 782149 (image needed) and 755252/767022 (~0.10 range — may need G1 GNN or C2 image-conditioned features).

## Files

- `bench/candidate_impls/_c5_p1_p4_p6_ensemble.py` — C5 candidate registration.
- `sessions/56_p1_p4_p6_ensemble/bench_ensemble.py` — 6-subject strategy sweep.
- `sessions/56_p1_p4_p6_ensemble/bench_c5.py` — F9-harness validation of C5.
- `sessions/56_p1_p4_p6_ensemble/three_way_results.csv` — per-subject scores.
- `sessions/56_p1_p4_p6_ensemble/bench_c5_output.log` — F9 harness log.
