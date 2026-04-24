# S55 — F7 per-method confidence calibration

## Goal

From S56's finding that P1/P4/P6 confidence scales are non-comparable (quantile distributions: P1 [0.000, 0.008, 0.233], P4 [0.611, 0.842, 0.928], P6 [0.420, 0.535, 0.680]), `union_conf` collisions can exploit uncalibrated per-method offsets. S55 fits per-method isotonic calibration (F7) and measures:

1. **Brier improvement** per method (LOSO across 6 subjects).
2. **Does calibration improve C5's ensemble r@20?** — the primary production question.
3. Ships calibrators as a pickled utility for downstream use (GUI thresholding, QF1 fallback).

## Method

- For each of 6 subjects × 3 methods (P1, P4, P6), run candidate via F9.
- Label each pair as correct := `distance(pred_hcr_centroid, GT_hcr_centroid) < 20 µm` for CZ cells present in `coreg_table.csv`. Drop pairs whose CZ isn't in GT.
- **LOSO** isotonic fit (sklearn `IsotonicRegression(out_of_bounds="clip")`): fit on 5 subjects, evaluate Brier on held-out.
- **Final** fit: all 6 subjects combined; pickle to `lib/calibrators/{p1,p4,p6}.pkl`.
- **Strategy sweep**: compare 8 merge strategies on 6 subjects.

## LOSO Brier

| Method | base rate | hold=788406 | hold=790322 | hold=755252 | hold=767022 | hold=767018 | hold=782149 |
|--------|-----------|-------------|-------------|-------------|-------------|-------------|-------------|
| P1     | 0.206 | 0.255 → 0.207 (−0.048) | 0.322 → 0.267 (−0.056) | 0.082 → 0.118 (+0.036) | 0.135 → 0.122 (−0.013) | 0.172 → 0.149 (−0.023) | SKIP (all neg) |
| **P4** | 0.150 | **0.541 → 0.151 (−0.390)** | **0.511 → 0.179 (−0.332)** | **0.635 → 0.103 (−0.532)** | **0.583 → 0.072 (−0.511)** | **0.469 → 0.238 (−0.230)** | SKIP (all neg) |
| P6     | 0.185 | 0.278 → 0.182 (−0.096) | 0.263 → 0.275 (+0.012) | 0.301 → 0.084 (−0.217) | 0.338 → 0.100 (−0.238) | 0.258 → 0.242 (−0.017) | SKIP (all neg) |

- **P4 is the big win** — raw Brier 0.47-0.64 drops to 0.07-0.24, because P4's raw scores are *uniformly high* (tight [0.61, 0.93] distribution) regardless of correctness. Isotonic essentially reassigns them to near the base rate.
- **P1 mostly improved** (4/5 positive, 755252 regresses because its correct-pair distribution is a thin tail of the ~0.99 conf values and isotonic over-flattens it).
- **P6 mostly improved** (3/5 positive, 790322 nearly flat, 767018 nearly flat).

## Strategy sweep: does calibration help the ensemble?

| Strategy | 788406 | 790322 | 755252 | 767022 | 767018 | 782149 | **sum r@20** |
|----------|--------|--------|--------|--------|--------|--------|----------|
| raw_up_P1_P4_P6 | 0.263 | 0.289 | 0.066 | 0.117 | 0.194 | 0 | 0.929 |
| raw_up_P4_P1_P6 | 0.197 | 0.224 | 0.095 | 0.079 | 0.308 | 0 | 0.903 |
| raw_up_P6_P1_P4 | 0.241 | 0.276 | 0.070 | 0.092 | 0.289 | 0 | 0.970 |
| raw_uc3         | 0.203 | 0.230 | 0.089 | 0.083 | 0.315 | 0 | 0.921 |
| cal_uc3         | 0.264 | 0.287 | 0.077 | 0.110 | 0.271 | 0 | 1.008 |
| **C5_auto** (current) | **0.263** | **0.289** | **0.095** | **0.117** | **0.315** | **0** | **1.080** |
| C5_auto_cal     | 0.263 | 0.289 | 0.095 | 0.117 | 0.271 | 0 | 1.036 |
| cal_uc3_mid_up  | 0.263 | 0.289 | 0.077 | 0.117 | 0.271 | 0 | 1.017 |
| **best_per_subject (oracle)** | 0.264 | 0.289 | 0.095 | 0.117 | 0.315 | 0 | **1.081** |

### Key findings

#### F1. C5_auto (1.080) is within 0.001 of the oracle (1.081)

The current C5 dispatch rule is essentially optimal for the 6-subject benchmark. Oracle-best only gains +0.001 on 788406 by substituting cal_uc3 (0.264) for up_P1_P4_P6 (0.263) — noise-level.

#### F2. Calibration uniformly (cal_uc3) lifts raw_uc3 from 0.921 → 1.008 (+0.088)

Confirming calibration works as intended on the uniform merge path: when all 6 subjects use 3-way confidence-sort, calibration recovers most of the lost performance (below C5-auto 1.080 but above raw_uc3 0.921). The missing 0.072 comes from the per-density dispatch rule, which calibration can't reproduce.

#### F3. C5_auto_cal regresses C5_auto on 767018 (0.315 → 0.271, −0.044)

Replacing the sparse-branch's raw uc3 with calibrated uc3 loses points on 767018 — a single-subject regime (sparse 9k HCR GFP+, n_cz=785 with only 267 labeled CZ cells in GT) where raw uc3 happens to pick up P6's high-confidence-mostly-wrong pairs *and* those pairs happen to be correct on this subject. Calibration correctly flattens P6's overconfidence in the training average but hurts this subject's specific regime.

#### F4. Calibration is a Brier win but not an r@20 win

- P4 Brier: 0.47-0.64 → 0.07-0.24 (2-7× improvement).
- P6 Brier: 0.26-0.34 → 0.08-0.27 (2-3× improvement on most, small regression on 2 subjects).
- P1 Brier: 0.08-0.32 → 0.12-0.27 (small improvements).
- r@20 impact on C5: **zero or slightly negative.** The per-density priority rule in C5 does most of the work; confidence-based tiebreak within union is only used on sparse subjects (767018, 782149), and on 767018 the calibrated tiebreak prefers the wrong regime-specific ordering.

## Ship decision

1. **Keep C5 default as-is** (no `calibrate` kwarg plumbed through). The 1.080 sum r@20 is within 0.001 of oracle; no reason to introduce an option that can't improve it.
2. **Ship calibrators as a library utility** — `lib/calibrators/{p1,p4,p6}.pkl` with a `load_calibrator(method_id)` helper in `lib/calibrators/__init__.py`. Usable by:
   - GUI thresholding (G1-review): set P@95 precision threshold on calibrated probabilities rather than raw confidence.
   - QF1 fallback classifier: use calibrated probabilities as a feature.
   - Any downstream code that needs absolute-probability semantics.
3. **Do NOT re-run S53 P6 or S47 P1 benchmark with calibration** — uncalibrated confidence is what C5 uses; calibration is for reporting, not ensemble logic.

## Files

- `lib/calibrators/__init__.py` — loader helper.
- `lib/calibrators/p1.pkl`, `p4.pkl`, `p6.pkl` — fitted `CalibrationResult` objects.
- `sessions/55_f7_calibration/fit_calibration.py` — LOSO + final fit + uc3 probe.
- `sessions/55_f7_calibration/strategy_sweep.py` — 8-strategy sweep across 6 subjects.
- `sessions/55_f7_calibration/labeled_pairs.csv` — per-pair (method, subject, conf, is_correct).
- `sessions/55_f7_calibration/loso_summary.csv` — per-method-per-held-out Brier.
- `sessions/55_f7_calibration/strategy_sweep.csv` — per-subject r@20 for 8 strategies.
- `sessions/55_f7_calibration/fit_calibration_output.log`, `strategy_sweep.log`.

## Follow-ups

- **G1-review threshold setting** — when the GUI ships (tier 1 per Grand Plan §9.1), wire calibrated P@95 thresholds into the accept/reject gate.
- **QF1 fallback classifier** — use calibrated probability as a per-pair feature when the fallback classifier is introduced (conditional on Grand Plan §9.6 stopping criterion).
- **782149 remains unreachable** — calibration is a confidence remap, not a matching rework. 782149 has zero correct pairs across all three methods so nothing to calibrate. Deferred to image-level I2/I3 track.
