# Subgoal 04/R1/02 — GFP+ threshold for intensity-only subjects

## Current state (after subgoal 01)

Subgoal 01 landed `gfp_threshold_method='yen_log'` as the default, applied
to **per-cell 488 spot counts**. It is only used when a spot CSV exists
(spot subjects: 788406, 790322, 767018, 782149).

Intensity-only subjects 755252 and 767022 are still loaded via the
`cell_data_mean_{subj}_R1.csv` fallback in `benchmark_data_loader._load_gfp()`
and have **no threshold applied at all** — every HCR cell is returned as
"GFP+" (n=77785 and 72213). This is now the dominant source of non-uniformity
in the benchmark: spot subjects have 10–16 % GFP+ (Yen), intensity subjects
have 100 %.

## Goal

Pick a distribution-driven threshold for intensity subjects that mirrors
subgoal 01's philosophy:

1. **Distribution-only** — no fixed percentile / fraction (future subjects
   have no coreg table to tune against).
2. **Maximise the threshold** among methods that pass the coverage bar.
3. **Coverage bar**: retain ≥ 95 % of coreg-table HCR IDs (validation-only).

## Data characterisation (scouted from this thread)

`cell_data_mean_{subj}_R1.csv` has columns
`[channel, cell_id, sum, count, mean, background]` with channel ∈ {405, 488,
514, 594}. Channel 488 is the GFP proxy.

Raw 488 `mean` percentiles:

| subject | p01 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| 755252 | 95 | 101 | 109 | 122 | 143 | 226 | 434 | 1262 |
| 767022 | 93 | 96 | 98 | 104 | 119 | 198 | 481 | 1442 |

Coreg-cell 488 `mean` (p5):

| subject | coreg p5 | coreg median |
|---|---|---|
| 755252 | 119 | 516 |
| 767022 | 123 | 585 |

Key observation: the 25–50th percentile of *all* cells (the
autofluorescence / unlabelled population) sits ~20–30 ADU below the 5th
percentile of coreg cells. The coreg-cell distribution is heavy-tailed,
log-normally shaped, separated from the bulk noise floor — similar in
character to the spot data.

## Scouting result — Yen alone doesn't work

Yen on `log(mean)` applied directly:

| subject | threshold | gfp_frac | coreg_coverage |
|---|---|---|---|
| 755252 | 233 | 9.6 % | **0.803** ✗ |
| 767022 | 186 | 10.4 % | **0.908** ✗ |

Both miss the 0.95 bar. Yen on intensity lands too aggressively in the
tail because the autofluorescence bulk has a heavier per-cell scatter
than the spot data's zero-count bulk.

## Candidate strategies to evaluate

Same distribution-driven family as subgoal 01, applied to `log(mean)`,
plus a few intensity-specific options:

| ID | Description |
|---|---|
| A | 2-comp GMM on `log(mean)` — should find autofluorescence vs signal modes |
| A3 | 3-comp GMM on `log(mean)` — admits a broader noise distribution |
| B | Otsu on `log(mean)` |
| Yen | Yen max-entropy on `log(mean)` — disqualified by scouting (see above) but keep for comparison |
| Isodata | Ridler–Calvard iterative mean on `log(mean)` |
| Triangle | Zack's triangle on `log(mean)` |
| **Bg** | Use `mean - background` (per-cell local background already provided in the CSV) as the feature; re-run Yen / Otsu / Isodata on `log(max(mean-background, ε))`. This directly targets the autofluorescence pedestal. |
| **MAD** | Threshold at `median(log mean) + k * MAD(log mean)` for `k ∈ {3, 5}` — a conservative outlier-detection baseline. Not the winner candidate, but useful as a lower-bound sanity check. |

The **Bg** family is the one to beat — it is the most direct analogue of
the spot-bearing cells' implicit zero-count floor. If `mean - background`
has a clean bimodal distribution, Otsu / Yen / GMM should all land on the
same threshold.

## Validation

Same as subgoal 01:

- Primary: `coreg_coverage` (must be ≥ 0.95 on both subjects).
- Secondary: `gfp_frac` (informational; no target — higher threshold
  preferred at matched coverage).
- Cross-subject stability: per-method `threshold_min` across the two
  subjects.

## Stopping condition

Exactly the subgoal-01 rule, adapted:

- A strategy *passes* iff `coreg_coverage ≥ 0.95` on both 755252 and
  767022.
- Among passers, the winner is the one with the highest
  `threshold_min` across subjects (tie-break by `threshold_mean`).
- Baseline ("no threshold") is a reference only, never the winner.

If nothing passes with raw `mean`, re-run with `mean - background` before
relaxing the coverage bar.

## Deliverables

1. `dev_code/04_r1_subgoal_02_intensity_threshold.py` — mirrors
   `04_r1_subgoal_01_gfp_threshold.py` in structure (per-subject
   analyser, strategy table, winner picker, figure output).
2. `sessions/04_R1_coarse_align/subgoal_02_intensity_threshold_results.json`
   — per-subject × per-strategy coverage + threshold.
3. `notebooks/04_R1_subgoal_02_intensity_threshold.ipynb` — narration +
   figures.
4. Loader extension: add an intensity-side method (e.g.
   `gfp_intensity_method='yen_log_bg'`) and wire the winning approach as
   default for the intensity fallback branch. Keep `None` / "no
   threshold" available as a legacy option.
5. Re-run `dev_code/04_r1_benchmark.py` and record the before/after
   origin error for 755252 and 767022.

## Risks / failure modes

- **Bimodality may be weak in raw `mean`.** Spot subjects had an implicit
  zero floor; intensity subjects have a distribution of autofluorescence.
  If no distribution method clears 0.95 coverage even on
  `mean - background`, consider whether `sum` or a SNR-like feature
  (`(mean - background) / sigma_bg`) separates better.
- **R1 origin error may swing.** R1's XY estimator is still `mean(HCR
  GFP+ centroid)`. On these subjects the "GFP+" set today is 100 % of
  HCR cells, so origin is the whole-cloud centroid. A tighter GFP+ set
  will shift the centroid — direction unknown. Document, do not tune
  against.
- **The coreg table bar may not be a hard bound.** For the spot subjects,
  Yen gave 0.96–1.00 coverage; for intensity, 0.803–0.908 on raw `mean`.
  If `mean - background` only recovers to ~0.92, we may need to either
  accept a lower bar for intensity subjects or pick a different feature.
  Document the trade-off explicitly before changing the bar.

## Handoff notes

- Load intensity data via `load_subject(sid).hcr_gfp_df` after setting
  `_load_gfp` source='intensity_r1'; the `mean` column is what the
  current code thresholds (trivially, at the moment).
- Keep the spot-side threshold (`yen_log`) untouched — this subgoal
  introduces a **parallel** method for the intensity path, selected by
  the `gfp_source` routing inside `_load_gfp`.
- Do not add a fixed-percentile / fixed-fraction method as a default
  (the same generalisability constraint applies as for subgoal 01).
