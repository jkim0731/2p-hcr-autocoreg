# Subgoal 04/R1/01 — Redefine GFP+ threshold

## Current state

`benchmark_data_loader._load_gfp()` applies `counts >= 5` as a hard cutoff
(`DEFAULT_GFP_MIN_SPOTS = 5`). That sits at roughly the 25th percentile of
spot-bearing cells → **54–68 % GFP+** on the four spot-data subjects
(788406, 790322, 767018, 782149). Far higher than the biological
expectation.

The manual pipeline (`step_3_more_iterations.ipynb`) uses a *relative*
threshold `min_counts = matched_cells.counts.min() * 1.2`, bootstrapped
from already-matched landmark cells. Circular for R1 — we have no matches
yet — so we cannot reuse it directly.

## Target

GFP+ fraction **~ 20 %** of all HCR cells on the four spot-data subjects,
while staying biologically coherent (pia-enriched spatial distribution,
stable threshold across subjects).

## Scope

- **In scope:** the four spot-data subjects (788406, 790322, 767018, 782149).
- **Out of scope (for now):** intensity-only subjects 755252 and 767022
  — handled in a follow-up subgoal.

## Candidate thresholding strategies

| # | Strategy | Intuition |
|---|---|---|
| A | Log-normal / GMM on `log(counts)` | Background (noise) and real GFP populations typically separate log-bimodally. Fit 2-component GMM, threshold at the crossover. Principled, no magic number. |
| B | Otsu on `log(counts)` histogram | Classical binary between-class-variance maximiser. Assumption-free; works when two modes exist. |
| C | Knee of sorted spot counts | Plot sorted counts descending; pick the knee (max curvature / Kneedle). Survives non-Gaussian tails. |
| D | Kittler–Illingworth min-error threshold | Like Otsu but optimal under Gaussian-mixture assumption — cross-check against A/B. |
| E | Fixed-fraction calibration (e.g. top 15 %) | Trivially hits the budget but not principled on its own. Useful as a sanity baseline. |
| F | `density`-column threshold | Manual workflow supports `feature='density'` as an alternative signal; compare against counts-only. |
| G | Joint counts **and** density | Require both `counts ≥ t_c` and `density ≥ t_d` (thresholds from A/B/C applied to each). Could tighten A/B/C further. |

## Validation metrics (per strategy, per subject)

1. **`gfp_frac`** = n_gfp / n_hcr_total  *(primary; target < 20 %)*
2. **`gfp_frac_top400`** — share of GFP+ cells at depth 0–400 µm
   *(should be enriched if GFP marks superficial-layer inhibitory neurons)*.
3. **`depth_cdf_curve`** — empirical depth CDF of GFP+ vs all HCR; GFP+
   should shift pia-ward.
4. **`cross_subject_std`** — std of derived thresholds across subjects;
   lower = more stable / less noise-driven.
5. **`landmark_coverage`** — fraction of active manual-landmark HCR cells
   that pass the threshold. Manual landmarks should be GFP+ by
   construction; a good threshold keeps **≥ 95 %** of them.
6. Qualitative plots: log-count histogram with threshold line overlaid;
   depth histogram of GFP+; XY scatter of GFP+ per subject.

## Deliverables

1. **`notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb`** — loads
   the four spot-data subjects, runs strategies A–G, tabulates the
   validation metrics above, and selects a winner.
2. **Recommendation write-up** — update `DEFAULT_GFP_MIN_SPOTS`, or
   introduce a distribution-based API
   (`gfp_threshold_method: "gmm" | "otsu" | "fixed" | ...`).
3. **R1 re-validation** — rerun `dev_code/04_r1_benchmark.py` with the
   new GFP+ definition and report the change in origin error per
   subject (expected: 782149 benefits most, since over-broad GFP+
   currently drags its centroid).

## Ambiguity to resolve

The user said *"tissue is from visual cortex top ~400 µm"*, but HCR
depths in the benchmark data run to ~1100–1300 µm. This plan treats
"top 400 µm" as an **enrichment target** (most GFP+ expected there), not
a hard tissue bound. Confirm or correct before we commit.
- Resolved: ~400 um is in czstack coordinate, and HCR data is expanded. Just use all data.

## Recommendation for notebook order

- Start with **A (GMM on log-counts)** and **B (Otsu on log-counts)** —
  the principled, constant-free candidates.
- Use **E (fixed-fraction @ 15 %)** as a baseline reference.
- **C / D** as sanity checks against A/B.
- **F / G** as refinements if A/B still overshoot 20 %.

## Stopping condition

At least one strategy achieves, on **all four** spot-data subjects
simultaneously:
- `gfp_frac < 30 %`,
- `landmark_coverage ≥ 95 %`,
- `cross_subject_std` no worse than the fixed-5 baseline.

If no strategy satisfies all four, fall back to the best-of-class and
document the residual gap; escalate the intensity-subjects subgoal.
