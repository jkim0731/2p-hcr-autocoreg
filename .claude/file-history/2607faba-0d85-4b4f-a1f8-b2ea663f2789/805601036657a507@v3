---
name: Session 13 pairwise-unmixing GFP+ via 07c BIC GMM — DOES NOT HELP
description: Pairwise-unmixing R*-488-GFP counts on 4 subjects (755252, 767022, 782149, 788406) tested by two metrics — depth-density CV (07c) and GT-matched-cell coverage (04/R1/01) — both lose to the existing `*_spot_488_counts.csv` density / `cell_data_mean_*_R1.csv` intensity baseline. Matched coverage drops 12–15 pp for the two intensity-only subjects and ~3 pp for the spot subjects; 21–28 % of GT-matched cells are entirely absent from the unmixed CSV (filtered or zero after re-attribution).
type: project
originSessionId: 0eccb3d7-adef-4802-b154-c928aa0e93fe
---
**What was tested.** Four pairwise-unmixing datasets
(`HCR_<sid>_pairwise-unmixing_2026-04-29_*`) provide per-cell unmixed
spot counts in `pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv`
under column `R{1,2}-488-GFP` (R1 for 782149/788406, R2 for the older
panel on 755252/767022). `cell_id` matches HCR `hcr_id`. Coverage of
the unmixed CSV is partial (47–66 % of total HCR cells per subject) —
unmixing QC drops the rest.

**Two diagnostics, same conclusion.**

**(1) Depth-density CV (07c gate).** Apply 07b's `fit_gmm_sweep`
K∈[2,6] BIC pipeline to log of unmixed counts (and to count/volume),
then 07c's GT-Procrustes `gt_density_gate` on the strict GFP+ set:

|              | 755252 | 767022 | 782149 | 788406 |
|--------------|-------:|-------:|-------:|-------:|
| baseline (density / intensity) CV | 0.??* | 0.??* | **0.332** | **0.226** |
| unmixed count CV      | 0.394 | 0.342 | 0.544 | 0.370 |
| unmixed count/vol CV  | n/a† | n/a† | 0.561 | 0.456 |
| 0.20 bar              | fail | fail | fail | fail |

\* Baseline intensity CV not recomputed here for 755252/767022 but the
matched-coverage diagnostic below makes the verdict clear.
† `cell_body_segmentation/metrics.pickle` missing for the older HCR
processings of 755252/767022, so volume could not be joined.

For 782149/788406 (direct apples-to-apples) CV gets ~+45–65 % worse
with unmixing.

**(2) GT-matched HCR coverage (04/R1/01 metric, target ≥95 %).**
Per-subject twinx histograms (figures
`figures/matched_vs_all/{sid}_{baseline,unmixed}.png`) of the BIC-GMM
threshold candidate, restricted both to (a) full HCR and (b) cells
inside the GT-Procrustes-mapped CZ overlap:

|         | feature   | matched-cells passing cutoff (full) | matched-cells passing cutoff (overlap) |
|---------|-----------|------------------------------------:|---------------------------------------:|
| 755252  | baseline (intensity) | **70 %** (448/639) | **80 %** (427/535) |
| 755252  | unmixed (count)      | 58 % (322/551) | 63 % (311/497) |
| 767022  | baseline (intensity) | **88 %** (696/793) | **95 %** (637/674) |
| 767022  | unmixed (count)      | 75 % (509/676) | 80 % (485/605) |
| 782149  | baseline (density)   | **97 %** (293/303) | **97 %** (271/278) |
| 782149  | unmixed (count)      | 97 % (218/224) | 97 % (199/205) |
| 788406  | baseline (density)   | **94 %** (742/787) | **94 %** (692/736) |
| 788406  | unmixed (count)      | 91 % (516/569) | 91 % (478/527) |

**Crucial side-effect: matched cells lost outright.** The unmixed CSV
is missing GT-matched cells entirely (or zeros them post-unmixing):

|         | matched / total | matched present in unmixed | matched dropped |
|---------|-------:|-------:|-------:|
| 755252  | 639 | 551 (+43 zeros)  | 88 (14 %) |
| 767022  | 793 | 676 (+29 zeros)  | 117 (15 %) |
| 782149  | 303 | 224 (+1 zero)    | 79 (26 %) |
| 788406  | 787 | 569 (+4 zeros)   | 218 (28 %) |

For the spot subjects 21–28 % of validated GFP+ cells are silently
killed by unmixing — denominator `n_matched_full` shrinks from 303→224
(782149) and 787→569 (788406). The 97 % / 91 % "passing" rates in the
unmixed column are computed against the *retained* matched cells only;
on the *original* matched set the coverage drops to 72 % (218/303) and
66 % (516/787).

**Why the unmixed feature loses on GMM-on-log.** Per-cell unmixed GFP
counts are integer-valued and dominated by a delta-spike at 0/1 spots
(frac_zero 0.44–0.70). On log axis this is a discrete spike at ln 1=0
plus a sparse tail; BIC happily picks K=6 to fit the spikes; bimodality
between background and GFP+ becomes far less clean than in the
continuous baseline density (`counts/volume`) or intensity histograms.

**Why the depth profile gets worse.** Unmixing demotes more cells in
dense mid-cortical layers (more crosstalk neighbours) than in sparse
deep layers, so the unmixed-GFP+ depth profile is *more* depth-biased
than the baseline (drifts toward deep cortex on 788406; shifts to
~300–500 µm on 782149). This worsens the M1-sz scale bias the
07c-era pipeline was already complaining about (~−71 %).

**Verdict.** Pairwise unmixing in its current form **does not** improve
GFP+ thresholding via BIC-GMM by either the depth-density CV or the
GT-matched-cell-coverage criterion. Adopting unmixed counts as a
drop-in replacement for `*_spot_488_counts.csv` /
`cell_data_mean_*_R1.csv` would *worsen* the threshold:
- Spot subjects: matched coverage falls 0–3 pp (or 25–31 pp if the
  dropped-by-unmixing matched cells are kept in the denominator).
- Intensity subjects: matched coverage falls 12–15 pp.
- Depth-density CV worsens by ~+45–65 % on the two subjects with a
  direct comparison.

**Possible follow-ups (NOT done):**
1. Investigate WHY the unmixing pipeline drops 14–28 % of GT-matched
   cells from `unmixed_all_cells.csv`. If those cells are
   distinguishable upstream (e.g., by `mixed_cell_by_gene_filtered.csv`
   filter), recover them with a less aggressive QC.
2. Test depth-stratified or ROI-quality-stratified unmixing thresholds
   — but this leaks the very signal we're measuring.
3. Combine baseline-density AND unmixed-density (intersection) for the
   spot subjects — keep cells flagged by both.

**Reusable artifacts:**
- `code/sessions/13_pairwise_unmix_gfp/run_compare.py` — baseline vs
  unmixed-count BIC-GMM + 07c CV diagnostic.
- `code/sessions/13_pairwise_unmix_gfp/run_unmixed_density.py` —
  unmixed-count / volume variant (GMM bimodality returns; CV still bad).
- `code/sessions/13_pairwise_unmix_gfp/run_matched_vs_all.py` —
  twinx histograms of all-HCR vs GT-matched (full + overlap-restricted),
  primary diagnostic; mirrors `04/R1/01_subgoal` figure layout.
- `code/sessions/13_pairwise_unmix_gfp/figures/matched_vs_all/{sid}_{baseline,unmixed}.png`,
  `results.json`, `results_unmixed_density.json`,
  `results_matched_vs_all.json`.

**Where:** `code/sessions/13_pairwise_unmix_gfp/`. Date: 2026-05-07.
