---
name: S11 v2 stitching-seam analysis
description: After 1011 labels, model systematically downgrades ROIs in tile seams; AUC drops 0.91→0.85 in-seam; merged calls in seams less likely to be confirmed.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
S11 v2 ROI quality — refit on 1011 active labels (was 552), then in-seam vs out-of-seam comparison (2026-04-30).

**Refit metrics (binary LOSO):** mean AUC 0.924 → **0.897** (decreased: positive class share dropped from 73 % to 51.5 % as 328 confirmed `merged` were added). 4-class macro-F1 **0.552 → 0.655**, with f1_merged 0.29 → **0.83**. Top features now `solidity_opened` (2714) > `frac_kept_opening` (519) > `c405_opened_std` (275).

**Stitching seam (population, 514k HCR cells across 6 subj):** 11.4 % fall in some seam (≈8 % per axis, 0.3 % corner).

**Model behaviour in seams (all 514k cells):**
- binary score mean: corner 0.66 / x_only 0.72 / y_only 0.73 / out 0.76
- p_merged mean: in-seam 0.05–0.06 vs out 0.04 (≈50 % relative ↑)
- p_bad mean: in-seam 0.19–0.25 vs out 0.17 (≈30–50 % relative ↑)

**Per-subject AUC on labelled cells (n_in=103 / n_out=908):** pooled in-seam AUC 0.853 vs out 0.906. Per-subject in/out: 755252 0.886/0.910, 767018 0.857/0.871, 767022 **0.692**/0.922, 782149 0.960/0.879, 788406 0.750/0.936, 790322 0.906/0.903.

**Labels by seam:** in-seam 4-class shares more `bad` (18–25 % vs 12 %), about same `good`/`merged`. χ²(in_seam vs none)=5.18, p=0.16 — not significant pooled.

**Why this matters:** model's *p_merged* is inflated in seams but confirmation rate is lower (31.7 % in-seam vs 35.7 % out) — i.e. some seam-merged calls are stitching artefacts, not true 2-cell fusions.

**How to apply:** add `in_x_seam`/`in_y_seam` (or distance-to-nearest-seam) as features in v3, OR post-hoc rescale scores for seam ROIs. To validate: target additional labels in seam zones (currently only 10 % of labels). Seam-zone definition is in `code/sessions/v3_S11_roi_quality/outputs/tile_boundaries.json` (level-2 voxel intervals along x and y per subject); analysis script is `07_stitching_boundary_analysis.py`.
