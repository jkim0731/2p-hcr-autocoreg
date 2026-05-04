---
name: S11 stage-3 v2 (image-quality + directional neighbor features)
description: v7 image-quality + rim-share directional neighbor aggregates baked in; binary AUC 0.943 (v5 0.945, -0.003), 4-class f1_macro 0.740 (v5 0.737, +0.003) — features pick up signal but mean regressed slightly.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
Stage-3 v2 lands 2026-05-01 on top of stage-2 v5, baking in two user
observations:

1. **Neighbor-class coupling** — "neighbor merged → host more likely bad_ok
   (donated soma); neighbor good → host more likely good at that boundary."
2. **Image-quality contrast** — intra-soma + boundary contrast as upstream
   cause of `bad_ok` calls.

**New feature families:**

- **v7 image-quality** (`code/dev_code/roi_v7_features.py`, 19 features):
  intra-soma stats over `mask_opened` (mean/sd/p10/p50/p90/p90-p10/CV/IQR
  over median), 1-vox inside-shell vs outside-shell intensity diff/ratio, and
  Sobel boundary gradient mean/p50/p90 — all on c405. Extraction
  `roi_quality_v7.py`; cache `*_features_v7.parquet`; 11 min wall (6 workers).

- **Rich neighbor aggregates** in `06b_train_stage3_iter_v2.py`:
  - `nbr_w_mean_p_*` — rim-share-weighted mean of neighbor stage-2 probas.
  - `host_rim_frac_with_p_{cls}_ge_{0.5,0.7}` — fraction of host's rim that
    contacts neighbors with p_class above threshold (4 classes × 2 thrs = 8).
  - `top_nbr_p_*` — the host's largest-rim-share neighbor's stage-2 probas
    (4 + binary) plus `top_nbr_rim_share`.
  - `total_other_rim_share` — fraction of host's rim contacting any other
    ROI (vs background).

**LOSO results (1274 labels, 6-subj):**
| | Stage-2 v5 | Stage-3 v2 | Δ |
|--|--|--|--|
| binary AUC | 0.9453 | 0.9427 | **-0.0026** |
| binary acc@0.5 | 0.870 | 0.860 | **-0.010** |
| 4-class f1_macro | 0.737 | 0.740 | **+0.003** |
| 4-class acc | 0.749 | 0.753 | **+0.004** |

Net: binary slightly regressed, 4-class slightly improved. Not a clear win
either way. v5 remains the binary production cut.

**Per-fold binary AUC** (vs v5):
755252 +0.005, 767018 -0.006, 767022 -0.012, 782149 -0.004, 788406 -0.009,
790322 +0.011. The two large negative folds (767022, 788406) early-stopped at
5 and 42 iters → tight overfit signal.

**Importance highlights — features ARE informative:**

Binary top-30:
- `intra_soma_iqr_over_median_405` rank **#8** (gain 91.6) — v7 IQ feature.
- `host_rim_frac_with_p_good_ge_0.5` rank **#14** — directional neighbor.
- `c405_inside_minus_outside_p90` rank **#17** — v7 IQ feature.
- `top_nbr_p_bad` rank **#27** — top-offender feature.
- The four stage-2 probas (`p_binary_pos`, `p_bad/good/merged/bad_ok`)
  occupy ranks 1-5 and dominate gain (5152 + 1149 = top features).

4-class top-30:
- `intra_soma_iqr_over_median_405` rank **#14** (gain 180.2).
- `c405_inside_minus_outside_p50` rank **#30**.
- No directional neighbor feature in top-30.

**Diagnosis:** stage-2 v5's own probas already encode most of what cell-level
features (own + neighbor) can say. With ~1000 training cells/fold, adding 70+
more features past stage-2's already-good probas causes mild overfit on some
folds. The 782149 bad_ok problem persists (f1_bad_ok = 0.22).

**How to apply:** stage-2 v5 remains the production cut. Stage-3 v2 artefacts
exist (`cached_roi_quality/*_stage3_*_v2.parquet`,
`roi_quality_stage3_{binary,4class}_v2.txt`,
`roi_quality_stage3_meta_v2.json`) but are not promoted. Useful as evidence
that v7 image-quality features carry real signal — could reincorporate them
into stage-2 v6 (treat as plain features without the neighbor stack). Next
lever for real lift is targeted relabelling of 782149 bad_ok cells, not
further stage-3 iteration.
