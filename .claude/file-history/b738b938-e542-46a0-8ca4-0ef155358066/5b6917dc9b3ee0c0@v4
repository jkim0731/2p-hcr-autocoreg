---
name: S11 v5 ROI-quality features
description: Protrusion-touches-other + within-subject percentile features; v5 binary AUC 0.945 (v4 0.943); 4-class f1_macro 0.737 (v4 0.744); label set grew 1011→1274.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
S11 v5 features land 2026-04-30 on top of v4 with the labelled set expanded
from 1011 → 1274 active labels (acc=0.755 → 0.749 4-class, AUC 0.943 → 0.945
binary).

**Two new feature families:**
1. **Protrusion-touches-other-ROI** (`code/dev_code/roi_v5_features.py`):
   protrusion = mask_raw & ~mask_opened; dilate 1 vox (cross), inspect
   seg labels of dilation rim ≠ {0, hcr_id}. Outputs:
   `n_protrusion_voxels`, `protrusion_voxel_frac`, `protrusion_rim_voxels`,
   `protrusion_rim_other_frac`, `protrusion_rim_bg_frac`,
   `protrusion_top_neighbor_frac`, `n_distinct_neighbor_ids_at_protrusion`,
   `protrusion_touches_other` (0/1), `protrusion_into_neighbor_frac`.
   Extraction reads cached masks + seg_orig per strip; 7 min wall (6 workers).
2. **Within-subject percentile** (computed at training time in
   `05e_train_stage2_v5.py::_add_within_subject_pct`): rank-pct of
   `volume_um3_opened`, `volume_vox_raw`, `axis3d_lambda1_um2`,
   `surface_area_um2_opened` within each subject's full ROI population.

**LOSO results (1274 labels, 6-subj):**
- binary AUC 0.945 (v4 0.943); AP 0.966; **acc@0.5 0.870** (v4 0.858, +0.012)
- 4-class f1_macro 0.737 (v4 0.744, slight regression)

**Importance highlights (4-class):**
- `axis3d_lambda1_um2_pct_subj` rank 3 (within-subject pct is HIGHLY informative)
- `protrusion_voxel_frac` rank 10, `n_protrusion_voxels` rank 12
- `sphericity_opened` still rank 1, `solidity_opened` rank 2

**Why marginal 4-class regression:** v2 already had `top_neighbor_overlap_frac`
and `surface_touching_frac` capturing similar adjacency signal; protrusion
features only sharpen on the merged-vs-others boundary. 782149 fold remains
the f1_bad_ok bottleneck (≈0.28) — small eval n, model under-calls.

**Redundancy audit (vs `volume_um3_opened`, Spearman ρ pooled across 542k ROIs):**
DROP candidates (ρ ≥ 0.95 AND 4-class importance rank ≥ 76):
volume_vox_raw, volume_vox_opened, volume_um3_raw, volume_um3_opened,
equivalent_diameter_um_raw, surface_area_um2_raw, surface_area_um2_opened.
KEEP (high importance despite high ρ): sa_to_vol_um_inv_*, axis3d_lambda1_um2.
KEEP (low ρ, high importance): sphericity_*, protrusion_*, within-subject pct.

**How to apply:** stage-2 production now ships v5
(`roi_quality_stage2_binary_v5.txt`, `roi_quality_stage2_4class_v5.txt`).
Trainer at `code/sessions/v3_S11_roi_quality/05e_train_stage2_v5.py`;
helpers `code/dev_code/roi_v5_features.py` + `roi_quality_v5.py`. Per-subject
oof at `cached_roi_quality/{sid}_stage2_*_v5.parquet` — these are the inputs
for any future stage-3 iterative-prediction model.

**v5c expansion-rate test (drop all 31 absolute-µm features, keep `*_pct_subj`
+ unitless ratios + voxel-unit cols):** binary AUC 0.937 (Δ −0.008), acc@0.5
0.847 (Δ −0.023); 4-class f1_macro 0.730 (Δ −0.008). Worst-hit fold = 782149
binary acc 0.864 → 0.756 (Δ −0.108). Trainer at `05g_train_stage2_v5c.py`;
artefacts under `cached_roi_quality/*_v5c.*` and `outputs/stage2/*_v5c.csv`.

**v5d follow-up (drop µm + add 28 voxel-companion features):** isolates
"per-sample expansion drift" from "lost feature signal" by recomputing the 13
no-companion µm features (surface_area, sa_to_vol, c405 4-vox shell, axis3d
extent/lambdas, proj extents/FWHM, n_neighbors_30vox, bbox extents,
equivalent_diameter) with `(vz, vy, vx) = (1, 1, 1)`. Voxel-companion
extractor `code/dev_code/roi_v6_voxel.py`; cache `*_features_v6_vox.parquet`
(~14 min/subj at level-2). Trainer `05h_train_stage2_v5d.py`.
LOSO results: binary AUC 0.943 (recovers ~75% of v5c gap vs v5), acc@0.5
0.860 (recovers ~50%), 4-class f1_macro 0.733 (recovers ~50%). 782149 binary
acc fully recovers: 0.756 → 0.858 ≈ v5's 0.864. Voxel-companion features rank
in top-25 importance: `sa_to_vol_vox_inv_*`, `c405_shell_minus_core4vox_p50`,
`c405_shell_over_core4vox_p50_ratio`, `axis3d_lambda1_vox2`,
`proj_xy_orth_fwhm_vox`. **Conclusion: per-sample expansion drift is ≤ 0.005
AUC; the v5c gap was mostly LOST FEATURE SIGNAL, not scale drift.** v5
remains the production cut; v5d is a viable alternative if downstream subjects
have larger expansion-factor variance than the 6-subject benchmark.
