# Co-registration Pipeline: Full Development Plan

## Context
The existing step_1→step_4 notebooks implement a **manual** workflow: a human places initial landmarks in BigWarp, auto-candidates are proposed per iteration, and the human QC's each iteration visually. Goal: make this **entirely automatic**, replacing all manual steps, with one final human QC on the output metrics table only.

6 completed manual co-registrations in `data/*_ctl-czstack-hcr-coreg_*` serve as ground-truth training data.

---

## Problem Definition

- **P**: ~600 GCaMP+ cell centroids, czstack pixel space (512×512×450 px, 0.78 µm/px XY, 1 µm/px Z)
- **Q**: ~1000–5000 GFP+ HCR cell centroids (thresholded by spot_488_counts), HCR pixel space
- **Unknown rigid transform**: ~180° rotation around Z axis + small tilts (pitch/roll) + unknown translation
- **Non-rigid deformation**: spatially local and variable; tissue shrinkage ex vivo produces anisotropic, position-dependent distortion (not simply a global Z compression)
- **~75% overlap**: some czstack cells have no GFP match; most HCR GFP+ cells are outside the czstack volume
- **Weak local distinctiveness**: cell patterns are recognizable with effort but not strongly unique

---

## Pipeline Overview

```
step_1_process_files.ipynb               [EXISTS, minor update]  czstack centroids + filepaths JSON
dev_step_0_training_analysis.ipynb       [NEW, one-time]         transform template + classifier training
dev_step_2_initial_alignment.ipynb       [NEW]                   rigid P → Q  (replaces manual BigWarp seeding)
dev_step_3_auto_matching.ipynb           [NEW]                   auto iterative TPS + verified matching
step_4_generate_coreg_table.ipynb        [minor update]          final coreg_table.csv
dev_step_5_qc_metrics.ipynb             [NEW]                   QC metrics + plots for final human review

New utility modules:
  coreg_data_loading.py    standardized loaders (filepaths, centroids, scales, landmarks, spot counts)
  coreg_alignment.py       rotation search, density volumes, cross-correlation, seed extraction
  coreg_verification.py   multi-scale patch NCC + constellation features + classifier
  coreg_matching.py        TPS fit, projection, iterative matching with verification gating
  coreg_metrics.py         QC metric computation for step_5
```

---

## Step 0 — `step_0_training_analysis.ipynb` (one-time, NEW)

Two goals: calibrate the rotation search (for step_2) and train the match classifier (for step_3).

### Part A: Transform Calibration

1. Load manual landmark CSV from each of 6 coreg directories
2. Fit rigid transform (SVD/Procrustes) on czstack_xyz → hcr_xyz per subject
3. Decompose rotation into Euler angles → measure deviation from 180°-Z + pitch/roll tilts
4. Compute per-subject Z scale ratio (physical units)
5. Compute translation range across subjects

**Output:** `data/coreg_transform_template.json`
```json
{
  "z_rotation_range_deg": [170, 190],
  "pitch_range_deg": [-10, 10],
  "roll_range_deg": [-10, 10],
  "z_scale_mean": 2.8,
  "z_scale_std": 0.3
}
```

### Part B: Match Classifier Training

For each of the 6 training pairs, extract labeled examples:
- **Positives**: all confirmed matched pairs from `{subject}_coreg_table.csv`
- **Negatives**: random non-matching cell pairs within 50 µm (5× more than positives)

**Feature set per candidate pair (czstack_i, hcr_j):**

| Feature group | Features | Source |
|---|---|---|
| Geometric | `distance_um`, `is_mutual_nn`, `nn_rank` | centroid math |
| Constellation (3 scales) | `constel_sim_20um`, `constel_sim_50um`, `constel_sim_100um` | relative neighbor positions |
| Image patch NCC (3 scales) | `ncc_20um`, `ncc_50um`, `ncc_100um` | GCaMP vs GFP 3D patches |
| Cell properties | `gfp_density`, `gfp_counts`, `hcr_volume_vox` | spot counts + metrics |

**Constellation similarity** (rotation-invariant): for each cell, compute sorted list of distances to its K=10 nearest already-matched neighbors → feature = correlation of the two distance vectors across modalities. Use multiple radii [20, 50, 100 µm]. If fewer than 3 neighbors exist at a given radius, mark as NaN.

**Image patch NCC**: extract aligned 3D patch (GCaMP channel from czstack TIFF vs GFP channel from HCR zarr level 3) at 3 scales: 20×20×20, 50×50×50, 100×100×50 µm. Compute normalized cross-correlation after intensity normalization.

**Multi-scale selection logic**: for each candidate, use the NaN pattern (which scales have enough neighbors) to weight features appropriately. Classifier handles NaN via median imputation or gradient boosting with native NaN support.

**Classifier**: gradient boosted trees (`sklearn.ensemble.GradientBoostingClassifier`) — handles NaN, small dataset, mixed feature types. Save model + scaler as `data/match_classifier.pkl`.

---

## Step 2 — `step_2_initial_alignment.ipynb` (NEW)

**Goal:** Automatically find the rigid transform P → Q, replacing manual BigWarp initial landmark placement.

### Method A — Rotation Search + 3D Density Cross-correlation *(recommended)*

**Algorithm:**
1. Load czstack centroids + HCR GFP+ centroids, convert to µm
2. Enumerate rotation candidates using template bounds: `θ_z ∈ [170°..190°, 5° steps] × θ_x ∈ [-10°..10°, 5° steps] × θ_y ∈ [-10°..10°, 5° steps]` = 125 candidates
3. For each rotation R:
   - Apply R to czstack centroids → P_rot
   - Build 3D Gaussian density volumes for P_rot and Q (5 µm voxels)
   - 3D FFT cross-correlation → peak = translation t
   - Score = # czstack cells with a GFP+ cell within 15 µm
4. Keep top-3 candidates, refine each with Nelder-Mead on (θ_z, θ_x, θ_y, tx, ty, tz)
5. Select global best (R*, t*)
6. Extract seed landmark pairs: MNN pairs within 15 µm → ~20–50 seed landmarks

**Output:** `{subject}_{date}_landmarks_initial.csv` (active=True, BigWarp format)

### Method B — CPD (Coherent Point Drift) *(fallback)*
If Method A overlap score < 50%: run `pycpd` rigid mode initialized at R*, t* from best Method A candidate.

### Method C — RANSAC + Rotation-invariant Descriptor Matching *(independent validation)*
For each cell: compute 10-dim descriptor (neighbor counts at [10,20,40,80] µm shells + distance statistics). Match descriptors → RANSAC (~1000 iterations, 3-point minimal set).

---

## Step 3 — `step_3_auto_matching.ipynb` (NEW)

**Goal:** Iteratively find and verify all matches automatically, replacing the manual BigWarp QC loop.

### Architecture: TPS Iteration + Classifier Gating

Each iteration:
```
1. Fit TPS (3× scipy.interpolate.Rbf thin_plate) from accepted landmarks
2. Project all unmatched czstack cells → HCR pixel space
3. Filter HCR pool: remove matched; apply GFP density threshold from accepted set × filter_rate
4. Forward match: choose_max_count_nearest_neighbor(k=5, feature='density')
5. Compute match features for each candidate:
   a. Geometric: distance_um, is_mutual_nn, nn_rank
   b. Constellation similarity (3 scales)
   c. Image patch NCC (3 scales)
6. Predict match probability using trained classifier (from step_0)
7. Accept candidates with probability > threshold (e.g., 0.6)
8. Add accepted to landmark pool
9. If active_landmarks > 500: apply grid_sample_landmarks()
```

**Convergence:** Stop when new accepted < 0.5% of total czstack cells OR 0 new matches OR max_iters (default: 10).

---

## Step 4 — `step_4_generate_coreg_table.ipynb` (minor update)

- Handle both `cz{N}-hcr{M}` and `qced_cz{N}-hcr{M}` naming (backward compatibility)
- Record `iter_matched` and `n_landmarks_at_match` per pair

---

## Step 5 — `step_5_qc_metrics.ipynb` (NEW)

**Metrics per match** (for human reviewer):

| Column | Computation |
|--------|-------------|
| `distance_um` | TPS projection error in µm |
| `nn_rank` | Rank of matched HCR by distance (1 = nearest) |
| `is_mutual_nn` | Final TPS reverse check |
| `match_probability` | Classifier output (from step_3) |
| `ncc_best_scale` | Best NCC score across scales |
| `constel_sim_best_scale` | Best constellation similarity |
| `gfp_counts`, `gfp_density`, `hcr_volume_vox` | Cell properties |
| `iter_matched` | Which iteration |
| `n_landmarks_at_match` | TPS quality proxy |
| `tps_loo_residual_um` | Leave-one-out TPS prediction error |

**QC Plots → `/results/`:**
1. `distance_um` histogram per iteration
2. Scatter: `distance_um` vs `match_probability`, colored by `nn_rank`
3. Scatter: `ncc_best_scale` vs `constel_sim_best_scale`
4. Histogram: `tps_loo_residual_um`
5. Cumulative match count per iteration
6. 3D scatter: matched (colored by `match_probability`) vs unmatched czstack cells

**Output:** `{subject}_coreg_table_with_metrics.csv` → `/results/`

---

## New Utility Modules

### `coreg_data_loading.py`
```python
load_filepaths(coreg_or_save_dir, subject_id, czstack_date, iter=False) -> dict[str, Path]
load_czstack_centroids(path) -> DataFrame          # czstack_cell_id, czstack_z/y/x
load_hcr_centroids(npy_path) -> DataFrame          # hcr_cell_id, hcr_z/y/x  (from N×4 .npy)
load_hcr_scales(fused_json_path) -> dict           # scale_x/y/z in µm/pixel
load_hcr_metrics(metrics_pickle_path) -> DataFrame # volume, bbox
load_spot_counts(hcr_dir, hcr_metrics_df, fallback_data_dir, subject_id) -> DataFrame
load_landmarks(csv_path) -> DataFrame              # ids, active, czstack_x/y/z, hcr_x/y/z
```

### `coreg_alignment.py`
```python
rotation_matrix_euler(theta_z_deg, theta_x_deg, theta_y_deg) -> ndarray (3×3)
apply_rotation(centroids_um, R, center_um=None) -> ndarray
centroids_to_density_volume(centroids_um, voxel_um, bounds=None) -> ndarray
cross_correlate_translation(vol_moving, vol_fixed) -> (tx, ty, tz) in µm
score_alignment(P_rot_um, Q_um, threshold_um) -> int
rotation_search(P_um, Q_um, template) -> (R_best, t_best, score_best)
extract_seed_landmarks(P_aligned_um, czstack_df, Q_um, hcr_df, scales, threshold_um) -> DataFrame
```

### `coreg_verification.py`
```python
extract_patch(volume_or_zarr, coord_px, patch_size_um, resolution_um) -> ndarray
compute_patch_ncc(czstack_vol, hcr_zarr, cz_coord_px, hcr_coord_px,
                  czstack_res, hcr_scales, patch_size_um) -> float
compute_constellation_similarity(czstack_cell_id, hcr_cell_id,
                                  czstack_projected, hcr_centroids,
                                  accepted_matches, radius_um) -> float
extract_candidate_features(candidates_df, czstack_df, hcr_df,
                            czstack_vol, hcr_zarr_path, scales,
                            accepted_matches, patch_sizes_um) -> DataFrame
train_match_classifier(features_df, labels) -> (model, scaler)
predict_match_probability(features_df, model, scaler) -> ndarray
```

### `coreg_matching.py`
```python
fit_tps(active_landmarks, scales) -> tuple[Rbf, Rbf, Rbf]
project_czstack_to_hcr(czstack_centroids, tps_models, scales) -> ndarray
check_mutual_nn(forward_hcr_indices, projected_czstack, hcr_pool) -> ndarray[bool]
run_one_iteration(state, czstack_vol, hcr_zarr_path, classifier, scaler, params) -> (new_matches_df, updated_state)
run_auto_matching(seed_landmarks, czstack_df, hcr_df, spot_counts, scales,
                  czstack_vol, hcr_zarr_path, classifier, scaler, params) -> all_accepted_df
```

### `coreg_metrics.py`
```python
compute_nn_rank(czstack_projected, matched_hcr_idx, hcr_pool_coords) -> Series
compute_mutual_nn_final(matched_df, czstack_projected_all, hcr_all_coords) -> Series[bool]
compute_tps_loo_residuals(active_landmarks, scales) -> Series[float]
compute_match_metrics(matched_df, hcr_full_df, spot_counts, hcr_metrics,
                      active_landmarks, scales) -> DataFrame
```

---

## Development Order

1. `coreg_data_loading.py`
2. `step_0_training_analysis.ipynb` (Part A: transform calibration)
3. `coreg_alignment.py` + `step_2_initial_alignment.ipynb`
4. `coreg_verification.py` — patch NCC + constellation features
5. `step_0_training_analysis.ipynb` (Part B: classifier training)
6. `coreg_matching.py` + `step_3_auto_matching.ipynb`
7. `step_4_generate_coreg_table.ipynb` (minor update)
8. `coreg_metrics.py` + `step_5_qc_metrics.ipynb`

---

## Files

| File | Action |
|------|--------|
| `code/coreg_data_loading.py` | CREATE |
| `code/coreg_alignment.py` | CREATE |
| `code/coreg_verification.py` | CREATE |
| `code/coreg_matching.py` | CREATE |
| `code/coreg_metrics.py` | CREATE |
| `code/dev_step_0_training_analysis.ipynb` | CREATE |
| `code/dev_step_2_initial_alignment.ipynb` | CREATE |
| `code/dev_step_3_auto_matching.ipynb` | CREATE |
| `code/dev_step_5_qc_metrics.ipynb` | CREATE |
| `code/step_1_process_files.ipynb` | minor: save czstack pixel dims to filepaths JSON |
| `code/step_4_generate_coreg_table.ipynb` | minor: handle both naming conventions; record iter_matched |
| `code/manual_coreg_utils.py` | no changes |
| `code/landmark_filtering.py` | no changes |
| `data/coreg_transform_template.json` | CREATE (output of step_0) |
| `data/match_classifier.pkl` | CREATE (output of step_0) |

---

## Verification

1. **Step 0 Part A**: Euler angles across 6 pairs confirm ~180° Z with small tilts; Z scale consistent (~2.5–3.0)
2. **Step 0 Part B**: Classifier AUC > 0.85 on leave-one-subject-out cross-validation
3. **Step 2**: On 790322 → overlap score of best rotation > 70% within 15 µm; seed landmarks visually close to manual `790322_2025-07-10_landmarks.csv`
4. **Step 3**: On 790322 → final match rate approaches 76.65% from manual pipeline; `match_probability` > 0.8 for most accepted matches
5. **Step 5**: `tps_loo_residual_um` larger in Z than XY; `match_probability` distribution bimodal (clearly accepted vs rejected)
6. **Cross-subject generalization**: Run full pipeline on 788406 (no initial landmarks file), using template + classifier from remaining 5 subjects
