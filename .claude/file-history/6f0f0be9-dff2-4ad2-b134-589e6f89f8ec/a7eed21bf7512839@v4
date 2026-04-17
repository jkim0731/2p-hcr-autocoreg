# Plan: Analyze Benchmark Data & Describe Data/Coregistration Features

## Context

The project aims to automate coregistration between 2-photon (2p) GCaMP cortical z-stack volumes and HCR light-sheet volumes. Currently this is done manually using BigWarp with iterative landmark matching. Before developing automatic registration, we need to thoroughly characterize the benchmark data to understand the problem's constraints, variability, and features that an algorithm must handle.

The existing `01 Data Description.md` contains approximate values. This plan produces a concrete, quantitative characterization from all 6 benchmark subjects.

**Constraint**: Per dev protocol, benchmark data is used ONLY for understanding the problem — not for algorithm development.

---

## Deliverables

1. **Analysis script**: `/root/capsule/code/dev_code/01_analyze_benchmark.py`
2. **Summary notebook**: `/root/capsule/code/notebooks/01_benchmark_data_analysis.ipynb`
3. **Updated doc**: `/root/capsule/code/docs/01 Data Description.md`
4. **Session log**: `/root/capsule/code/sessions/01_analyze_benchmark_data/log.md`

---

## Step 1: Create directories and data loader

Create `/root/capsule/code/dev_code/` and `/root/capsule/code/notebooks/`.

Write `dev_code/benchmark_data_loader.py` with functions to load per-subject data:

- **CZStack centroids**: `data/{subj}*coreg*/*czstack_cell_centroids.csv` — cols: `[czstack_cell_id, czstack_z, czstack_y, czstack_x]` (pixel coords)
- **HCR centroids**: `data/HCR_{subj}*/cell_body_segmentation/cell_centroids.npy` — cols: `[z, y, x, id]` (pixel coords)
- **HCR GFP+ data**: `data/{subj}*coreg*/*spot_488_counts.csv` for 782149/788406/790322 (cols: `[hcr_id, counts, volume, global_bbox, density]`); `data/cell_data_mean_{subj}_R1.csv` for 755252/767022 (intensity-based); 767018 TBD
- **Landmarks**: All `*landmarks_matched_ext_iter*` CSV files. Format: `[ids, active, czstack_x, czstack_y, czstack_z, hcr_x, hcr_y, hcr_z]`. Manual landmarks start with `Pt-`, auto-matched start with `cz*-hcr*`.
- **Coreg table**: `*coreg_table.csv` — cols: `[czstack_id, hcr_id]`
- **Resolution metadata**: CZ from `session.json` (`fov_scale_factor`); HCR from `fused_ng.json` (`dimensions.x[0]` in meters)

Handle format variations:
- 767018: no date prefix in coreg dir, HCR centroids in CSV (not NPY in coreg dir), no `fused_ng.json`
- 755252/767022: GFP+ via intensity instead of spot counts
- Older subjects: `_reordered_qced` suffix on landmarks

Convert all coordinates to physical (um) using per-subject resolutions.

### Known data values to validate against:
| Subject | CZ res (um/px) | HCR XY res (um/px) | HCR Z res | CZ image | 
|---------|---------------|--------------------|-----------|---------| 
| all     | 0.78          | 0.2451-0.2474      | 1.0       | 512x512 |

---

## Step 2: Per-subject summary table

Compute and tabulate for all 6 subjects:

| Metric | Source |
|--------|--------|
| CZ cell count | czstack centroids CSV (line count - 1) |
| CZ volume extent (pixels & um) | centroid range * resolution |
| HCR total cell count | cell_centroids.npy shape[0] |
| HCR volume extent (pixels & um) | centroid range * resolution |
| HCR GFP+ cell count | spot_488_counts or intensity threshold |
| GFP+ fraction | GFP+ / HCR total |
| Coreg match count & rate | coreg_table / CZ total |
| Manual landmark count | grep `Pt-` in iter1 landmarks |
| Total active landmarks (final iter) | count `active=True` in final landmarks |

### Expected values (verified from data):
| Subject | CZ cells | HCR total | HCR GFP+ | Matched | Rate   |
|---------|----------|-----------|-----------|---------|--------|
| 755252  | 835      | 84,233    | ~intensity| 639     | 76.5%  |
| 767018  | 785      | 108,506   | TBD       | 273     | 34.7%  |
| 767022  | 926      | 76,336    | ~intensity| 793     | 85.6%  |
| 782149  | 894      | 39,291    | 29,557    | 303     | 33.8%  |
| 788406  | 932      | 127,275   | 90,815    | 787     | 84.4%  |
| 790322  | 1016     | 106,379   | 86,353    | 778     | 76.6%  |

HCR physical volume sizes:
- 755252: ~454 x 454 x 1247 um
- 767022: ~455 x 455 x 1174 um
- 782149: ~565 x 566 x 878 um (notably thin)
- 788406: ~571 x 571 x 1329 um
- 790322: ~566 x 565 x 1160 um

---

## Step 3: Cell density analysis

For each subject (in physical um coordinates). Depth profiles use the **surface-anchored depth coordinate** from Step 4b(i), not raw z.

a) **3D cell density** (cells/mm^3) for CZ and HCR  
b) **Depth profiles**: bin cells in ~20 um bins of `depth = z - z_surface(x, y)`, plot density vs depth for CZ and HCR_GFP+ overlaid, also rescaled onto a common depth axis using the Z expansion factor (~2.8×)  
c) **XY density maps**: 2D histogram projections (top-down, pia-parallel)  
d) **GFP+ fraction by depth**: discriminative signal for matching  
e) **Cell count ratio**: HCR_GFP+ / CZ_cells — characterizes search space size  

Plots:
- 6-panel depth profiles (one per subject, CZ vs HCR GFP+ overlaid)
- GFP+ fraction bar chart across subjects
- XY density comparison (CZ vs HCR, physical coords)

---

## Step 4: Coordinate system and anisotropic scaling analysis

### Key insight — resolution conventions (from step_1 and step_2 notebooks)

From `manual workflow/step_2_automatic_mapping_for_qc.ipynb` cell 6:
```python
scale_x = data['dimensions']['x'][0] * 4e6   # NOT 1e6
scale_y = data['dimensions']['y'][0] * 4e6   # NOT 1e6
scale_z = data['dimensions']['z'][0] * 1e6   # standard m->um
```

This means **HCR cell centroids in `cell_centroids.npy` are stored at level-2 pyramid pixels**, where XY has been downsampled 4× from raw. Effective resolution in the centroid pixel space:
- HCR: XY ≈ 0.988 um/pixel, Z = 1.0 um/pixel  (nearly isotropic in pixel space)
- CZ:  XY = 0.78 um/pixel, Z ≈ 1.0 um/pixel   (slightly anisotropic in pixel space)

CZ options from step_1 (`czstack_xy_size`): 400 or 700 um FOV over 512 pixels → 0.78 or 1.37 um/px. All current benchmark subjects use 400 (0.78 um/px).

**The data loader must use these conventions exactly.** Build two coordinate spaces per subject:
- `pixel` coords as stored in files (used by manual workflow for RBF/landmarks)
- `physical (um)` coords using the per-modality scales above

### Anisotropic expansion from landmarks

**Critical new finding** (verified on 788406, 604 active landmarks): converting landmarks to physical um, the ex-vivo HCR volume is **expanded anisotropically** relative to in-vivo CZ:
- XY expansion factor ≈ 1.9×
- Z expansion factor ≈ 2.8×

This anisotropy (likely from CLARITY-style clearing + axial shrinkage/expansion) has been absent from the data description and is essential context for algorithm design. Per-subject expansion factors must be measured and reported.

### Computations

a) **Per-axis expansion factor from active landmarks**:
   - Convert CZ landmarks to physical um (× 0.78, × 0.78, × 1.0)
   - Convert HCR landmarks to physical um (× 0.988, × 0.988, × 1.0)
   - After removing the ~180° XY rotation (Procrustes), fit per-axis scale: `s_x, s_y, s_z`
   - Report XY vs Z expansion ratio: `(s_x + s_y)/2 / s_z`  (expected ≠ 1)

b) **Rotation estimation**: SVD-based Procrustes on centered landmark pairs. Confirm rotation angle near 180° around Z. Report actual angle per subject.

c) **Anisotropy between modalities (in pixel space)**: document the pixel-space scaling that the RBF transform has to absorb:
   - XY pixel ratio: 0.988 / 0.78 ≈ 1.27
   - Z pixel ratio: 1.0 / 1.0 = 1.0
   - Combined with physical expansion, the effective CZ-pixel → HCR-pixel XY scale ≈ 1.9 / 1.27 ≈ 1.50, Z scale ≈ 2.8. These are what the landmark-based RBF is doing.

d) **Spatial variation of expansion**: check whether the local scale factor varies across the volume by fitting local affines in spatial subregions. This characterizes how far a global affine is from a good fit.

### Known values to verify (measured on 788406 final iter):
| Quantity | Value |
|---|---|
| HCR level-2 XY res | ≈ 0.988 um/px |
| HCR physical volume | ≈ 2.3 × 2.3 × 1.3 mm (not ~1×1×0.5!) |
| XY expansion (CZ→HCR) | ≈ 1.9× |
| Z expansion (CZ→HCR) | ≈ 2.8× |
| Anisotropy ratio (XY:Z) | ≈ 0.68 |

---

## Step 4b: Additional candidate features for automatic coregistration

These features are useful as building blocks for localization/registration algorithms. Analyze each on all 6 subjects and report feasibility + cross-subject stability.

### (i) Pia surface estimation from ROI segmentations

Both modalities "start from cortex surface" but the exact surface is not at z=0: the top is a blank buffer of unknown thickness (CZ: ~50 um nominal; HCR: variable/slanted).

Approach:
1. Filter **out-of-tissue false positives**: HCR in particular may contain scattered segmentations above the pia (in agarose/medium) or below tissue. Strategies:
   - Density threshold: require minimum local neighbor count within radius R (remove isolated cells)
   - Largest connected component in binarized density volume
   - Robust top envelope: fit plane (or low-order surface) to top 5–10% of cells per (x, y) tile, with RANSAC or IRLS to reject outliers
2. **Fit surface** z = f(x, y):
   - Lowest-order: plane (3 params; tilt)
   - Flexible: low-order polynomial or thin-plate spline
3. Report per subject: plane normal vector, tilt angle, surface roughness (residual std)
4. Compare surface tilt between CZ and HCR — HCR is known to be slanted across samples

Output features for algorithm design:
- `z_surface_cz(x, y)`, `z_surface_hcr(x, y)` — depth-from-surface coordinate
- Tilt angle distribution across subjects
- Surface-anchored z coordinate will remove a large fraction of alignment ambiguity

### (ii) Local ROI density profile along depth-from-surface

After surface estimation, reparametrize z as `depth = z - z_surface(x, y)`.

Compute depth profiles (density vs depth, bin ~20 um):
- CZ ROI density vs depth
- HCR all-cell density vs depth
- HCR GFP+ density vs depth
- Rescale by anisotropic expansion to put CZ and HCR on comparable depth axes

Hypothesis: cortical layers produce a reproducible 1D density signature. Cross-correlating CZ and HCR depth profiles provides a cheap, robust 1D localization along the axial direction.

Per-subject output:
- Depth profiles (1D arrays)
- Layer boundaries (peaks/troughs in density)
- Cross-correlation of CZ vs HCR_GFP+ depth profiles after expansion correction → optimal z-offset

### (iii) Other candidate features

a) **Cell nearest-neighbor distances**: distribution of 1-NN, k-NN (k=5) distances per modality. Characterizes sparsity/packing. Useful for detecting constellations (unusually tight clusters).

b) **ROI volume distribution** (available in HCR `metrics.pickle`, CZ segmentation mask): mean / variance of per-cell volumes. Mismatch between modalities informs matching confidence weighting.

c) **GFP+ spatial clustering** in HCR: GCaMP+ inhibitory neurons are sparse and may form recognizable constellations. Compute GFP+ cell density maps and detect local density peaks (candidate constellations for initial alignment).

d) **Segmentation error rate estimation**: using active landmarks as ground truth, compute what fraction of matched cells are geographically plausible under the affine — reports how forgiving the matcher must be to missing / spurious ROIs.

e) **Depth-dependent spacing**: mean NN distance as a function of depth. Cortical layers differ not just in density but in cell-cell spacing patterns.

f) **GFP+ count distribution per depth**: high-count cells tend to be more reliably matched (manual protocol preferentially selects them). Report GFP count vs depth and GFP count vs match success.

g) **Anisotropy of expansion across the volume**: fit local affines in spatial octants, check whether XY/Z expansion factors are spatially constant or drift (gradient across volume).

h) **Top-of-volume buffer thickness**: distance from z=0 to pia surface in each modality. Report variability and whether it is predictable.

i) **Surface-normal alignment**: pia normal vectors of CZ and HCR should agree after the correct rotation. Provides an initial-alignment cue independent of cell positions.

j) **ROI morphology features** (optional): elongation, sphericity from segmentation masks. Could support per-cell matching confidence.

### Deliverable

A catalog of features with per-subject values, variability assessment, and a recommendation of which features look most stable/informative for downstream algorithm design (localization vs registration vs matching stages).

---

## Step 5: Transform analysis from landmarks

For each subject:

a) **Fit 3D affine transform** to all active landmarks (physical coords) via least-squares/Procrustes (SVD). Decompose into rotation R, scale s, translation t.  
b) **Residuals after affine**: compute residual vectors (predicted - actual) for each landmark. Report mean/median/max residual in um. This quantifies the nonrigid deformation magnitude.  
c) **Spatial pattern of residuals**: quiver plots in XY and XZ projections. Look for systematic patterns (e.g., larger residuals at edges).  
d) **Deformation smoothness**: pairwise landmark distance vs pairwise residual difference scatter plot. This informs kernel size for nonrigid registration.  
e) **Iteration progression**: count active landmarks per iteration (1 through 4-6).  

Key outputs for algorithm design:
- Affine residual magnitude range → how much nonrigid correction needed
- Smoothness scale → kernel size for deformation field
- Rotation angle confirmation → can hardcode 180-deg or needs estimation

---

## Step 6: Matching statistics

a) **Match rate vs subject characteristics**: correlate match rate with CZ cell count, HCR GFP+ count, GFP+ fraction, HCR volume, manual landmark count  
b) **Distance distribution of matched pairs**: after affine transform, compute CZ→HCR distances for matched cells. Report distribution in um.  
c) **Spatial distribution**: plot matched vs unmatched CZ cells in 3D. Check if unmatched cells cluster (near edges, deep layers, sparse regions).  
d) **GFP+ counts for matched cells**: compare spot counts distribution of matched vs all GFP+ HCR cells (for subjects with spot data)  

---

## Step 7: Cross-subject variability summary

- Compile all per-subject metrics into comparison table/plots
- Compute coefficient of variation for: cell count, density, GFP+ fraction, match rate, affine residual
- Identify outliers: 782149 (thin section, low match rate), 767018 (low match rate, older pipeline)
- Summarize "problem difficulty" parameters per subject

---

## Step 8: Update `01 Data Description.md`

Rewrite `/root/capsule/code/docs/01 Data Description.md` with:
- **Corrected HCR resolution** (level-2 pixel ≈ 0.988 um/px XY, 1 um/px Z) and the `4e6` / `1e6` convention from `fused_ng.json`
- **Corrected HCR physical volume** (~2.3 × 2.3 × 1.3 mm, not ~1×1×0.5)
- **Anisotropic expansion factors** (XY ~1.9×, Z ~2.8×) per subject, with range across subjects
- Per-subject resolution values (CZ 400 vs 700 um FOV options)
- Volume extents in both pixel and physical coordinates
- Cell count statistics with GFP+ fractions per subject
- Transform characteristics (rotation ~180°, anisotropic scale, nonrigid residual magnitudes in um)
- Coordinate-system conventions (column ordering per data source, landmark column ordering `[cz_x, cz_y, cz_z, hcr_x, hcr_y, hcr_z]`)
- Surface features (pia tilt, buffer thickness) from Step 4b
- Candidate features catalog (depth profile, constellations, NN distances, etc.)
- Cross-subject variability ranges
- Data format notes (767018 older pipeline, 755252/767022 intensity vs spot data)

---

## Step 9: Write session log

Save session log at `/root/capsule/code/sessions/01_analyze_benchmark_data/log.md` documenting:
- Goal, approach, key findings
- Computed metrics tables
- Insights for algorithm development
- Next steps

---

## Critical files to read/modify

| File | Action |
|------|--------|
| `code/docs/01 Data Description.md` | Update with concrete numbers |
| `code/manual workflow/manual_coreg_utils.py` | Reference for landmark parsing, coordinate conventions |
| `code/manual workflow/step_1_process_files.ipynb` | Reference for data loading patterns |
| `code/dev_code/benchmark_data_loader.py` | Create new — data loading utilities |
| `code/dev_code/01_analyze_benchmark.py` | Create new — analysis functions |
| `code/notebooks/01_benchmark_data_analysis.ipynb` | Create new — main analysis notebook |
| `code/sessions/01_analyze_benchmark_data/log.md` | Create new — session log |

## Verification

1. Run the analysis script end-to-end on all 6 subjects
2. Verify computed cell counts match expected values (table in Step 2)
3. Verify rotation angle is ~180 degrees for all subjects
4. Verify XY scaling ratio is ~3.17 in pixel space
5. Check that notebook renders with all plots
6. Review updated data description for accuracy and completeness

---

# Sub-plan: explore surface detection using ROI segmentation

## Context

The current image-based pia surface (combined channels + 5% relative margin)
gives `frac_above_pia` of 2–11% for HCR and 0–4% for CZ. Two open questions:

- Can we do better by using the **ROI segmentation data** itself? The premise:
  if segmentation were perfect, the shallowest ROI-centroid-z per (x, y)
  column would be the pia.
- In practice, HCR segmentation produces out-of-tissue false positives that
  sit above the real pia. Before we try a new fitter, we need to
  **characterize how many, and whether they have distinguishing features**.
  If simple ROI features (volume, local density, spot counts, etc.)
  separate them from normal ROIs, a filter-then-shallowest-z fitter may
  match or beat the image-based method, and can also help the image-based
  method by narrowing its z search.

Focus is HCR (which is where the problem is concentrated). CZ is already
good under the existing image-based fitter.

## Deliverables

1. Analysis script `code/dev_code/02_surface_detection_exploration.py`
2. Figures + tables under `code/sessions/01_analyze_benchmark_data/figures/`
   and `surface_exploration_*.csv`
3. New section in `code/notebooks/01_benchmark_data_analysis.ipynb`
4. If one method clearly wins: promote it to the default in
   `analyze_subject` (`code/dev_code/benchmark_analysis.py`), refresh
   the data description doc and session log.

## Stages (priority ordered — stop early if a stage already fixes it)

### Stage A (diagnostic; simplest). Characterize ROI segmentation error

For each subject, using the current **image-based pia as reference**:

- Classify each HCR ROI as `above_pia` (`depth < −5 um`) or `in_tissue`.
- Compute per-ROI features:
  - `volume` (from `hcr_dir/cell_body_segmentation/metrics.pickle`, already
    used by `_aggregate_spots_from_hcr` in `benchmark_data_loader.py`).
  - `bbox_dims` in pixels (from `global_bbox` in the same metrics).
  - `local_density`: number of ROI neighbors within 30 um (reuse
    `filter_in_tissue` from `benchmark_analysis.py`).
  - `spot_488_counts`, `spot_488_density`: already on `SubjectData.hcr_gfp_df`.
- For each subject, plot distributions of features for above-pia vs
  in-tissue groups (histograms, box plots). Same per-feature ROC-style
  separability.
- Measure what fraction of above-pia ROIs can be removed by simple
  thresholds on each feature alone and in combination.

Outcome: a quantitative answer to "does ROI segmentation actually disrupt
surface estimation, and can features identify the bad ROIs?"

### Stage B (simple filter → shallowest-z plane)

Candidate filter sets, ordered by simplicity:

1. **Density filter only** (already in `filter_in_tissue`): require ≥ 6
   neighbors within 30 um, or auto-scaled radius (`3 × median 1-NN`).
2. **Density + volume** outlier: drop ROIs outside IQR on volume.
3. **Density + volume + connected component**: build a coarse 3D
   ROI-density grid (~30 um voxels), threshold, `scipy.ndimage.label`,
   keep only ROIs in the biggest component.

For each surviving ROI set, fit the surface with two sub-methods:

- `shallowest_quantile_q_tile` with q ∈ {0, 1, 2, 5} % per (x, y) tile of
  size 100–150 um (reuses the existing `estimate_pia_surface` tile logic).
- `min_z_per_tile` directly (q = 0).

Measure `frac_above_pia`, tilt, roughness, number of valid tiles. Compare
against the current image-based baseline.

### Stage C (hybrid: ROI coarse prior + image refinement)

If Stage B improves over baseline but is still limited by segmentation
residuals, try:

- Use the best Stage-B surface as a **coarse prior** `z_prior(x, y)`.
- In `estimate_pia_surface_from_image` (or a new variant), restrict the
  per-column search to z ∈ [z_prior − 100 um, z_prior + 100 um].
- Fit the plane to that constrained first-crossing set.

This combines the global stability of ROI-based coarse localization with
the sharp boundary of image intensity.

### Stage D (final comparison + integration)

- Single table across methods for all 6 subjects:
  `method`, `HCR c`, `HCR tilt`, `HCR rough`, `HCR frac_above_pia`.
- Visual: pia overlay panels for 788406 (flat) and 782149 (tilted) showing
  lines from image-based, best Stage-B, and best Stage-C methods.
- Depth-profile falloff plot for the best method.
- Promote the winner to `analyze_subject` if it beats the image-based
  baseline by > 2 percentage points on median `frac_above_pia` **and** does
  not worsen any single subject by more than 2 percentage points.

## Critical files and reused utilities

| File | Role |
|------|------|
| `code/dev_code/benchmark_analysis.py` | **reuse**: `filter_in_tissue`, `estimate_pia_surface` (tile+quantile structure), `_robust_plane_fit`, `estimate_pia_surface_from_image`, `load_hcr_combined`, `depth_from_surface`, `depth_profile` |
| `code/dev_code/benchmark_data_loader.py` | **reuse**: `load_subject`, `SubjectData.hcr_gfp_df` (spot counts), metrics.pickle loading pattern in `_aggregate_spots_from_hcr` |
| `code/dev_code/02_surface_detection_exploration.py` | **new**: stage A/B/C experiments and figures |
| `code/notebooks/01_benchmark_data_analysis.ipynb` | **update**: add a surface-exploration section referencing the new figures |
| `code/docs/01 Data Description.md` | **update** if default changes |
| `code/sessions/01_analyze_benchmark_data/log.md` | **update** with findings |

## Verification

- `surface_exploration_methods.csv` covers all 6 subjects × all methods with
  numeric metrics.
- `frac_above_pia` for the winning method ≤ current image-based baseline
  on every subject; median ≥ 2 pp better.
- Pia overlays visually confirm the winning line sits at the image's
  tissue boundary for 788406 (flat) and 782149 (strongly tilted).
- Depth-profile density for HCR all cells falls to near 0 at depth = 0 for
  every subject.
- `analyze_subject` default surface agrees with the winning method; the
  notebook re-executes cleanly.

## Why this order

Stage A is cheap and answers whether the problem is "bad ROIs" or
"fundamentally hard geometry". If a single feature separates the groups
cleanly, Stage B with a 1-line filter may already win. Stage C is reserved
for the case where neither method alone is good enough — it is the most
code but also the most powerful.
