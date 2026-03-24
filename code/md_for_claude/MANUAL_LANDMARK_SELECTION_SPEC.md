# Manual Landmark Selection — Specification for Automation

> This document unambiguously describes the human manual process for
> placing initial co-registration landmarks between a czstack (2P in-vivo)
> volume and an HCR (lightsheet ex-vivo) volume. It serves as the ground
> truth specification for the automated replacement.

---

## 1. Inputs

| Item | Format | Notes |
|---|---|---|
| `czstack_seg_path` | 3D label TIFF (z × y × x) | Integer cell IDs per voxel; 0 = background |
| `czstack_reg_path` | 3D TIFF (z × y × x) | Raw GCaMP fluorescence |
| `czstack_centroid_path` | CSV (czstack_cell_id, czstack_z/y/x) | Centroids in CZ pixel coords |
| HCR GFP zarr | Zarr multiscale pyramid | Use pyramid level ≈ 1 µm/px; GFP (488 nm) channel only |
| `hcr_centroid_path` | `.npy` (N × 4) | hcr_cell_id, hcr_z/y/x in native HCR pixels |
| `hcr_segmentation_metrics_path` | `.pickle` | volume, bbox per HCR cell |
| `calibration_results.csv` | CSV | Per-subject rotation angles, z_scale, translation |
| `transform_analysis_summary.csv` | CSV | Robust tissue bounds, anchor fraction, z_scale_fit |

### Coordinate conventions
- CZ pixel resolution: XY = 0.78 µm/px, Z = 1.0 µm/px
- HCR zarr at ≈ 1 µm/px for all dimensions at the working pyramid level
- **Z direction is NOT flipped**: small CZ z → small HCR z (confirmed by
  `hcr_z = 3.05 × cz_z − 5.2`, positive slope, report_03 section 7)

---

## 2. Preprocessing — Rolling Max-Projection (MIP) of HCR

**Purpose**: CZ two-photon axial resolution (~10–20 µm PSF) makes many
cells visible in a single z-plane. HCR lightsheet axial resolution is
~1 µm/px. The rolling MIP collapses the HCR stack into windows that
match the CZ axial blur, allowing direct visual comparison.

**Two variants** (both should be implemented and compared):

### 2a. Image-based MIP
- Source: HCR GFP zarr at ≈ 1 µm/px
- For each window centre z₀ (step = 1 µm, covering full HCR z range):
  `MIP[z₀] = max(GFP[z₀ − W/2 : z₀ + W/2], axis=z)`
- Output: 3D array (n_windows × y × x), one 2D projection per µm

### 2b. Centroid-density MIP
- Source: HCR GFP+ centroids (cells with spot_count > threshold)
- For each window centre z₀:
  render a 2D binary or Gaussian-density image of all centroid XY
  positions whose z falls within [z₀ − W/2, z₀ + W/2]
- Output: 3D array (n_windows × y × x)

**Window size W**:
- Default: tunable parameter (suggest 10–20 µm to start)
- **Self-calibrated** from initial landmarks (Pt-0 to Pt-N, see §4):
  after the first matched set is found, compute
  `W_calibrated = max(hcr_z of Pt-0…Pt-N) − min(hcr_z of Pt-0…Pt-N)`
  This equals the effective CZ axial PSF projected into HCR z units.
  Use W_calibrated for all subsequent MIP generation.

---

## 3. Step A — Select Initial Constellation in CZ

**Goal**: choose a compact, distinctive group of CZ cells to use as the
search template.

### 3a. Choose the z-plane
- Search within **20–50 µm above the robust CZ tissue surface**
  (`cz_z_surface` from `robust_tissue_bounds`, n_edge = 5)
- Rationale: surface cells are sparser → more distinctive patterns;
  corresponds to small HCR z values (no coordinate flip)
- *Quantify the exact depth range from manual landmark data in
  analysis notebook `analysis_summary_report_04_initial_constellation.ipynb`*

### 3b. Define cells in the plane

**Image-based**: cells with non-zero pixels in `segmentation_masks.tif`
at the chosen z-slice.

**Centroid-based**: cells with centroid z within a ~10 µm z-window
centred on the chosen z, covering ~15% of CZ XY area,
forming a spatially compact group.

### 3c. Select the constellation
Choose a group of **5–10 cells** satisfying all of:

1. **No gaps** (convex hull criterion): no other GFP+/GCaMP+ cells
   fall inside the convex hull of the selected group
2. **Unique local pattern** (most important): the geometric arrangement
   is asymmetric and unlikely to be reproduced elsewhere
3. **Sparse region** (second): chosen from a low-density area of the
   volume to reduce false matches in HCR
4. **Spatially spread** (third): cells span as large an area as
   possible while satisfying criteria 1–3

### 3d. Constellation ranking
Score each candidate constellation and rank by:
```
score = w1 × uniqueness + w2 × sparsity + w3 × spread
```
where uniqueness > sparsity > spread in weight (tune w1:w2:w3 from data).

---

## 4. Step B — Search for Matching Constellation in HCR

**Goal**: find the HCR location corresponding to the CZ constellation.

### 4a. Apply rotation to CZ constellation
Rotate the CZ centroid coordinates using the known rotation matrix
`R = rotation_matrix_euler(euler_z, euler_x, euler_y)` from
`calibration_results.csv`. This removes the ~175° rotation and
aligns the two coordinate frames approximately.

### 4b. Define HCR search region

**Z range**: search within
`[hcr_z_surface, hcr_z_surface + D_search]`
where `D_search` is inferred from report_03 as
`D_search = cz_constellation_depth × z_scale_fit + margin`
≈ 100–150 µm. Compute per-subject from `transform_analysis_summary.csv`.

**XY range**: the CZ volume is approximately centred in HCR XY
(translation ty ≈ 470 µm, tx ≈ 475 µm from calibration). Search
within the HCR XY region corresponding to the CZ footprint
(≈ 400 × 400 µm centred at the expected translation position).

### 4c. Search method (implement and compare both)

**Method A — Image patch matching**:
- Extract a 2D patch from the CZ plane (segmentation outline or raw fluorescence)
- Slide over the rolling HCR MIP within the search region
- Score by normalised cross-correlation (NCC)
- Top-N NCC peaks = candidate matches

**Method B — Geometric centroid descriptor matching**:
- Represent constellation as rotation-normalised pairwise distance/angle descriptor
- Match against all subsets of GFP+ HCR centroids projected into 2D
  (within the MIP window at each candidate z)
- Score by descriptor similarity

**Method C — Hybrid**: use Method B for candidate generation,
Method A for verification.

### 4d. Candidate evaluation
For each candidate match position, score by:
- Descriptor / NCC similarity of the constellation itself
- Absence of contradicting nearby cells (false extra matches inside hull)

### 4e. Constellation switching
If no candidate exceeds the acceptance threshold after exhausting the
full search region (computational budget equivalent to ~5 min human
search time), discard this constellation and return to Step A to pick
a new one. Record which constellations were tried and failed.

---

## 5. Step C — Confirm Initial Match (Pt-0 to Pt-N)

**Goal**: verify the tentative constellation match before committing
landmarks.

Place the matched cell pairs as initial landmarks and apply a TPS
transform using only these N points (N = 4–6).

**Confirmation metrics** (implement and test all; report which works best):

1. **Local MNN distance drop**: compute mean mutual nearest-neighbour
   distance between TPS-projected CZ cells and HCR GFP+ cells,
   restricted to a local neighbourhood around the initial landmarks
   (radius ≈ 2× constellation diameter). Accept if the drop vs.
   pre-TPS is substantial.

2. **Local residual smoothness**: after TPS, the per-cell residual
   vectors in the neighbourhood should be spatially smooth (low
   divergence), not random.

3. **Constellation self-consistency**: the matched cells' relative
   geometry in HCR should match the CZ constellation within expected
   expansion (xy_scale ≈ 1.82×, z_scale ≈ 2.88×).

If confirmation fails, discard this match and continue searching (§4c).

---

## 6. Step D — Iterative Landmark Addition

**Goal**: improve registration coverage one landmark at a time.

### Phase 1 — Interior improvement
Select the next landmark as the CZ cell satisfying:
```
priority = match_confidence × local_TPS_error
```
- `match_confidence`: how clearly the CZ cell has an identifiable
  HCR counterpart (constellation similarity, MNN rank, NCC)
- `local_TPS_error`: current TPS projection error at that cell's
  location (distance from projected position to nearest HCR cell)

Pick the highest-priority cell, find its HCR match, add as landmark,
re-fit TPS. Continue until interior is well-covered.

### Phase 2 — Edge expansion
Once interior is registered, shift priority toward CZ cells near
the volume edges (outside or near the boundary of the current
landmark convex hull). Selection criterion remains the same
(confidence × local error) but weighted by distance to hull boundary.

### Confirmation after each landmark
After adding each new landmark, verify locally:
- TPS residual in the neighbourhood of the new landmark should
  decrease vs. before
- The new landmark should not worsen residuals elsewhere (TPS
  global check with outlier detection)

---

## 7. Step E — Stopping Criterion

### For the automated full-pipeline approach (implement first)
Do not stop. Feed all found landmarks directly into the iterative
TPS matching pipeline (step_2 → step_3 → step_4). Let the downstream
pipeline determine convergence.

### For the manual-mimic approach (implement second)
Stop when ALL of:
- (a) Landmark convex hull covers the CZ volume edges (hull boundary
  within ~50 µm of CZ volume boundary)
- (b) Local TPS residual < threshold everywhere within the hull
- (c) Landmark count is 50–100

---

## 8. Vasculature
Human visual cue only. **Not used in the automated pipeline.**

---

## 9. Open Implementation Questions (to be resolved by testing)

| Question | Options | Decision |
|---|---|---|
| Image vs centroid MIP | Both implemented | Compare on all 5 subjects |
| Confirmation metric | MNN drop, residual smoothness, geometry check | Test all, pick best |
| Constellation score weights w1:w2:w3 | To be calibrated | From analysis notebook |
| Search budget (computational equivalent of ~5 min) | Number of candidate positions | Calibrate from data |
| MIP window default before calibration | 10 µm or 20 µm | Infer from analysis notebook |
| CZ surface search depth (exact range) | ~20–50 µm | Quantify from manual landmark data |

---

## 10. Analysis Notebook (required before implementation)

`code/analysis_summary_report_04_initial_constellation.ipynb`

Quantify from the 6 manual landmark CSVs:
- Depth of Pt-0 to Pt-N relative to `cz_z_surface`
- HCR z-spread of Pt-0 to Pt-N (→ MIP window W)
- Constellation size: cell count, XY area, z-span
- Inter-cell spacing distribution
- How many constellations were typically tried (if recoverable from
  iteration history)
