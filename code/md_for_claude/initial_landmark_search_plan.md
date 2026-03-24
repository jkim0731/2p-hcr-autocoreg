# Initial Landmark Search — Sub-Development Plan

> **Scope**: everything needed to automatically produce the initial seed landmarks
> that replace manual BigWarp placement (Pt-0…Pt-5). This is Step 2 of the
> larger pipeline. Validation is exclusively against `coreg_table.csv` for all 6
> subjects. Nothing here concerns the iterative TPS matching (Step 3+).

---

## 1. Problem Statement

Given:
- **CZ centroids** (~600 cells, pixel space, 0.78 µm/px XY, 1 µm/px Z)
- **HCR GFP+ centroids** (~1000–5000 cells, pixel space, ~1 µm/px all dims)
- **Calibration template** (`coreg_transform_template.json`) with rotation prior and
  Z-scale estimate

Produce: a seed landmark CSV (BigWarp format: ids, active, czstack_x/y/z, hcr_x/y/z)
with ≥ 10 correct matches that are spatially distributed enough to initialise TPS.

"Correct" is defined relative to `{subject}_coreg_table.csv` (ground truth from
manual co-registration).

---

## 2. Ground-Truth Data (from `analysis_summary_report_04`)

| Subject | Initial pts | CZ depth (from surface) | HCR z-spread | Z-scale obs. | Hull frac |
|---------|-------------|-------------------------|--------------|--------------|-----------|
| 755252  | 5           | 0 – 16 µm (mean 5)      | 52.7 µm      | 2.68         | 0.03      |
| 767018  | 6           | 17 – 108 µm (mean 62)   | 306.7 µm     | 3.35 (deep)  | 0.17      |
| 767022  | 2 only!     | 267 µm                  | 0.5 µm       | —            | —         |
| 782149  | 6           | 19 – 39 µm (mean 28)    | 57.9 µm      | 2.98         | 0.02      |
| 788406  | 6           | 11 – 20 µm (mean 14)    | 18.0 µm      | 1.94         | 0.13      |
| 790322  | 4           | 155 – 158 µm (mean 157) | 22.5 µm      | 9.14 (tiny z-span) | 0.06 |

**Key derived facts:**
- Reliable MIP window W: ~50–60 µm (median across non-outlier subjects;
  ignore 767018 z-spread=307 and 790322 z-span=2.5 µm which both reflect
  manual quirks, not true axial spread)
- NN spacing: 68.3 ± 20.2 µm
- Constellation cell count: 4–6 (hard cap Pt-0…Pt-5, i.e. 7 points)
- CZ search depth (from surface): most subjects within 40 µm; two outliers at
  62 µm (767018) and 157 µm (790322) — the algorithm must handle these
- Hull fraction: 0.02–0.17 (very variable; compact clusters DO exist but are
  not always used manually)

**Implication:** depth-from-surface alone cannot select the right z-slab.
The algorithm must either (a) scan multiple depths or (b) not require the
initial constellation to be near-surface.

---

## 3. What Already Exists in `coreg_alignment.py`

| Function | Approach | Status |
|---|---|---|
| `rotation_search` | 3D rotation grid (125+ cands) + Nelder-Mead; scores by mutual-NN | Implemented |
| `find_anchor_slice_alignment` | Scan (CZ_z, HCR_z) pairs with 2D CC; seeds from full-volume MNN | Implemented, **primary in step_2** |
| `find_initial_alignment_xycorr` | Two-stage 2D XY FFT CC (coarse footprint + fine subsampled) | Implemented |
| `find_seed_constellation` | Random constellation sampling + pairwise distance matching + SVD | Implemented |
| `extract_seed_landmarks_zxy` | Z-restricted XY mutual-NN seed extraction | Implemented |
| `score_mutual_nn_xy` | 2D XY mutual-NN scoring | Implemented |
| `_find_hcr_constellations` | Pairwise XY distance matching with recursive pruning | Implemented |
| `ransac_rigid_match` | RANSAC rigid from descriptor-matched correspondences | Implemented |

None of these have been systematically validated across all 6 subjects.

---

## 4. Known Failure Modes of Existing Methods

### 4a. `find_anchor_slice_alignment`
- **Uses template rotation mean only** (no grid search). If the true rotation
  differs by >5° from mean, XY alignment can fail.
- **CZ scan depth hardcoded to 200 µm**. 767022 (depth=267 µm) and 790322
  (depth=157 µm) are near or beyond this limit.
- **Sparse HCR z-slabs**: some HCR z levels have few cells, making CC
  unreliable at those levels.
- **CC objective ≠ match objective**: the best CC peak is not always the best
  MNN alignment.

### 4b. `find_initial_alignment_xycorr`
- **Z slab requires knowing z_scale**: uses `z_scale_mean=2.8` to estimate
  which HCR z-range corresponds to the CZ volume. If z_scale is wrong, the
  HCR pool contains the wrong cells.
- **Dense HCR pool**: with 1000–5000 cells, subsampling introduces noise.
- **No rotation grid search**: uses mean rotation only.

### 4c. `find_seed_constellation`
- **Template-dependent search region**: needs `margin_z_min_frac_mean` from
  the template. If this is wrong, the HCR pool is wrongly placed.
- **Random constellation sampling**: not guaranteed to find the most unique
  constellation first.
- **pairwise distance matching only in XY**: sensitive to rotation error;
  a 5° rotation error at 100 µm radius = 8.7 µm XY drift, comparable to
  the 10 µm tolerance used.

### 4d. `rotation_search`
- **Slow**: 3D rotation grid × centroid alignment per candidate. Suitable for
  small Q (few hundred cells), but HCR has 1000–5000 cells → ~125 × 5000
  distance ops per evaluation — not the bottleneck but Nelder-Mead adds 500
  evals per candidate.
- **Z-scale ambiguity**: scoring with 3D mutual-NN penalises the Z dimension,
  which is wrong during initial alignment before z-scale is known.
- **The existing function uses 3D MNN, not 2D XY-only**: the Z-scale is
  2.34–3.46×, so 3D distance is dominated by Z error; XY scoring is better.

---

## 5. Approaches to Develop

Three complementary approaches are described below. They differ in how they
search: by rotation grid, by Z scan, or by constellation geometry.
All three should be implemented and benchmarked.

---

### Approach A — Constellation Selection + Hough XY Vote (NEW, PRIMARY)

**Rationale:** A small group of cells with an empty convex hull is a rare,
locally unique pattern. Once the ~180° rotation is applied (known to ~5°),
the XY pairwise distances between cells are preserved. A Hough vote finds the
XY translation quickly. The Z offset is scanned exhaustively because z_scale
is unknown.

#### Part A1 — `select_constellation(cz_df_um, cz_z_surface_um, dz_um=15.0, n_cells=6)`

**Goal**: find k=4–6 near-surface CZ cells whose XY convex hull contains no
other CZ cells.

```
1. Compute cz_z_surface from robust_tissue_bounds (already exists in notebooks;
   move to coreg_alignment.py as standalone function)
2. Filter CZ to slab [cz_z_surface, cz_z_surface + dz_um]
   - If fewer than k+2 candidates, expand by 5 µm steps up to dz_max=60 µm
3. Enumerate C(N, k) subsets for k in {6, 5, 4}:
   - If N ≤ 35: enumerate all subsets directly (max C(35,6) ≈ 1.6M — may be
     slow; use k=4 or pre-filter if too large)
   - If N > 35: first reduce to ~30 candidates via XY k-means on the z-slab
     cells, then enumerate from reduced set
4. Score each subset:
   Primary:   count CZ cells inside XY convex hull (exclude selected k) → minimise
   Secondary: XY hull area → minimise (tiebreak)
5. Return subset with lowest intruder count (ties broken by hull area)
```

**Expected N_candidates**: from report_04, ~12–34 CZ cells per 10 µm z-window.
For N=20, k=6: C(20,6)=38760 subsets — fast to enumerate.
For N=34, k=6: C(34,6)=1344904 — too large; apply XY k-means pre-filter first.

**Constellation-switching**: if best constellation has >2 intruders, increase
dz_um by 5 µm and retry (up to dz_max=60 µm). This handles the case where the
surface is very dense.

**Why not use depth as primary criterion?**
Report_04 shows 790322 has constellation at 157 µm depth. The empty-hull
criterion selects the *most distinctive* constellation regardless of depth;
restricting to the top slab first is a heuristic default, not a hard rule.
The dz expansion loop handles the outlier cases.

---

#### Part A2 — `match_constellation(constellation_um, hcr_df_um, template_path, W_um=55.0, z_step_um=25.0, match_threshold_um=15.0)`

**Goal**: for each (rotation, HCR z-level) candidate, use Hough XY voting to
find the best translation, then score by cell-match count.

```
Inputs:
  constellation_um  : (k, 3) array, z/y/x in µm (from select_constellation)
  hcr_gfp_um        : (M, 3) array, HCR GFP+ cells in µm
  template          : dict with pitch_range_deg, z_rotation_range_deg, roll_range_deg
  W_um              : MIP z half-window (default 55 µm; covers ~50-60 µm MIP)
  z_step_um         : step between HCR z-center candidates (default = W_um/2 = 27.5 µm)
  match_threshold_um: XY threshold for counting a match (default 15 µm)

Algorithm:
  Build rotation grid from template (same as rotation_search):
    theta_x in pitch_range (±186°, step 5°) — this is the ~180° rotation
    theta_z in z_rotation_range (±15°, step 5°)
    theta_y in roll_range (±15°, step 5°)
    → ~125 rotation candidates

  For each rotation R in grid:
    Apply R to constellation (k points) → constellation_rotated (z/y/x µm)
    constellation_xy = constellation_rotated[:, 1:]  # (k, 2) YX

    For each z_center in arange(hcr_z_min, hcr_z_max, z_step_um):
      hcr_window = hcr_gfp_um where |hcr_z - z_center| <= W_um/2 → (M_w, 3)
      if M_w < 3: skip
      hcr_window_xy = hcr_window[:, 1:]  # (M_w, 2)

      Hough XY vote:
        For each pair (cz_i, hcr_j): vote += 1 at (hcr_j_y - cz_i_y, hcr_j_x - cz_i_x)
        Bin size: 5 µm
        (t_y, t_x) = argmax of vote accumulator

      Score: count i in 0..k-1 where min_j |hcr_window_xy[j] - (constellation_xy[i] + [t_y,t_x])| <= match_threshold_um

  Collect top-5 (R, z_center, t_y, t_x, score) by score
  Return top-5 as list of dicts
```

**Compute budget:**
- 125 rotations × 17 z-levels × (6 × ~150 HCR pairs per window) ≈ 1.9M ops
- Fully vectorisable (broadcasting): constellation_xy (k,2) vs hcr_window_xy (M_w,2)
  → all k×M_w difference vectors in one operation
- Expected runtime: < 1 second

---

#### Part A3 — Verification and Seed Expansion

For each of the top-5 hypotheses from `match_constellation`:

```
1. Apply full transform (R, t=[z_center, t_y, t_x]) to ALL CZ cells
2. Z-restricted XY mutual-NN: for each CZ cell, find HCR match within
   z_window=150 µm and xy_threshold=22 µm
3. Accept hypothesis if ALL of:
   a. All k constellation cells matched within 25 µm 3D distance
   b. For each pair (i,j) in constellation:
      |dist_hcr(matched_i, matched_j) - dist_cz_transformed(i, j)| <= 20 µm
      (pairwise distance consistency check)
   c. Full-volume XY-MNN score >= 10

4. Best-scoring consistent hypothesis → R_best, t_best
5. Extract full seed landmark set using extract_seed_landmarks_zxy
```

---

### Approach B — Anchor-Slice 2D CC (Existing, Improved)

The existing `find_anchor_slice_alignment` is the current primary method.
Improvements needed:

1. **Rotation grid**: instead of using the template mean rotation, run the CC
   at the top-3 rotation candidates from a coarse grid (5° steps). Total:
   3 rotations × (N_cz_levels × N_hcr_levels) = 3× cost, still fast.

2. **Extend CZ scan depth**: increase `cz_anchor_max_depth_um` from 200 µm to
   300 µm to cover 790322 (157 µm) and edge cases.

3. **Score primary = MNN, not CC**: the CC peak is a noisy proxy. After computing
   CC-derived (ty, tx), immediately compute MNN; use MNN as primary sort key
   rather than raw CC peak.

4. **Multi-level Z initialisation**: instead of a single tz = hcr_anchor - cz_anchor,
   try 3 values of tz (±50 µm from the CC-derived one) and pick the MNN-best.

---

### Approach C — Full Rotation Search + 2D XY Score (Existing, Use As Fallback)

The existing `rotation_search` already does this but uses 3D MNN scoring
(wrong, because z_scale is unknown). Switch to XY-only scoring:
- Replace `score_mutual_nn` calls with `score_mutual_nn_xy` in the grid loop
- Use as fallback if Approach A and B both fail (score < 10)

This is slower but more exhaustive: it searches the full 6-DOF space.

---

## 6. Validation Protocol

### 6.1 Ground-truth reference

`{subject}_coreg_table.csv` contains all QC-accepted czstack_cell_id →
hcr_cell_id pairs from the completed manual pipeline.  The seed landmark `ids`
column encodes both IDs directly in the format `cz{czstack_cell_id}-hcr{hcr_cell_id}`
(produced by `extract_seed_landmarks_zxy`).  Validation therefore requires no
transform fitting — just ID lookup.

```python
# Parse seed IDs
seed_df["czstack_cell_id"] = seed_df["ids"].str.extract(r"cz(\d+)").astype(int)
seed_df["hcr_cell_id"]     = seed_df["ids"].str.extract(r"hcr(\d+)").astype(int)

# Build ground-truth set
gt = set(zip(coreg_table["czstack_cell_id"], coreg_table["hcr_cell_id"]))

# Evaluate
seed_df["correct"] = seed_df.apply(
    lambda r: (r.czstack_cell_id, r.hcr_cell_id) in gt, axis=1
)
```

### 6.2 Per-subject metrics

| Metric | Definition |
|--------|-----------|
| `n_seeds` | Number of seed landmarks produced |
| `n_correct` | Seeds whose (czstack_cell_id, hcr_cell_id) pair is in coreg_table |
| `precision` | n_correct / n_seeds |
| `seed_coverage` | Fraction of CZ XY bounding box covered by seed convex hull |
| `runtime_s` | Wall-clock time |

**Success criterion**: precision ≥ 0.85 and n_seeds ≥ 10 across all 6 subjects
with a single set of parameters.

### 6.3 Validation notebook cell

`code/dev_step_2_initial_alignment.ipynb` should have a **validation cell**
after each approach that:
1. Loads `{subject}_coreg_table.csv`
2. Parses czstack_cell_id / hcr_cell_id from `seed_df["ids"]`
3. Looks up each pair in the coreg_table set
4. Prints: n_seeds, n_correct, precision
5. Plots: seed XY positions coloured correct/incorrect over all CZ cells

This runs after each approach so all 3 can be compared side by side.

---

## 7. Function Signatures

### New in `coreg_alignment.py`

```python
def robust_tissue_bounds(z, y, x, n_edge=5) -> float:
    """Robust CZ z-surface estimate (already used in notebooks, move here)."""

def select_constellation(
    cz_df_um: pd.DataFrame,      # czstack_z/y/x in µm
    cz_z_surface_um: float,
    dz_um: float = 15.0,
    dz_max_um: float = 60.0,
    n_cells: int = 6,
    kmeans_N_threshold: int = 35,
) -> pd.DataFrame:
    """Return k-cell constellation with minimum CZ intruders in XY hull."""

def hough_vote_xy(
    constellation_xy: np.ndarray,  # (k, 2) YX in µm
    hcr_window_xy: np.ndarray,     # (M, 2) YX in µm
    bin_size_um: float = 5.0,
) -> tuple[float, float, int]:
    """Vectorised Hough XY vote. Returns (t_y, t_x, peak_count)."""

def match_constellation(
    constellation_um: np.ndarray,  # (k, 3) z/y/x in µm
    hcr_gfp_um: np.ndarray,        # (M, 3)
    template: dict,
    W_um: float = 55.0,
    z_step_um: float = 25.0,
    match_threshold_um: float = 15.0,
) -> list[dict]:
    """Return top-5 hypotheses with keys: R, z_center, t_y, t_x, score."""

def verify_constellation_match(
    hypothesis: dict,              # from match_constellation output
    constellation_um: np.ndarray,  # (k, 3)
    hcr_gfp_um: np.ndarray,        # (M, 3)
    dist_threshold_3d_um: float = 25.0,
    dist_consistency_um: float = 20.0,
) -> bool:
    """True if all k constellation cells matched and pairwise distances consistent."""
```

### Reused (existing)

- `rotation_matrix_euler` — unchanged
- `apply_rotation` — unchanged
- `score_mutual_nn_xy` — unchanged
- `extract_seed_landmarks_zxy` — unchanged
- `find_anchor_slice_alignment` — extend with rotation grid
- `_rigid_svd_3d` — unchanged (used for validation)

---

## 8. Parameter Defaults Justified by Data

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `dz_um` (constellation slab) | 15.0 µm | ~12–34 cells per 10 µm window (report_04); 15 µm gives ~18–50 candidates |
| `n_cells` | 6 | Mode of manually-placed initial landmark count |
| `W_um` (MIP window) | 55.0 µm | Median HCR z-spread excluding outliers (52.7, 57.9, 18.0 µm across subjects) |
| `z_step_um` | 25.0 µm | W_um/2 = half-window step ensures contiguous coverage |
| `match_threshold_um` | 15.0 µm | Smaller than NN spacing (68 µm) but larger than cell centroid jitter |
| `bin_size_um` (Hough) | 5.0 µm | ~1/4 of match_threshold; fine enough to localise translation peak |
| `dist_threshold_3d_um` (verify) | 25.0 µm | 3D distance; generous to allow for Z-scale uncertainty in z |
| `dist_consistency_um` | 20.0 µm | Pairwise distance tolerance; ~30% of typical NN spacing |
| `xy_threshold_um` (seed extract) | 22.0 µm | ~1/3 of NN spacing (68 µm); tight enough to avoid false matches |
| `z_window_um` (seed extract) | 150.0 µm | Large to absorb z_scale uncertainty; MNN in XY enforces precision |

---

## 9. Implementation Order

1. **`robust_tissue_bounds`** → move from notebook to `coreg_alignment.py`
2. **`select_constellation`** → new function
3. **`hough_vote_xy`** → new function
4. **`match_constellation`** → new function
5. **`verify_constellation_match`** → new function
6. **Update `dev_step_2_initial_alignment.ipynb`**:
   - Cell A: run Approach A (constellation + Hough)
   - Cell B: run Approach B (anchor-slice CC, existing)
   - Cell C: validation cell — compare both against coreg_table
7. **Improve `find_anchor_slice_alignment`** (optional, after validation shows it needs it)

---

## 10. Edge Cases and Failure Strategies

| Case | Issue | Strategy |
|------|-------|----------|
| 767022 (2 initial pts, depth=267 µm) | Constellation at extreme depth; `select_constellation` default slab misses it | The algorithm does NOT need to match the manual choice. Any correct k=4+ constellation at any depth is acceptable. If surface slab is empty, scan all depths. |
| 790322 (4 pts, depth=157 µm) | Deep constellation | Same: scan up to dz_max=300 µm if needed |
| Very dense z-slab (>50 cells) | Many C(N,k) subsets → slow | K-means pre-filter to 30 candidates before enumeration |
| No constellation with 0 intruders | Entire CZ surface is dense | Relax to 1, then 2 intruders; or try a different z-slab |
| Top-5 hypotheses all fail verification | Wrong rotation or z-level | Fall back to `find_anchor_slice_alignment` or `rotation_search` |
| HCR z-window empty | HCR z-step landed in a gap | Reduce z_step_um to 15 µm (overlap increases); also add fallback to use any HCR window with ≥ 3 cells |

---

## 11. Deliverables

| File | Change |
|------|--------|
| `code/coreg_alignment.py` | Add: `robust_tissue_bounds`, `select_constellation`, `hough_vote_xy`, `match_constellation`, `verify_constellation_match` |
| `code/dev_step_2_initial_alignment.ipynb` | Rewrite: orchestrate Approach A + B, validation cell, plots |
| `code/analysis_summary_report_04_initial_constellation.ipynb` | Already done — no changes |

**No new Python modules** needed for this sub-task. All additions go into
`coreg_alignment.py`.

---

## 12. What Is NOT in Scope

- Iterative TPS matching (Step 3)
- Classifier training (Step 0 Part B)
- Full rotation search with Nelder-Mead refinement (Step 2 Method A full pipeline)
- CPD / pycpd fallback
- RANSAC descriptor matching (remains as existing code, not used in primary path)
- step_4 / step_5 notebook updates
