# Session 07b — Stricter (scale-only) GFP+ + k-NN / density-ratio scale

## Context

Session 07 (anisotropic ICP) and session 06 (k-NN ratio) both failed
on the CZ GCaMP+ ↔ HCR GFP+ centroid inputs. Session 07 Part D
diagnosed the cause: the current project-wide GFP+ set sampled the
truth population (matched-HCR + unmatched-CZ mapped into HCR) with
subject-specific, depth-dependent bias — integrated GFP+/truth
0.46→7.31 across 6 subjects, per-bin CV 0.33–1.12.

**Hypothesis.** A stricter GFP+ cutoff at the Bayes-optimal
intersection between the rightmost and second-rightmost GMM
components on `log(feature)` deflates the false-positive tail. If
the stricter set approximates the truth population to within a
subject- and depth-independent scalar, per-axis k-NN distance ratios
and a global xy/z density ratio can recover `(sxy, sz)`.

The new cutoff is **scale-only** — it does not replace the
project-wide GFP+ definition.

## Stopping condition (tightened)

- **Pass** iff **every** subject has `|rel_err_sxy| ≤ 5 %` AND
  `|rel_err_sz| ≤ 5 %` on at least one of M1 or M3.
- GT: landmark-Procrustes `fit_anisotropic_similarity(landmark_pairs_um(s,
  active_only=True))`. Used **only** for scoring.
- If 07b fails, **stop** and log. Do not chain into another
  surface-based estimator in this session — the proposed 07c was
  rejected (pia plane tilt is erased by R1 alignment, and
  z-extent-based sz is contaminated by modality-specific imaging
  depth cuts). The next candidate approach (image-level 488
  correlation after full surface+normal alignment) is substantive
  enough to justify its own session; I will describe it in the log
  but not implement it here.

## Files

```
dev_code/07b_gfp_intersection_threshold.py   (NEW — threshold + plots)
dev_code/07b_scale_from_clean_gfp.py         (NEW — driver)
sessions/07b_scale_clean_gfp/
    log.md
    results.json
    depth_density_clean_summary.json
    figures/                                 (histograms, depth_density, scales)
    notebook.ipynb
```

## Reused infrastructure (read-only)

- `benchmark_data_loader.load_subject` + `cz_px_to_um`, `hcr_px_to_um`.
- `benchmark_data_loader._aggregate_spots_from_hcr` for 767018.
- `benchmark_data_loader.DATA_DIR / "cell_data_mean_{sid}_R1.csv"`
  for 755252 / 767022 intensity features.
- `benchmark_analysis.analyze_subject`, `fit_anisotropic_similarity`,
  `landmark_pairs_um(active_only=True)` — GT only.
- `dev_code/07_depth_density_diagnosis.py::analyze_subject_depth`.
  We monkey-patch `load_subject` in its module namespace so it reads
  our strict-GFP+ `SubjectData` without editing the file.
- `dev_code/local_distance_scale.py::estimate_local_distance_scale`.
- `r1_revised.coarse_align_revised`, `apply_coarse_affine`.

## Step 1 — GMM-intersection threshold
`dev_code/07b_gfp_intersection_threshold.py`

**Intersection formula.** Two weighted 1D Gaussians
`w_i·φ(x; μ_i, σ_i)` cross where

    A·x² + B·x + C = 0
      A = 1/(2σ₂²) − 1/(2σ₁²)
      B = μ₁/σ₁² − μ₂/σ₂²
      C = log(w₁/w₂) + 0.5·log(σ₂²/σ₁²) + μ₂²/(2σ₂²) − μ₁²/(2σ₁²)

Pick the root inside `[min(μ₁, μ₂), max(μ₁, μ₂)]`. Degenerate
`σ₁=σ₂` → linear: `x = 0.5·(μ₁+μ₂) + σ²·log(w₂/w₁)/(μ₂−μ₁)`. If no
interior root, fall back to the midpoint and flag
`no_interior_root=True`.

**Spot subjects (788406, 790322, 767018, 782149): GMM-4 on
`log(density > 0)`.**
Feature source:
1. `*spot_488_counts.csv` in coreg_dir → has `density`.
2. Else `_aggregate_spots_from_hcr(hcr_dir)` → same schema.

Fit `GaussianMixture(n_components=4, n_init=5, random_state=0)`.
Sort components by mean ascending. Rightmost = K−1; next = K−2.
Intersection → linear cutoff. Filter `density >= cutoff`.

**Intensity subjects (755252, 767022): GMM-2 on
`log10(mean − bg > 0)`.**
Read `cell_data_mean_{sid}_R1.csv`, keep `channel==488`, compute
`feature = mean − background`. Fit GMM-2. Intersection in
log10-domain → `10^x` linear cutoff. Filter `feature >= cutoff`.

**Per-subject figures and diagnostics.**
Histogram of `log(feature)` (100 bins), overlay weighted GMM
components, intersection line, v2.2 threshold line. Table values:
`cutoff_strict`, `cutoff_v22`, `n_strict`, `n_v22`, `n_coreg_kept`,
`coreg_coverage_strict`, `sigma_right/sigma_next`,
`no_interior_root`.

**Per-subject sanity gate** (blocks scale estimation for that
subject):
- intersection lies strictly between μ_right and μ_next;
- `n_strict ≥ 300`;
- `coreg_coverage_strict ≥ 0.80`.

If any subject fails the sanity gate, note it in the log and skip
scale estimation on that subject (a subject that fails this gate
cannot contribute to a 6/6 pass regardless).

## Step 2 — Driver `07b_scale_from_clean_gfp.py`

Per subject:

1. `s = load_subject(sid)` (v2.2 defaults).
2. Compute strict cutoff via step-1 function; filter `s.hcr_gfp_df`
   to cells passing. Replace `s.hcr_gfp_df` in-memory.
3. `analyze_subject(s)` → `info['gfp_xyz']` is the strict set.
4. **Depth-density gate** — import `07_depth_density_diagnosis`,
   monkey-patch `load_subject` to return our patched `s`, run
   `analyze_subject_depth(sid)`. Save the 5-cohort profile with
   the strict GFP+ replacing the v2.2 one.
   - **Gate A (per-bin uniformity):** CV of per-bin GFP+/truth
     ratio (over bins where `ρ_truth > p25(ρ_truth)`) `≤ 0.20`
     per subject — **tighter than session 07** because the 5 %
     scale bar requires a far flatter detection profile than the
     20 % bar did.
   - **Gate B (cross-subject offset):** integrated GFP+/truth
     ∈ [0.8, 1.25] across all 6 subjects (±25 % of unity).
   - If fewer than 6/6 pass both gates, log the per-subject
     values and still run step-5 scale estimation (so we can report
     where the passing subjects land); but declare 07b a failure.

5. **M1 — axis-separated k-NN distance ratio.**
   `estimate_local_distance_scale(cz_xyz_um, gfp_xyz_um_strict,
   coarse_fit=r1_minimal, sxy_upper_feasibility=L_hcr_xy/L_cz_xy)`.
   Defaults: k=5, k_density=10, sxy_lower_crop=1.2,
   sz_lower_crop=1.2, crop_margin_factor=1.3, max_iter=8.
   No GT anywhere in this leg.

6. **M3 — global xy/z density ratio (the simplest proposed).**
   Use the overlap AABB in the HCR µm frame from the depth-density
   diagnostic. Both CZ-mapped and HCR GFP+-strict are clipped to
   it.
   - `sxy_m3 = sqrt( (N_cz_in_box · L_hcr_xy_area) /
                      (N_hcr_in_box · L_cz_xy_area_rotated) )`
     where the rotated-CZ area is the R1-rotated-footprint area.
     (If same AABB used for both, this reduces to
     `sqrt(N_cz / N_hcr) · L_hcr_xy_area/L_cz_xy_area_rotated`.)
   - `sz_m3 = (N_cz_in_box / L_cz_z_rotated) /
              (N_hcr_in_box / L_hcr_z)`, i.e. 1D-density ratio
     along the HCR-pia-normal axis, on cells inside the same AABB.
   - This is the "even simpler density in xy and z matching"
     requested.

7. GT + scoring: landmark-Procrustes; record
   `sxy_m1, sz_m1, sxy_m3, sz_m3, sxy_gt, sz_gt, rel_err_*`. Per
   subject, report whether M1 or M3 cleared ±5 %.

## Step 3 — Write-up

- `sessions/07b_scale_clean_gfp/log.md` —
  - Part A: plan + self-critique (from user critique + my own).
  - Part B: step-1 GMM diagnostic table + figures.
  - Part C: step-2 depth-density gate results (strict vs v2.2 side
    by side, per-subject CV, integrated ratio).
  - Part D: scale results, M1 vs M3 table vs GT.
  - Part E: pass/fail and next-candidate proposal (image-level 488
    correlation after R1 + surface-normal alignment; detailed
    design sketch only, not implemented here).
- `sessions/07b_scale_clean_gfp/results.json` — everything
  machine-readable.
- `sessions/07b_scale_clean_gfp/depth_density_clean_summary.json`.
- `sessions/07b_scale_clean_gfp/figures/` — GMM histograms,
  depth-density-clean, scale comparison plots.
- `sessions/07b_scale_clean_gfp/notebook.ipynb` built via
  `_build_notebook.py` (mirrors session 07's pattern).

## Verification

1. **Synthetic sanity.** Stretch a CZ copy by `(1.77, 1.77, 2.82)`;
   feed directly to M1 and M3 (no GFP+). Expect ≤ few % recovery.
2. **v2.2 baseline in the same notebook.** Run depth-density with
   v2.2 GFP+ (baseline) and strict GFP+ side-by-side so the
   flattening is visible.
3. **No GT leakage.** Grep the driver for
   `landmark_pairs_um|fit_anisotropic` to confirm it appears only
   in the scoring block.

## Self-critique (post user review)

- **Why 07c was dropped.** z-extent-based sz is contaminated by
  FOV-dependent depth cut (CZ and HCR stop at different depths for
  reasons unrelated to tissue expansion). Pia-plane-tilt-based
  sz/sxy ratio collapses because R1 already aligns the two planes.
  Density-shape matching along the pia-normal might recover sz but
  not sxy — a partial solution that can't meet the 5 %/both-axes
  bar by itself.
- **Weakest 07b assumption.** The top-two GMM components bracket
  the noise/signal boundary (vs splitting the signal peak in two).
  Step-1 sanity gate + histogram plots catch the failure mode.
- **5 % bar implication.** This is aggressive. Even a perfect
  scale estimator with N-dependent sampling noise in median
  distances will have `~1/sqrt(N)` variability. For N ≈ 1000 the
  sampling variability on `median_knn_xy2d` is already ~3 %. A
  stricter GFP+ set lowers N, which tightens the requirement on
  bias reduction. Plan to estimate sampling variability via
  bootstrap in the notebook and label any 5 % pass that is within
  sampling error as "consistent with 5 %" rather than "passes
  cleanly."

## Next candidate (only if 07b fails — NOT implemented in this session)

Image-level 488 intensity correlation along the pia-normal axis:
1. Align surfaces properly: apply R1 (R, t) to CZ; rotate both so
   the HCR pia-normal is along +z.
2. In a matched xy ROI in that frame, build 1D intensity profiles
   `I_cz(z)` and `I_hcr_488(z)` by summing image intensity over xy
   slices.
3. Find `sz` that maximises `NCC(I_cz(z·sz), I_hcr_488(z))`.
4. For `sxy`, do the same along x and y in thin z-slabs at matched
   depth. `sxy` = stretch factor that maximises 2D NCC between CZ
   and HCR xy slabs.
5. Requires re-reading zstack + HCR 488 channel images; infrastructure
   in session 05 R1-revised gives a starting point, but its density-
   map NCC failed without surface alignment — pre-aligning surfaces
   is the key new contribution.

This remains a future session (07c'). Not implemented here.
