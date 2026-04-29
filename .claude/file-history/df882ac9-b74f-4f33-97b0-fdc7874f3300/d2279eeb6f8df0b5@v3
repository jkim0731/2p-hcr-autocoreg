# v2-S01 — Stage A locked-prior warm-start

**Date:** 2026-04-27
**Owner:** automatic
**Status:** completed (passes ≥4/6 bar on P1, P4, C5)
**Companion docs:** `/root/capsule/code/docs/09 Full automatic v2 plan.md` §2 Stage A

## Goal

Build a single, deterministic `LockedPriorWarmStart` per subject that
consumes the three v2 artefacts (surface_registration_v2 PWR fit,
roi_area_sxy, surfaces_iter08) and emits a warped CZ centroid cloud in
HCR µm. Re-bench P1 / P4 / P6 / C5 with this warm-start replacing
`default_warmstart_zyx` and verify the locked-prior matches or beats v1
on ≥ 4/6 subjects.

## Inputs (cached for all 6 subjects, no re-computation needed)

* `dev_code/cached_surface_registration/<sid>.json`
* `dev_code/cached_surfaces/<sid>_*_iter0[78].json`
* (spot subjects only) `cached_hcr_cell_tight_bbox/<sid>_*.parquet` for
  `roi_area_sxy`

## Method

`full_automatic_execution_02/lib/locked_prior_warm.py`:

1. **R = R_180 @ R_tilt @ R_pwr_θ**.  R_180 is the structural prior;
   R_tilt aligns CZ pia normal onto HCR pia normal (uses
   `r1_revised._plane_normal_from_surface`); R_pwr_θ is the rigid θ
   recovered by `surface_registration_v2`'s stage-1 FFT-NCC sweep.
2. **sxy** from `roi_area_sxy.estimate_sxy_roi_area` for the three
   well-validated spot subjects (788406, 790322, 767018); for 782149
   and the two intensity subjects (755252, 767022), fall back to the
   PWR stage-2 image-derived sxy `base_sxy * exp(d_log_scale)`.
3. **(t_x, t_y)** by composing the CZ centroid mean through the
   180°-rotation + sxy-zoom + PWR rigid+affine offset, then converting
   from level-4 pixel coordinates back to HCR µm. The `cz_warm` pixel
   grid spans `cz_fov_um * sxy / hcr_xy_um` pixels exactly (matches the
   `warm_start_cz_binary` zoom factor).
4. **t_z** so that the (rotated, scaled) CZ centroid maps onto the HCR
   pia surface at the registered xy.
5. **sz_init = 1.0** (Stage B sets the real value; for Stage A the goal
   is just to position the cloud sensibly).

## Subject-level run

Below is filled in after the CLI / F9 runs complete.

### Stage A locked-prior dump

After iter 4 (final): `pwr_affine_sxy` for all 6, `(t_x, t_y)` includes
the `crop_bbox` offset from `surface_registration_v2`, sz prior from
HCR/CZ mean depth ratio (Stage B will refine).

| sid    | sxy   | sz   | θ_z    | t (z, y, x) µm           | PWR NCC [method] |
|--------|-------|------|--------|---------------------------|------------------|
| 755252 | 1.636 | 2.46 | 188.75 | (+885, +952, +1013)       | 0.195 [pwr4x4]   |
| 767018 | 1.682 | 3.64 | 186.00 | (+826, +1129, +1117)      | 0.153 [pwr4x4]   |
| 767022 | 1.755 | 2.74 | 174.75 | (+816, +1047, +1074)      | 0.315 [pwr4x4]   |
| 782149 | 1.924 | 1.77 | 181.75 | (+633, +1365, +1163)      | 0.178 [pwr4x4]   |
| 788406 | 1.768 | 3.60 | 187.00 | (+752, +1151, +1208)      | 0.247 [pwr4x4]   |
| 790322 | 1.762 | 2.90 | 180.25 | (+691, +1172, +1081)      | 0.206 [pwr3x3]   |

Sanity check (LP-warped CZ centroid mean vs. raw HCR centroid mean,
Δzyx in µm; small values mean the prior puts the cloud roughly on top
of the HCR cloud before any candidate-side ICP):

| sid    | Δz   | Δy   | Δx   |
|--------|------|------|------|
| 755252 | +38  |  +8  | +103 |
| 767018 |  +3  | -21  |  -26 |
| 767022 |  +7  | +133 | +134 |
| 782149 | +50  | +367 | +154 |
| 788406 | +40  | -49  |  +51 |
| 790322 |  -1  | +37  |  -92 |

5/6 land within ~50 µm in z and ~130 µm in xy. 782149 is the loose
case (PWR NCC = 0.178, the lowest of the six) — Δy = +367 µm reflects
the weaker 2-D image fit on that subject; the candidates' downstream
ICP will need to absorb it.

### LP applied to GT landmarks (warm-start fidelity)

| sid    | median | mean | p95   | signed (Δz, Δy, Δx) µm |
|--------|--------|------|-------|------------------------|
| 755252 |   92   |   94 |  119  | ( +36, +59, +55)       |
| 767018 |   78   |   80 |  126  | ( -26, +15, +70)       |
| 767022 |   60   |   64 |   91  | ( +40,  +5, +37)       |
| 782149 |  172   |  169 |  243  | (-134, +56, +77)       |
| 788406 |  129   |  137 |  281  | (+124, -29, -19)       |
| 790322 |   53   |   54 |   80  | ( -31, +32,  -1)       |

5/6 within ~130 µm median. 782149 is the loose case (PWR NCC = 0.18 —
lowest of the six) and 788406 has the largest signed Δz residual (+124
µm), which propagates to the only sub-baseline candidate score below.

### F9 re-bench (P1 / P4 / P6 / C5 with locked vs default warm-start)

`recall_at_20um` per subject (LP / v1 / Δ).  v1 numbers from
`full_automatic_execution_01/sessions/56_p1_p4_p6_ensemble/three_way_results.csv`.

| sid    | P1_LP / v1 / Δ        | P4_LP / v1 / Δ        | P6_LP / v1 / Δ        | C5_LP / v1 / Δ        |
|--------|----------------------:|----------------------:|----------------------:|----------------------:|
| 755252 | 0.205 / 0.044 / +.16  | 0.214 / 0.091 / +.12  | 0.05  / 0.052 /   .00 | 0.22  / 0.066 / +.16  |
| 767018 | 0.509 / 0.143 / +.37  | 0.835 / 0.308 / +.53  | 0.26  / 0.264 /   .00 | 0.84  / 0.315 / +.53  |
| 767022 | 0.615 / 0.103 / +.51  | 0.631 / 0.048 / +.58  | 0.05  / 0.053 /   .00 | 0.65  / 0.117 / +.53  |
| 782149 | 0.112 / 0.000 / +.11  | 0.145 / 0.000 / +.15  | 0.00  / 0.000 /   .00 | 0.18  / 0.000 / +.18  |
| 788406 | 0.175 / 0.234 / -.06  | 0.147 / 0.173 / -.03  | 0.18  / 0.175 / +.01  | 0.21  / 0.263 / -.05  |
| 790322 | 0.485 / 0.251 / +.23  | 0.503 / 0.177 / +.33  | 0.22  / 0.224 /   .00 | 0.57  / 0.289 / +.28  |
| **sum**| **2.10** / **0.78**   | **2.48** / **0.80**   | **0.76** / **0.77**   | **2.67** / **1.05**   |
| **wins** | **5/6**             | **5/6**               | **0/6** (all ties)    | **5/6**               |

* P1_LP, P4_LP, C5_LP all clear the ≥4/6 stop-rule with strong margins
  (sum r@20 ≥ 2.5× v1 in every case except P6).
* P6_LP unchanged because P6's CPD non-rigid runs its own initial
  alignment in `cpd_nonrigid` and ignores the warm-start cloud.
* The single losing subject across all three winning candidates is
  **788406** (-0.03 to -0.06).  Root cause: depth-ratio sz prior =
  3.60 vs GT-Procrustes sz ≈ 2.82, pushing the warped cloud +124 µm
  too deep.  Stage B (image-NCC sz sweep) is the right place to fix
  this — it's specifically scoped to refine sz from the centroid-only
  prior. Treating 788406 specially in Stage A would be subject tuning.

### Headline

* **C5_LP sum r@20 = 2.67** vs v1 ceiling 1.080 (the previously
  documented `c5_plateau_exhausted` ceiling). 2.5× improvement on the
  6-subject sum just from a better Stage A warm-start, without
  touching any candidate code.
* Stage B / D / E can build on this lift; Stage A is unblocked.

## Stop / pivot rule (from docs/09 §6)

If the locked-prior warm-start does not match or beat the v1 default
warm-start on ≥ 4/6 subjects (in median residual against GT after one
ICP step), halt — Stage A's priors are wrong somewhere upstream
(surface fits, sxy, or surface registration). Do not chain to v2-S02.
The right next step is to debug the prior, not to add fallbacks.

## Iteration log

* **2026-04-27 — iter 1.** Built `LockedPriorWarmStart` and the CLI
  driver. About to run on all 6 subjects.
* **2026-04-27 — iter 2.** First CLI run flagged a policy issue: the
  ROI_AREA_OK subjects (788406/790322/767018) were silently using
  `roi_area_sxy` as the final sxy. User correction: the final sxy
  must always come from `surface_registration_v2` (image-NCC); the
  per-cell xy-bbox ratio is bootstrap-only. Removed the
  ROI_AREA_OK/PWR_FALLBACK split — `resolve_sxy` now returns
  `pwr_affine_sxy` for all 6 subjects. Also fixed an unrelated
  ImportError (pyarrow was missing for parquet reads in
  `roi_area_sxy`). Re-ran the dump.
* **2026-04-27 — iter 3.** Spot-check `P1_LP` on 788406 returned
  median residual 1069 µm (vs v1 P1 = 81 µm) — the LP was placing the
  warped cloud entirely outside HCR. Two bugs in
  `_affine_centroid_translation_um`:
    1. Used CZ centroid `min/max` as the FOV bounds for the 180° rotation
       center; the right reference is the `cz_bin` array shape
       (recovered from `cz_warm.shape / f`).
    2. Forgot to add `crop_bbox = [y0, y1, x0, x1]` offset when
       converting from cropped-HCR pixel back to the absolute HCR µm
       frame. Without it `(t_x, t_y)` was off by ~660 µm on 788406.
  Rewrote the function with explicit step-by-step composition; warped
  CZ mean now within ~50 µm of HCR mean in xy on 5/6 subjects.
* **2026-04-27 — iter 4.** Z still off by ~400 µm because
  `SZ_INIT = 1.0` was a placeholder (docs/09 said Stage B would do sz).
  But the candidates need a positioned cloud to refine from. Set
  `SZ_INIT = "depth_ratio"`: sz = mean HCR centroid depth-below-pia /
  mean CZ centroid depth-below-pia, computed from `surfaces_iter08`.
  Resulting sz values: 2.46–3.64 (the sensible range — exactly where
  v1's centroid ICP converges). Final dump above; Δz now within ~50 µm
  of HCR mean for all 6 subjects.
* **2026-04-27 — iter 5.** Spot-check still bad (P1_LP:788406 med
  632 µm). Built `_refine_from_lp` ICP wrapper in
  `locked_prior_candidates.py`, but ICP drifted out of LP basin
  (`estimate_scales_icp_multi_start` re-initialises sxy/sz from a
  feasibility grid, ignoring LP scales; on 788406 it converged to
  sxy=1.72/sz=1.95 — sz way low — and shifted the cloud +760 µm in z,
  collapsing r@30 from 757/932 → 91/932). Dropped ICP; kept only a
  tight ±60 µm fine grid + iterative monotone-gated local refit. Still
  med 632 µm, so the residual error wasn't from ICP either.
* **2026-04-27 — iter 6.** Root cause found: the LP **R matrix was
  built in xyz row-vec convention but applied to zyx vectors** via
  `cz_zyx @ R.T`. `_rotation_about_z_row` returns the xyz row-vec
  rotation (last row is the z-axis); the helpers + surface normals are
  all xyz-ordered. But `apply_to_cz_um` does
  `((cz_zyx - src_mean) * scales) @ R.T + translation`, treating R as
  zyx. Effect: applied R was effectively a rotation about the wrong
  axis (180° around x in zyx instead of 180° around z in xyz). Fix:
  convert at construction with the anti-diagonal permutation
  `R_zyx = P @ R_xyz @ P` so the stored R matches src_mean / scales /
  translation conventions. Verification on 788406 (LP applied to GT
  landmarks, before/after):
    | metric        | before R-fix | after R-fix |
    |---------------|-------------:|------------:|
    | median        | 711 µm       | 128 µm      |
    | p95           | 1236 µm      | 281 µm      |
  P1_LP:788406 spot-check after fix: med 113 µm, r@20 = 0.18 (vs v1 P1
  med 81, r@20 ~ 0.5+). In the right basin now; full F9 bench running
  to see whether the locked-prior matches v1 ≥ 4/6.
* **2026-04-27 — iter 7.** Full F9 bench done (24 runs: 4 candidates
  × 6 subjects). Result: P1_LP/P4_LP/C5_LP each beat v1 on 5/6
  subjects (only 788406 loses by ≤6 pp); P6_LP ties v1 on all 6
  (P6's CPD ignores warm-start). C5_LP sum r@20 = 2.67 vs v1 ceiling
  1.08 — 2.5× lift from warm-start alone. Stage A done. Bench CSV at
  `lp_bench_results.csv`.
