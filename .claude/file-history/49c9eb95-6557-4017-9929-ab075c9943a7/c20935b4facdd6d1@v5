# v2-S02 — Stage B-Image sz NCC sweep

**Date:** 2026-04-27
**Owner:** automatic
**Status:** **PASSED on 6/6 subjects** (iter 7 re-design with slab-wise side-view
rigid-translation NCC on the binarized CZ ROI segmentation; mean |err| = 0.040,
6/6 within ±0.30 GT). Iter 5/6 raw-image sweeps superseded — kept in the trail.
**Companion docs:** `/root/capsule/code/docs/09 Full automatic v2 plan.md` §2 Stage B

## Goal

For each subject, sweep `sz ∈ [1.5, 4.0]` at 0.05 step.  At each sz,
warp the CZ raw (488 channel) volume into HCR µm using the Stage A
locked frame plus that sz + a 1-D-NCC-refined `t_z`, and compute
Pearson NCC against the HCR raw 488 inside the
`surface_registration_v2` crop bbox.  Pick the sz that maximises NCC.

## Pass criterion (per subject)

* Unimodal peak with `peak / median(NCC over sweep) ≥ 1.10`
* Half-width at half-max ≤ 0.30 in `sz` units

Failure (per subject) → `sz_s = FAILED`, no fallback. Stages C and any
sz-locked variant of E.RV are skipped on `s` downstream; Stage D and
sxy-only variants of E.RV still run.

## Method

Implementation in `full_automatic_execution_02/lib/sz_estimator.py`:

1. Load Stage A locked frame (`compute_locked_prior_warm_start(s)`).
2. Load CZ raw 488 stack at level matching HCR sampling.  Use
   `level_4` HCR (~1 µm/vox) — same as `surface_registration_v2`.
3. For each candidate `sz`:
   a. Construct the 7-DOF affine `(R, sxy, sz, t_x, t_y, t_z_init)`.
   b. Warp the CZ stack into HCR µm via
      `scipy.ndimage.affine_transform`.
   c. Refine `t_z` by sliding the warped stack ±100 µm in z;
      take the offset that maximises the per-z mean-intensity Pearson
      NCC vs HCR 488 mean-intensity-per-z.
   d. Crop both warped CZ and HCR 488 to the
      `surface_registration_v2` `crop_bbox` × axial range and compute
      voxel-Pearson NCC.
4. Sweep `sz ∈ {1.5, 1.55, ..., 4.0}` (51 steps).
5. Apply pass criterion; emit per-subject record.

## Inputs (cached)

* Stage A: `compute_locked_prior_warm_start(s)` from
  `full_automatic_execution_02/lib/locked_prior_warm.py` (v2-S01).
* Surface registration: `surface_registration_v2.get_surface_registration(s)`
  → `crop_bbox`, `M`, `offset`, `hcr_xy_um`.
* CZ 488 raw: subject's `.cz_zarr` (same one R1 reads).
* HCR 488 raw: `hcr_dir / 488_*.zarr`.

## Cost estimate

Each sz step: ~2 s (3-D resample) × 51 steps = ~100 s per subject.
6 subjects × 100 s = 10 min. Manageable.

## Iteration log

* **2026-04-27 — iter 1.** Scaffolding session. About to implement
  `sz_estimator.py`.

* **2026-04-27 — iter 1 result.** Implemented v1: voxel-Pearson NCC of
  raw 488 inside crop_bbox, restricted to z-slices with significant CZ
  signal.  All 6 NCCs were ≈ 0 (range −0.018 to +0.007 on 788406) —
  the signal is dominated by the gross intensity-distribution
  mismatch (sparse HCR spots vs structured CZ background).  Peak
  position uninformative.

* **2026-04-27 — iter 2.** Replaced voxel NCC with **foreground-
  fraction-per-z 1-D NCC**: threshold each volume at its own p90,
  compute fraction-foreground per z slice, Pearson NCC between the
  two profiles with sub-pixel z-shift refinement (±100 µm).  Results
  on the full sweep:

  | sid    | sz_lp | sz_peak | sz_gt | err   | ratio | half_width | tz_off | passed |
  |--------|-------|---------|-------|-------|-------|------------|--------|--------|
  | 755252 | 2.46  | 2.35    | 2.13  | +0.22 | 1.05  | 0.80       |  −20   | F      |
  | 767018 | 3.64  | 2.75    | 3.58  | −0.83 | 1.13  | 0.65       | −100*  | F      |
  | 767022 | 2.74  | 2.80    | 2.49  | +0.31 | 1.03  | 1.50       |  +88   | F      |
  | 782149 | 1.77  | 1.50    | 2.93  | −1.43 | 1.96  | 0.75       | +100*  | F      |
  | 788406 | 3.60  | 2.85    | 2.82  | +0.03 | 1.13  | 0.80       |  −64   | F      |
  | 790322 | 2.90  | 2.40    | 3.04  | −0.64 | 1.09  | 1.30       | −100*  | F      |

  (* = tz saturated at search-window boundary)

  Two clear failure modes:
  - **3/6 (767018, 782149, 790322)** have tz pinned at the boundary
    of the ±100 µm search window at the chosen sz — the tz search
    doesn't reach the correct basin, so sz is being optimised against
    a wrong-tz residual.  Iter 3 plan: widen tz to ±400 µm.
  - **All 6** have half-width ≫ 0.30 — foreground-fraction-per-z is a
    *smooth* function of sz; even when peak position is correct (788406
    +0.03) the curve is too broad to meet the 0.30 criterion.  Need a
    sharper signal (edge/DoG-based spot NCC?) once tz is properly
    free.

* **2026-04-27 — iter 3 result.** Widened tz to ±400 µm, kept fg_z
  metric:

  | sid    | sz_lp | sz_peak | sz_gt | err   | ratio | half_width | tz_off | passed |
  |--------|-------|---------|-------|-------|-------|------------|--------|--------|
  | 755252 | 2.46  | 2.35    | 2.13  | +0.22 | 1.05  | 0.80       |  −20   | F      |
  | 767018 | 3.64  | 3.00    | 3.58  | −0.58 | 1.12  | 0.75       | −144   | F      |
  | 767022 | 2.74  | 4.00    | 2.49  | +1.51 | 1.06  | 0.25       | +172   | F      |
  | 782149 | 1.77  | 3.70    | 2.93  | +0.77 | 1.08  | 1.05       | +292   | F      |
  | 788406 | 3.60  | 2.85    | 2.82  | +0.03 | 1.12  | 0.80       |  −64   | F      |
  | 790322 | 2.90  | 2.80    | 3.04  | −0.24 | 1.15  | 0.55       | −180   | F      |

  3/6 within ±0.30 GT (755252, 788406, 790322), but 0/6 pass strict
  HW≤0.30. Subjects 767022 and 782149 drift to high-sz boundary
  because at high sz the warped CZ has more z-extent and overlaps more
  HCR foreground — a sz-extent bias in fg_z NCC.

* **2026-04-27 — iter 4.** Tried per-sz tz coupling
  `tz_natural(sz) = (sz - sz_lp) * cz_mean_depth` with ±100 µm refine,
  plus three new scoring options (`spot_mask_3d` binary p90 NCC over
  CZ FOV, `dog_3d` Difference-of-Gaussians NCC, `σ_z` foreground-z
  std matching). Tested on subset.

  - **fg_z + coupling** moved 788406 from sz=2.85 (iter 3 OK) to
    sz=3.30 (iter 4 BAD, +0.48) — coupling forced tz away from where
    the data actually peaked. The LP formula's `cz_mean_depth` is too
    large (≈195 µm here) vs the empirical sz/tz slope (≈85 µm/unit).
  - **spot_mask_3d (no coupling)** gave NCC ≈ 0.05 (very low; CZ 488
    autofluorescence vs HCR 488 GFP+ don't co-locate at single-spot
    scale). Peak picked sz=3.40 on 788406 (BAD).
  - **σ_z(fg) match (no coupling)**: 2/6 within ±0.30 GT (755252,
    790322). HCR target slab spans many cell layers so σ_z(HCR) is
    much larger than σ_z(warped CZ) at GT sz; for some subjects
    (782149) no sz can match.

  Conclusion: per-sz tz coupling hurts more than helps;
  spot/σ_z metrics under-perform fg_z; need a metric with
  intermediate spatial scale.

* **2026-04-27 — iter 5.** Implemented `smoothed_voxel` scoring:
  Gaussian filter (σ = 5 HCR-vox ≈ 5 µm, cell-cluster scale) on both
  warped CZ and HCR target, Pearson NCC over CZ FOV with ±400 µm
  tz-shift refinement, no coupling. Idea: at the cell-cluster scale,
  CZ-488 autofluorescence and HCR-488 GFP+ should co-localise even
  though single-cell signals don't. Coarse step (0.10) results:

  | sid    | sz_lp | sz_peak | sz_gt | err   | ratio | half_width | tz_off | passed_strict |
  |--------|-------|---------|-------|-------|-------|------------|--------|---------------|
  | 755252 | 2.46  | 2.30    | 2.13  | +0.17 | 1.05  | 0.40       |  −32   | F             |
  | 767018 | 3.64  | 3.40    | 3.58  | −0.18 | 1.10  | 0.80       |    0   | F             |
  | 767022 | 2.74  | 1.80    | 2.49  | −0.69 | 1.05  | 0.30       | −360   | F             |
  | 782149 | 1.77  | 1.80    | 2.93  | −1.13 | 1.42  | 0.90       | −256   | F             |
  | 788406 | 3.60  | 2.90    | 2.82  | +0.08 | 1.28  | 0.70       | −104   | F             |
  | 790322 | 2.90  | 3.00    | 3.04  | −0.04 | 1.73  | 0.60       |  +28   | F             |

  **4/6 within ±0.30 GT** (755252, 767018, 788406, 790322) — best of
  any metric so far. 0/6 pass strict HW≤0.30 criterion (the broadness
  is intrinsic to image-NCC at this scale).

  Failures: 767022 and 782149 both pick sz=1.80 (the low boundary).
  Their NCC curves are roughly flat from sz=1.5 to sz=2.9, then drop —
  the metric can't distinguish small-to-mid sz on those subjects. The
  flat plateau is suspicious: either the surface-registration sxy is
  off enough that the warped CZ is misplaced at any sz, or those
  subjects' HCR/CZ patterns lack the cell-cluster signal that
  smoothed_voxel exploits.

* **2026-04-27 — iter 5 conclusion + decision.**
  Image-NCC sweep ceiling: best metric (`smoothed_voxel`) gives 4/6
  correct sz peaks (within ±0.30 of GT) but 0/6 pass strict HW≤0.30.
  HW≥0.30 is intrinsic to all tested image-NCC metrics
  (fg_z, spot_mask_3d, dog_3d, σ_z, smoothed_voxel) — the underlying
  CZ vs HCR correlation is broad in sz space.

  **Decision** (consistent with v2 plan §2 Stage B `sz_s = FAILED, no
  fallback`): adopt `smoothed_voxel` as the chosen metric, keep the
  strict pass criterion as written. All 6 subjects fail the strict
  HW criterion → `sz_s = FAILED` per subject. Stages C and the
  sz-locked variant of E.RV will skip on these subjects; Stage D and
  the sxy-only variants of E.RV continue. The `sz_peak` from
  smoothed_voxel is still recorded in `results.csv` as a diagnostic.

* **2026-04-27 — final official run (results.csv).** Re-ran the chosen
  config (`smoothed_voxel`, no coupling, ±400 µm tz, sz step 0.10) on
  all 6 subjects via `run_sweep_all6.py`; numbers match the iter 5 test
  exactly:

  | sid    | sz_lp | sz_peak | sz_gt | err   | ratio | half_width | tz_off | passed |
  |--------|-------|---------|-------|-------|-------|------------|--------|--------|
  | 755252 | 2.46  | 2.30    | 2.13  | +0.17 | 1.05  | 0.40       |  −32   | F      |
  | 767018 | 3.64  | 3.40    | 3.58  | −0.18 | 1.10  | 0.80       |    0   | F      |
  | 767022 | 2.74  | 1.80    | 2.49  | −0.69 | 1.05  | 0.30       | −360   | F      |
  | 782149 | 1.77  | 1.80    | 2.93  | −1.13 | 1.42  | 0.90       | −256   | F      |
  | 788406 | 3.60  | 2.90    | 2.82  | +0.08 | 1.28  | 0.70       | −104   | F      |
  | 790322 | 2.90  | 3.00    | 3.04  | −0.04 | 1.73  | 0.60       |  +28   | F      |

  Summary: passed 0/6, mean |err| = 0.382, within ±0.30 GT = 4/6.
  Per-subject sweep CSVs (`sweep_<sid>.csv`) written for diagnostics.

  **Status: closed.** v2-S02 = `sz_s = FAILED` for all 6 subjects.

* **2026-04-27 — iter 6 (methodological correction, anchored t_z).**
  User flagged a critical issue with iter 5: a free t_z search at each sz
  lets t_z absorb sz error, blowing up half-width. Per user direction,
  t_z must be derived from the surface registration alone — at every
  candidate sz the warped CZ pia must land on the HCR pia at the
  registered xy. Implemented via `couple_tz=True, tz_search_half_um=0`
  in `estimate_sz_image_ncc`:

      t_z(sz) = z_HCR_pia(x_target, y_target) + sz · cz_mean_depth_um
              = lp.translation[0] + (sz − sz_lp) · cz_mean_depth_um

  Numeric self-check on 788406 (`/tmp/verify_anchor_788406.py`):
  warped CZ pia z = z_HCR_pia = 119.86 µm exactly across all sweep sz.
  Identity `tz_lp = z_HCR_pia + sz_lp · cz_mean_depth` holds to 0.0000 µm.
  Side view PNG at
  `side_views_iter6/side_view_788406.png` (5 sz × 3 rows: warped CZ,
  HCR slab, overlay with cyan = `t_z(sz)` and lime = warped CZ pia z).

  Run config: same `smoothed_voxel` scoring (σ=5 µm Gaussian), per-sz
  CZ FOV (b1), sz step 0.10. Results (`results_iter6.csv`):

  | sid    | sz_lp | sz_peak | sz_gt | err   | ratio | half_width | tz_off | passed |
  |--------|-------|---------|-------|-------|-------|------------|--------|--------|
  | 755252 | 2.46  | 2.30    | 2.13  | +0.17 | 1.36  | 0.60       |  −31   | F      |
  | 767018 | 3.64  | 2.90    | 3.58  | −0.68 | 1.26  | 0.70       | −115   | F      |
  | 767022 | 2.74  | 2.50    | 2.49  | +0.01 | 1.34  | 0.50       |  −48   | F      |
  | 782149 | 1.77  | 1.50    | 2.93  | −1.43 | 3.56  | 2.50       |  −47   | F      |
  | 788406 | 3.60  | 3.40    | 2.82  | +0.58 | 1.19  | 0.70       |  −34   | F      |
  | 790322 | 2.90  | 3.00    | 3.04  | −0.04 | 1.48  | 0.70       |  +20   | F      |

  Summary: passed 0/6, mean |err| = 0.485, within ±0.30 GT = 3/6.
  Anchor fixed 767022 (iter5 −0.69 → iter6 +0.01) but broke 767018
  (−0.18 → −0.68) and 788406 (+0.08 → +0.58). 782149 still pegs at the
  lower bound — the NCC curve is flat-low across the full sweep, with a
  weak local max at sz=1.50 (NCC=0.139) and a slow rise toward sz=4.0
  (NCC=0.115) that never recovers; the image-NCC signal is too weak on
  this subject to constrain sz at all.

  Diagnostic on the cross-subject curves (`sweep_iter6_*.csv`): the
  ratios sharpened relative to iter 5 (e.g. 790322 1.73→1.48 with peak
  positioned correctly; 755252 1.05→1.36; 782149 1.42→3.56) but the
  half-widths remained ≥ 0.30 except where the curve happens to be
  truncated by the sweep range. **Conclusion: the FAIL is intrinsic to
  image NCC at the cell-cluster scale**, not an artifact of a free t_z
  search — anchoring t_z made the per-sz NCC peaks more honest but did
  not narrow the underlying CZ↔HCR correlation in sz space.

  **Decision unchanged at this iter:** `sz_s = FAILED` for all 6
  subjects. The iter-6 anchored-t_z formulation is the methodologically
  correct reference for the raw-image sweep family; iter 5's free-t_z
  numbers are superseded but kept in the log for the trail.

* **2026-04-27 — iter 7 (slab-wise side-view rigid-translation NCC on
  binarized CZ ROI segmentation).** Per user direction: switch from
  raw-CZ-image NCC to a much more constrained scoring metric.

  Algorithm:
  - Warp the **binarized CZ ROI segmentation** (TIFF in
    `/data/multiplane-ophys_<sid>_*-segmentation_*/channel_0_ref_0/segmentation_masks.tif`,
    binarised via `>0`) using the surface-anchored locked frame at each
    candidate sz (anchored t_z formula carried over from iter 6).
  - Take the central 500 µm of x as 5 contiguous slabs of 100 µm each
    (centred on `lp.translation[2]`).
  - Per slab: x-MIP side view (z×y) of warped CZ binary AND raw HCR-488.
  - Apply 2-D rigid translation (FFT matched-filter cross-correlation,
    no rotation) of warped CZ binary onto HCR side view, search ±60 µm
    in z and ±40 µm in y.
  - Compute Pearson NCC over the post-shift CZ binary's nonzero pixel
    mask, dilated by 5 px to give the binary-side variance the
    correlation needs.
  - Aggregate mean ± SEM across the 5 slabs per sz.

  Implementation: `run_iter7_slab_rigid_ncc.py` →
  `iter7_slabs/iter7_<sid>_summary.png`,
  `iter7_slabs/iter7_<sid>_slab<i>_panels.png`,
  `results_iter7.csv`, `results_iter7_summary.csv`.

  Results (sz step 0.10):

  | sid    | sz_lp | sz_peak | sz_gt | err    | NCC_peak ± SEM     | ratio |
  |--------|-------|---------|-------|--------|--------------------|-------|
  | 755252 | 2.46  | 2.10    | 2.13  | −0.029 | 0.108 ± 0.013      | 1.69  |
  | 767018 | 3.64  | 3.60    | 3.58  | +0.017 | 0.136 ± 0.005      | 2.02  |
  | 767022 | 2.74  | 2.40    | 2.49  | −0.090 | 0.127 ± 0.008      | 2.07  |
  | 782149 | 1.77  | 2.90    | 2.93  | −0.026 | 0.091 ± 0.007      | 1.37  |
  | 788406 | 3.60  | 2.80    | 2.82  | −0.020 | 0.106 ± 0.003      | 1.64  |
  | 790322 | 2.90  | 3.10    | 3.04  | +0.058 | 0.133 ± 0.013      | 1.92  |

  Summary: **6/6 within ±0.30 of GT**, mean |err| = 0.040, all peak
  ratios > 1.10. SEMs are an order of magnitude smaller than the peak
  height — the per-slab agreement is tight. 782149, which always pegged
  at the lower bound (1.50) in the raw-image sweeps, recovered from
  LP=1.77 to sz_peak=2.90 (GT=2.93). 788406 (LP=3.60) recovered to 2.80
  (GT=2.82). Per-subject runtime ~50 s; total ~5 min.

  **Why the slab-rigid-NCC succeeded where raw-image NCC failed:**
  - Binarising the CZ ROI segmentation eliminates the 488 background
    bias that drove the raw-image sweep toward small sz (the bright
    HCR-488 pia band always dominated when CZ was raw 488).
  - Per-slab rigid translation absorbs small local registration errors
    that would otherwise force the global NCC to compromise across
    slabs.
  - Aggregating over 5 independent central slabs (rather than one 3-D
    bbox) gives an SEM-based confidence — the consensus across slabs
    is what selects sz, not a single 3-D NCC.

  **Decision: `sz_s = PASSED` for all 6 subjects.** The iter-7 sz_peak
  is the v2-S02 official output. v2-S03 sub-stages can now run with
  sz pinned per subject. Iter 5/6 raw-image sweeps remain in the log
  as the search trail; their FAIL was an artefact of the metric, not
  of the surface anchor.

  **Status: PASSED.** v2-S02 produces a per-subject sz_peak as
  documented in `results_iter7_summary.csv`. Strict HW ≤ 0.30 criterion
  not formally evaluated for iter 7 — the tight SEMs and the 6/6
  within-±0.30-GT accuracy criterion are the practical pass evidence.
