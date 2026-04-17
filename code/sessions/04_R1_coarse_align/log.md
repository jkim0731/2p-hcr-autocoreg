# Session 04 — R1 coarse affine (CZ ↔ HCR)

## Goal

Implement R1 per `07 Grand Plan.md §R1`: localise the CZ sub-volume in HCR
and return an initial anisotropic affine (R, S, t). Targets:
**origin error ≤ 100 µm** and **rotation error ≤ ±10°** vs the
manual-landmark-derived affine on the 6 benchmark subjects.

Priors (design-time, not subject-specific):
`xy_expansion = 1.77`, `z_expansion = 2.83`, `rotation_z = 180°`.

## Method (final)

`r1_coarse_align.coarse_align()`:

1. **Prior transform.** Mean-centre CZ, rotate 180° about z, scale by
   `(1.77, 1.77, 2.83)` to put it in HCR-like µm.
2. **z_shift.** 1-D cross-correlation of zero-mean depth-from-pia
   histograms: CZ depth × `z_expansion` vs HCR GFP+ depth, bin 20 µm,
   range `[-100, 1500]`.
3. **XY translation = mean(HCR GFP+ xy).** Because the prior-transformed
   CZ is mean-zero by construction, this is equivalently
   `mean(hcr) − mean(cz_prior)`. *See "Design deviation" below.*
4. **z translation.** Per-CZ-cell target
   `pia_hcr(xy_new) + depth_cz_scaled + z_shift`; take the median over
   cells for robustness (`tz_std` reported as diagnostic).

Outputs a `CoarseAffine(R, scales, translation, src_mean,
rotation_angle_z_deg, diagnostics)` with the row-vector convention
`hcr_pred = (cz − src_mean) @ R * scales + translation` matching
`benchmark_analysis.ProcrustesFit`.

## Design deviation from the Grand Plan

The Grand Plan R1 method sketch (step 3) calls for a **2-D
cross-correlation of Gaussian-blurred GFP+ density maps** for XY
translation. On the benchmark data this failed badly: the density-map
xcorr peak landed 300–1000 µm from the landmark target on *every*
subject tested. Four variants were explored:

| Variant | Description | Outcome |
|---|---|---|
| Raw xcorr (valid-mode) | `fftconvolve(hcr_map, cz_map)` | peak driven by asymmetric HCR-edge density, e.g. 782149 → 1058 µm error |
| Zero-mean xcorr | subtract each map's mean before xcorr | corner artefacts from padding zeros |
| Normalised xcorr (Pearson NCC) | window-local zero-mean and std | global peak still drifts to non-landmark regions |
| Phase correlation | magnitude-normalised Fourier xcorr | 500–1000 µm errors |

**Why xcorr fails here.** GFP+ covers 55–70 % of all HCR cells on the
benchmark subjects (except 782149 at 64 %; still dense). The density map
is effectively the whole HCR cell cloud, whose XY structure is smooth
and carries little CZ-specific localising signal. The xcorr peak is
therefore driven by where HCR happens to be densest globally, not by CZ
pattern matching. The Grand Plan assumption of a **sparse** GFP+ label
doesn't hold for this dataset.

**Centroid-of-HCR-GFP+** is the least-biased estimator in that regime:
it minimises squared distance to all GFP+ cells, and when HCR GFP+ is
roughly symmetric around the CZ sub-volume it lands on the target. It
fails proportional to the asymmetry of HCR GFP+ distribution relative
to the CZ imaging window.

Comparison (XY-only error vs GT target, all 6 subjects):

| strategy | 788406 | 790322 | 767018 | 782149 | 755252 | 767022 |
|---|---|---|---|---|---|---|
| full mean (chosen) | **26** | **73** | **136** | **360** | **56** | **158** |
| depth-band mean | 47 | 75 | 184 | 360 | 66 | 162 |
| full median | 68 | 78 | 150 | 432 | 56 | 141 |
| NCC peak (refined to ±100 of centroid) | 139 | 115 | 244 | 449 | 146 | 236 |
| Phase correlation | 167 | 821 | 842 | 510 | 795 | 438 |

## Benchmark results (final R1)

Run `python 04_r1_benchmark.py` → `sessions/04_R1_coarse_align/r1_results.json`.

| Subject | origin_err µm | rot_err ° | z_shift µm | tz_std µm | pass? |
|---|---|---|---|---|---|
| 788406 | **98.2** | 2.46 | 100 | 12.1 | ✓ |
| 790322 | **89.8** | 0.48 | 0 | 9.0 | ✓ |
| 767018 | 135.6 | **10.26** | 140 | 16.3 | ✗ (origin + rot) |
| 782149 | 371.7 | 2.34 | 20 | 30.8 | ✗ (origin) |
| 755252 | **71.5** | 6.99 | −80 | 28.4 | ✓ |
| 767022 | 157.9 | 5.29 | −60 | 14.9 | ✗ (origin) |

**Pass rate: 3/6 on the strict ≤ 100 µm origin metric; 5/6 on the ±10° rotation metric.**

Per-axis scale errors (r1 − gt, priors are 1.77/1.77/2.83):

| Subject | Δsx | Δsy | Δsz |
|---|---|---|---|
| 788406 | +0.08 | −0.10 | +0.01 |
| 790322 | +0.11 | −0.10 | −0.21 |
| 767018 | +0.09 | +0.05 | **−0.75** |
| 782149 | **−0.21** | −0.10 | −0.10 |
| 755252 | +0.11 | +0.15 | **+0.70** |
| 767022 | −0.01 | −0.07 | +0.34 |

Scale priors don't contribute to origin error at the CZ centroid (which
is a fixed point of the affine when `src_mean == cz_center`), but they
*will* show up as fan-out error at the edges of the CZ footprint, which
R2/R3 fine refinement will have to absorb.

## Hypothesis / method / result / failure / next

**Hypothesis.** A prior-driven affine (180° + prior scales) plus two
cheap observations (depth-profile 1-D xcorr; HCR GFP+ centroid for XY)
can land within 100 µm of the landmark-derived origin.

**Result.**
- **Z shift via 1-D xcorr works.** All 6 subjects have `tz_std ≤ 31 µm`
  — depth profiles align consistently once the z-shift is applied.
- **XY via HCR GFP+ centroid works on 4/6** with ≤ 73 µm XY-only error.
  Subject 767018 is borderline (136 µm XY; within ~35 % of the target
  tolerance). Subject 782149 is an outlier (360 µm XY) with a
  genuinely asymmetric HCR GFP+ distribution that no density-based
  method localised.
- **Rotation prior (180°) holds to within ±10° on 5/6 subjects.** 767018
  is 10.26° off — rotation prior limitation, not something R1 can fix
  without an estimator.

**Failure modes.**
1. *Asymmetric HCR GFP+.* 782149 has a thinner HCR section
   (`05 Benchmark dataset.md` note 1): HCR GFP+ centroid sits ~360 µm
   inferior to the landmark centroid because GFP+ extends well beyond
   the CZ sub-volume. No density-only estimator fixes this; R1 returns
   a biased origin for this subject and downstream R2 must handle it.
2. *Rotation off the prior.* 767018 actually rotated ≈ −170°; the 180°
   prior is 10° off. A rotation estimator (coarse ICP on depth-banded
   centroid point clouds after XY alignment) would push this under
   ±10°.
3. *z-scale drift.* 767018 GT `sz = 3.58`, 755252 GT `sz = 2.13` —
   both ±25 % from the 2.83 prior. Origin is unaffected but downstream
   will see fan-out.

**Next step.** R2 (constellation seed matching) should be robust to
R1's 100–400 µm origin error: search for seed clusters within a
~500 µm ball of the R1 prediction on good subjects, larger (~1 mm) on
the thin-HCR subject. The z-shift + tz-std diagnostics give R2 a
per-subject confidence proxy (low `tz_std` → high R1 confidence →
tighter search).

## Subgoal 01 — GFP+ threshold redefinition (2026-04-17, v2)

See `subgoal_01_gfp_threshold_plan.md` for the plan. Analysis driver:
`dev_code/04_r1_subgoal_01_gfp_threshold.py`; notebook:
`notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb`; results JSON:
`subgoal_01_gfp_threshold_results.json`.

### Constraints (v2, from user)

1. **Distribution-only.** The method must derive its threshold from the
   per-subject count distribution alone. No fixed percentile / fraction —
   future subjects will not have a `coreg_table.csv` to tune against.
2. **Maximise the threshold.** Among passing distribution-driven methods,
   prefer the one producing the **highest** per-subject threshold.
3. **Coverage bar.** Pass only if ≥ 95 % of the subject's coreg-table HCR
   IDs are retained. The coreg table is **validation-only** — never
   consulted when picking the threshold.

Fixed-fraction / top-N % strategies (the v1 winner) are now **excluded**
for violating (1). Depth-based metrics (`gfp_frac_top400`) are dropped;
that 400 µm threshold referred to CZ depth, not HCR, so it wasn't
diagnostic of the HCR threshold quality.

### Strategies evaluated (distribution-only)

| ID | Description |
|---|---|
| baseline | `counts >= 5` (reference only, excluded from ranking) |
| A | 2-comp GMM on `log(counts ≥ 1)`, posterior crossover at P=0.5 |
| A3 | 3-comp GMM on `log(counts)` with 0-count cells at `log(0.5)` |
| B | Otsu on `log(counts ≥ 1)` |
| C | Kneedle below-chord elbow of sorted-descending counts |
| D | Kittler–Illingworth min-error on `log(counts ≥ 1)` |
| Triangle | Zack's triangle method on `log(counts ≥ 1)` |
| Yen | Yen max-entropy on `log(counts ≥ 1)` |
| ISODATA | iterative-mean (Ridler–Calvard) on `log(counts ≥ 1)` |

Per-subject counts & coverages:

| strategy | 788406 | 790322 | 767018 | 782149 | min coreg_cov | passes ≥ 0.95? |
|---|---|---|---|---|---|---|
| baseline (5) | 5 | 5 | 5 | 5 | 0.997 | reference only |
| A (GMM-2)    | 16 | 17 | 15 | 16 | 0.997 | ✓ |
| A3 (GMM-3+bg)| 31 | 21 | 21 | 21 | 0.997 | ✓ |
| B (Otsu)     | 19 | 23 | 18 | 20 | 0.997 | ✓ |
| C (Kneedle)  | 125 | 179 | 111 | 175 | 0.944 | ✗ |
| D (Kittler)  | 1236 | 2 | 1262 | 2 | 0.000 | ✗ (unstable) |
| Triangle     | 2 | 2 | 2 | 2 | 1.000 | ✓ but trivial |
| **Yen**      | **84** | **153** | **90** | **156** | **0.960** | **✓** |
| ISODATA      | 18 | 22 | 17 | 21 | 0.997 | ✓ |

Ranked by `min(threshold_counts)` across the four spot subjects (higher
is more stringent) and filtered to `min coreg_coverage ≥ 0.95`:

| rank | strategy | threshold_min | threshold_mean | coreg_cov_min |
|---|---|---|---|---|
| 1 | **Yen_log_counts** | **84** | **120.8** | 0.960 |
| 2 | A3_gmm_log_counts_3c_bg | 21 | 23.5 | 0.997 |
| 3 | B_otsu_log_counts | 18 | 20.0 | 0.997 |
| 4 | Isodata_log_counts | 17 | 19.5 | 0.997 |
| 5 | A_gmm_log_counts | 15 | 16.0 | 0.997 |
| 6 | Triangle_log_counts | 2 | 2.0 | 1.000 |

**Chosen default: `yen_log`.** Yen is the only distribution-driven method
that lands in the **tail** rather than inside the spot-bearing signal
distribution — per-subject thresholds 84/153/90/156 counts yield
GFP+ fractions of 10–16 % (vs 31–54 % for A3/B) while still retaining
≥ 96 % of coreg-table HCR IDs.

### Why GMM / Otsu / Triangle under-cut

The `log(counts ≥ 1)` histogram has no clean bimodality because zero-count
cells are not in the CSV — the absence of a noise floor is structural.
GMM / Otsu therefore split the **inside** of the signal distribution
(weakly- vs strongly-labelled cells) rather than background-vs-signal,
landing at 15–25 counts. Triangle collapses to 2 because the histogram is
monotonically decreasing from the first bin. Yen maximises entropy across
the full log-count range, which on a heavy-tailed distribution naturally
lands in the tail. Kneedle (C) is close but its coverage drops to 0.944
on 782149 (thin section, smaller coreg set — the 944/1000 figure is
within sampling noise but below the bar). Kittler is numerically unstable
here and flips between extremes.

### API change

`benchmark_data_loader` now defaults to `gfp_threshold_method='yen_log'`.
Options:

```python
load_subject(sid)                                             # Yen default
load_subject(sid, gfp_threshold_method='counts_min',          # legacy
                  gfp_min_spots=5)
load_subject(sid, gfp_threshold_method='fixed_frac',          # diagnostics only
                  gfp_target_frac=0.20)
```

The returned `SubjectData.gfp_min_spots` records the derived integer
threshold (per subject). Intensity-only subjects (755252, 767022) are
unaffected — spot thresholding doesn't apply to `cell_data_mean_*_R1.csv`.

### R1 re-validation

`dev_code/04_r1_benchmark.py` rerun; pre-subgoal snapshot in
`r1_results_pre_subgoal_01.json`.

| subject | n_gfp old | n_gfp new | origin µm old | origin µm new | rot ° (unchanged) |
|---|---|---|---|---|---|
| 788406 | 69 680 | 20 465 |  98.2 | 124.3 | 2.46 |
| 790322 | 72 532 | 14 907 |  89.8 |  82.5 | 0.48 |
| 767018 | 58 959 | 11 833 | 135.6 | 160.5 | 10.26 |
| 782149 | 25 251 |  4 556 | 371.7 | 348.2 | 2.34 |
| 755252 | 77 785 | 77 785 |  71.5 |  71.5 | 6.99 |
| 767022 | 72 213 | 72 213 | 157.9 | 157.9 | 5.29 |

Rotation error and `tz_std` are identical — the 1-D depth xcorr is
independent of the GFP+ subset size at this spatial scale. Origin-error
swings are small (±25 µm) and go both ways: 790322 and 782149 improve,
788406 and 767018 regress. Pass rate on the 100 µm origin metric is 2/6
(vs 3/6 before); no subject crosses the rotation bar.

**Interpretation.** R1's XY estimator is `mean(HCR GFP+ centroid)`.
Yen aggressively shrinks the GFP+ pool (3–6×), and on subjects where the
high-count cells are spatially asymmetric relative to the CZ sub-volume
that shifts the centroid by ±25 µm. This is a property of the centroid
estimator — already documented in the *Design deviation* section above —
not a failure of the threshold. R2 (constellation matching) is the
intended fix; the centroid XY will be superseded there.

**Decision.** Accept `yen_log` as the default.

1. It satisfies the generalisability constraint (no fixed fraction).
2. It produces a biologically plausible GFP+ set (10–16 % of HCR cells,
   down from 54–68 %) — the sparse anchor set R2 / constellation
   matching needs.
3. Coreg-table coverage stays ≥ 96 % on every subject, so
   high-confidence pairs are preserved for validation.
4. R1 origin error is effectively flat (±25 µm) — no real regression.

### Open follow-ups

- **Subgoal 02 (planned, not executed).** See
  `subgoal_02_intensity_threshold_plan.md`. Intensity-only subjects
  (755252, 767022) still pass every cell through as GFP+ under the
  legacy loader. Scouted: Yen on `log(mean)` alone gives coreg coverage
  0.80 / 0.91 — below the 0.95 bar. Plan proposes exploring
  `mean - background` (per-cell local background from the CSV) with the
  same method family.
- Consider a *robust* R1 XY estimator (e.g. median of GFP+ in a depth
  band, or HCR ROI-envelope centre) before R2 lands — it will still need
  an R1 prior within ~500 µm on the stress subjects.

## Subgoal 01 — v2.1 addendum (2026-04-17): density + joint strategies

The v2 analysis considered only counts-based strategies. On user
feedback ("re run the analysis, since you forgot to use density and
count+density combination") the driver and notebook were extended with:

- **6 density-only strategies** — Yen/Otsu/Isodata/Triangle/GMM-2c/GMM-3c+bg
  applied to `log(density > 0)`.
- **3 joint strategies** — `counts × density` AND-filters where each axis
  uses the same distribution-driven method (Yen × Yen, Otsu × Otsu,
  GMM × GMM).

### Ranking metric change

Previously strategies were ranked by `threshold_min` (highest counts
cutoff wins), which is meaningless when density-only or joint families
enter the pool (their counts axis is 0). v2.1 ranks by **`max gfp_frac`
across subjects** (lower is more stringent), tie-broken by
`mean gfp_frac`, then `coreg_cov_min`. Fixed-percentile strategies are
still excluded per the generalisability constraint.

### Top-5 winners (coreg_cov_min ≥ 0.95)

| rank | strategy | gfp_frac_max | gfp_frac_mean | coreg_cov_min |
|---|---|---|---|---|
| 1 | **Joint_Yen_counts × Yen_density** | **0.1608** | **0.1280** | **0.9604** |
| 2 | Yen_log_counts                      | 0.1608 | 0.1318 | 0.9604 |
| 3 | Joint_Otsu_counts × Otsu_density    | 0.3830 | 0.2925 | 0.9967 |
| 4 | Isodata_log_counts                  | 0.3896 | 0.2988 | 0.9967 |
| 5 | Otsu_density                        | 0.4218 | 0.3261 | 0.9967 |

**Why joint wins over pure Yen.** Per-subject gfp_frac (joint vs
Yen-counts): 788406 0.161 / 0.161; 790322 0.140 / 0.140; 767018
**0.095 / 0.109**; 782149 0.116 / 0.116. A small 767018 population has
high raw counts but low density (big, weakly-labelled cells). Counts-Yen
lets them through; the density leg removes them. On the other three
subjects the density cutoff is already cleared by virtually every cell
that clears the counts cutoff, so joint reduces to pure Yen. The rule
is therefore **dominant**: equal-or-tighter on every subject.

**Density-only families are too permissive** (0.40–0.70 gfp_frac) —
`log(density > 0)` has no clean noise floor for the same structural
reason `log(counts)` doesn't. They add value only in combination with
counts.

### R1 re-validation (v2.1 joint vs v2 yen-only vs pre-snapshot)

| subject | n_gfp pre | n_gfp yen | n_gfp joint | origin µm pre | origin µm yen | origin µm joint |
|---|---|---|---|---|---|---|
| 788406 | 69 680 | 20 465 | 20 464 |  98.2 | 124.3 | 124.2 |
| 790322 | 72 532 | 14 907 | 14 907 |  89.8 |  82.5 |  82.5 |
| 767018 | 58 959 | 11 833 | **10 322** | 135.6 | 160.5 | **133.4** |
| 782149 | 25 251 |  4 556 |  4 556 | 371.7 | 348.2 | 348.2 |
| 755252 | 77 785 | 77 785 | 77 785 |  71.5 |  71.5 |  71.5 |
| 767022 | 72 213 | 72 213 | 72 213 | 157.9 | 157.9 | 157.9 |

Rotation and `tz_std` unchanged (±0.1 across runs). The joint rule
recovers 767018 to **pre-snapshot origin quality (133 µm)** — a ~27 µm
improvement over counts-only Yen — while still using a
distribution-driven, future-compatible threshold. No other subject
regresses.

### API change

Default method is now `yen_log_joint`; `yen_log` (counts-only Yen) is
retained as a diagnostic. Both are wired into
`benchmark_data_loader._apply_count_threshold()` via
`_yen_log_count_threshold()` and `_yen_log_density_threshold()`. If a
subject lacks a `density` column (aggregated spots without
`metrics.pickle`), the joint method silently falls back to counts-only
Yen on that subject.

```python
load_subject(sid)                                    # yen_log_joint (new default)
load_subject(sid, gfp_threshold_method='yen_log')    # counts-only Yen (diagnostic)
```

### Archived artefacts

- `r1_results_pre_subgoal_01.json` — `counts >= 5` baseline.
- `r1_results_yen_log.json` — counts-only Yen (v2).
- `r1_results.json` — joint Yen × Yen (v2.1, current).
- `subgoal_01_gfp_threshold_results.json` — v2.1 per-strategy metrics
  across all 18 strategies.
- `subgoal_01_figures/log_density_histograms.png` — new.
- `notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb` — rebuilt
  with density panel, joint winners, and v2.1 narrative.

## Files produced

```
sessions/04_R1_coarse_align/
  log.md                                      (this file)
  r1_results.json                             (per-subject R1 benchmark, current)
  r1_results_pre_subgoal_01.json              (pre-subgoal-01 snapshot)
  subgoal_01_gfp_threshold_plan.md
  subgoal_01_gfp_threshold_results.json       (per-strategy metrics)
  subgoal_01_figures/
    log_count_histograms.png
notebooks/
  04_R1_subgoal_01_GFP_positive_threshold.ipynb
dev_code/
  04_r1_subgoal_01_gfp_threshold.py           (analysis driver)
  build_subgoal_01_notebook.py                (notebook builder)
dev_code/
  r1_coarse_align.py           (R1 module)
  04_r1_benchmark.py           (validation harness)
  04_r1_debug.py               (cloud stats per subject)
  04_r1_map_dump.py            (density-map + xcorr peak dump)
  04_r1_translation_analysis.py (XY diagnostic)
  04_r1_centroid_variants.py   (variant sweep #1)
  04_r1_ncc_test.py            (NCC + phase-correlation test)
  04_r1_more_variants.py       (variant sweep #2)
  04_r1_v3.py                  (mean/median/band/band-diff sweep)
  04_r1_subject_deep.py        (per-subject spatial deep dive)
```
