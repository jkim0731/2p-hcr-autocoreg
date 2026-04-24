---
name: 03 image-based surface (curved)
description: Session 03 — image-only curved surface. v2.1 + v2.2 auto-select + iter07 patch-MAX + IRLS-Huber deg-2 poly covers pia for 6/6 HCR and CZ. iter08 CZ 50 µm prior selection + HCR bottom boundary (z-flipped detector) is the PROMOTED main-pipeline surface: `info["cz_surface"]` / `info["hcr_surface"]` default to iter08/iter07 via `surfaces_iter08.py`, cached in `code/dev_code/cached_surfaces/`. iter09 baseline-anchored variant was built but NOT promoted — user kept iter08.
type: project
originSessionId: cabb1567-11f8-49c0-8a28-82d7327a0415
resumedFromSessionDeath: true
---
Session `03_image_based_surface_estimation` implements
`estimate_surface_and_l2_image_based()` in
`code/dev_code/03_image_based_surface.py`.  Five-stage pipeline:

1. **Global baseline** from the top 1–2 z-planes (median + 1.4826·MAD;
   skip planes ≥ 95 % zero; HCR pure-padding top → fallback
   `σ = 0.02 · percentile(vol, 99)`).  Used as diagnostic only.
2. **Stage-2 anchor** — per-column first-sustained-above-thr over the
   whole z on a sparse grid (`column_stride_um = 10`), fit quadratic
   IRLS-Huber to get a rough pia plane.
3. **Stage-3 per-column tops in a ±150 µm band** around the anchor
   (`column_margin = 0.25`, `min_signal_abs_frac = 0.20`,
   `column_min_thick_um = 10`).
4. **Stage-4 quadratic quantile-regression IRLS** at
   `target_quantile = 0.85` on stage-3 tops (module default; CZ
   driver opts down to 0.70 — see v2.1 below).
5. **Stage-5 column-top clamp** (image-only analogue of N22's ROI
   envelope clamp): 120-µm tiles, 10 % quantile of z per tile,
   `lift = max(positive_deficits) − safety_offset`.

Driver + figures in `03_image_based_surface_run.py` /
`03_image_based_surface_figs.py`; results + log in
`sessions/03_image_based_surface_estimation/`.

**v2 fix (2026-04-18):** max-lift over-lifts on HCR 755252/767022
because column tops are contaminated by shallow autofluorescence
(e.g. 755252 has 63 % of tiles with z10 above the real pia).  Added
two knobs to `_clamp_to_column_tops`:
* `lift_q` (default 1.0) — use `quantile(pos_deficits, q)` instead
  of max
* `lift_iqr_k` (default 1.5) — cap lift at Tukey fence
  `Q75 + k·IQR`, but only when `len(pos_deficits) ≥ 50`
  (gates out CZ's ~25 tiles where IQR is too noisy)

**v2.1 fix (2026-04-18):** raised the HCR `target_quantile` default
from 0.70 → 0.85.  Closed the 790322 above_frac gap (1.13 % → 0.06 %)
without regressing any other subject by more than a single onset bin.
755252 onset improved 72.5 → 62.5 µm but remains AF-limited.
CZ driver keeps `target_quantile = 0.70` (no AF issue; v2 CZ metrics
are already clean).

**How to apply:** Five of six HCR subjects now match N22 quality
(above_frac ≤ 0.35 %, onset ≤ 12.5 µm).  Only remaining stubborn
case: **755252 onset = 62.5 µm** — autofluorescence on 63 % of tiles
is irreducible without centroid data or a soma-level image primitive
(ruled out: 3-D peak detection / first_substantial / deeper anchor /
narrower band / per-column thick filter — every parameter that helps
755252 breaks ≥ 2 clean subjects).

**Do NOT** expect image-only to ever fully match N22 on every subject
— image-only cannot distinguish soma from autofluorescence the way
centroid positions do.  Treat as a complementary image-driven
cross-check for pia, not a replacement for N22.

**Future-work lead: per-subject channel selection.** On 755252,
`load_hcr_combined(channels=["488","514"])` followed by the normal
v2.1 pipeline yields surface c=153.9, onset=2.5 µm, above 0.51 %
— i.e. the pia IS recoverable when the AF-carrying channels (405,
594) are dropped. But the "good subset" is subject-specific: the
same `488+514` subset is catastrophic on 767018 (above 10 %),
782149 (above 13 %), etc.  An auto-channel-selection rule using a
confidence metric (e.g. below-vs-above-surface contrast ratio)
could close 755252's gap without regressing the others.  Out of
scope for v2.1 — genuine structural change.

**v2.2 (session 03c, 2026-04-19):** implemented the above future-work
lead.  Two new pure public functions in `03_image_based_surface.py`:
- `score_surface_quality(surface, vol, z_um, xy_um, ...) -> dict`
  with `Q = median(below_near)`, `Qabove = median(above_near)`,
  `Qcon_n = median((b-a)/(b+a+ε))` in windows [10,50] µm below and
  [-50,-10] µm above the candidate surface.
- `auto_select_surface(candidates, ref_vol, z_um, xy_um, *,
  qcon_n_threshold=0.9) -> (name, result, scores_df)`: argmax Q among
  candidates with `Qcon_n ≥ 0.9`.  The contrast gate separates three
  regimes — pia (Qcon_n ≈ 1), gap/AF-top (≈ 0), deep-in-tissue
  (0.1–0.55) — and correctly rejects both failure modes.

Convenience wrapper `estimate_pia_surface_image_autoselect(s, *,
level=4, qcon_n_threshold=0.9)` in `benchmark_analysis.py` builds the
per-subject channel-subset bank (`all_tq{0.70,0.85,0.95}`, `no405`,
`mid488` with per-subject 514/561 pick, `only488`, `only594`) and
returns `(surface, info, scores)`.

**Result.**  On the 6-subject HCR benchmark, auto-select matches N22
onset quality (`onset ≤ 2.5 µm`, `above_frac ≤ 1.13 %`) on **all**
subjects including 755252 — the fix the "do NOT expect" paragraph
above ruled out with v2.1 alone is now achievable via candidate-bank
selection rather than a single parameter choice.  Per-subject picks:
755252→`mid488_tq0.85`, 767018→`only594_tq0.85`, 767022→`all_tq0.70`,
782149→`no405_tq0.85`, 788406→`no405_tq0.85`, 790322→`all_tq0.70`.

Caveat: `argmax Q` is slightly deep-biased (deeper = brighter tissue),
so selected `c` values are within ~30 µm of N22's but can overshoot
(790322: +16 µm; above_frac 1.13 %).  All still within the onset/above
gates.  If tighter control is needed later (e.g. GFP surfaces), add
a shallow-bias term or weight Q by Qcon_n.

**Iter 5–6 (2026-04-20) — OOT-aware scoring.**  Added
`min_col_max_frac` (default 0.05) + optional shared
`global_max` / `global_max_percentile` kwargs to
`score_surface_quality`.  Columns whose column-max is below
`min_col_max_frac × global_max` are dropped before Q / Qabove / Qcon
/ Qcon_n medians; scoring dict exposes `n_oot`.  `auto_select_surface`
computes `global_max = np.percentile(ref_vol.max(axis=0), 99.5)`
once and shares it across the bank so every candidate uses the
same mask.

**Reference-choice design (critical).**  Under `vol.max()` as the
global-max reference, a handful of saturated voxels on
`load_hcr_combined` output (`vol.max()` = 32–253 × a typical tissue
column) inflate the 5 % threshold above most real columns — 100 %
OOT on 788406, 99.75 % on 790322; the scoring raises
`All-NaN slice encountered`.  Under `np.percentile(vol, 99.5)` the
reference is a typical bright *voxel* (~0.3) rather than a typical
bright *column* (~1.0) over the >1 B voxels of a level-4 HCR volume,
so 5 % of it passes every column including truly empty ones.
`np.percentile(colmax, 99.5)` is robust to single-voxel saturation
(each column contributes at most one bright-pixel's worth to
`colmax`) and samples the correct scale.  **Always use this form**
when adding OOT masks downstream.

**How to apply.**  On the 6-subject HCR benchmark the mask drops
0–9 of 400 grid columns per subject (15 %-padded interior grid is
mostly all-tissue).  Iter 5's widely-quoted "52–94 % OOT" was a
reference-inflation artefact from using `vol.max()`; the OOT
*loophole* that iter 3/4 failed to close was never the real issue on
this benchmark — but the mask is still worth keeping as (i) a
safety net against saturated-voxel crashes, (ii) a principled tissue
gate for future narrow-FOV or partial-crop data, and (iii) it
handles bottom-OOT implicitly via the downstream `Qcon_n ≥ 0.9`
gate (below-tissue surfaces read `Qcon_n ≈ -1`).  Selections under
the new scoring are unchanged on 6/6 subjects vs iter 4.

Session artefacts: `code/sessions/03c_onset_features/` — iterations
01–07 with logs, notebooks, figures, and CSV data.

**Iter 7 (2026-04-20) — curvature-capable image-only pia (TPS + patch-MAX).**
Motivation: user audit of iter 6 auto showed v2.1/v2.2 are planar
quadratics, and the real pia is curved.  Specifically (a) **755252**
N22 *under-estimates* the pia (real surface deeper); quadratics can't
track the curvature, (b) **790322** iter-6 auto sits too deep.  Both
failures point at the IRLS-quadratic fit, not at candidate-bank
scoring.

**Final detector (v6 after six sub-iterations):**
1. Log of the combined `load_hcr_combined` volume, offset by `EPS = 1e-3`.
2. Per grid point, take the **log-MAX over a 15 × 15 xy patch**
   (`PATCH_W = 7` half-width ≈ 60 µm at level 4).  *Not* mean — mean
   smears sparse L1 cells; max preserves them, which is what closes
   the 790322 gap.
3. Smooth the resulting column with `gaussian_filter1d`
   (`SMOOTH_Z_UM = 5.0`).
4. Column-adaptive threshold:
   `thr = max(THR_FLOOR = -6.3, col_p90_log − DELTA_LOG = 3.0)`.
   Cliff columns (755252, col_p90 ≈ -1) → thr ≈ -4 at cliff bottom;
   gradient columns (790322, col_p90 ≈ -3) → thr ≈ -6 at rise start.
5. First z where smoothed column stays above `thr` for
   `SUSTAIN_Z_UM = 15` → transition z for that column.
6. Fit **bivariate cubic polynomial** (`POLY_DEGREE = 3`, 10
   coefficients) via IRLS-Huber (`HUBER_K = 1.5`, 8 iterations) on
   the 400 transition points.  Coordinates normalised to
   `[-1, 1]` over the data range so extrapolation to the full FOV
   stays in `[-1.43, 1.43]` — bounded for degree ≤ 3.
   Implementation: `fit_polysurf` / `eval_polysurf` in
   `iter07_compute.py` (duplicated in `iter07_multiy.py` for the
   verification render).

   **Model evolution on this grid.**  TPS via
   `scipy.interpolate.RBFInterpolator(smoothing=200 → 10000)` was
   the first try; user review flagged the TPS surface as
   intensity-sensitive and wiggly even at smoothing=10000.  Switched
   to a 10-parameter cubic polynomial on 2026-04-21 because (a) 10
   degrees of freedom vs 400 points is strongly under-parameterised,
   so high-frequency wiggles are forbidden by construction, (b) IRLS
   downweights outlier transition z's (AF-latched columns, noise)
   without a hard cut, (c) boundary extrapolation for a low-degree
   polynomial is stable and monotonic.  Degree 2 is also viable if
   degree 3 is still too flexible on future data.

**Why patch-MAX (not patch-mean, not single-voxel):** log of the mean
dilutes isolated bright cells; max preserves them.  For cliff
subjects the max ≡ mean once tissue is dense, so the result is
unchanged.  Single-voxel columns are dominated by whichever voxel
happens to land at the grid point and miss sparse L1 cells entirely.

**Why column-adaptive threshold (not absolute):**  v3 absolute thr =
−4 worked on 755252 (+28 µm vs N22 ✓) but failed on 790322 (+94 µm
vs auto ✗).  Both cliff and gradient column profiles now fire at
"first meaningful signal above the column's own OOT noise".

**6-subject HCR bench (`iter07_summary.csv`, cubic poly IRLS-Huber):**

| subject | median_z_um | Δ vs N22 | Δ vs auto | note |
|---|---:|---:|---:|---|
| 755252 | 400.0 | +26.5 | +27.8 | x-slope preserved at all y |
| 767018 | 276.0 | +11.7 | +12.2 |  |
| 767022 | 256.0 |  +4.1 | +26.3 |  |
| 782149 | 308.0 | +21.2 | +14.0 | diagonal tilt captured |
| 788406 | 112.0 | +24.8 | +30.9 | thinnest subject |
| 790322 | 152.0 | +29.5 |  +8.4 | mild bow preserved |

All 6 subjects land 4–30 µm below N22 consistently — N22 is a
slightly-shallow estimator on this bench.  Valid-transition count is
400/400 per subject; no OOT drops or detector failures.  Median
transition z unchanged from the TPS pass (detector is identical);
surface-model change only affects the TPS→poly evaluation step.

**Status.**  Iter 7 is a standalone detector in `iter07_compute.py`,
not yet promoted into the public `benchmark_analysis` API.  Before
promoting:
- validate patch-MAX on subjects outside the 6-subject HCR bench
  (mouse GFP+, E15.5 cortex, non-HCR modalities).
- decide whether iter07 replaces the quadratic candidate bank
  entirely or becomes a parallel `_polysurf` entry point.
- the iter-6 auto path is still correct for centroid-comparable
  outputs (iter07 is systematically 4–29 µm deeper, so applications
  expecting "onset near N22" should keep using auto until iter07's
  deeper bias is characterised against external truth).

**Degree switch + CZ port (2026-04-21).**

*Degree.*  Side-by-side of `POLY_DEGREE=2` vs `POLY_DEGREE=3`
(`iter07_degcompare.py`) showed deg-3 introduced a mild edge-bow on
767018 and 767022 that didn't match the image (10 coefs with 4 cubic
terms overfit corners).  Deg-2 (6 coefs: `1, x, y, x², xy, y²`) keeps
one concavity per axis plus an xy twist — still curvature-capable,
no edge bow.  **Locked `POLY_DEGREE = 2`** for HCR and CZ.

*Threshold generalisation.*  Original
`thr = max(THR_FLOOR, col_p90 - DELTA_LOG=3)` failed hard on CZ: CZ
log p90 ≈ 7.5-8.7, offset drops thr below the camera-offset log
~5.5, so every column exceeded threshold from z=0 (median_z = 0 µm on
all subjects).  Replaced with a distribution-driven **range-relative**
rule (honours CLAUDE feedback on threshold generalisability):

```python
col_p10 = percentile(smooth, 10)
col_p90 = percentile(smooth, 90)
thr = max(thr_floor, col_p10 + TRANS_FRAC * (col_p90 - col_p10))
# TRANS_FRAC = 0.5 — "halfway between OOT floor and in-tissue ceiling"
```

`THR_FLOOR` now serves as a pure sanity floor (HCR `-6.3`, CZ
`4.5 ≈ log 90 DN`) just to reject columns that never see tissue.  HCR
regression check: 400/400 valid on all 6 subjects, median_z within
±4 µm of the DELTA_LOG result; no subject regressed.

*CZ port* (`iter07_cz_proto.py`, `data/iter07_cz_summary.csv`).  CZ
raw uint16 stacks (`*reg-dim-swapped.ome.tif`) have single-channel
intensities with camera offset ~30-100 DN and tissue ~500-1500 DN,
so `np.log(vol + EPS)` gives a narrower (p10 ≈ 6, p90 ≈ 7-9) range
than HCR combined.  Same detector (patch-MAX 15×15, smooth 5 µm,
sustain 15 µm, range-relative thr) + same poly-deg2 surface model.

CZ bench (400/400 valid per subject, all 6 HCR-named subjects):

| subject | log_p10 | log_p90 | thr | median_trans_z (µm) |
|---|---:|---:|---:|---:|
| 755252 | 6.51 | 7.97 | 8.03 |  58.0 |
| 767018 | 7.96 | 8.69 | 8.81 |  42.0 |
| 767022 | 6.30 | 7.68 | 7.52 | 102.5 |
| 782149 | 6.38 | 7.22 | 7.25 |  52.0 |
| 788406 | 6.55 | 7.07 | 7.12 |  49.0 |
| 790322 | 6.36 | 7.36 | 7.36 |  54.0 |

Visual check on `iter07_cz_<sid>.png` (YZ log-slice × 4 y positions,
with CZ centroid = green, CZ image_ceiling = cyan, iter07 poly-deg2 =
red): on all 6 subjects red tracks green centroid closely and sits
at visible tissue onset, below cyan (image_ceiling is the aggressive
column-top bound; red is the structural pia).  767022 has the
deepest CZ transition (102.5 µm), consistent with an OOT slab above
tissue — red still in tissue on every y.

Same detector + surface model (patch-MAX + range-relative threshold
+ IRLS-Huber deg-2 poly) now works on both HCR combined and CZ raw
with only a modality-specific `THR_FLOOR` sanity floor.

**Iter 8 (2026-04-22, resumed after session crash) — CZ 50 µm prior +
HCR bottom.**

*CZ prior selection* (`iter08_cz_prior.py`).  User noted CZ
acquisitions place pia at z ≈ 50 µm by construction.  Single
`TRANS_FRAC` can't cover everyone: `0.5` lands 767022 at 102.5 µm
(OOT slab), `0.02` lands 34-46 µm (latches on pre-pia AF).  Solution:
sweep `TRANS_FRAC ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}`,
precompute patch log-max columns once, select the candidate whose
median per-column z is closest to 50 µm.  MAD-gate columns at ±75 µm
around the prior before the same IRLS-Huber deg-2 poly fit.

Bench:
| subject | selected TF | median surf z | |Δ prior| |
|---|---:|---:|---:|
| 755252 | 0.20 | 48.5 | 1.5 |
| 767018 | 0.50 | 41.7 | 8.3 |
| 767022 | **0.15** | **50.2** | **0.2** (was 102.5) |
| 782149 | 0.50 | 45.9 | 4.1 |
| 788406 | 0.50 | 48.5 | 1.5 |
| 790322 | 0.30 | 50.0 | 0.0 |

767018's 8 µm offset is intrinsic (tissue literally at slice 0 edge);
the prior selects the detector, not the answer.

*HCR bottom* (`iter08_hcr_bottom.py`).  `load_hcr_combined` clips OOT
to 0 at BOTH ends, so the bottom is a symmetric cliff.  Per column,
patch log-max 15 × 15 xy, reverse in z, call `col_detect_transition`
unchanged (HCR params: `TRANS_FRAC=0.5`, `THR_FLOOR=-6.3`), then map
`z_flip → Z-1-z_flip` for original-volume coordinates.  MAD-gate ±3
× MAD before IRLS-Huber deg-2 poly fit.

Bench (bottom surface vs HCR centroid z p99):
| subject | top (µm) | bottom (µm) | thickness (µm) | Δ centroid p99 |
|---|---:|---:|---:|---:|
| 755252 |  396 | 1230 |  845 | −81 |
| 767018 |  268 | 1287 | 1014 | −6 |
| 767022 |  252 | 1255 | 1006 | −18 |
| 782149 |  300 |  856 |  553 | −3 |
| 788406 |  108 | 1258 | 1158 | −25 |
| 790322 |  144 | 1148 | 1005 | +9 |

All bottom surfaces within 25 µm of centroid p99 except 755252 (80
µm above, tissue genuinely extends past labelled cells there).
400/400 valid columns on every subject.  Patch-MAX at the bottom is
safe because tissue is dense neuropil (MAX ≡ mean locally), no
sparse-cell asymmetry like top L1.

**Key added params.**

```python
# iter08 CZ
CZ_TARGET_Z_UM = 50.0
TRANS_FRAC_BANK = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
GATE_UM = 75.0

# iter08 HCR bottom
MAD_GATE_K = 3.0
# detector params unchanged from iter07
```

**Artefacts.** `iterations/iter08_cz_prior.py`,
`iterations/iter08_hcr_bottom.py`; `data/iter08_cz_selection.csv`,
`data/iter08_cz_sweep.csv`, `data/iter08_hcr_bottom_summary.csv`;
`figures/iter08_cz_<sid>.png`, `figures/iter08_hcr_bottom_<sid>.png`.

**Iter 9 (2026-04-22) — CZ baseline-anchored detector, 50 µm demoted to a weak parameter.**

Motivation: iter08 forces the CZ surface to 50 µm via TRANS_FRAC
*selection*, so when the experimenter's 50 µm target actually *fails*
on a given brain (user-flagged case: 782149), iter08 still returns
~50 µm by picking a detector that does so.  iter09 uses 50 µm only
to bound the search and to size the baseline window — never as a
correction to the detector output.

**Detector** (`iterations/iter09_cz_baseline.py`).  Per column:
1. `log_col = log(patch-MAX(15×15) + ε)` (same as iter07/08).
2. Gaussian smooth 5 µm in z.
3. **Baseline** from `smoothed[z ∈ [0, Z_BASELINE_UM=30 µm]]`, take
   the darkest half (`≤ median`), robust-estimate `μ_b = median`,
   `σ_b = max(0.05, 1.4826·MAD)`.
4. **Threshold** `thr = μ_b + max(K_SIGMA·σ_b, LOG_MARGIN)` with
   `K_SIGMA = 3.0`, `LOG_MARGIN = 0.7` log-units (≈ 2.0× linear
   fold-change).  The log-margin floor is load-bearing: on clean
   dark air `σ_b → 0.05` is *genuine* Poisson noise so `3σ_b ≈ 0.15`
   corresponds to only a 16 % linear rise — any pre-tissue
   scattered-light ramp crosses it.  The `K·σ_b` term handles
   subjects whose baseline window is contaminated by shallow tissue
   (on 782149, `σ_b = 0.97` alone raises thr to 8.57).  Using `max`
   of the two means each subject is protected by whichever failure
   mode actually applies.
5. Bounded search `z ∈ [0, Z_MAX_UM = 120 µm]` (2.4× the 50 µm
   target) for first sustained crossing of `SUSTAIN_Z_UM = 15 µm`.
   Return NaN if no crossing.

Surface fit: IRLS-Huber bivariate deg-2 poly (unchanged), MAD-gated
around the per-subject median (no reference to 50 µm).

**Tuning.**  LOG_MARGIN sweep: at 0.0 (no floor), 5/6 subjects fired
at 22–38 µm — too shallow because 3σ_b ≈ 0.15 on clean baselines
catches scattered-light ramp.  At 1.0 (≈ 2.7×), 788406 overshot to
85 µm with only 77/400 valid columns (thr above most of that
dimmer stack's onset ramp).  **0.7 landed all 6 subjects at 39–52 µm
with 133+ valid columns; this is the chosen default.**

**Bench** (`data/iter09_cz_summary.csv`):

| subject | μ_b  | σ_b  | thr  | n_valid | med col z | iter09 surf | iter08 surf |
|---------|-----:|-----:|-----:|--------:|----------:|------------:|------------:|
| 755252  | 6.58 | 0.05 | 7.30 | 266     | 43.0      | 41.1        | 48.5        |
| 767018  | 7.93 | 0.05 | 8.64 | 352     | 39.0      | 38.6        | 41.7        |
| 767022  | 6.13 | 0.05 | 6.83 | 398     | 44.0      | 44.1        | 50.2        |
| 782149  | 5.68 | 0.97 | 8.57 | 133     | 37.0      | 39.6        | 45.9        |
| 788406  | 6.54 | 0.05 | 7.28 | 177     | 50.0      | 52.4        | 48.5        |
| 790322  | 6.29 | 0.05 | 7.06 | 237     | 48.0      | 47.2        | 50.0        |

All 6 subjects land 39–52 µm *without using 50 µm as a selection
criterion*.  The independent convergence validates that the 50 µm
target was genuinely hit on 5/6 subjects; 782149 and 767018 end up
9–11 µm shallower, and on 782149 the large `σ_b` is the signature
that the baseline window already contained tissue — the detector
picked this up from the data, not from the prior.

**How to apply.**  Use iter09 as the CZ-surface detector of record.
iter08 remains useful as (i) a prior-agreement sanity check and (ii)
the basis for the `select_trans_frac` sweep whose statistics help
diagnose whether a subject's 50 µm target actually held.  Do NOT
re-introduce the 50 µm prior as a selection criterion on any
downstream detector — keep it as a loose search bound only.

**Artefacts.** `iterations/iter09_cz_baseline.py`;
`data/iter09_cz_transitions_<sid>.npz`,
`data/iter09_cz_summary.csv`; `figures/iter09_cz_<sid>.png`.

**Key added params.**

```python
Z_BASELINE_UM = 30.0         # darkest-half window for μ_b, σ_b
Z_MAX_UM = 120.0             # bounded search cap (2.4 × target)
K_SIGMA = 3.0                # σ_b multiplier
LOG_MARGIN = 0.7             # floor on the above-baseline gap (log units)
BASELINE_LOWER_FRAC = 0.5
SIGMA_FLOOR_LOG = 0.05       # Poisson-noise floor for σ_b
```

**Iter 9 outcome (2026-04-22): NOT PROMOTED.**  User reviewed and chose
to stick with iter08 as the main-pipeline CZ surface.  iter09 remains
available as a research artefact in
`iterations/iter09_cz_baseline.py` / `iter09_cz_summary.csv` for
future re-visits (particularly useful if a future subject
genuinely has pia far from 50 µm and iter08's prior-selection would
force it toward 50).  Do not reach for iter09 by default.

---

### Promotion to main pipeline (2026-04-22)

**Module**: `code/dev_code/surfaces_iter08.py` — public module with:
- `compute_cz_surface_iter08(s)` → iter08 CZ prior-selected surface
- `compute_hcr_top_surface_iter07(s)` → iter07 HCR top surface
- `compute_hcr_bottom_surface_iter08(s)` → iter08 HCR bottom surface
- `get_*(s, use_cache=True)` cache-aware accessors (load from JSON)
- `save_cached_surface`, `load_cached_surface`
- `build_main_surface_store()` batch builder
- `_polyfit_to_abcpqr(fit)` — exact converter from iter07/08's
  normalised-coef polyfit dict to the canonical `{a,b,c,p,q,r}`
  format; verified numerically to ~1e-13 agreement with
  `eval_polysurf`.

**Cache**: `/root/capsule/code/dev_code/cached_surfaces/`
- `<sid>_cz_iter08.json`
- `<sid>_hcr_top_iter07.json`
- `<sid>_hcr_bottom_iter08.json`
- `main_surfaces_summary.csv` (all 6 subjects × all 3 surfaces).

Load cost is ~0.1-0.3 ms per surface per subject (JSON parse); no
CZ/HCR volume load on cache hit.

**analyze_subject defaults changed** (`code/dev_code/benchmark_analysis.py`):

- `cz_surface_method` default: `"image_ceiling"` → `"iter08"`
- `hcr_surface_method` default: `"quantile_ceiling"` → `"iter07"`

New keys added to `info`:
- `info["cz_surface_iter08"]`
- `info["hcr_top_surface_iter07"]`
- `info["hcr_bottom_surface_iter08"]`

Legacy keys (`cz_surface_image_ceiling`, `hcr_surface_quantile_ceiling`,
…) are still computed and returned so downstream ablations that
explicitly reference them continue to work.

**Compatibility**: all surfaces carry canonical `{a,b,c,p,q,r}` keys
(no `type` override), so `depth_from_surface`,
`_plane_normal_from_surface`, `_surface_z_at`, `coarse_align`,
`coarse_align_revised` all consume iter08 transparently.  End-to-end
smoke tests on 767022 (both `r1_coarse_align.coarse_align` and
`r1_revised.coarse_align_revised`) pass.

**How to apply.**  Downstream sessions (04, 05, 07, …) consume
`info["cz_surface"]` / `info["hcr_surface"]` unchanged — they
automatically get iter08/iter07 surfaces now.  To pin a specific
legacy method for regression work, pass
`cz_surface_method="image_ceiling"` etc. to `analyze_subject`.
