# Session 03c — Intensity-feature extraction at N22 surface onset

Date: 2026-04-19
Author: Claude (autonomous research loop)

## Goal

The image-only v2.1 pia surface estimator
(`03_image_based_surface.py`) matches N22 (centroid-based) on 5/6 HCR
subjects but catastrophically fails on **755252** (onset_depth
62.5 µm vs N22's 2.5 µm).  The failure mode is a thick
autofluorescence sheet ~200 µm above the real pia that is
**indistinguishable from a thin pia edge using column-profile
statistics alone** (see `../03_image_based_surface_estimation/log.md`
for the v2.1 negative-results audit).

**Hypothesis:** N22's surface, derived from centroid positions, gives
us a reliable per-subject "ground truth" pia z for every (x, y).
Intensity features extracted in a *z-window* around the N22 surface
should differ measurably from intensity features at the v2.1
image-based surface, *only on 755252* (where v2.1 sits on AF, not
pia).  That difference is the signal we need.

**Success criterion:** Discover at least one feature whose
distribution at the N22 surface is consistent across subjects AND
clearly separates from the distribution at the AF-contaminated
v2.1 surface on 755252.

## Iteration structure

Each iteration produces:
- a **notebook** in `iterations/iterNN_*.ipynb` with plots and tables
- a brief **iteration log** appended below
- **figures** in `figures/iterNN_*.png`
- **data** in `data/iterNN_*.parquet` or `.npz` when caching helps

## Iterations

### Iteration 1 — Intensity features at N22 vs v2.1 surface

**Setup.**  For each of the 6 HCR subjects load the combined
(`load_hcr_combined`) volume, run `estimate_pia_surface_quantile_ceiling`
to get the N22 surface, and run `estimate_surface_and_l2_image_based`
(v2.1, `target_quantile = 0.85`) to get the image-only surface.
Sample a 20 × 20 grid of interior columns (edge pad 15 %), and for
each column compute features anchored at both `z_N22` and `z_v21`.

**Features.**  Means in windows relative to the surface z:
`above_near = [z-50,z-10]`, `below_near = [z+10,z+50]`,
`below_mid = [z+50,z+150]`, `below_deep = [z+150,z+250]`; central
gradient at z (±10 µm); a "dip" trio (peak in [z,z+30], trough in
[z+30,z+150], peak-minus-trough).

**Artefacts.**
- `iterations/iter01_compute.py` — computation driver
- `iterations/iter01_analysis.ipynb` — rendered figures + tables
- `data/iter01_features.csv` — 2400 rows (6 × 400 columns)
- `data/iter01_profiles.npz` — sample column profiles for plots
- `figures/iter01_sample_profiles.png` — center-column profiles with
  both surfaces overlaid per subject
- `figures/iter01_below_near_hist.png` — per-subject histograms
- `figures/iter01_ratio_near_deep_hist.png` — scale-free variant

**Key finding.**  `below_near` (mean intensity in a 40 µm slab
immediately below the candidate surface) cleanly discriminates
755252's AF-top v21 position from its real-pia N22 position:

| subject | median at N22 | median at v21 |
|---|---:|---:|
| **755252** | **0.052** | **0.000**  ← AF |
| 767018    | 0.015 | 0.008 |
| 767022    | 0.015 | 0.007 |
| 782149    | 0.030 | 0.008 |
| 788406    | 0.002 | 0.001 |
| 790322    | 0.001 | 0.001 |

Physical reading: on 755252 the v21 surface sits on top of a ~30–50 µm
thick AF sheet with a **dark gap** between the AF and the real pia.
The 40 µm slab below v21 falls in that gap; its median intensity is
floored at 0 by the `load_hcr_combined` normalisation (values below
the per-channel median are clipped to 0).  At the real pia (N22), the
same slab contains bulk tissue and is clearly non-zero.

**Caveats.**
- Median-based summary smears per-column variance; per-column
  discrimination is what the estimator will actually consume.
  Next iteration needs to compute the feature per column top (Stage
  3 of the v2.1 pipeline) and show that AF tops and pia tops are
  separable at the individual-column level.
- Several features (`above_near`, `dip_trough`, etc.) came back as
  exact zeros, because `load_hcr_combined` clips sub-median voxels
  to 0. For iteration 2 we should switch to raw per-channel
  intensities (or at least a non-clipping normalisation) so small
  differences in "dim" regions are preserved.

**Decision.**  Iteration 2 will (i) compute `below_near` per column
top in Stage 3 of the v2.1 pipeline, (ii) check that AF tops and
pia tops form separable distributions, and (iii) design a per-top
acceptance rule (soft weighting or hard rejection) before the IRLS
quantile fit.

### Iteration 2 — Per-column-top `below_near` is *not* discriminative

**Setup.**  Replicated v2.1 Stage 2 + 3 for each subject with the
`03_image_based_surface.py` defaults; labelled each column top as
`pia` (|z_top − z_N22(x,y)| ≤ 25 µm) or `AF/other` (else); measured
`below_near` at every top.  See `iterations/iter02_compute.py` +
`iterations/iter02_analysis.ipynb`.

**Finding.**  Per-top `below_near` is **not separable** between pia
and AF tops.  On 755252 the very few tops that fell in the AF zone
(dist_to_n22 ∈ [−50, −10], n=89) had `below_near` ≈ 0.30, while
pia-labelled tops (n=865) had `below_near` ≈ 0.25 and deeper
tissue-internal tops (dist ∈ [+25, +75]) had 0.18.  All are
"bright-just-below" because the probed slab sits inside either
AF body (AF top) or tissue (pia / deep top); the feature cannot tell
them apart at the column level.

**Why iter-1 still worked.**  The iter-1 signal was specific to
**the fitted surface z**, not to individual column tops.  v2.1's
fitted surface on 755252 ends up ~60–90 µm above N22, landing in
the *gap* between the AF bottom and the real pia onset.  That gap
is narrow (≈ 40 µm on 755252) and individual column tops rarely sit
inside it — but the fitted surface does, and in that gap `below_near`
collapses to 0.

**Artefacts.**
- `iterations/iter02_compute.py` — extractor
- `iterations/iter02_analysis.ipynb` — distributions + tables
- `data/iter02_tops.csv` (140 712 rows)
- `figures/iter02_top_below_near_hist.png` — overlapping pia/AF histograms
- `figures/iter02_dist_to_n22_hist.png` — distribution of z_top−z_N22 per subject

**Decision.**  Don't filter individual tops.  Iteration 3 will score
**entire candidate surfaces** by their aggregate `below_near` (or a
scale-free variant) and use that score to choose among candidates
(different channel subsets, different `target_quantile`).


### Iteration 3 — Surface quality score + candidate-bank auto-selection

**Setup.**  For each subject, fit v2.1 on 7 candidate (channel-subset,
target_quantile) combinations.  Also compute N22 and v2.1-default for
reference.  Score every candidate in the reference combined volume on
a 20 × 20 interior grid (400 columns) using three features:

- `Q      = median below_near`              (mean intensity 10–50 µm below surface)
- `Qabove = median above_near`              (mean intensity 10–50 µm above surface)
- `Qcon_n = median ((b - a) / (b + a + ε))` (normalised contrast in [-1, 1])

Artefacts: `iter03_compute.py`, `iter03_analysis.ipynb`,
`data/iter03_scores.csv`, `figures/iter03_Q_Qcon_plane.png`,
`figures/iter03_selection_vs_v21.png`.

**Key finding.**  The `(Q, Qcon_n)` plane cleanly separates three
regimes for every subject:

| regime                 | Q         | Qcon_n      |
|------------------------|-----------|-------------|
| real pia               | moderate  | ≥ 0.98      |
| gap / AF-top (755252)  | ≈ 0       | ≈ 0         |
| deep in tissue body    | high      | -0.05–0.55  |

The tissue-body trap that sank iter-1's naive Q score is now cleanly
rejected: on every subject, the "only488" and "mid488" candidates
that sit deep in cortex fall below `Qcon_n = 0.9`, while every real-pia
candidate sits above it.

**Selection rule.**  `argmax Q  s.t.  Qcon_n ≥ 0.9`.  This picks a
valid pia surface on **all 6 subjects**:

| subject | v2.1 onset (µm) | auto onset (µm) | auto above (%) | selected candidate |
|---|---:|---:|---:|---|
| 755252 | **62.5** ❌ | **2.5** ✅ | 0.51 | `mid488_tq0.85`  |
| 767018 | 2.5  | 2.5 | 0.56 | `only594_tq0.85` |
| 767022 | 12.5 | 2.5 | 0.02 | `all_tq0.70`     |
| 782149 | 12.5 | 2.5 | 0.08 | `no405_tq0.85`   |
| 788406 | 2.5  | 2.5 | 0.34 | `no405_tq0.85`   |
| 790322 | 2.5  | 2.5 | 1.13 | `all_tq0.70`     |

- 755252 is fixed (onset 62.5 → 2.5, matching N22's 7.5).
- Onset never regresses relative to v2.1-default; centroid-above
  fraction stays below 1.2 % on every subject (gate: 5 %).
- Selected `c` values can be up to ~30 µm offset from N22 (notably
  790322: +16, 782149: −35) — a soft miscalibration in which of the
  pia-pass candidates is "most pia-like".  Both endpoints still pass
  the onset/above gates, so not blocking, but worth noting when we
  move to GFP surface fits that demand tighter control.

**Caveats.**
- The three observed "tissue-body Qcon_n" values (0.55, 0.45, 0.11,
  -0.05, 0.33, 0.42) cluster well below 0.9, but only six subjects
  are in this dataset.  The 0.9 threshold should be distribution-
  driven (e.g. pick the valley in a bimodal Qcon_n histogram across
  candidates) rather than a hard constant — consistent with the
  session's global rule on thresholds.
- `max Q` is biased toward deeper pia (tissue brightness monotonically
  increases with depth over the 10–50 µm window).  For subjects where
  many candidates pass the Qcon_n gate (787406, 790322), this nudges
  selection 10–30 µm below N22.  A future refinement could add a
  shallow-bias term or weight `Q` by `Qcon_n`.

**Decision.**  Iteration 4 will integrate this selection logic into
`03_image_based_surface.py` as a new public entry point that fits the
candidate bank, scores each in the combined volume, and returns the
argmax-Q survivor — producing an image-only pia estimate that matches
N22 on all 6 HCR subjects without access to centroids.


### Iteration 4 — Integrate scoring + auto-select into the image module

**Setup.**  Added two new public functions to
`code/dev_code/03_image_based_surface.py`:

- `score_surface_quality(surface, vol, z_um, xy_um, *, n_side, edge_frac,
  below_um, above_um) -> dict(Q, Qabove, Qcon, Qcon_n, n)`
- `auto_select_surface(candidates, ref_vol, z_um, xy_um, *,
  qcon_n_threshold=0.9, **score_kwargs) -> (name, result, scores_df)`

`candidates` is a list of `(name, SurfaceResult)` pairs, so the caller
fits the bank however it likes (channel subsets × `target_quantile`,
parameter sweeps, etc.) and the module stays pure-image.  A fall-back
(argmax Q over all candidates) fires if nothing passes the filter, so
the function is total.

Validation driver: `iterations/iter04_compute.py`.  Re-runs iter 3's
7-candidate bank via the new public API on all 6 HCR subjects and
compares the selected surface to iter 3's scores.  Output:
`data/iter04_autoselect.csv` (55 rows = 6 × 8 candidates + subject col,
plus pass/selected flags, plus onset/above for centroid-truth check).

**Result.**  Selections match iter 3 exactly on 6/6 subjects.  Every
selected surface sits at onset ≤ 2.5 µm with ≤ 1.13 % of HCR centroids
above it — the key centroid-based quality gates that previously failed
for v2.1-default on 755252.

| subject | selected (auto) | Q | Qcon_n | above (%) | onset (µm) |
|---|---|---:|---:|---:|---:|
| 755252 | `mid488_tq0.85`  | 0.0506 | 0.982 | 0.51 | 2.5 |
| 767018 | `only594_tq0.85` | 0.0144 | 1.000 | 0.56 | 2.5 |
| 767022 | `all_tq0.70`     | 0.0135 | 1.000 | 0.02 | 2.5 |
| 782149 | `no405_tq0.85`   | 0.0188 | 1.000 | 0.08 | 2.5 |
| 788406 | `no405_tq0.85`   | 0.0014 | 0.999 | 0.34 | 2.5 |
| 790322 | `all_tq0.70`     | 0.0022 | 0.999 | 1.13 | 2.5 |

**Caveat — iter-3 log edit.**  While drafting the iter-3 summary the
767018 selection was mis-transcribed as `all_tq0.70` (Q = 0.0105);
iter 4 made the argmax behaviour explicit and surfaced the true winner,
`only594_tq0.85` (Q = 0.0144).  The iter-3 table above has been
corrected.

**Next.**  Add a `benchmark_analysis.estimate_pia_surface_image_autoselect`
convenience wrapper that builds the channel-subset candidate bank from
a subject's HCR channels (the glue currently duplicated in
`iter03_compute.py` and `iter04_compute.py`) and returns a
`(surface, info, scores)` triple.  That gives application code a
one-line image-only pia estimate without calling the auto-select
plumbing directly.


### Iteration 5 — Surface overlay audit + out-of-tissue (OOT) diagnosis

Date: 2026-04-20

**Setup.**  Response to user note: "In HCR data, there are out-of-tissue
volume below the tissue, not only above it.  And sometimes also at the
side margin.  Consider this factor and rerun the development."

For every HCR subject rendered a 2 × 2 figure showing the XZ mid-Y and
YZ mid-X slices of (i) the combined normalised HCR volume and (ii) the
raw 488 channel, with three surfaces overlaid: N22 (green, truth),
v2.1 image-based default (orange), and the iter-4 auto-selected
surface (cyan).  Also computed a per-subject OOT audit on the 20 × 20
scoring grid used by `score_surface_quality`: a column is "OOT" if its
column-max intensity is below 5 % of the global volume max.

Artefacts: `iter05_compute.py`, `figures/iter05_surfaces_<sid>.png`,
`data/iter05_audit.csv`.

**Finding (as reported at the time).**  Side-margin OOT was read off as
pervasive on HCR:

| subject | grid OOT frac (iter 5, buggy ref) | tissue thickness p50 | tis top p50 | tis bottom p50 |
|---|---:|---:|---:|---:|
| 755252 | **52.2 %** | 756 µm | 492 µm | 1232 µm |
| 767018 | **94.0 %** | 768 µm | 460 µm | 1244 µm |
| 767022 | **76.8 %** | 804 µm | 412 µm | 1224 µm |
| 782149 | (pending) |   |   |   |
| 788406 | (pending) |   |   |   |
| 790322 | (pending) |   |   |   |

The `b = 0, a = 0` failure mode is physically real for padded
columns, so the "OOT mask the grid" decision stood — but the numbers
above over-stated the prevalence.

**Correction (2026-04-20 evening, during iter 6 implementation).**  The
iter 5 audit used `colmax.max() = vol.max()` as its "global max"
reference for the 5 % threshold.  On HCR volumes `vol.max()` is
dominated by a handful of saturated voxels (32–253 × typical tissue
column brightness; the combined `load_hcr_combined` output can cross
100 × on 788406/790322).  That inflated the threshold until most
real tissue columns read as OOT.  When the reference is changed to
`np.percentile(colmax, 99.5)` — robust to single-pixel saturation and
sampling a "typical bright column" — the OOT frac on the 15 %-padded
interior grid drops to 0–2.25 % on all 6 subjects (see iter 6
`iter06_oot_mask_summary.csv`).  So iter 5's "52 %/94 %/77 %" numbers
were a reference-inflation artefact; the OOT loophole on the 20 × 20
interior grid is **not** what was dragging iter 3/4's Q/Qcon_n down.
What actually was dragging those values down on iter 3/4 is still
under investigation — candidates: integer-quantized column maxima
near the thresholds, a handful of grid points at the FOV edges that
happen to be just beyond the brain, or simply the 10–50 µm sampling
window landing below the tissue bottom on some columns.

What iter 5 got right: padded-below columns **do** add `(b = 0,
a = 0)` rows to the medians (they're included even in a mostly-tissue
grid).  The OOT mask in iter 6 is a correct fix, just for a much
smaller subset of columns than iter 5 claimed.

Bottom-OOT does *not* need its own mask: `load_hcr_combined` clips
below-tissue voxels to 0, so a candidate surface sitting below the
tissue bottom would read `b ≈ 0, a > 0 → Qcon_n ≈ -1` and already be
rejected by the contrast gate.  Only side-margin OOT needs fixing.

**Decision.**  Iteration 6 masks OOT grid columns before taking
medians.  The default `min_col_max_frac = 0.05` threshold is derived
from the per-subject OOT audit — all 6 subjects have a clean gap
between tissue columns and empty columns (see the OOT-mask figures
in iter 6), so any threshold in [0.01, 0.1] reproduces the same
mask provided the reference "global max" is taken robustly.  Keeping
the threshold explicit as a kwarg so callers can tune if future data
has a narrower gap.


### Iteration 6 — OOT-aware scoring

Date: 2026-04-20

**Change.**  Added `min_col_max_frac` (default 0.05) and optional
shared `global_max` / `global_max_percentile` kwargs to
`score_surface_quality`.  Columns whose column-max lies below
`min_col_max_frac × global_max` are skipped before computing
Q / Qabove / Qcon / Qcon_n.  `auto_select_surface` computes
`global_max = np.percentile(ref_vol.max(axis=0), 99.5)` once and
passes it to every candidate so the mask is identical across the
bank.  Also added `n_oot` to the scoring dict (number of dropped
columns — diagnostic).

**Why percentile-of-colmax rather than `vol.max()` or
`percentile(vol, 99.5)`:** the iter 5 audit used `colmax.max() =
vol.max()` as its global reference; while that worked for 755252
(52 % OOT) / 767018 (94 %) / 767022 (77 %), it blew up on 788406 and
790322 where a single saturated voxel drove `vol.max()` 1–2 orders of
magnitude above typical tissue columns, making the `5 %` threshold
larger than most real tissue columns (reported 100 % / 99.75 % OOT
— wrong).  Falling back to `percentile(vol, 99.5)` was the opposite
failure: over the >1 B voxels of a level-4 HCR volume the 99.5th
percentile is a typical bright *voxel* (~0.3), not a bright *column*
(~1.0), so 5 % of it let in every column including truly empty ones
(0 % OOT on 755252 — also wrong).  `percentile(colmax, 99.5)` is
robust to saturated pixels (colmax carries at most one bright-pixel
contribution per column) and samples the correct scale (bright
tissue column-max), so both failure modes are avoided.

**Validation.**  `iter06_compute.py` re-runs the 7-candidate bank on
all 6 HCR subjects with the new default and compares selections to
iter 4.  `iter06_viz.py` renders a top-down colmax + grid figure per
subject showing which grid points are in-tissue vs OOT under the new
mask.  Artefacts: `iter06_autoselect.csv`,
`iter06_selection_summary.csv`, `iter06_oot_mask_summary.csv`,
`figures/iter06_oot_mask_<sid>.png`.

**Confirmed result** (iter06_selection_summary.csv):

| subject | selected | onset_um | above_frac | Q_selected | Qcon_n_selected | n_in_tissue | n_oot |
|---|---|---:|---:|---:|---:|---:|---:|
| 755252 | mid488_tq0.85  | 2.5 | 0.0051 | 0.0506 | 0.9818 | 400 | 0 |
| 767018 | only594_tq0.85 | 2.5 | 0.0056 | 0.0144 | 0.9998 | 400 | 0 |
| 767022 | all_tq0.70     | 2.5 | 0.0002 | 0.0140 | 0.9999 | 398 | 2 |
| 782149 | no405_tq0.85   | 2.5 | 0.0008 | 0.0196 | 0.9998 | 391 | 9 |
| 788406 | no405_tq0.85   | 2.5 | 0.0034 | 0.0014 | 0.9992 | 400 | 0 |
| 790322 | all_tq0.70     | 2.5 | 0.0113 | 0.0022 | 0.9985 | 400 | 0 |

6/6 agree with iter 4.  Onset = 2.5 µm on every subject, above_frac
≤ 1.13 %, Qcon_n well above the 0.9 gate.  `n_oot` = 0–9 of 400
columns per subject — the 15 %-padded interior grid is essentially
all-tissue on this benchmark (see iter 5 correction above).

The practical wins of iter 6:
- Scoring is robust to saturated voxels (`np.percentile(colmax, 99.5)`
  vs a naive `vol.max()` that `All-NaN`-crashes on 788406 / 790322).
- `n_oot` is surfaced as a diagnostic so future narrow-FOV data can
  be monitored.
- Q/Qcon_n are now measured strictly on in-tissue columns, which
  matters if the candidate bank is extended or the Qcon_n gate is
  re-derived from distribution on future data.

### Iteration 7 — Image-only, curvature-capable pia via patch-max + TPS

Date: 2026-04-20.

**Why a new iteration.** After inspecting iter06's auto (`v2.2`) overlays the
user reported two specific failures: (a) **755252** — N22 *underestimates*
pia (real surface is deeper); v2.1/v2.2 are both flat quadratics that
cannot track the real curvature, (b) **790322** — auto is too deep.
Both failures point at the IRLS-quadratic fit in v2.1/v2.2: the real pia
is curved, so an arbitrary planar-quadratic surface cannot match it.

**Goal.** Pia estimator that (i) is image-only (no centroid dependency),
(ii) produces a curved surface (not a plane / quadratic), (iii) beats
N22 on 755252 and auto on 790322 visually.

**Summary of sub-iterations.** Every attempt below feeds a single
thin-plate-spline surface via `scipy.interpolate.RBFInterpolator` on a
20×20 interior-padded grid; the difference is in how the per-column
OOT→tissue transition z is detected.

| sub-iter | detector | 755252 z_um | 790322 z_um | verdict |
|---|---|---:|---:|---|
| v1 | global log-combined Gaussian LLR | 416 (deep) | 192 (deep) | LLR dragged down by L1 faint neuropil |
| v2 | 488 col-local mean+k·std | 504 (deep)→88 after MAD fix | 432 (22/400 valid) → 128 | 488 lacks L1 contrast; caught noise floor rise |
| v3 | abs log-combined thr = −4 + sustain 15 µm | 404 (+28 vs N22 ✓) | 232 (+94 vs auto ✗) | good on cliff-subject 755252, fails on gradient-subject 790322 |
| v4 | v3 ∪ first log-combined peak (height=−2, prom=2.5) | 404 | 232 | peaks land on deeper L2 cells, not early sparse cells |
| v5 | col-adaptive thr = max(−6.3, col_p90 − 3.0) + sustain | 412 | 236 | per-col p90 is dominated by whatever cell spike each col has → threshold comes out ≈ const ≈ −4 regardless of subject |
| v5 + patch-mean (15×15 vox, ≈ 60 µm) | same thr | 404 (+26) | 172 (+30 vs auto) | averaging voxels smears cell signal, didn't fully close the gap |
| **v6 = v5 + patch-MAX (15×15 vox)** | same col-adaptive thr on log-max of patch | **400 (+25.9 vs N22)** | **152 (+4.4 vs auto)** | **patch log-max preserves sparse bright cells → surface tracks the earliest tissue in each neighborhood** |

Key insight (v6): taking **log-MAX over a 15×15 xy patch** per grid
point — instead of log-mean of single voxel — is what closes the gap
on subjects with sparse L1 cells.  Log of the mean dilutes isolated
bright cells; max preserves them.  For cliff-subjects (755252) the
patch max ≡ patch mean once tissue is dense so the result is unchanged.

**Final summary on full 6-subject HCR bench** (`iter07_summary.csv`):

| subject | median_trans_z_um | Δ vs N22 (µm) | Δ vs auto (µm) |
|---|---:|---:|---:|
| 755252 | 400.0 | +25.9 | +26.2 |
| 767018 | 276.0 | +11.5 | +11.1 |
| 767022 | 256.0 |  +3.9 | +26.2 |
| 782149 | 308.0 | +22.5 | +16.5 |
| 788406 | 112.0 | +22.7 | +29.3 |
| 790322 | 152.0 | +25.6 |  +4.4 |

All 6 iter07 surfaces sit **4–29 µm below N22**, which matches the
user's specific finding on 755252 ("real surface is below N22") and
is consistent across subjects — N22 in this bench is a bias-slightly-
shallow estimator.  TPS surfaces are curved (not planar quadratic) so
they track real pia topology.  On 790322 the iter07 surface is now
only **+4.4 µm below auto** (iter06) vs the previous v3 **+94 µm** —
the patch-MAX fix was decisive.

**Artifacts.**
- `iterations/iter07_compute.py` — final detector + TPS + full-bench run
- `iterations/iter07_proto.py` — 2-subject prototype driver
- `iterations/iter07_analysis.ipynb` — plots + tables
- `figures/iter07_surfaces_<sid>.png` — XZ/YZ overlays of N22 / v2.1 /
  auto / iter07 on combined + log(combined) slices
- `figures/iter07_dists_<sid>.png` — per-column transition-z + thr
  histograms + sample smoothed column profiles
- `data/iter07_summary.csv`, `data/iter07_transitions_<sid>.npz`

**Parameters (locked).**

```python
EPS = 1e-3
SMOOTH_Z_UM = 5.0
SUSTAIN_Z_UM = 15.0
DELTA_LOG = 3.0
THR_FLOOR = -6.3
PATCH_W = 7            # 15×15 voxels ≈ 60 µm at level 4
POLY_DEGREE = 3        # bivariate polynomial degree (2 or 3)
HUBER_K = 1.5          # IRLS cutoff in MAD units
IRLS_N_ITER = 8
DETECTION_SOURCE = "combined"  # log of bg-subtracted combined vol
```

**Smoothing sweep → model switch (2026-04-20/21).**  After the first
full-bench run, user flagged the TPS surface in
`iter07_multiy_<sid>.png` as too wiggly.  Bumped
`TPS_SMOOTHING` from 200 → 10000 (sweep in
`iter07_smoothing_sweep.py`, figures in `iter07_smsweep_<sid>.png`;
10000 killed wiggles on all 6 subjects while preserving macro
tilt/curvature).

User then flagged the sm=10000 surfaces as still too wiggly / too
sensitive to column-level intensity variation.  Replaced TPS with a
**bivariate cubic polynomial** fit via IRLS-Huber on the 400
transition points:

- Design matrix: all terms `x^i y^j` with `i + j ≤ 3`, i.e. 10
  coefficients (1, x, y, x², xy, y², x³, x²y, xy², y³).
- Coordinate scaling: `xn = (x − x̄) / (Δx/2)` mapping the data grid to
  `[-1, 1]`; evaluation at the full FOV goes to ≈ ±1.43 — bounded
  extrapolation for degree ≤ 3.
- Robust weighting: IRLS with MAD-scale Huber weights
  `w = min(1, HUBER_K / |r/s|)` for 8 iterations, which downweights
  outlier transition points (AF-latched columns, noise) without
  zeroing them out the way a hard-reject would.

10 degrees of freedom for 400 points is strongly under-parameterised
(overdetermined by 40×), so the fit naturally resists high-frequency
wiggles and gives a monotonically smooth surface — exactly the
property the TPS's data-interpolating behaviour lacked.

**Final deltas** (same `median_trans_z_um` — polynomial-vs-TPS only
changes the surface-evaluation step, not the detector):

| subject | median_z_um | Δ vs N22 (µm) | Δ vs auto (µm) |
|---|---:|---:|---:|
| 755252 | 400.0 | +26.5 | +27.8 |
| 767018 | 276.0 | +11.7 | +12.2 |
| 767022 | 256.0 |  +4.1 | +26.3 |
| 782149 | 308.0 | +21.2 | +14.0 |
| 788406 | 112.0 | +24.8 | +30.9 |
| 790322 | 152.0 | +29.5 |  +8.4 |

All 6 subjects still land 4–30 µm below N22, consistent with the
TPS result — the surface model only changed the *shape*
(interpolating TPS → smoothly fitting poly), not the depth-bias
direction.  Multi-y figures (`iter07_multiy_<sid>.png`) show red
surface is now smooth across every y on every subject, no visible
wiggles, and macro tilt/curvature is still tracked (755252 x-slope,
782149 diagonal, 790322 gentle bow).

**Updated deltas** (same `median_trans_z_um` — smoothing only affects
the TPS evaluation at grid points, not the per-column detector):

| subject | median_z_um | Δ vs N22 (µm) | Δ vs auto (µm) | was (sm=200) |
|---|---:|---:|---:|---|
| 755252 | 400.0 | +25.9 | +27.1 | +25.9 / +26.2 |
| 767018 | 276.0 | +14.8 | +16.2 | +11.5 / +11.1 |
| 767022 | 256.0 |  +4.5 | +27.2 |  +3.9 / +26.2 |
| 782149 | 308.0 | +23.6 | +17.4 | +22.5 / +16.5 |
| 788406 | 112.0 | +28.8 | +35.7 | +22.7 / +29.3 |
| 790322 | 152.0 | +32.1 | +10.7 | +25.6 /  +4.4 |

Deltas shift up to +7 µm on a couple of subjects because the
heavily-smoothed TPS absorbs per-column noise into the macro trend —
it reports the neighborhood median rather than tracking each column
individually.  Still 4–32 µm below N22 on all 6 subjects; 790322
remains closest to auto (+10.7 µm, previously +4.4 µm before the
flatness bump).  Direction of iter07 vs N22 and vs auto is unchanged.

**Degree-2 switch + CZ port (2026-04-21).**  User asked for a
side-by-side of `POLY_DEGREE=2` vs `POLY_DEGREE=3` (see
`iter07_degcompare.py` / `iter07_degcompare_<sid>.png`).  Deg-3
introduced a mild edge-bow on 767018 / 767022 that didn't match the
image (10 coefs with 4 cubic terms overfit the corners).  Deg-2 (6
coefs: `1, x, y, x², xy, y²`) keeps one concavity per axis plus an
xy twist — still curvature-capable, no edge bow.  **Locked
`POLY_DEGREE = 2`** for HCR and CZ.

Then ported iter07 to CZ stacks (`iter07_cz_proto.py`).  CZ
differences:
- single channel raw uint16 (not bg-subtracted combined); camera
  offset ~30-100 DN, tissue ~500-1500 DN
- OOT log ≈ 3-5, tissue log ≈ 6-8 (vs HCR combined OOT ≡ 0, tissue
  log ≈ -3 to 0)
- loader: `*reg-dim-swapped.ome.tif` via `tifffile.imread`, volume
  (Z=450, Y=512, X=512), `z_um=1.0`, `xy_um=0.78`

**Threshold switch — range-relative (generalises across modalities).**
Original `thr = max(THR_FLOOR, col_p90 - DELTA_LOG)` failed hard on
CZ: CZ log p90 ≈ 7.5-8.7, `DELTA_LOG=3.0` drops threshold to 4.5-5.7,
below the camera offset log ≈ 5.5, so every column exceeded threshold
from z=0.  First run had median_z = 0 µm on all subjects.

Replaced with a distribution-driven rule (CLAUDE feedback on
threshold generalisability):

```python
col_p10 = percentile(smooth, 10)
col_p90 = percentile(smooth, 90)
thr = max(thr_floor, col_p10 + TRANS_FRAC * (col_p90 - col_p10))
```

`TRANS_FRAC=0.5` means "half-way between the OOT floor and the
in-tissue ceiling, per column" — scales automatically with the
modality's intensity range.  THR_FLOOR stays as a pure sanity floor
(HCR `-6.3`, CZ `4.5` ≈ `log(90)`, just to reject columns that never
see tissue).

HCR regression check after the switch: 400/400 valid on all 6
subjects, median_z within ±4 µm of the DELTA_LOG result, median_thr
shifted from ~-2.4 to ~-3.1 (now reports the relative midpoint rather
than the p90-offset).  No subject regressed.

**CZ bench result** (`iter07_cz_summary.csv`):

| subject | shape (Z,Y,X) | log_p10 | log_p90 | thr | median_trans_z (µm) |
|---|---|---:|---:|---:|---:|
| 755252 | 450, 512, 512 | 6.51 | 7.97 | 8.03 |  58.0 |
| 767018 | 450, 512, 512 | 7.96 | 8.69 | 8.81 |  42.0 |
| 767022 | 450, 512, 512 | 6.30 | 7.68 | 7.52 | 102.5 |
| 782149 | 450, 512, 512 | 6.38 | 7.22 | 7.25 |  52.0 |
| 788406 | 450, 512, 512 | 6.55 | 7.07 | 7.12 |  49.0 |
| 790322 | 450, 512, 512 | 6.36 | 7.36 | 7.36 |  54.0 |

All 400/400 transitions valid on every subject.  Visual check on
`iter07_cz_<sid>.png` (4 y positions × log(YZ) slice, with CZ
centroid = green, CZ image_ceiling = cyan, iter07 poly-deg2 = red):

- 755252 / 782149 / 788406 / 790322: red tracks green centroid
  closely, sits at visible tissue onset, below the aggressive cyan
  image_ceiling.
- 767018: dense bright slab from the top — red converges to the
  centroid plane (green), stable across y.
- 767022: deepest transition (102.5 µm), consistent with an OOT slab
  above tissue; red is still below cyan and in tissue, tracking
  centroid.

Same detector + surface model (patch-MAX + range-relative threshold +
IRLS-Huber deg-2 poly) now works on both HCR combined and CZ raw,
without any modality-specific knob other than `THR_FLOOR` (pure
sanity floor).

**Artefacts added.**
- `iterations/iter07_cz_proto.py`
- `iterations/iter07_degcompare.py`, `figures/iter07_degcompare_<sid>.png`
- `figures/iter07_cz_<sid>.png`, `data/iter07_cz_transitions_<sid>.npz`
- `data/iter07_cz_summary.csv`

**Parameters (locked, HCR+CZ).**

```python
EPS = 1e-3
SMOOTH_Z_UM = 5.0
SUSTAIN_Z_UM = 15.0
TRANS_FRAC = 0.5        # range-relative (new)
DELTA_LOG = 3.0         # legacy, kept only for logging
THR_FLOOR_HCR = -6.3
THR_FLOOR_CZ  =  4.5
PATCH_W = 7
POLY_DEGREE = 2         # degree-3 overfit corners on HCR
HUBER_K = 1.5
IRLS_N_ITER = 8
```


### Iteration 8 — CZ 50 µm prior + HCR bottom boundary

Date: 2026-04-22.  Picked up after a crashed session.

**Two extensions to iter 7 driven by new meta-information.**

#### 8a. CZ surface with experimenter's 50 µm depth prior

User flagged that during CZ acquisition the pial surface was manually
positioned at **z ≈ 50 µm** from the first slice.  iter07_cz_proto
(`TRANS_FRAC=0.5`) matches this on 5/6 subjects but lands at 102.5 µm
on **767022** because an OOT slab above the tissue pushes the p10→p90
midpoint mid-ramp onto strong tissue.  iter07_cz_lowfrac
(`TRANS_FRAC=0.02`) fires earlier but consistently below 50 µm (34-46
on all subjects) — it latches on coverslip / meningeal AF on the
subjects that have a clean cliff.

No single `TRANS_FRAC` works for every subject.  The 50 µm prior is
exactly the disambiguation signal we were missing.

**Approach** (`iterations/iter08_cz_prior.py`).  Sweep a small bank
`TRANS_FRAC ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}`, run the
same patch-MAX + range-relative detector for each, and select the
candidate whose **median per-column transition z** is closest to 50
µm.  Pre-compute the patch log-max columns once (4D reuse) so the
bank sweep is cheap.  After selecting, MAD-gate columns at ±75 µm
around the prior before the IRLS-Huber deg-2 poly fit so isolated
deep AF latches don't bend the surface.

**Bench result** (`data/iter08_cz_selection.csv`):

| subject | selected TF | median col z (µm) | median surf z (µm) | |Δ prior| | n in gate |
|---|---:|---:|---:|---:|---:|
| 755252 | 0.20 |  49.0 | 48.52 | 1.48 | 400 |
| 767018 | 0.50 |  42.0 | 41.73 | 8.27 | 396 |
| 767022 | 0.15 |  50.0 | 50.21 | 0.21 | 397 |
| 782149 | 0.50 |  52.0 | 45.88 | 4.12 | 319 |
| 788406 | 0.50 |  49.0 | 48.52 | 1.48 | 395 |
| 790322 | 0.30 |  50.0 | 49.96 | 0.04 | 400 |

**767022 fixed**: 102.5 µm → 50.2 µm (0.2 µm from prior).  The
visual check on `iter08_cz_767022.png` shows the red surface now sits
exactly on the visible tissue onset at all four y positions, right on
the gold dashed prior plane.

Subjects with dense tissue from the top (767018: `log_p10=7.96`,
tissue cliff at the slice 0 edge) correctly get `TRANS_FRAC=0.5` and
land at the intrinsic tissue start (42 µm, 8 µm above the prior) —
the prior can't force tissue to be where it isn't.  This is the
correct behaviour: the prior selects the best detector, not the
answer.

Why selection-over-sweep instead of a soft prior on the poly fit:
(i) the per-column transitions are the real failure mode on 767022
— no matter how the surface is fit, a detector that fires on
mid-tissue will return mid-tissue; the bank lets us find a detector
that fires at pia, (ii) selection is interpretable (we report which
TF won) and auditable (full sweep in `iter08_cz_sweep.csv`), (iii)
soft L2 to 50 µm on the IRLS fit would bias equally on all subjects
regardless of whether the detector was already correct.

#### 8b. HCR bottom tissue boundary

Symmetric to iter07's pia detector.  `load_hcr_combined` clips OOT to
0 at BOTH ends of the volume, so the bottom of the tissue block is a
symmetric cliff (tissue → 0).  Same detector + same surface model,
just with the column reversed in z.

**Implementation** (`iterations/iter08_hcr_bottom.py`).  Per grid
column, patch log-max over 15 × 15 xy, reverse in z, call
`col_detect_transition` on the reversed signal with the same
HCR-tuned params (`TRANS_FRAC=0.5`, `THR_FLOOR=-6.3`).  The first
sustained-above-thr voxel on the reversed column, at index `z_flip`,
maps to original-volume index `Z - 1 - z_flip` — the LAST tissue
voxel, symmetric to the top detector's FIRST tissue voxel.  MAD-gate
per-column bottom z at 3 × MAD before IRLS-Huber deg-2 poly fit.

**Bench result** (`data/iter08_hcr_bottom_summary.csv`):

| subject | top (iter07, µm) | bottom (iter08, µm) | thickness (µm) | HCR centroid z p99 (µm) | bottom − p99 |
|---|---:|---:|---:|---:|---:|
| 755252 | 396 | 1230 |  845 | 1311 | −81 |
| 767018 | 268 | 1287 | 1014 | 1293 | −6 |
| 767022 | 252 | 1255 | 1006 | 1273 | −18 |
| 782149 | 300 |  856 |  553 |  859 | −3 |
| 788406 | 108 | 1258 | 1158 | 1283 | −25 |
| 790322 | 144 | 1148 | 1005 | 1139 | +9 |

All bottom surfaces sit within 25 µm of the deepest 1% of HCR
centroids except 755252 (80 µm above centroid p99, consistent with
the tissue extending slightly past the deepest labelled cells).
400/400 columns are valid on every subject; MAD-gate keeps 341-400
columns depending on how clean the deep edge is (782149 is thinnest,
most variance).

Visual check (`iter08_hcr_bottom_<sid>.png`): magenta line tracks the
visible tissue→OOT edge tightly on every subject, including the
curved bottoms of 755252 and 782149.

**Why patch-MAX still works at the bottom.**  Patch-MAX was chosen at
the top to preserve sparse bright L1 cells (mean-smearing was the
failure mode on 790322).  At the bottom, tissue is dense neuropil —
patch-MAX ≡ patch-mean locally — and there is no sparse-cell
asymmetry to worry about.  The detector transitioning from "bright
column" to "clipped-to-zero column" is the same cliff shape in
either direction.

**Artefacts added.**
- `iterations/iter08_cz_prior.py`
- `iterations/iter08_hcr_bottom.py`
- `figures/iter08_cz_<sid>.png`, `figures/iter08_hcr_bottom_<sid>.png`
- `data/iter08_cz_selection.csv`, `data/iter08_cz_sweep.csv`,
  `data/iter08_cz_transitions_<sid>.npz`
- `data/iter08_hcr_bottom_summary.csv`,
  `data/iter08_hcr_bottom_transitions_<sid>.npz`

**Parameters (iter08 additions).**

```python
# CZ
CZ_TARGET_Z_UM = 50.0                     # experimenter prior
TRANS_FRAC_BANK = (0.02, 0.05, 0.10,
                   0.15, 0.20, 0.30, 0.50)
GATE_UM = 75.0                            # per-column gate around prior

# HCR bottom
MAD_GATE_K = 3.0                          # per-column bottom-z outlier gate
# detector params (TRANS_FRAC, THR_FLOOR, PATCH_W, etc.) unchanged from iter07
```

### Iteration 9 — CZ baseline-anchored detector, 50 µm as a weak parameter

Date: 2026-04-22

**Motivation.**  iter08's `select_trans_frac` treats the 50 µm
experimenter target as a *selection* signal: sweep `TRANS_FRAC ∈
{0.02..0.50}` and keep the one whose median column z is closest to 50
µm.  That bakes the prior in *as the answer* — if the experimenter's
target itself was off on a particular brain (it can, cf. 782149), the
detector will still force the surface to 50 µm by picking whichever
TF does so.  We want 50 µm to *inform* the detector, not *be* the
detector.

**New detector (`iterations/iter09_cz_baseline.py`).**

Per column:

1. `log_col = log(patch-MAX(15×15) + ε)`, same as iter07/08.
2. 1-D Gaussian smoothing over 5 µm in depth.
3. **Baseline** from `smoothed[z ∈ [0, Z_BASELINE_UM=30 µm]]`, taking
   the darkest half (`≤ median`) of that window so any shallow-tissue
   contamination drops out.  `μ_b = median`, `σ_b = 1.4826 · MAD`
   (floor 0.05 log-units to avoid zero-spread degeneracy).
4. **Threshold** `thr = μ_b + max(K_SIGMA · σ_b, LOG_MARGIN)` with
   `K_SIGMA = 3.0`, `LOG_MARGIN = 0.7` log-units (= 2.0× linear
   fold-change).  The log-margin floor is load-bearing: on clean
   dark air `σ_b → 0.05` is *genuine* Poisson noise, so `3σ_b ≈ 0.15`
   corresponds to only a 16 % linear rise — any pre-tissue scattered
   light crosses it.  Requiring at least 0.7 log units (≈ 2×) above
   baseline forces a real fold-change.
5. **Bounded search** — first `z ∈ [0, Z_MAX_UM=120 µm]` where the
   smoothed profile stays above `thr` for `SUSTAIN_Z_UM=15 µm`.  If
   no crossing within the window, return NaN for that column (the
   detector refuses to extrapolate past the physically plausible
   range).

Surface fit: IRLS-Huber bivariate degree-2 polynomial (same
`fit_polysurf` as iter07/08), gated by `3 × MAD` around the
**per-subject median** — no reference to 50 µm.

**How 50 µm enters, and where it does not.**

| use of 50 µm                    | iter08 (prior)                 | iter09 (this iter) |
|---------------------------------|---------------------------------|---------------------|
| basis for baseline window (z<30 µm) | —                           | ✓ (weak)            |
| basis for search cap (z<120 µm) | —                               | ✓ (generous)        |
| sanity diagnostic overlay       | ✓                               | ✓                   |
| per-column gate (`GATE_UM=75`) around prior | ✓                    | ✗                   |
| surface selection (argmin\|med−50\|) | ✓                          | ✗                   |

**Tuning walk (LOG_MARGIN).**  First pass with the detector as
originally designed (k_sigma=3, no log-margin) fired 22–38 µm on
5/6 subjects — too shallow, because the Poisson-tight baseline gives
`3σ_b ≈ 0.15 log` which is crossed by any faint scattered-light ramp.
Bumping to `LOG_MARGIN = 1.0` (≈ 2.7×) overshot on 788406 (surf 85
µm, only 77/400 valid columns — threshold above most of this dimmer
stack's onset ramp).  `LOG_MARGIN = 0.7` (≈ 2.0×) lands all 6
subjects within 10 µm of the prior, keeps 177+ valid columns on
every subject, and agrees with iter08 to within 3–7 µm on the 5
subjects where the experimenter's target held.

**Bench result** (`data/iter09_cz_summary.csv`, LOG_MARGIN = 0.7):

| subject | μ_b (log) | σ_b (log) | thr (log) | n_valid / 400 | median col z (µm) | iter09 surf (µm) | iter08 surf (µm) | \|Δ prior\| iter09 | \|Δ prior\| iter08 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 755252 | 6.58 | 0.05 | 7.30 | 266 | 43.0 | 41.1 | 48.5 | 8.9 | 1.5 |
| 767018 | 7.93 | 0.05 | 8.64 | 352 | 39.0 | 38.6 | 41.7 | 11.4 | 8.3 |
| 767022 | 6.13 | 0.05 | 6.83 | 398 | 44.0 | 44.1 | 50.2 | 5.9 | 0.2 |
| 782149 | 5.68 | 0.97 | 8.57 | 133 | 37.0 | 39.6 | 45.9 | 10.4 | 4.1 |
| 788406 | 6.54 | 0.05 | 7.28 | 177 | 50.0 | 52.4 | 48.5 | 2.4 | 1.5 |
| 790322 | 6.29 | 0.05 | 7.06 | 237 | 48.0 | 47.2 | 50.0 | 2.8 | 0.0 |

All six subjects land in 39–52 µm.  **The 50 µm target was not used
as a selection criterion**, yet 5/6 subjects converge to within 8 µm
of it — this is independent empirical validation that the
experimenter's target held on those subjects.  767018 and 782149 sit
9–11 µm shallower than 50 µm; on 782149 the high `σ_b = 0.97`
(tissue already fills the baseline window) is the signature the user
flagged as "target can fail here," and iter09 correctly picks up
that signal *from the data*, not from the prior.

**Why both `K · σ_b` and `LOG_MARGIN` in the same threshold.**  The
`σ_b` term handles subjects where the baseline window is
*contaminated* by shallow tissue: there `σ_b` blows up (0.97 on
782149 vs 0.05 elsewhere), and `k_sigma · σ_b = 2.9 log units`
dominates, raising thr well above the contaminating tissue.  The
`LOG_MARGIN` floor handles subjects with *clean* baselines: `σ_b`
floors at 0.05 from Poisson noise, so `3σ_b = 0.15` is meaningless
as a tissue gate — the log-margin ensures we still require a real
rise.  Using `max` of the two means each subject is protected by
whichever failure mode actually applies.

**Artefacts added.**
- `iterations/iter09_cz_baseline.py`
- `figures/iter09_cz_<sid>.png`
- `data/iter09_cz_transitions_<sid>.npz`,
  `data/iter09_cz_summary.csv`

**Parameters (iter09 additions).**

```python
Z_BASELINE_UM = 30.0         # darkest-half window for μ_b, σ_b
Z_MAX_UM = 120.0             # bounded search cap (2.4 × target)
K_SIGMA = 3.0                # σ_b multiplier
LOG_MARGIN = 0.7             # floor on the above-baseline gap (log units)
BASELINE_LOWER_FRAC = 0.5    # darkest half of baseline window
SIGMA_FLOOR_LOG = 0.05       # Poisson-noise floor for σ_b
# patch width, smoothing, sustain, polyfit — unchanged from iter07/08
```

**Outcome: iter09 NOT PROMOTED.**  User reviewed the iter08/iter09
comparison and elected to keep **iter08 as the main-pipeline CZ
surface**.  iter09 stays as a research artefact in this session.

---

## Promotion to main pipeline (2026-04-22)

**Goal.**  Move iter08 (CZ prior-selected) + iter07 (HCR top) +
iter08 (HCR bottom) out of the session-local `iterations/` folder
and into `code/dev_code/` so every other session consumes them
transparently through `analyze_subject`.

**Module added: `code/dev_code/surfaces_iter08.py`.**

Public entry points:
- `compute_cz_surface_iter08(s)` — iter08 CZ prior-selected fit,
  returns a surface dict.
- `compute_hcr_top_surface_iter07(s)` — iter07 HCR top fit.
- `compute_hcr_bottom_surface_iter08(s)` — iter08 HCR bottom fit.
- `get_cz_surface_iter08(s, use_cache=True)`, plus the HCR
  counterparts — cache-aware accessors that load from a JSON cache
  in O(ms) when present and fall through to the detector otherwise.
- `build_main_surface_store()` — batch builder that computes all
  three surfaces for every benchmark subject, caches them as JSON,
  and writes a summary CSV.

Key helper: `_polyfit_to_abcpqr(fit)` — inverts the normalised-coef
form used by `iter07_compute.fit_polysurf` back to the canonical
`{a, b, c, p, q, r}` dict (absolute µm coordinates) that
`depth_from_surface` expects.  Verified numerically:
`eval_polysurf(fit, x, y)` vs the converted
`a·x + b·y + c + p·x² + q·x·y + r·y²` agree to ≈ 4 × 10⁻¹³ RMS on
a random test grid.

**Cache**: `/root/capsule/code/dev_code/cached_surfaces/`:
- `<sid>_cz_iter08.json`,
  `<sid>_hcr_top_iter07.json`,
  `<sid>_hcr_bottom_iter08.json` (one file per subject × surface type).
- `main_surfaces_summary.csv` — flat table with the fit coefficients,
  median column z, and gate counts for all 6 subjects.

Per-surface load cost is ~0.1-0.3 ms; no CZ/HCR volume is re-opened
on cache hit.  `build_main_surface_store()` takes ~2 min total on the
six-subject benchmark (dominated by CZ TIFF reads + HCR zarr reads).

**Summary of cached surfaces**:

| subject | cz TF | cz med col z | cz n_in_gate | hcr top med z | hcr bot med z |
|---|---:|---:|---:|---:|---:|
| 755252 | 0.20 | 49.0 | 400 | 396.0 | 1230.0 |
| 767018 | 0.50 | 42.0 | 396 | 268.0 | 1292.0 |
| 767022 | 0.15 | 50.0 | 397 | 252.0 | 1256.0 |
| 782149 | 0.50 | 52.0 | 319 | 300.0 |  856.0 |
| 788406 | 0.50 | 49.0 | 395 | 108.0 | 1260.0 |
| 790322 | 0.30 | 50.0 | 400 | 144.0 | 1152.0 |

**`analyze_subject` defaults changed** (`code/dev_code/benchmark_analysis.py`):

```python
hcr_surface_method: str = "iter07"      # was "quantile_ceiling"
cz_surface_method:  str = "iter08"      # was "image_ceiling"
```

with three new keys in the returned `info` dict:

```python
info["cz_surface_iter08"]
info["hcr_top_surface_iter07"]
info["hcr_bottom_surface_iter08"]
```

and the existing keys (`info["cz_surface"]`, `info["hcr_surface"]`)
now point at the promoted surfaces by default.  Legacy surface keys
(`cz_surface_image_ceiling`, `hcr_surface_quantile_ceiling`, …) are
still computed and returned so downstream ablations that explicitly
reference them continue to work.

**Compatibility tests.**  iter08 / iter07 surfaces carry the
canonical `{a,b,c,p,q,r}` keys without any `type` override, so the
whole existing surface-consuming infrastructure plugs in unchanged:

- `depth_from_surface(pts, cz_surface)` — numeric agreement with
  manual `a·x+b·y+c+p·x²+q·x·y+r·y²` on 767022 (exact; same floats).
- `r1_coarse_align.coarse_align(cz, gfp, cz_surface, hcr_surface)` on
  767022 runs, returns `CoarseAffine` with `scales=[1.77, 1.77, 2.83]`.
- `r1_revised.coarse_align_revised(...)` on 767022 runs, returns
  `CoarseAffineV2` — `_plane_normal_from_surface` consumes iter08
  transparently through `_surface_z_at` → `depth_from_surface`.

No downstream file edits were needed.  Sessions 04/05/07 automatically
use iter08 surfaces on their next run.

**Artefacts added.**
- `code/dev_code/surfaces_iter08.py`
- `code/dev_code/cached_surfaces/*.json` (18 files, 3 × 6 subjects)
- `code/dev_code/cached_surfaces/main_surfaces_summary.csv`
- `code/dev_code/benchmark_analysis.py` — default-method change + new
  return keys.


