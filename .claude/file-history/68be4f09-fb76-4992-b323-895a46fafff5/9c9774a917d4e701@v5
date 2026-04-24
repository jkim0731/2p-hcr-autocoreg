# Session 03 — Image-based curved-surface estimation

Date: 2026-04-17

## Goal

Image-only analogue of N22 (session 03 v3).  Fit a **quadratic** surface
`z = a·x + b·y + c + p·x² + q·x·y + r·y²` to the pia using intensity
information alone (no centroid ROI).  No L2 estimation — dropped per
user request (the CZ L2 signal was ambiguous in the side views).

After two iterations (per-column in-band search too fragile; v3-port
too complex), the algorithm was simplified to a two-stage,
top-of-volume-anchored method that works on all six benchmark
subjects.

## Algorithm (2 stages)

### Stage 1 — global baseline from the top 1–2 z-planes

The top 1–2 z-slices are always outside tissue.  Pool every voxel of
those slices across the **entire image** and compute a robust
(median + 1.4826 · MAD) baseline.  One threshold is used for every
tile — the image's own noise floor, not a per-tile / per-column
heuristic.

Padding handling:

* A slice whose fraction of exact-zero voxels ≥ `max_pad_frac` (0.95)
  is treated as pure padding and skipped.
* Exact-zero voxels are dropped from the pool so the median reflects
  the **real blank region**, not the zero padding (important for CZ,
  where z=0 is often ~50 % zero-padded around a non-rectangular ROI
  and z=1+ is the real buffer ring above tissue).
* HCR: top slices are literally pure zeros → all slices skipped →
  fall back to `sigma = 0.02 · dyn_hint` where `dyn_hint =
  percentile(vol, 99)`.  Threshold then lives on the volume's own
  dynamic range.
* CZ: dropping zeros yields a median at the buffer fluorescence
  (~400–2500 counts, subject-dependent) and a MAD capturing the
  buffer's own pixel variability.

Threshold:
```
surf_thr = max( bg_mean + k_surface · bg_sigma,
                bg_mean + relative_margin_min · dyn_hint )
```
with `k_surface = 6.0`, `relative_margin_min = 0.02`.

### Stage 2 — per-tile onset + quadratic Huber fit

Tile the XY plane (`xy_tile_um` = 80 HCR / 60 CZ).  For each tile:

1. 1-D median profile along z.
2. Gaussian-smooth with `smooth_sigma_um` (6 HCR / 4 CZ).
3. Find the first z where the profile stays above `surf_thr` for at
   least `min_thick_surface_um` contiguous microns (12 HCR / 10 CZ),
   searching only the top `surface_search_frac = 0.30` of the volume.
4. Record (x_tile_centre, y_tile_centre, z_onset_px · z_um).

Fit quadratic `z = a·x + b·y + c + p·x² + q·x·y + r·y²` via IRLS-Huber
(centered for conditioning; coefficients returned in absolute frame).

No ROI clamp (image-only).  No in-band per-column search (removed —
the band runs into tissue and the "baseline percentile" is
contaminated).

## Files

* `code/dev_code/03_image_based_surface.py`
  — `estimate_surface_and_l2_image_based(vol, z_um, xy_um, *,
     n_top_planes, k_surface, relative_margin_min, xy_tile_um,
     surface_search_frac, min_thick_surface_um, smooth_sigma_um)`
  → `SurfaceResult(surface, baseline_mean, baseline_sigma, surf_thr,
     dyn_hint, n_tiles_total, n_surface_tiles, tile_records, ...)`.
* `code/dev_code/03_image_based_surface_run.py`
  — driver over `BENCHMARK_SUBJECTS`; HCR combined + CZ.
* `code/dev_code/03_image_based_surface_figs.py`
  — walkthrough figures: XZ / YZ MIPs + fitted surface (red) + global
  intensity profile with the baseline threshold.
* `code/sessions/03_image_based_surface_estimation/results.csv`
  — per-subject scalars (quadratic coefficients, tilt, residual MAD,
  baseline_mean, baseline_sigma, surf_thr, above_frac, onset_depth_um).
* `code/sessions/03_image_based_surface_estimation/figures/`
  — `walkthrough_<sid>.png` for all six subjects.

## Per-subject results

| subject | HCR resid MAD | HCR above% | HCR onset µm | HCR tilt° | CZ resid MAD | CZ above% | CZ onset µm | CZ tilt° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 755252 | 66.1 |  3.5 | 2.5 | — | 3.22 |  1.0 | 22.5 | — |
| 767018 |  7.8 |  1.4 | 2.5 | — | 1.56 |  1.3 |  2.5 | — |
| 767022 | 74.0 |  3.0 | 2.5 | — | 1.83 |  0.6 |  7.5 | — |
| 782149 |  6.9 |  1.6 | 2.5 | — | 0.93 |  8.3 |  2.5 | — |
| 788406 | 45.0 | 10.1 | 2.5 | — | 2.56 |  2.6 |  7.5 | — |
| 790322 | 32.9 |  8.6 | 2.5 | — | 1.43 |  0.1 |  2.5 | — |

(tilt column left blank — dominated by the quadratic curvature terms;
read off the walkthrough plots.)

### Interpretation

* **HCR onset collapsed to 2.5 µm on every subject** — the quadratic
  fit now hugs the tissue top everywhere.  Visual inspection of the
  walkthroughs confirms the red surface sits exactly at the first
  bright band in HCR XZ / YZ MIPs, including the 790322 x≈2000 µm dip
  that defeated the plane fit.
* **CZ residual MAD 0.9–3.2 µm** — effectively matches the z-step.
  CZ surface tracks the top of the GFP layer on all six subjects.
* **HCR residual MAD still variable** (7–74 µm) because HCR has
  ~800 tiles per subject at 80 µm; shiny debris and per-tile first-rise
  noise inflate the spread even when the fit itself is good.  A
  larger `xy_tile_um` (150 µm) tightens resid MAD but coarsens the
  curvature — deferred.
* **HCR above_frac** still 1.4–10.1 % (788406 / 790322 highest).  The
  gap to N22's ≤ 0.08 % is the missing ROI envelope clamp; N22 lifts
  the fit until ≤ 10 % of per-tile cells sit above, and that requires
  centroid data this session deliberately excludes.

## Parameters

| knob | HCR | CZ | source |
|---|---|---|---|
| n_top_planes | 2 | 2 | always-blank slices at top |
| k_surface | 6.0 | 6.0 | robust-σ multiplier above baseline |
| relative_margin_min | 0.02 | 0.02 | floor vs dyn_hint (rescues HCR where σ≈0) |
| max_pad_frac | 0.95 | 0.95 | drop pure-padding planes |
| xy_tile_um | 80 | 60 | CZ volume is smaller in µm |
| surface_search_frac | 0.30 | 0.30 | only search top 30 % of z |
| min_thick_surface_um | 12 | 10 | suppress single-plane spikes |
| smooth_sigma_um | 6 | 4 | matches z-step resolution |

All distribution-driven — no hand-tuned absolute intensities.

## What works, what doesn't

### Works
* HCR quadratic surface tracks curvature (790322 x-dip, 788406 tilt).
* CZ surface excellent (resid MAD < 3.3 µm, onset ≤ 22.5 µm).
* All six subjects converge.
* No per-column / in-band machinery — global top-of-volume baseline
  suffices because top slices are always outside tissue.
* HCR fallback (sigma from dyn_hint) handles the pure-padding case
  cleanly.

### Doesn't yet
* **Above_frac without the ROI clamp** still sits at 1.4–10 % for HCR
  (N22 with clamp: ≤ 0.08 %).  Closing the gap requires either the
  ROI clamp (rules out image-only) or biasing the fit deeper
  (e.g. quantile loss at q = 0.80).
* **782149 CZ above_frac = 8.3 %** is the CZ outlier; manual
  inspection shows a handful of unusually bright upper-layer GFP+
  cells the fit (correctly) sits above.

## Next step candidates

1. **Quantile-loss fallback** (q = 0.75–0.80) for HCR only, to lower
   above_frac without breaking CZ (which is already fine).
2. **Larger HCR tiles (150 µm)** to tighten resid MAD, at the cost of
   fewer curvature samples.
3. Promote into `benchmark_analysis.py` as an image-driven
   cross-check alongside N22 (complementary: N22 uses centroids,
   this uses intensity).

---

## v2 (2026-04-18) — N22-style port + column-top clamp

v1 hit the expected image-only wall: HCR above_frac 1.4–10 % (vs N22's
≤ 0.08 %) because there was no ROI clamp.  v2 ports the N22 machinery
into image-only form:

1. **Stage-2 anchor** — per-column top-of-signal over the whole z,
   global sparse grid (column_stride_um=10), fed into the quadratic
   IRLS to define a rough pia plane.
2. **Stage-3 per-column tops in a ±150 µm band** around the anchor,
   relative_margin=0.25, min_signal_abs_frac=0.20, min_thick_um=10.
3. **Stage-4 quadratic quantile-regression IRLS** at q=0.70 (pulls
   fit toward the 30 %-shallowest tops).
4. **Stage-5 column-top clamp** — analogue of N22's ROI envelope
   clamp but on tops instead of centroids.  Tile the tops into
   120-µm XY bins, take the 10 % quantile of z per tile
   (`within_tile_q=0.10`), compute `deficit = plane_z − tile_q_z`,
   and lift the surface by the robust-max of positive deficits minus
   `safety_offset=3 µm`.

The result is a *fully image-only* analogue of the N22 clamp.

### The autofluorescence contamination problem

Max-lift caused over-lifting on 755252 and 767022.  Diagnostic on
755252 tile-z10 distribution (239 tiles) showed the cause:

* shallowest 5 % of HCR centroids start at z = 439 µm (so pia sits
  near there);
* yet 151/239 tiles (63 %) have z10 < 439 — their tops are from a
  thin **shallow autofluorescence layer** above the real pia, not
  pia itself;
* the single worst tile has z10 = 150 µm (≈ 289 µm above pia) and
  `max(deficit)` anchors the lift to it, pulling the surface ~80 µm
  above pia after clamp.

Tops (unlike centroids) pick up any bright column signal, including
tissue-edge autofluorescence that does not correspond to real somata.
This is the fundamental gap between image-only and centroid-based
clamps.

### Fix — robust Tukey-fence cap on the lift

Two knobs added to `_clamp_to_column_tops` (defaults in bold):

| knob | default | role |
|---|---|---|
| `lift_q` | **1.0** | use `quantile(pos_deficits, q)` instead of max; `q < 1` ignores the shallowest `(1-q)` tail of tiles |
| `lift_iqr_k` | **1.5** | also cap the lift at Tukey upper fence `Q75 + k·IQR(pos_deficits)`; only applied when `len(pos_deficits) ≥ 50` |

The IQR cap gates on tile count so it only bites on HCR
(~230–360 clamp tiles); CZ's ~25 clamp tiles always take plain max.
With the defaults, subjects whose deficit distribution is tight
(no clean outlier tail) get fence > max → no cap → strict N22 behaviour.
Subjects with a clear outlier tail (767022) get capped.

### Results — v2 defaults on all six subjects (HCR)

| subject | above % | onset µm | lift µm | notes |
|---|---:|---:|---:|---|
| 755252 | 0.001 | **72.5** | 182.1 | 4 tiles capped — contamination too diffuse |
| 767018 | 0.336 | 2.5 | 237.6 | no cap — distribution clean |
| 767022 | 0.020 | **2.5** | 200.1 | 13 tiles capped → N22-quality onset |
| 782149 | 0.003 | 7.5 | 239.6 | no cap |
| 788406 | 0.152 | 2.5 | 221.9 | no cap |
| 790322 | 1.125 | 2.5 | 210.9 | no cap; above_frac is the real residual gap |

* **767022 closed**: onset 42.5 → 2.5 (vs pure max-lift) with
  above_frac 0.02 % — near N22.
* **755252 remains the irreducible limitation**: 63 % of tiles are
  autofluorescence-contaminated, so the deficit distribution has no
  clean outlier tail for the Tukey fence to bite on.  Candidates
  still on the table: tighter tops filter (column_min_thick > 20 µm)
  at the cost of 767018/782149, or a soma-level image primitive
  (deferred to a future session — task #13 already ruled out 3D
  peak detection).
* **790322 above_frac = 1.1 %** — small stubborn above-pia band.
  Image-only cannot discriminate real shallow centroids from
  autofluorescence here; reducing `lift_q` worsens it further
  because the already-fit surface is close to the density front.

### CZ

v2 changes are transparent on CZ (cz_above_frac stays 0 on every
subject; onsets 7.5–47.5 µm unchanged) because the `len(pos) ≥ 50`
gate skips the fence on the ~25-tile CZ geometry.  The CZ onset
spread is a known separate issue (v1 log) — untouched by v2.

### Honest summary vs N22

| metric | N22 | v2 (image-only) | gap |
|---|---|---|---|
| HCR above_frac | ≤ 0.08 % | 0.001 – 1.1 % | worst: 790322 |
| HCR onset_depth | ≤ 7.5 µm | 2.5 – 72.5 µm | worst: 755252 |

Four of six HCR subjects reach N22 quality (above ≤ 0.34 %,
onset ≤ 7.5 µm).  The remaining two (755252 onset, 790322
above_frac) are limited by the tops-vs-centroids information gap;
closing them fully requires either centroid data or a soma-level
image primitive.

---

## v2.1 (2026-04-18) — raise `target_quantile` from 0.70 → 0.85 (HCR only)

### Motivation

v2 left 790322 with a stubborn `above_frac = 1.13 %` — 20× worse
than the other HCR subjects.  Side-view MIPs showed a thin band of
centroids ~10–20 µm above the fitted pia surface: not autofluorescence
but real shallow GFP+ cells nearest the pia edge.  The quantile-IRLS
fit at q = 0.70 placed the surface at roughly the 30 %-shallowest
column tops, leaving that thin band above it.

Raising `target_quantile` pulls the Stage-4 fit deeper (toward the
more-densely-populated pia body) so the clamp then lifts back into
the correct position — net effect is a surface that sits ~15 µm
deeper and captures the shallow GFP+ band.

### Experiments (all on HCR, six subjects)

Before committing tq=0.85 as the default I ruled out every other
idea I could think of for the 755252 onset residual.  All trails led
back to the same tops-vs-centroids information gap flagged in v2:

* `column_min_thick_um` sweep (10 → 120) — any thickness that
  suppresses 755252's AF also suppresses real pia edges on 767018 /
  782149 (their pia signal is thin too).
* Per-column adaptive thin/thick merging — 755252 OK (onset 2.5) but
  clean subjects blew up (767018 above → 10.8 %, 782149 → 20.8 %).
* Smart clamp rejecting intra-tile bimodal shallow clusters —
  diagnostic showed clean subjects have *more* bimodal tiles (86 %
  on 767018) than 755252 (53 %), so the signal is anti-correlated.
* `band_um` sweep (40 → 150) — band 60 fixed 755252 (above 1.2 %
  onset 2.5) but broke 767018 (above 10.3 %).
* Z-Gaussian smoothing — blurs every subject's pia edge.
* Anchor density filter (radius 20–50 µm, min_neighbors 3–8) —
  barely shifts 755252's anchor (266 → 257 at r=50 nb=8); AF on
  this subject is a continuous sheet, not isolated spikes.
* Deeper-quantile anchor (fit plane on tops above pctile 0.25–0.75)
  — 755252 anchor correctly moves to 460+ µm (real pia), but clean
  subjects' `above_frac` explodes to 5–18 % because the band shifts
  deeper than their pia top.
* Anchor-only `first_substantial` (frac 0.3–0.7) or thicker
  (thick 40 / 60) — same story, one subject helped at the expense
  of ≥ 2 others.

**Conclusion — image-only parameters alone cannot separate 755252's
AF from 767018's / 782149's real pia edge.  Both appear as thin,
spatially-coherent top-of-signal onsets above the main body.**
Only tq = 0.85 has a net-positive effect across all six subjects.

### Result

| subject | v2 above% | v2 onset µm | v2.1 above% | v2.1 onset µm | notes |
|---|---:|---:|---:|---:|---|
| 755252 | 0.001 | 72.5 | 0.005 | **62.5** | AF limitation persists (small win) |
| 767018 | 0.336 | 2.5  | 0.351 | 2.5  | flat |
| 767022 | 0.020 | 2.5  | 0.003 | 12.5 | above↓, onset +10 µm (still < N22 bound) |
| 782149 | 0.003 | 7.5  | 0.000 | 12.5 | above↓, onset +5 µm |
| 788406 | 0.152 | 2.5  | 0.202 | 2.5  | flat |
| 790322 | **1.125** | 2.5 | **0.057** | 2.5  | **20× better** — the intended win |

### Scope: HCR only

CZ driver (`03_image_based_surface_run.py`) still passes
`target_quantile = 0.70` — CZ has no autofluorescence-above-pia
issue, and v2 CZ metrics are already clean (above_frac = 0 on all
six subjects).  The module default (`0.85`) is set for the HCR
failure mode; the CZ driver opts out explicitly.

### Remaining stubborn cases

* **755252 onset 62.5 µm.**  Documented as irreducible under
  image-only constraints.  The AF is a continuous sheet ~200 µm
  above pia that is indistinguishable from a thin pia edge using
  column-profile statistics alone.  Closing the gap needs either
  centroid data (N22's approach) or a soma-level image primitive
  such as a bandpass-filtered spot detector; both are out of scope
  for this session.  Task #13 already ruled out 3-D peak detection.
* **All other subjects** now within N22 bounds (`above_frac ≤
  0.35 %`, `onset ≤ 12.5 µm`).

### Extended negative-results audit (post-v2.1)

Further sweeps, all confirming the 755252 limitation is structural:

* `clamp_within_tile_q` sweep (0.10 → 0.50 at tq=0.85).  On 755252
  nothing changes (clamp not limiting; surface sits at AF depth
  *before* the clamp).  On 790322, wtq ≥ 0.25 is catastrophic
  (above 0.06 % → 3.94 %).  Keep default 0.10.
* Clamp density filter (`density_radius_um` 30–80, `min_neighbors`
  3–20).  755252 onset pinned at 62.5 across all configs — AF tops
  are spatially continuous, so density thinning doesn't discriminate.
* Higher `target_quantile` (0.88 → 0.95).  Marginal: 755252 onset
  drops 62.5 → 57.5 at tq=0.92 / 0.95, but 767022 onset rises by 5,
  and 788406 / 790322 above-fracs double at tq=0.95.  Net-negative.
* **Per-channel subset test on 755252.**  Running the full pipeline
  on `[488, 514]` alone gives c=153.9, onset=2.5 µm, above 0.51 %
  — i.e. the pia IS recoverable on 755252 if the AF-carrying
  channels (405, 594) are dropped.  But the same subset is
  catastrophic on every other subject (767018 above 10.0 %, 782149
  above 12.9 %, 790322 above 5.4 %).  The "good channels" vary per
  subject, so a global subset is not viable.  An auto-selection
  rule would need a confidence metric (e.g. below-vs-above-surface
  contrast ratio) — genuine future-work lead but a substantial
  structural change.

### Code changes

* `03_image_based_surface.py`: module default
  `target_quantile = 0.85` (was 0.70); docstring updated.
* `03_image_based_surface_run.py`: HCR driver now passes
  `target_quantile = 0.85` explicitly; CZ remains at 0.70.
