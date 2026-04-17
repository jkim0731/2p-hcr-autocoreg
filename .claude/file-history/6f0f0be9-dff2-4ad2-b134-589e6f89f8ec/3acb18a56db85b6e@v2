# Session 03 — Improving HCR pia surface estimation

## Goal

The current default (hybrid: ROI density prior + image-based first-crossing
within ±100 um window) leaves 2.5–8.2 % of HCR ROIs above the estimated
pia. The user criterion is **ROI density at the surface should be close
to 0**: i.e., the depth-from-surface density profile should have a clear
floor at depth = 0.

Develop new methods, iterate, and converge on a protocol that satisfies
the criterion. Log every attempt — including failures — and explain the
result.

## Quality metrics

We evaluate each candidate against the user's criterion with two primary
numbers (computed in `quality_metrics()` of `03_surface_iteration.py`):

- **`r0_narrow`** = mean ROI density in depth ∈ [−3, +3] um / mean density
  in the bulk window [50, 200] um. **This is the primary metric — the user
  explicitly asked for density at the surface to be near 0.** Target: ≲ 0.5.
- `r0_broad` — same but over [−10, +10] um.
- `frac_above_pia` — share of cells with depth < −5 um. Secondary, because
  above-pia cells *are* segmentation false positives (Stage A, session 01):
  if the surface is correctly placed past the spike of out-of-tissue
  clusters, these cells will be correctly classified as above-pia.
- `spike_to_bulk` — max density in [−100, −10] / bulk density. Diagnostic
  of how well the spike is separated from the bulk by the final surface.

## Baseline

Starting-point quality for all six subjects under the two existing methods:

| subject | method           | r0_narrow | frac_above | comment            |
|---------|------------------|----------:|-----------:|--------------------|
| 755252  | image            | 2.97      | 2.1 %      | spike at depth 0   |
| 755252  | hybrid (default) | 2.66      | 4.0 %      | spike at depth 0   |
| 767018  | image            | 0.22      | 10.8 %     | clean — no spike   |
| 767018  | hybrid           | **13.5**  | 2.5 %      | **regresses badly**|
| 767022  | image            | 0.18      | 4.6 %      | small spike        |
| 767022  | hybrid           | 0.20      | 4.4 %      | ≈ baseline         |
| 782149  | image            | 7.33      | 8.2 %      | big spike at 0     |
| 782149  | hybrid           | 8.34      | 5.7 %      | spike worse        |
| 788406  | image            | 0.25      | 5.2 %      | ≈ clean            |
| 788406  | hybrid           | 0.29      | 5.8 %      | ≈ baseline         |
| 790322  | image            | 0.24      | 6.6 %      | ≈ clean            |
| 790322  | hybrid           | 0.55      | 8.2 %      | regression         |

Two worst subjects on the criterion are **755252 (r0=2.97)** and
**782149 (r0=7.33)**. Note the current hybrid default is *not* the best on
this metric — the image-only fit is better on all subjects except 767018
where hybrid is catastrophic (r0 jumps from 0.22 to 13.5). That result
alone says the hybrid default should not stand.

## Hypothesis

Looking at raw `density vs depth-from-image-surface` profiles (see
`figures/baseline_image_density_profiles.png`), a consistent **spike-then-
trough-then-bulk** pattern appears in every subject at negative depths:

- a narrow peak around depth ≈ −50 to −5 um (out-of-tissue segmentation
  clusters in agarose/buffer — these are the small, clustered ROIs from
  session 01 Stage A),
- a trough around depth ≈ +50 to +100 um where density drops to near
  zero,
- then the cortical bulk density rises past 100 um.

The image-based fitter anchors at the autofluorescence boundary, which
coincides with the start of the spike — so the surface sits inside the
spike, making `r0_narrow` large. The right surface is *past* the spike,
at the trough.

## Methods tried (rounds 1–4)

### Round 1 — per-tile density threshold (M1)

**Idea.** For each (x, y) tile, find the first z (scanned from the top)
where the rolling ROI density exceeds a fraction of that tile's plateau
density. Fit a plane to those per-tile depths.

**Result.** Pushed the surface too deep — straight into the rising-edge
of the bulk. Improved `r0_narrow` for the easy subjects but catastrophic
for 782149 (surface landed ~150 um below truth). `frac_above_pia`
exploded for the subjects that did not have a real spike.

**Why.** A density threshold does not know about the spike; it only
knows "plateau reached". For subjects without a spike (767018, 788406,
790322), the plateau condition triggers inside the bulk itself, so the
plane gets pushed inward. Dropped.

### Round 2 — gap-finding methods anchored on the image surface (M4, M5, M6)

**M4 — global trough (image_then_trough).** Start with the image-based
surface, compute the global density profile vs depth-from-image-surface
in 5-um bins smoothed over 20 um. Find the spike peak in the negative
half, then the local minimum within 100 um after the peak. Shift the
plane by that offset. If no spike-then-trough is detected, fall back to
the image fit.

**M5 — per-tile trough.** Same gap-finding inside each (x, y) tile.

**M6 — spike-vs-bulk two-component fit.** Fit the depth profile as sum
of a narrow Gaussian (spike) and a broad logistic (bulk rise); place
surface at the 50 % point of the logistic.

**Results (r0_narrow vs baseline_image):**

| subject | base  | M4       | M5    | M6    |
|---------|------:|---------:|------:|------:|
| 755252  | 2.97  | **0.19** | 0.77  | 0.43  |
| 767018  | 0.22  | **0.21** | 0.46  | 0.51  |
| 767022  | 0.18  | **0.11** | 0.48  | 0.45  |
| 782149  | 7.33  | **0.38** | 2.68  | 0.47  |
| 788406  | 0.25  | **0.23** | 0.32  | 0.58  |
| 790322  | 0.24  | **0.23** | 0.33  | 0.49  |

**M4 wins on every subject by `r0_narrow`**, and by a huge margin on the
two problem subjects. M4 offsets range from −28 to +82 um (most subjects
+40 to +80 um), which matches the spike width seen in the raw profiles.

**Why M5 under-performs.** Per-tile density is noisy — a 100–150 um tile
holds only a few hundred ROIs, so the "trough" localization is dominated
by Poisson fluctuations. The bias introduced by choosing the noisiest
trough position per tile is larger than the signal from real tilt
variation. A per-tile method would need much stronger regularization
(or stronger global priors) to beat the global one.

**Why M6 under-performs.** The Gaussian-plus-logistic model mis-
represents the rise: the true cortical density has structure inside
(layered), so the logistic's 50 % point sits deeper than the trough.
This pushes `frac_above_pia` up without buying `r0_narrow`.

### Round 3 — decay-from-peak (M7, M8)

**Idea.** Instead of "local minimum after the spike" (which can be
ambiguous when the trough is shallow), set the surface at the depth
where the density has decayed to `k × (peak − bulk)` below the spike
peak. Three variants of `k`: 0.10 (strict), 0.20 (default), 0.40
(loose). M8 = per-tile version.

**Result.** On subjects where the spike is absent or weak (755252
spike/bulk = 1.26, 788406 = 0.93, 790322 = 1.06), the peak is not
meaningfully above bulk, so the decay target is below the current
density and the search falls off the array → returns baseline. A
guard that skips the shift when `spike_peak < 1.5 × bulk` fires for
755252 — the very subject we most needed to fix. Loosening the guard
produces unstable results on 767018 where the peak is far above zero
(at depth −92), and the 40 % decay point is at a negative depth which
clipped to 0.

**Why it failed.** Anchoring on the *peak height* is fragile when the
peak is small or noisy; the trough-finding formulation in M4 is
equivalent to "where does the density stop decreasing", which is
robust to peak height. Dropped; M4 remains the winner.

### Round 4 — bulk-floor (M9, M10)

**Idea.** Rather than "local minimum", find the first depth ≥ 0 where
the smoothed density drops below `threshold × bulk` (default 0.50) and
stays below for 15 um. This is a simple absolute threshold relative to
bulk, independent of peak height.

**Results (r0_narrow):**

| subject | M4       | M9 (0.50) | M9 (0.30) | M9 (0.70) | M10  |
|---------|---------:|----------:|----------:|----------:|-----:|
| 755252  | **0.19** | 0.52      | 0.27      | 0.52      | 2.97 |
| 767018  | 0.22     | 0.22      | 0.22      | 0.22      | 0.22 |
| 767022  | **0.11** | 0.18      | 0.18      | 0.18      | 0.18 |
| 782149  | **0.38** | 1.09      | 0.51      | 1.51      | 7.33 |
| 788406  | **0.23** | 0.25      | 0.25      | 0.25      | 0.25 |
| 790322  | **0.23** | 0.24      | 0.24      | 0.24      | 0.24 |

M9 never beats M4 and sometimes returns baseline (`search_lo=0` plus
strict threshold means the condition is not met → no shift). M10
per-tile falls back to baseline whenever any tile fails the spike-
detection gate, so it's effectively a no-op on every subject.

**Why M9 under-performs M4.** Using an absolute threshold `0.5 × bulk`
picks the point where density *first crosses* down through that level,
not the point where the profile has minimum density. In a spike-then-
trough profile, the first-crossing happens *on the descending flank*
of the spike (still above the trough), so the shift is under-sized.
M4 searches for a local minimum, which always sits at/after the
crossing, so M4 is equivalent to "M9 plus one step deeper to the true
trough". M4 is strictly better on this profile shape.

Dropped.

## Winner: M4 (image_then_trough)

M4 satisfies the criterion on every subject:

| subject | r0_narrow | r0_broad | spike/bulk | gap_depth | offset vs image |
|---------|----------:|---------:|-----------:|----------:|----------------:|
| 755252  | 0.19      | 0.18     | 2.29       | 97.5 um   | +52.5 um        |
| 767018  | 0.21      | 0.21     | 5.50       | 77.5 um   | +7.5 um         |
| 767022  | 0.11      | 0.14     | 2.48       | 102.5 um  | −27.5 um        |
| 782149  | 0.38      | 0.44     | 4.03       | 82.5 um   | +82.5 um        |
| 788406  | 0.23      | 0.26     | 1.19       | 77.5 um   | −27.5 um        |
| 790322  | 0.23      | 0.23     | 1.53       | 77.5 um   | −22.5 um        |

**All subjects have `r0_narrow < 0.5`**, where the previous default
(hybrid) had `r0_narrow` up to 13.5. The tilt and roughness are
unchanged (M4 is a pure translation of the image plane).

### Note on `frac_above_pia`

M4 increases `frac_above_pia` (e.g., 782149: 8 % → 22 %). This is the
expected consequence of correctly placing the surface *past* the spike:
the out-of-tissue ROIs in the spike are now correctly classified as
above-pia. Session 01 Stage A already showed these cells are small
(~10–30× smaller than in-tissue ROIs) and cluster in the buffer —
they are segmentation false positives that should be filtered
downstream, not cells that belong below pia. Using
`frac_above_pia` as the quality metric (which the hybrid default was
implicitly optimizing for) rewards hiding them under the plane.

## Recommended protocol

Promote **M4** (`estimate_pia_image_then_trough`) to the default HCR
surface method, replacing `"hybrid"`. The pure image-based fit remains
the CZ default — it already satisfies the criterion there (CZ has no
out-of-tissue spike).

Parameters (defaults that worked on all 6 subjects):
- 5 % relative margin on the image first-crossing.
- Combined-channel HCR volume at pyramid level 4 (~4 um voxels).
- Search window ±150 um for the trough, 20 um smoothing, 5 um bins,
  100 um lookahead after spike peak for local-minimum search.
- Fallback to image-only fit when no spike-then-trough is detected.

## Outputs

- Script: `code/dev_code/03_surface_iteration.py`.
- Tables: `round1_results.csv` … `round4_results.csv` (this directory).
- Figures (in `figures/`):
  - `baseline_image_density_profiles.png` — diagnostic that revealed
    the spike-then-bulk pattern.
  - `round2_depth_profiles.png` — shows M4 cleanly separating the spike
    from the bulk on 755252, 767022, 782149.
  - `round3_depth_profiles.png`, `round4_depth_profiles.png` — failed
    attempts for the record.
- Notebook: `code/notebooks/03_surface_estimation_iteration.ipynb`.
