# Session 03 v2 — HCR/CZ pia surface under corrected criterion

## Correction from v1

The v1 ("M4 trough") protocol was wrong because I misread the criterion.

**Corrected criterion (from user).** ROI density should be close to 0 **at
AND above** the surface — i.e., the cumulative share of ROIs at depth < 0
(and the density around depth = 0) should be small. The out-of-tissue
spike of segmentation false positives is still a collection of ROIs; the
pia plane should sit **above** that spike, not below it.

v1 M4 shifted the plane DEEPER (trough between spike and bulk), leaving
the spike at negative depth. In v2 we shift the plane SHALLOWER so that
the spike falls at depth > 0 (below pia).

Other constraints the user called out:
- Surface is not always flat — must fit tilts.
- Multiple channels should be considered for the image signal.
- A separate protocol is needed for CZ.
- Outputs must live under `/root/capsule/scratch/...`, not under
  `/root/capsule/code/...`.

## Metrics (new)

- `above_frac` — share of cells with depth < 0. Primary. Target: ≪ 1 %.
- `above_frac_5` — share with depth < −5 um (matches v1 `frac_above_pia`).
- `cum_above_ratio` — mean density in [−200, 0] / mean density in
  [50, 200]. Expresses the *total* above-pia density relative to bulk.
  Target: ≪ 1.
- `onset_depth_um` — depth ≥ 0 where smoothed density first reaches 50 %
  of bulk. Indicates how quickly cortex density rises after the plane.
- `above_peak_pos` — argmax of density in [−200, 0]. Stays at −200 (empty
  or negligible) when no spike remains above pia.

## HCR methods tried

All methods were evaluated on the same 6 benchmark subjects with
density-filtered ROI centroids in physical um; image volumes loaded at
pyramid level 4 (~4 um voxels). Results in
`/root/capsule/scratch/03_surface_v2/hcr_results.csv` (see `/tmp/...`
until `/scratch` is writable for this user).

### Baselines
- **`baseline_image_5pct`** — current image-based fit (relative_margin 5 %).
  Has `above_frac` 2.7–10.7 %, `cum_above_ratio` 0.30–0.94. The spike
  of out-of-tissue ROIs shows as `above_peak_pos` at depth −2 to −113.

### N1 — low image margin
Decrease `relative_margin` to 1 %, 0.5 %, 0.1 %. Each step makes the
image crossing shallower (catches the outermost intensity edge), so the
plane moves toward the buffer. Helps but is not enough by itself:
`above_frac` falls to 0.9–5.9 %. The problem is that the image fit can
*miss* columns where autofluorescence is weak, and those columns keep
the plane below the ROI cluster there.

### N2 — per-channel shallowest
For each HCR channel (405/488/561 or 514/594 — whichever exist), run
`estimate_pia_surface_from_image` at low margin and keep the channel
that produces the smallest `c`. Result: for 4/6 subjects the fit
degenerates (channel with near-zero edge padding → tilt ≈ 0, c ≈ 0),
which puts the plane at the very top of the volume and the cortex
onset depth at 140–170 um (too deep). Dropped — per-channel fits are
too easily dominated by a single noisy channel.

### N3 — ROI leading-edge plane with offset
Density-filter ROIs (≥ 3 neighbours in 30 um to keep clusters, drop
singletons), take the per-tile min-z, fit a plane, shift by a fixed
offset (−15 or −30 um toward buffer). With a 30 um offset it is
competitive (`above_frac` 0.08–2.75 %, `cum_above_ratio` 0.018–0.313)
but still fails 755252 (the subject with the highest-z ROI cluster in a
cortex-edge tile). The fixed offset is a hyper-parameter that doesn't
generalize.

### N4 — image + per-tile ROI ceiling (winner)
Low-margin image fit gives the tilted plane; density-filtered ROIs give
a per-tile minimum-z set; the plane is then **lifted** by the maximum
tile-deficit (where plane z was below some tile's min-z ROI) plus a
fixed 5 um safety offset. The plane's *tilt* comes from the image, but
its *intercept* is clamped to stay above every ROI-bearing tile.

Results for `N4_image_ceiling_0p5pct_off5`:

| subject | c (um) | tilt (°) | lift (um) | above_frac | cum_above_ratio | onset (um) |
|---------|-------:|---------:|----------:|-----------:|----------------:|-----------:|
| 755252  |   97.7 |   9.48   |   ~60     |    0.00 %  |       0.000     |    57.5    |
| 767018  |   97.6 |   6.69   |   ~35     |    0.01 %  |       0.001     |    12.5    |
| 767022  |  191.0 |   3.46   |   ~45     |    0.00 %  |       0.000     |    12.5    |
| 782149  |  −55.8 |  11.64   |   ~70     |    0.00 %  |       0.000     |    32.5    |
| 788406  |  −39.2 |   2.97   |  ~110     |    0.00 %  |       0.000     |    32.5    |
| 790322  |  −50.7 |   3.29   |  ~155     |    0.00 %  |       0.000     |    82.5    |

All subjects satisfy the criterion. The tilts match the image-fit tilts
(up to 11.6° for 782149). The large lift for 788406/790322 shows these
subjects have substantial ROI clusters well above the autofluorescence
boundary — N4 correctly places the plane above those clusters.

Aggressive (0.1 % margin, 10 um safety) gives identical above_frac ≈ 0
with deeper image starts and therefore larger lifts; the 0.5 %+5 um
variant is the smallest-change version that still clears the criterion
and is chosen as the default.

### N5–N20 — exploration beyond N4 (mostly for 790322)

N4 satisfies the corrected criterion on every subject (above_frac ≈ 0),
but **790322** retained `onset_depth_um = 77.5 µm` vs ≤ 32.5 µm for
every other subject — a localised dip at x ≈ 2000 µm that the
quadratic fit couldn't follow. N5–N20 are all attempts to reduce that
790322 gap; none solved it on their own. Full numbers for the
surviving variants are in `explore_790322_stats.csv`; code lives in
`dev_code/03_surface_790322_explore.py`.

(N5 was never defined; the numbering jumps from N4 to N6.)

- **N6 — `v_n6`: ROI-quadratic ceiling anchor.** Density-filtered ROI
  min-z per tile → robust quadratic. Used downstream as an "anchor"
  that later variants refine. On 790322: onset 122.5 µm (too deep).
- **N11 — `v_n11`: N6 anchor + ±150 µm band per-column image tops.**
  For each xy column search within ±150 µm of the N6 anchor z for the
  top-of-signal crossing (relative_margin + min_thick + min_signal
  thresholds), then fit a robust quadratic to those tops. This is the
  shipping default. 790322 onset 77.5, x_spread 50 (wide and wrong).
- **N12 — `v_n12_asym`: asymmetric band (deep below, narrow above)
  around N6 anchor.** Same fit but search window is tilted deeper
  because the N6 anchor itself is biased shallow. Did not shift the
  790322 onset materially. Dropped.
- **N13 — `v_n13_n4_anchor`: N4 image plane as anchor, ±150 µm band,
  multi-channel per-column tops, IRLS-Huber quadratic.** The key
  finding here: switching the anchor from the ROI-quadratic (N6) to
  the image plane (N4) cuts `onset_x_spread` on 790322 from 50 → 15 —
  i.e. the fit *shape* correctly follows the dip. But `onset_depth`
  stays at 77.5 (the whole surface is offset). User confirmed N13 has
  the best shape of all variants in `figures/diagnose_790322.png`.
- **N14 — `v_n14_iterative` / `v_n14_n4_iter`: two-pass band.** First
  pass ±150 µm around N6/N4 anchor, second pass ±80 µm around the
  first fit. 790322 onset 67.5–82.5; no reliable win; variance across
  starting anchors. Dropped.
- **N15 — `v_n15_imgfirst`: purely image-first.** No ROI anchor; per-
  column image crossing with low relative_margin. Degenerates on
  subjects with weak autofluorescence columns. 790322 onset 87.5–92.5.
- **N16 — `v_n16_n4_plus_imgfirst`: N4 first, then image-first refine
  inside a generous z-window.** Kept N4's protection but added column
  flexibility. 790322 onset 97.5 — worse. Dropped.
- **N17 — `v_n17_grid_smooth`: non-parametric grid.** Per-column tops
  in a 40 µm xy grid, Gaussian-smoothed with σ ∈ {80, 100, 150} µm,
  holes filled by confidence-weighted nearest. Could in principle
  follow the dip exactly, but the smoothing radius has to be large
  enough to close holes and that re-introduces the offset. 790322
  onset 57.5–72.5 depending on σ; edges noisy. Dropped.
- **N18 — `v_n18_quad_plus_grid_residual`: robust quadratic base +
  Gaussian-smoothed residual grid.** Additively correct the quadratic
  with a smoothed per-column residual (σ 80 or 120 µm). Corrects the
  dip direction but also rides above-surface noise; no reliable win.
- **N19 — `v_n19_selective_residual`: like N18 but with a
  hard-threshold, positive-only, confidence-weighted residual.** Only
  apply the residual correction where |r| > 30 µm and the residual is
  positive (i.e. only pull the surface *down*, never up). Helped on
  790322's dip but fragile on other subjects. Dropped in favour of a
  cleaner approach.
- **N20 — `v_n20_n4_asymmetric`: N13 pipeline + asymmetric outlier
  rejection.** Iteratively drop tops with residual < −shallow_reject
  (shallow outliers only), refit Huber. Swept rejection thresholds,
  min-thickness and min-ratio (`/tmp/n20_results.out`). Rejection
  barely flagged anything (kept ≥ 92 % of columns) because the
  shallow noise is dense, not sparse — symmetric robust loss already
  absorbed it. 790322 onset stuck at 77.5 across every parameter combo.
  Root cause: the wrong question — the noise isn't "outliers", it's a
  systematic bulk-bias in the Huber loss.

**Takeaway.** N6–N20 were all attempts to fix the *surface fit*.
The real problem is the final ROI-envelope *clamp* (see Addendum
below): N21 (quantile regression instead of Huber) attacks the bulk
bias correctly, but the clamp was lifting the corrected fit back up
by ~90 µm. N22 is N21 + a loosened clamp and is the fix.

## CZ methods tried

CZ has no out-of-tissue spike (segmentation is cleaner) but a few
subjects (notably 782149, 5.8 %) still have ROIs above the 5 %-margin
surface. Results in `/root/capsule/scratch/03_surface_v2/cz_results.csv`.

### CZ baseline
`cz_baseline_5pct` (relative_margin 0.05, min_signal_abs 50): above_frac
0.0–5.8 %, cum_above_ratio 0.0–0.115.

### CZ low margin (1–2 %)
Small improvement; 782149 still shows 5.6 %.

### CZ image_ceiling (winner)
Same structure as HCR N4: low image margin (2 %) + per-tile ROI
minimum-z ceiling with a 3 um safety offset and 80 um tiles. Achieves
`above_frac = 0.00 %`, `cum_above_ratio = 0.000`, `onset_depth`
7–42 um on every subject while preserving tilt (CZ tilts are smaller,
≤ 3°).

## Promoted protocol

### HCR (benchmark_analysis.estimate_pia_surface_image_ceiling)
1. Load combined-channel HCR at pyramid level 4.
2. Image fit with `relative_margin=0.005`, `min_signal_abs=0.05`.
3. Density-filter HCR ROI centroids (≥ 3 in 30 um).
4. Tile (120 um) the filtered ROIs; take min-z per tile.
5. Lift the image plane to clear every tile's min-z by +5 um.

### CZ (same function, CZ-tuned parameters)
1. CZ registered z-stack.
2. Image fit with `relative_margin=0.02`, `min_signal_abs=50`.
3. Tile (80 um) all CZ ROI centroids (no density filter needed — CZ is
   already clean); take min-z per tile.
4. Lift by +3 um safety.

Both paths are wired into `analyze_subject()`:
- `hcr_surface_method="image_ceiling"` (default)
- `cz_surface_method="image_ceiling"`  (default)

## Outputs

Under `/tmp/03_surface_v2/` (mirrored to `/root/capsule/scratch/03_surface_v2/`
when `/scratch` becomes writable for this user):
- `hcr_results.csv` — per-subject, per-method metrics for all HCR methods.
- `cz_results.csv` — per-subject, per-method metrics for CZ methods.
- `figures/hcr_depth_profiles.png` — normalised depth profiles for all
  HCR methods across all 6 subjects.
- `figures/cz_depth_profiles.png` — same for CZ.
- `log.md` — this file.

## Why v2 works where v1 did not

v1 treated the spike as "junk to be hidden below the plane" and moved
the plane deeper; the user's criterion is the opposite — the spike
(whatever its biology) represents real ROIs and those ROIs must end up
BELOW pia. The only direction that satisfies the criterion is
shallower, and the only reliable way to keep the plane shallower than
every ROI cluster is to *clamp* the image plane against a per-tile
minimum-z envelope of the density-filtered ROIs. That is what N4 does.

---

## Addendum — N21 and N22 (fixing 790322)

### Motivation
N11/N13 all landed at `onset_depth_um = 77.5 µm` on subject 790322
(every other subject ≤ 17.5 µm). Diagnosis from
`figures/diagnose_790322.png`: the per-column tops track the real pia
— including a localised dip at x ≈ 2000 — but IRLS-Huber fits the
*bulk* of those tops, biased shallow by above-surface noise. Symmetric
outlier rejection (N20) did not help.

### N21 — quantile regression on tops
Same pipeline as N13 (N4 anchor → ±150 µm band → per-column top) but
replace IRLS-Huber with an IRLS quantile regression at
`target_quantile = 0.70`:

```
w_i = q         if r_i > 0 (top deeper than current fit)
w_i = 1 − q     if r_i < 0 (top shallower)
w_i /= max(|r_i|, eps)
```

Intuitively: "70 % of tops should end up *above* the fit", so the fit
is pushed *deeper* — away from shallow above-surface noise, onto the
real tissue boundary.

### Diagnostic: the clamp was masking the N21 gain
Raw (no-clamp) N21 fit on 790322 gives
`onset = 2.5 µm, x_spread = 0 µm, above% ≈ 1.4 %` — i.e. the density
rises exactly at the surface, perfectly. After running the existing
`_clamp_to_roi_envelope` the surface was *lifted by 90 µm* to keep
every tile's 2nd-percentile cell below it. One shallow-outlier ROI per
tile was dictating the lift, so the clamp re-introduced the whole
77.5 µm offset. See `figures/n21_clamp_790322.png`: magenta (no
clamp) at tissue boundary; green (clamped) 90 µm above.

### Clamp sweep
Swept `within_tile_q ∈ {0.02, 0.10, 0.15}`,
`lift_q ∈ {1.0, 0.98, 0.95}`,
`filter_in_tissue.min_neighbors ∈ {3, 8}`,
`min_tile_n ∈ {5, 20}`, at `target_quantile ∈ {0.55, 0.70, 0.80}`.

Best config: **`within_tile_q = 0.10`** alone. It ignores the
shallowest ~10 % of cells per tile — exactly the above-surface noise —
while still catching real cell clusters that violate the fit.
`nb8_wtq10` (stricter neighbourhood filter alone) did **not** help
790322: the shallow outliers there are well-clustered, not isolated.

### N22 — N21 fit + loosened clamp
`v_n22_n4_quantile_loose_clamp`:
identical to N21 except `_clamp_to_roi_envelope(within_tile_q=0.10)`
(was `0.02`).

### Head-to-head (all 6 benchmark subjects, q=0.70)

onset_depth_um (µm):

| subject | N11 | N13 | N21 | **N22** |
|--------:|----:|----:|----:|--------:|
| 755252  | 27.5 | 22.5 | 22.5 | **7.5** |
| 767018  |  7.5 |  2.5 |  7.5 | **2.5** |
| 767022  |  2.5 |  2.5 |  2.5 | **2.5** |
| 782149  | 12.5 | 12.5 | 17.5 | **2.5** |
| 788406  | 12.5 | 17.5 | 17.5 | **2.5** |
| **790322**  | **77.5** | **77.5** | **72.5** | **2.5** |

above_frac (%):

| subject | N11 | N13 | N21 | **N22** |
|--------:|-----:|-----:|-----:|--------:|
| 755252  | 0.000 | 0.001 | 0.000 | 0.044 |
| 767018  | 0.006 | 0.006 | 0.006 | 0.073 |
| 767022  | 0.000 | 0.000 | 0.000 | 0.003 |
| 782149  | 0.000 | 0.000 | 0.000 | 0.000 |
| 788406  | 0.000 | 0.000 | 0.000 | 0.026 |
| 790322  | 0.003 | 0.002 | 0.002 | 0.055 |

**Reading.** N22 collapses `onset_depth_um` to 2.5–7.5 µm across all 6
subjects (primary target: tissue boundary = depth 0). 790322's 77.5
µm gap closes entirely without regressing any other subject's onset.
`above_frac` rises from ≤ 0.006 % to 0.000–0.073 %, all well within
the "close to 0, not exactly 0" criterion the user specified for
primary fit quality.

### Source
- Variants & clamp: `03_surface_790322_explore.py`
  (`v_n21_n4_quantile`, `v_n22_n4_quantile_loose_clamp`,
  `_clamp_to_roi_envelope(within_tile_q=…)`).
- Comparison driver: `03_surface_790322_compare.py`.
- Stats: `compare_variants_stats.csv`.
- Figures: `figures/compare_790322.png`,
  `figures/compare_all_density.png`,
  `figures/n21_clamp_790322.png` (clamp diagnostic).
