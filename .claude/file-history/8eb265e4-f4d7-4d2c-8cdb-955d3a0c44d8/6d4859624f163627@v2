# Session 07d — Image-level 488 depth-profile NCC for sz

## Goal

Estimate `sz` (CZ→HCR z-scale) from image intensity correlation
after R1-identity alignment, bypassing the GFP+/GFP− centroid
threshold-bias that blocked sessions 06, 07, 07b, and 07c.

## Stopping condition

`|rel_err_sz| ≤ 5 %` on every subject (6/6).

## Design

1.  Apply R1 (identity scales) — rotation + translation only — to every
    CZ voxel so CZ and HCR share the xy centroid and pia-aligned z
    axis of HCR.
2.  For each CZ voxel: compute `depth below HCR pia`
    = `hcr_z − (a·hcr_x + b·hcr_y + c)`, using
    `analyze_subject(s)['hcr_surface']` as `(a, b, c)`.
3.  Histogram CZ intensity by that depth in 10-µm bins → 1D profile
    `I_cz(d)`.  Also track the xy AABB of the CZ footprint in HCR
    frame to clip the HCR profile to the same xy region.
4.  Repeat for the HCR volume (single 488 channel *or* combined
    multi-channel) → `I_hcr(d)` on the same depth grid.
5.  Stretch CZ-profile depth by sz around the R1 translation anchor
    `d_anchor = r1.translation.z − (a·tx + b·ty + c)`.  For each
    candidate sz, evaluate NCC on the overlap where both profiles
    are positive.
6.  Report argmax sz and compare to the GT anisotropic-similarity
    (landmark-Procrustes) sz.  GT is diagnostic-only.

## Why this might work

Intensity is linear in signal, so the threshold-dependent detection
bias that blocked session 07/07b/07c (GFP+ counts) is bypassed: every
voxel — whether it would or would not have been called "GFP+" — contributes
its full brightness.  If CZ and HCR share a common cortical-layer
structure (bright layer at L2/3, dimmer deep cortex, sharp fall at
tissue end), the stretch that maximises NCC should equal the
physical tissue expansion sz.

## Variants explored

Three preprocessing variants were tested end-to-end:

1. **488-only, no baseline subtract** (archived initial run on the
   wider `sz ∈ [1.0, 5.0]` grid).
2. **combined multi-channel, baseline-subtracted p10** — all four
   405/488/514/561/594 channels summed after per-channel
   background/norm, minus a p10 pedestal.
3. **488-only, baseline-subtracted p10** — 488 alone, minus a p10
   pedestal of the nonzero profile. Grid tightened to
   `sz ∈ [1.5, 4.5]`.

## Results

### 488-only, no baseline subtract (archived)

Source: `archive_488_only/sz_ncc_summary_488.json` (grid [1.0, 5.0]).

| subject | sz_gt | sz_ncc | err (%) | NCC |
|---------|-------|--------|---------|-----|
| 755252  | 2.129 | 1.840  | −13.6   | 0.903 |
| 767018  | 3.583 | 5.000  | +39.5 @grid edge | 0.630 |
| 767022  | 2.490 | 3.140  | +26.1   | 0.646 |
| 782149  | 2.926 | 1.620  | −44.6   | 0.651 |
| 788406  | 2.820 | 3.340  | +18.4   | 0.838 |
| 790322  | 3.042 | 2.940  | −3.3    | 0.818 |

Pass rate: **1/6** within ±5 %.

### Combined multi-channel + baseline subtract

Source: `sz_ncc_summary_combined_baseline.json` (grid [1.5, 4.5]).

| subject | sz_gt | sz_ncc | err (%) | NCC |
|---------|-------|--------|---------|-----|
| 755252  | 2.129 | 1.520  | −28.6   | 0.960 |
| 767018  | 3.583 | 2.280  | −36.4   | 0.828 |
| 767022  | 2.490 | 2.120  | −14.9   | 0.613 |
| 782149  | 2.926 | 1.840  | −37.1   | 0.416 |
| 788406  | 2.820 | 2.360  | −16.3   | 0.928 |
| 790322  | 3.042 | 4.500 @grid edge | +47.9 | 0.749 |

Pass rate: **0/6**. Combined channel is *worse* than 488-only —
the 405/514/561/594 structure (e.g., autofluorescence-like signal
near pia in 405) does not match CZ GCaMP and actively biases the
depth profile.

### 488-only + baseline subtract (primary)

Source: `sz_ncc_summary_488_baseline.json` (grid [1.5, 4.5]).

| subject | sz_gt | sz_ncc | err (%) | NCC |
|---------|-------|--------|---------|-----|
| 755252  | 2.129 | 2.040  | **−4.2 ✓** | 0.871 |
| 767018  | 3.583 | 3.600  | **+0.5 ✓** | 0.727 |
| 767022  | 2.490 | 2.020  | −18.9   | 0.425 |
| 782149  | 2.926 | 1.560  | −46.7   | 0.306 |
| 788406  | 2.820 | 3.440  | +22.0   | 0.825 |
| 790322  | 3.042 | 4.500 @grid edge | +47.9 | 0.745 |

Pass rate: **2/6**. Best variant — but still fails the 6/6 bar.

### Cross-variant per-subject comparison

| subject | GT   | 488-only | 488+baseline | combined+baseline | Best err |
|---------|------|----------|--------------|-------------------|---------|
| 755252  | 2.13 | −13.6    | **−4.2 ✓**   | −28.6             | −4.2 |
| 767018  | 3.58 | +39.5    | **+0.5 ✓**   | −36.4             | +0.5 |
| 767022  | 2.49 | +26.1    | −18.9        | −14.9             | −14.9 |
| 782149  | 2.93 | −44.6    | −46.7        | −37.1             | −37.1 |
| 788406  | 2.82 | +18.4    | −16.3        | −16.3             | −16.3 |
| 790322  | 3.04 | **−3.3 ✓** | +47.9      | +47.9             | −3.3 |
| union   |      | 1/6      | 2/6          | 0/6               | 3/6 |

No single variant reaches 3/6. Even the union across variants only
clears 3/6, and choosing the winning variant per subject requires
GT knowledge — so this is not a valid estimator.

## Z-extent analysis (CZ fixed 450 µm, HCR variable)

| subject | cz_z | hcr_z | hcr_z / cz_z | GT sz | HCR truncated? |
|---------|------|-------|--------------|-------|----------------|
| 755252  | 450  | 1640  | 3.64 | 2.13 | No  |
| 767018  | 450  | 1460  | 3.24 | 3.58 | Yes (≈9 % short of GT-expected) |
| 767022  | 450  | 1500  | 3.33 | 2.49 | No  |
| 782149  | 450  | 1004  | 2.23 | 2.93 | **Yes, 24 % short**  |
| 788406  | 450  | 1516  | 3.37 | 2.82 | No  |
| 790322  | 450  | 1352  | 3.00 | 3.04 | Marginal (≈1 %) |

When HCR-observed z-extent < GT_sz × CZ z-extent, the CZ-stretched
profile at GT sz spills past the HCR observation region and NCC
prefers a *smaller* sz that keeps stretched CZ inside HCR — this is
the structural failure mode for 782149 and partly 767018.

## Failure modes (from figures)

- **782149** (err −37 to −47 %, all variants): HCR depth truncated
  at ~1004 µm, can never observe the GT-stretched CZ extent
  (~1318 µm). NCC finds a sz that fits CZ inside HCR, not the
  physical expansion. **Structural failure — no preprocessing
  can fix this.**
- **767022** (err −14.9 to +26.1 %): HCR 488 has a fall-off at
  ~1200 µm, so raw 488 drives sz high; baseline subtraction
  erodes CZ signal (NCC drops to 0.425) and flips sz low — neither
  matches GT. Profile mismatch independent of sz.
- **788406** (err −16.3 to +22.0 %): HCR has a distinct fall edge
  at ~1600 µm; NCC prefers larger sz to stretch CZ to meet that
  edge. Baseline subtraction partially helps but overshoots.
- **790322** (err −3.3 → +47.9 %): Raw 488 has a clean plateau that
  matches CZ shape (488-only passes). Baseline subtraction creates
  artificial peaks at tissue ends, degrading the match and pushing
  sz to the grid edge. Best with **no preprocessing**.
- **755252** (err −28.6 → −4.2 %): Raw 488 has a broad plateau;
  baseline subtraction sharpens it into a bell that matches CZ
  GCaMP shape (passes). Best **with preprocessing**.
- **767018** (err +39.5 → +0.5 %): Raw 488 is nearly flat — NCC
  has no depth feature to anchor on and railroads the grid top.
  Baseline subtraction reveals latent structure (fall at deep end
  of cortex) that anchors NCC at GT sz. Best **with preprocessing**.

The pattern: whether baseline subtraction helps depends on the raw
HCR 488 profile shape of the specific subject. No universal
preprocessing rule works for all six.

## Verdict

**FAIL at the 5 %/6-of-6 bar.** Best variant is 488-only +
baseline-subtract: **2/6** pass. Even a per-subject oracle over the
three variants reaches only **3/6**.

Root causes of the failures:

1. **Structural truncation (782149):** HCR z-extent insufficient to
   contain the GT-stretched CZ. NCC cannot report a scale larger
   than what fits in the observed HCR volume. This subject will
   fail *any* method that expects full overlap between stretched CZ
   and observed HCR. The only escape is to constrain sz from a
   feature that does *not* require observing the full stretched
   extent — e.g., matching a localised near-pia landmark, or
   co-using sxy + cell-count conservation.

2. **Profile-shape mismatch (767022, 788406):** CZ GCaMP and HCR
   488 have qualitatively different depth shapes (GCaMP: shallow
   expression peak; 488 HCR: layer-dependent dye retention and
   tissue fall). NCC picks the sz that best aligns *these different
   shapes*, which is not necessarily the physical stretch.

3. **Preprocessing-specific (755252/767018 vs 790322):** Baseline
   subtraction is modality-specific. A CZ- and HCR-symmetric
   preprocessing that survives all six subjects would require
   per-subject tuning that we refuse on principle (would be
   GT-tuned).

Intensity NCC bypasses the *threshold-bias* that blocked centroid
methods, but surfaces a new bias — **shape mismatch between GCaMP
and 488 HCR stain** — that is at least as severe and has no
threshold knob to turn.

## Where

- `dev_code/07d_image_ncc_sz.py` — estimator (supports
  `488_only / 488_baseline / combined_baseline` variants).
- `dev_code/07d_probe_images.py` — one-subject probe script.
- `sessions/07d_image_ncc_scale/figures/sz_ncc_<variant>_<sid>.png`
  — per-subject profile comparison + NCC-vs-sz curve for the two
  primary variants.
- `sessions/07d_image_ncc_scale/sz_ncc_summary_488_baseline.json`
  — best result (2/6).
- `sessions/07d_image_ncc_scale/sz_ncc_summary_combined_baseline.json`
  — 0/6 (worse than 488-only; archived for completeness).
- `sessions/07d_image_ncc_scale/archive_488_only/` — initial
  488-only no-baseline run (1/6) on the wider `[1.0, 5.0]` grid.

## Next candidate (NOT implemented here)

Intensity NCC along the pia-normal depth axis is *too coarse* — a
single scalar profile collapses all the lateral structure. Two
independent directions to try in a future session:

**(a) 2D xy-depth NCC.** Build `I(x, d)` and `I(y, d)` projections
rather than `I(d)`, and optimise `(sxy, sz)` jointly by 2D NCC.
This recovers lateral structure (cell-dense columns vs
cell-sparse gaps) that a 1D profile averages away, and is robust
to subjects where the 1D profile is nearly flat (767018).
Cost: 2D grid search → 30× more NCC evaluations, but still
tractable.

**(b) Segmentation-based depth features.** Run a cortical-layer
segmentation (e.g., DAPI-based layer boundaries on HCR, or
intensity-peak detection on CZ) and match *layer boundary depths*
across modalities rather than raw intensity profiles. This
side-steps the GCaMP-vs-488 shape mismatch: both modalities
report the *same* layer boundary in µm, regardless of stain
intensity. Cost: requires a robust layer segmentation, which is
its own project.

**(c) Drop sz from image and co-estimate from sxy + cell-count
conservation.** If sxy is well-constrained (e.g., from a 2D
surface-projected matching), and total cell count is preserved
under coregistration, then sz follows from
`sz = N_ratio / sxy²`. This bypasses the 782149 truncation
problem entirely since it never requires matching stretched
depth profiles.

These should each be their own session. Image-level 488 NCC is
falsified as a standalone sz estimator at the 5 % bar.
