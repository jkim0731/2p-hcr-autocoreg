# Session 07b — Stricter (scale-only) GFP+ + k-NN / density-ratio scale

**Verdict: FAIL.** Stopping condition (6/6 at ±5 % on sxy and sz) not met.
- GMM-intersection sanity: **5/6** (767022 degenerate — no interior root).
- Depth-density uniformity gate (CV ≤ 0.20, integrated ∈ [0.8, 1.25]): **0/5**.
- Scale estimators (M1 or M3 clearing ±5 %): **0/5** subjects.

Session 06 (k-NN), session 07 (anisotropic ICP), session 07b (strict-GFP+ +
k-NN / density ratio) have all failed on centroid features. Per the approved
plan, we stop here and document. The next candidate (image-level 488
intensity NCC after R1 + surface-normal alignment) is described in Part E
but **not implemented** — it is a new session.

---

## Part A. Plan recap + self-critique

Full plan: `/root/capsule/.claude/plans/snoopy-bubbling-cherny.md`.

**Hypothesis.** A stricter GFP+ cutoff at the Bayes-optimal intersection
between the rightmost and second-rightmost GMM components on `log(feature)`
deflates the false-positive tail. If the stricter set approximates the truth
population (matched-HCR + unmatched-CZ mapped) to within a subject- and
depth-independent scalar, per-axis k-NN distance ratios (**M1**) and a
global xy/z density/span ratio (**M3**) can recover `(sxy, sz)` within ±5 %.

**Scoring.** Landmark-Procrustes `fit_anisotropic_similarity(landmark_pairs_um(s,
active_only=True))` — scoring only, no leakage into estimators.

**Stopping condition.** Pass iff every subject clears `|rel_err| ≤ 5 %` on
both axes on at least one of M1/M3. If 07b fails, stop and log; do not chain
into a surface-based estimator (07c was rejected: pia-plane tilt erased by
R1 alignment, z-extent-based sz contaminated by modality-specific imaging
depth cuts).

**Pre-run self-critique.** Weakest assumption: the top-two GMM components
bracket the noise/signal boundary rather than splitting the signal peak. The
5 % bar is also aggressive — even a perfect estimator has `~1/sqrt(N)`
sampling variability on median k-NN distances (~3 % for N ≈ 1000).

---

## Part B. GMM-intersection threshold

Code: `dev_code/07b_gfp_intersection_threshold.py`. Figures:
`figures/gmm_threshold_<sid>.png`.

| subject | class | cutoff_strict | cutoff_v2.2 | n_strict | n_v2.2 | coreg_cov | sanity |
|---------|-------|--------------:|------------:|---------:|-------:|----------:|:------:|
| 755252  | intensity | 34.6    | 20.7   | 19919 | 30804 | 0.90 | ok |
| 767018  | spot      | 1.08e-3 | 8.0e-4 |  7903 |  9161 | 0.98 | ok |
| 767022  | intensity | 10.6    | 28.4   | 28574 | 14239 | 0.98 | **FAIL** |
| 782149  | spot      | 1.71e-3 | 1.3e-3 |  3450 |  3831 | 0.97 | ok |
| 788406  | spot      | 1.09e-3 | 6.7e-4 | 12508 | 17427 | 0.96 | ok |
| 790322  | spot      | 1.79e-3 | 1.5e-3 |  9459 | 10131 | 0.96 | ok |

**767022 failure mode.** The GMM-2 fit on `log10(mean − bg)` produces a
narrow left component (σ=0.57, μ=0.83) and a very wide right component
(σ=1.08, μ=1.22). Their Gaussians don't cross between their means — both
roots of the quadratic fall outside `[μ₁, μ₂]`. The fallback midpoint
(10.6 linear) is *lower* than the v2.2 threshold (28.4), producing a
**larger** strict set than v2.2 (28,574 vs 14,239). This contradicts the
"deflate false positives" intent and confirms the weakest-assumption risk
from the plan. 767022 is skipped in downstream scale estimation.

For the other five subjects, the strict cutoff raises the threshold modestly
(intensity ×1.67; spot ×1.19–×1.62) and shrinks n_strict by 11–34 %.

---

## Part C. Depth-density gate (strict GFP+ vs truth)

Code: `dev_code/07_depth_density_diagnosis.py` (re-run against a monkey-
patched `load_subject` returning the strict-GFP+ `SubjectData`).
Figures: `figures/depth_density_strict_<sid>.png`. Baseline figures for the
v2.2 GFP+ set remain in `sessions/07_scale_failure_diagnosis/figures/`.

| subject | v2.2 integrated | **strict integrated** | v2.2 per-bin CV | **strict per-bin CV** | Gate A (CV ≤ 0.20) | Gate B (ratio ∈ [0.8, 1.25]) |
|---------|---:|---:|---:|---:|:---:|:---:|
| 755252  | 7.31 | **4.75** | 0.47 | **0.41** | ✗ | ✗ |
| 767018  | 1.32 | **1.11** | 0.55 | **0.49** | ✗ | ✓ |
| 782149  | 0.46 | **0.41** | 1.12 | **1.09** | ✗ | ✗ |
| 788406  | 3.64 | **2.42** | 0.47 | **0.37** | ✗ | ✗ |
| 790322  | 1.01 | **0.93** | 0.37 | **0.35** | ✗ | ✓ |

**Observation.** The strict threshold nudges the per-bin CV down (mild,
~5–17 % reduction) and the integrated ratio toward 1 (moderate), but the
underlying **subject-specific and depth-dependent detection bias is still
severe**: 755252 still over-detects 4.75× vs truth (v2.2 was 7.31×);
782149 still under-detects at 0.41× vs truth. Per-bin CV never gets close
to the 0.20 bar.

This means the GMM-intersection hypothesis is **falsified**: discarding the
rightmost-component tail does not convert the HCR GFP+ set into a
subject-independent, depth-independent subsample of the truth population.
The residual bias is orders of magnitude too large to satisfy the 5 % scale
bar downstream.

---

## Part D. Scale results (M1 k-NN + M3 span ratio) vs GT

Synthetic check (random cube stretched by (1.77, 1.77, 2.82)):
- M1 = (1.804, 2.849)  — 1.9 % / 1.0 % error
- M3 = (1.770, 2.820)  — 0.0 % / 0.0 % error

Both estimators are **mathematically sound** on ideal sampling.

Real subjects:

| subject | GT sxy | GT sz | M1 sxy | M1 sz | rel_err M1 sxy | rel_err M1 sz | M3 sxy | M3 sz | rel_err M3 sxy | rel_err M3 sz | pass5 |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 755252 | 1.64 | 2.13 | 0.89 | 0.43 | −46.0 % | **−80.0 %** | 4.62 | 2.08 | +182 % | −2.4 % | ✗ |
| 767018 | 1.70 | 3.58 | 1.93 | 1.06 | +13.4 % | **−70.5 %** | 5.65 | 2.76 | +232 % | −22.8 % | ✗ |
| 767022 | — | — | — | — | — | — | — | — | — | — | skipped |
| 782149 | 1.92 | 2.93 | 3.02 | 0.61 | +56.9 % | −79.0 % | 5.86 | 1.40 | +205 % | −52.0 % | ✗ |
| 788406 | 1.78 | 2.82 | 1.75 | 0.80 | **−1.5 %** | −71.7 % | 5.73 | 3.02 | +222 % | +7.2 % | ✗ |
| 790322 | 1.76 | 3.04 | 2.08 | 0.92 | +18.1 % | −69.7 % | 5.86 | 2.52 | +232 % | −17.2 % | ✗ |

Figure: `figures/scales_comparison.png`.

### M1 failure pattern
`sz_M1` is **always low by ~70 %** on real data (factor ~3.3×
underestimate). Strict-GFP+ retains high-feature cells, which cluster
shallower in depth; their z-spread is narrower than the CZ-mapped cloud's
z-spread → median nearest-neighbour z-distances on GFP+ are too *small*
relative to CZ → `sz = median_hcr_z / median_cz_z < 1` instead of ~3.
This is a direct consequence of the depth-dependent bias shown in Part C:
the strict set is not a uniform z-subsample of the truth, so per-axis k-NN
ratios measure detection bias, not scale.

`sxy_M1` is closer — two subjects (788406, 767018) land within 15 % and one
(788406) within 2 % — but the remaining three (755252, 782149, 790322) are
off by 18–57 %. No subject clears both axes.

### M3 failure pattern
`sxy_M3` is **always too large (5.6–5.9 vs GT ~1.7)** — a factor ~3.3
overestimate. HCR GFP+ covers the full HCR field-of-view in xy, which is
much wider than the R1-rotated CZ sub-volume's xy footprint; the span ratio
captures **FOV coverage, not tissue expansion**.

`sz_M3` is surprisingly competitive: 3/5 subjects within 23 % and two
(755252 at −2.4 %, 788406 at +7.2 %) within the target range. This
suggests the z-extent is dominated by true tissue depth (both modalities
image a similar physical slab), not detection bias. But sxy can't be
rescued without removing the FOV mismatch.

### Near-misses
- 788406 M1 sxy = −1.5 % (inside 5 % bar). Paired with M1 sz = −71.7 %.
- 755252 M3 sz = −2.4 % (inside 5 % bar). Paired with M3 sxy = +182 %.
- 788406 M3 sz = +7.2 % (outside 5 % but close). Paired with M3 sxy = +222 %.

Two different subjects get one axis right by accident, but no subject clears
both axes with either method. The "near-misses" don't show a consistent
combinable structure.

---

## Part E. Why 07b fails and what's next

### Root cause
The per-subject depth-dependent detection bias in HCR GFP+ (Part C) is
**larger than the scale effect we're trying to measure**. Integrated bias
spans 0.41×–4.75× across subjects even at the strict threshold; the true
scales we want to recover span 1.6×–1.9× (sxy) and 2.1×–3.6× (sz). You
cannot estimate a 2–3× ratio from clouds whose per-bin detection ratio
itself varies by 4×+ within a single subject and by 10×+ across subjects.

This rules out **all centroid-only** scale estimators on the current HCR
GFP+ signal, whether k-NN-based (session 06), ICP-based (session 07), or
density/span-ratio-based (session 07b). Session 07's "passing" configuration
was demonstrably GT-tuned (see `memory/project_07_anisotropic_icp_scales.md`).

### 07c was rejected
Surface-only estimation cannot recover both axes:
- Pia-plane tilt differences are **erased by R1**, which aligns the two
  pia planes before any scale estimate.
- z-extent from the pia-normal axis to the bottom of imaging is **modality-
  dependent**: CZ and HCR each stop imaging at different physiological or
  optical depths for reasons unrelated to tissue expansion. Using the
  extent ratio would multiply scale into an FOV term.
- Surface-pattern matching (e.g. distances among pia sample points) could
  recover sxy but leaves sz unconstrained.

### Next candidate (new session — NOT implemented here)

**Image-level 488 NCC after surface + normal alignment.**

The centroid-feature detection bias is baked into the localisation step.
Raw image intensity is not: a 488 spot at half detector brightness still
contributes its intensity. If we match CZ GCaMP intensity against HCR 488
intensity directly, we bypass the GFP+/GFP− binary and the threshold bias.

Sketch:
1. Align surfaces properly: apply R1 (R, t) to CZ, then rotate both frames
   so the HCR pia-normal is along `+z`.
2. In a matched xy ROI in that frame, build 1D intensity profiles
   `I_cz(z)` and `I_hcr_488(z)` by summing image intensity over xy slices.
3. Find `sz` that maximises `NCC(I_cz(z·sz), I_hcr_488(z))`.
4. For `sxy`, do the same along `x` and `y` in thin z-slabs at matched
   depth. `sxy` = stretch that maximises 2D NCC between CZ and HCR xy
   slabs.

**Why this might work where centroids fail.**
- Intensity integration is linear — false negatives in the spot detector
  don't disappear from the profile, they contribute at full intensity.
- Depth-dependent attenuation affects both modalities, and NCC is invariant
  to per-signal linear gain (so bulk attenuation differences only bias, not
  break, the scale estimate).
- Per-slab 2D NCC de-couples sxy from sz.

**What it requires.** Re-reading zstack + HCR 488 channel images. Session
05 `r1_revised` has the zstack infrastructure; the density-map NCC used in
session 05 failed without pia-normal alignment — pre-aligning surfaces is
the key new contribution.

This is a substantive engineering task with different data-loading
requirements than 07/07b. Opening it here would violate the plan's
"if 07b fails, stop" condition. Deferring to a fresh session (07c').

### What is safe to use from 07b
- `dev_code/07b_gfp_intersection_threshold.py` — the GMM-intersection
  threshold machinery and the 767022-degenerate warning are reusable as a
  diagnostic tool even if not as a scale pre-step.
- `dev_code/07b_scale_from_clean_gfp.py` — the depth-density monkey-patch
  pattern, `_patched_loader` trick, and the M1/M3 driver structure are
  reusable. The M3 span-ratio formula should **not** be trusted as a scale
  estimator (FOV-contaminated).

---

## Files

```
sessions/07b_scale_clean_gfp/
    log.md                         (this file)
    results.json                   (machine-readable: GMM + gate + scales)
    figures/
        gmm_threshold_<sid>.png    (6 subjects)
        depth_density_strict_<sid>.png  (5 subjects)
        scales_comparison.png

dev_code/
    07b_gfp_intersection_threshold.py
    07b_scale_from_clean_gfp.py
    07b_scale_comparison_plot.py
```

## Conclusion

Session 07b **fails** the 6/6 ±5 % stopping condition:
- 1 subject (767022) fails the GMM sanity gate outright
- 0 of 5 remaining subjects pass the depth-density uniformity gate
- 0 of 5 remaining subjects pass the ±5 % scale bar on both axes via M1 or M3

Centroid-feature scale estimation is blocked on the HCR GFP+ detection
bias. Surface-only estimation cannot cover both axes. The next candidate
is image-level 488 intensity NCC after R1 + pia-normal alignment — a new
session.
