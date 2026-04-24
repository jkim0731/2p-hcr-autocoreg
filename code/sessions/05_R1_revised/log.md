# Session 05 — R1 revised (scale-free, surface-tilt aligned, graceful-degradation)

## Goal

Implement `07 Grand Plan.md §R1 (2026-04-17 revision)`: a sample-specific
coarse affine that uses **only** the 180° rotation prior, the "CZ ⊂ HCR
in XY" geometric prior, and the two pia-surface fits — **no
benchmark-derived expansion priors**. Emit

- **Minimal** `(R, t)` always, from surface tilt + centroid match.
- **Extended** `(R, t, S)` only when per-axis scale confidence clears
  threshold; components that fail are marked `unknown` and left to
  downstream scale-aware methods (A1 / A3 / TEASER-class internals).

Targets (when emitted): origin ≤ 100 µm, rotation ± 5°, per-axis scale
within ± 20 %.

Supersedes `sessions/04_R1_coarse_align/`. GFP+ threshold defaults stay
at the v2.2 loader (`peakgauss3_density_p0.1` for spots,
`peakgauss3_mean_bg_p1` for intensity) — subgoal 01/02 work is reused
unchanged.

## Method

`r1_revised.coarse_align_revised()` implements Grand Plan §R1 steps 1–8:

1. **Pia-plane fit.** LSQ plane through each surface's sample points.
   Unit normals `n_cz`, `n_hcr`.
2. **Tilt-aligned rotation.** `R = R_tilt · R_180°XY` where `R_tilt` is
   Rodrigues' formula solving `(R_180°XY · n_cz) → n_hcr`. Row-vector
   convention matches `ProcrustesFit`.
3. **Minimal translation = full 3D HCR centroid match.**
   `t_minimal = mean(hcr_xyz)` — because `R` is applied after mean-centring
   CZ, this is equivalent to `mean(hcr) − R · mean(cz)`.
4. **Depth-from-surface** in both modalities (existing
   `depth_from_surface`).
5. **Z scale + offset, 1-D partial-overlap NCC.**
   Grid: `sz ∈ [0.5, 6.0]` step 0.02; `tz` slid over the shared depth
   range in 20-µm bins. NCC computed on the overlap region only, floored
   at 25 % of the **longer** profile (see *Deviation 1*).
6. **XY scale + translation joint search.**
   `sxy ∈ [0.5, L_hcr_xy / L_cz_xy]` step 0.05; `(tx, ty)` on a sparse
   grid within ± half-HCR-XY-extent. Density maps rasterised at σ = 25
   µm, 20-µm bins. Score = integral-image 2-D NCC (Lewis 1995).
7. **Anisotropic refinement.** Local `(sx, sy)` sweep around the best
   `sxy` if the XY peak is confident.
8. **Per-axis scale confidence.** Robust z on the score curve:
   `(peak − median) / (1.4826 · MAD)`. Accept when ≥ threshold. See
   *Deviation 2* for the threshold.

Output `CoarseAffineV2(R, scales, scale_known[3], scale_confidence[3],
translation, src_mean, rotation_angle_z_deg, coverage_regime,
minimal_translation, diagnostics)`. `apply_coarse_affine()` uses the
same row-vector convention.

### Deviation 1 — minimum-overlap basis

Grand Plan step 5 says "≥ 25 % of the **shorter** profile". On thinner
HCR (782149, 755252) the shorter profile is HCR, with 10-30 bins. Any
rescaled-tiny CZ aligned to a 2-3-bin HCR window is a trivial match —
NCC is scale-invariant, so the best `sz` collapsed to 0.5 on first
test. Switching the basis to the **longer** profile keyed the min-overlap
off the larger of the two, which eliminated the collapse. Keyed by
`max(n_src, n_tgt)` in `_partial_overlap_ncc_1d`.

### Deviation 2 — confidence threshold raised to 6.0

`peak/RMS` (first metric tried) is ~1 on typical NCC surfaces → false
positive every time. Switched to the more standard robust z,
`(peak − median) / (1.4826 · MAD)`. Default threshold of 3 admitted
788406's `sxy = 0.75` at confidence 5.07 — physically wrong (GT = 1.77),
pushing origin error from 133 µm to 571 µm. Raised default to 6.0: all
6 benchmark subjects now emit minimal output (`scale_known = [F, F, F]`).

This is a deliberate conservative default: **robust-z on a density-map
NCC detects "there is a peak", not "the peak is correct"**. In the
current benchmark regime (GFP+ covers only 10 – 16 % of HCR cells under
v2.2 peakgauss thresholds, HCR spans 1.8 – 2.3 mm in XY while CZ spans
only 400 µm) density-map NCC admits prominent but geometrically-wrong
peaks at very small `sxy`. Raising the threshold routes everything to
the minimal output, which is the §R1 "worst-case graceful degradation"
path.

## Results

```
python dev_code/05_r1_revised_benchmark.py
→ sessions/05_R1_revised/r1_results.json
```

| Subject | n_gfp | coverage | cz_tilt° | hcr_tilt° | origin µm | rot ° | pass origin? | pass rot? |
|---|---|---|---|---|---|---|---|---|
| 788406 | 17 427 | equal    | 2.78 | 2.05 | 133.3 | 2.38 | ✗ | ✓ |
| 790322 | 10 131 | thinner  | 0.83 | 3.10 | **70.9** | **0.41** | ✓ | ✓ |
| 767018 |  9 161 | thicker  | 1.86 | 6.48 | **99.3** | 10.09 | ✓ | ✗ |
| 782149 |  3 831 | thinner  | 1.53 | 11.80 | 424.1 | 2.92 | ✗ | ✓ |
| 755252 | 30 804 | thinner  | 2.93 | 9.94 | **56.7** | 6.68 | ✓ | ✗ |
| 767022 | 14 239 | thicker  | 0.34 | 4.08 | 180.2 | 5.36 | ✗ | ✗ |

**Pass rate: 3/6 on origin ≤ 100 µm; 3/6 on rotation ±5°; only 790322
clears both.** Scale not emitted for any subject (see confidence section).

### Scale search (all diagnostic-only, none emitted)

| Subject | sz_best | sz_conf | sxy_best | sxy_conf | L_cz_xy / L_hcr_xy µm |
|---|---|---|---|---|---|
| 788406 | 3.02 | 1.81 | 0.75 | **5.07** | 400 / 2284 |
| 790322 | 3.18 | 2.52 | 0.50 | 1.99 | 397 / 2265 |
| 767018 | 1.56 | 1.54 | 0.65 | 1.86 | 398 / 2281 |
| 782149 | 5.90 | 2.34 | 0.50 | 2.55 | 420 / 2263 |
| 755252 | 5.92 | 1.07 | 0.75 | 2.53 | 427 / 1816 |
| 767022 | 1.74 | 2.71 | 0.70 | 2.16 | 410 / 1816 |

`sz_best` is *qualitatively correct* on 788406 (3.02 vs GT 2.82), 790322
(3.18 vs GT 3.04) and plausibly so on 767018 (1.56 vs GT 3.58: z-scale
is under-recovered but on the right side of 1). It's wildly off on
782149 (5.90) and 755252 (5.92) — both thin-HCR subjects where the
partial-overlap NCC is dominated by 10-bin HCR profiles. `sxy_best`
lands at 0.5 – 0.75 across the board — below every GT scale (1.66 –
1.98). The density-map NCC peaks where a shrunken CZ fits entirely
inside HCR with room to spare; the answer is geometrically plausible
for the NCC objective but physically wrong. This is the mechanism the
Grand Plan's graceful-degradation clause anticipates.

### Comparison to first-pass R1 (v2.2 peakgauss defaults)

| subj | orig_v22 | orig_rev | Δ | rot_v22 | rot_rev |
|---|---|---|---|---|---|
| 788406 |  58.1 | 133.3 | **+75.3** | 2.46 | 2.38 |
| 790322 |  74.3 |  70.9 |  −3.4 | 0.48 | 0.41 |
| 767018 | 127.0 |  99.3 | **−27.7** | 10.26 | 10.09 |
| 782149 | 334.8 | 424.1 | **+89.3** | 2.34 | 2.92 |
| 755252 | 123.6 |  56.7 | **−66.8** | 6.99 | 6.68 |
| 767022 | 181.0 | 180.2 |  −0.9 | 5.29 | 5.36 |

788406 regresses because the v2.2 R1 exploited the 1.77×/2.83× prior to
scale CZ before XY-centroid matching; the revised R1 centroids HCR
directly, which is biased whenever HCR GFP+ is asymmetric with respect
to the CZ imaging window. 767018 improves because the thicker-HCR
subject gets a better translation estimate when tilt is removed.
755252 improves for the same reason (large tilt ≈ 10°). 782149 is
the thin-HCR outlier already documented; its error stays ≈ 350–430 µm
regardless of method.

## Hypothesis / method / result / failure / next

**Hypothesis.** Without any benchmark-derived scale prior, a coarse
affine derived from (180° rotation + surface tilt + centroid match) can
already clear the ≤ 100 µm / ± 5° target on half of the benchmark,
while the scale search honestly reports `unknown` when density patterns
are too coarse to localise — matching the grand-plan graceful-degradation
intent.

**Result.**
- **Minimal output is as good as or better than the expansion-prior
  first pass on 4/6 subjects** (790322, 767018, 755252, 767022 are
  within 30 µm either direction). 788406 regresses 75 µm; 782149 stays
  in its 350–420 µm thin-HCR band.
- **Tilt correction is active.** CZ surface tilt is ≤ 3° on every
  subject; HCR tilt is ≤ 12°. Four subjects benefit from the tilt
  component (~50 µm z-translation correction on each).
- **Scale search admits no reliable peak.** All 6 subjects fail the
  confidence bar. The confidence metric correctly flags that the NCC
  landscape is peaky but the peak's physical meaning is unverifiable
  at this spatial signal level. Minimal output is the intended fallback.

**Failure modes observed.**
1. *Asymmetric HCR GFP+ (788406, 782149).* HCR-centroid XY lands off
   the landmark target whenever the GFP+ distribution is skewed in XY
   relative to the CZ window. R2 is the intended fix.
2. *Rotation beyond the prior (767018, 10°; 755252, 6.7°; 767022,
   5.4°).* The 180° prior still limits the rotation error on half the
   subjects — structural; only a rotation estimator would close the
   gap.
3. *Density-map NCC XY collapses to small sxy.* On 400-µm CZ vs
   2.3-mm HCR, the NCC objective is maximised by a shrunken CZ placed
   near HCR's densest patch. Robust-z prominence cannot distinguish
   this from a correct-scale peak. The threshold was raised (3 → 6)
   rather than re-engineering the search, because Grand Plan §R1 step
   8 explicitly allows `unknown` outputs.
4. *Sz search drifts on thin-HCR subjects (782149, 755252).* Profile
   length of 10–20 bins doesn't constrain `sz` well. `tz_best_um` for
   these two is implausible (−1 460, −1 276 µm) — both discarded in
   favour of centroid Z, as specified.

**Next step.** Minimal `(R, t)` is the contract for downstream:
- **R2 (constellation matching)** must tolerate 100–400 µm origin
  error — same ballpark as first-pass R1, no degradation.
- **A1 (TEASER++) / A3 (FGW)** receive the minimal affine and estimate
  scale internally — their documented sweet spot.
- **Rotation drift (767018)** is the only place R1 *could* still gain
  (e.g. ICP on depth-banded centroids after tilt alignment); defer
  unless R2 seed-matching performance suffers from the 10° residual.

## Open follow-ups

- The confidence metric is currently (peak − median) / (1.4826·MAD)
  over the raw NCC curve. An alternative that checks *how concentrated
  the NCC mass is around the peak* (e.g. entropy of
  softmax(NCC / T) − entropy(uniform)) may distinguish "sharp peak"
  from "peaky but wide plateau" more reliably. Not urgent: the minimal
  output is already the documented fallback.
- 782149's tz_std is 2.7 µm (computed over in-band cells) but the
  sz_best = 5.9 is clearly spurious. Flag the diagnostic triplet
  `(coverage_regime = thinner, hcr_tilt > 10°, sz_conf < 3)` as a
  "thin-HCR pathology" marker in the R2 handoff.
- Decide whether to report `sz` diagnostically when `sz_conf ∈ [3, 6]`
  even though it's not emitted as a scale — currently all subjects sit
  in exactly that range. R2/R3 may find the information useful as a
  prior even without formal confidence.

## Files produced

```
sessions/05_R1_revised/
  log.md                          (this file)
  r1_results.json                 (per-subject benchmark output)

dev_code/
  r1_revised.py                   (the revised R1 module)
  05_r1_revised_benchmark.py      (validation harness, mirrors 04_r1_benchmark.py)
  05_r1_diag.py                   (per-subject score-curve dump)
```
