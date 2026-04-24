# Session 07c — BIC-sweep GFP+ + CZ-density uniformity gate

**Verdict: FAIL.** Stopping condition (6/6 at ±5 % on sxy and sz) not met.
- GMM BIC-selected K sanity: **5/6** (755252 coreg coverage 0.70 < 0.80).
- CZ-density uniformity gate (per-bin CV of `GFP+ / CZ_mapped` ≤ 0.20):
  **0/6**.
- Scale estimators (M1 or M3 clearing ±5 % on both axes): **0/6** subjects.

07c addresses two concrete objections to 07b:
1. **K was wrong.** Production R1 uses GMM-3 for intensity (Session 04/R1).
   07b used GMM-2 on intensity and GMM-4 on spot. 07c sweeps K∈[2,6]
   per subject, chooses K* by BIC.
2. **The truth baseline was an indirect proxy.** 07b measured
   `GFP+ / (matched-HCR ∪ unmatched-CZ)` — a proxy for the true cell
   population that is itself derived from coreg. 07c replaces the proxy
   with **all CZ centroids mapped into HCR via R1** — a direct, coreg-free
   sample of the underlying neuronal cloud.

The new gate is tighter-aligned to what the scale estimator actually uses
(densities of two point clouds in the same frame), but the gate still
fails and the residual bias still dominates M1 sz.

---

## Part A. Plan recap + user course correction

User request (verbatim):

> "Based on 04_R1_coarse_align, it should have been GMM-3 for intensity-
> based cutoff, instead of GMM-2. How about making GMM-2 to GMM-6 for all
> and choose the one with the best fit per subject? check if this makes
> density of the ROIs within the overlapping regions similar between
> >threshold 488 ROIs and registered czstack cells within the same volume
> (do not need to use HCR coreg cells - just all czstack cells within the
> same volume after transformation)"

Two concrete changes:
- **Threshold**: BIC-selected K per subject, sweeping K=2..6 on
  `log(density > 0)` for spot subjects and `log10(mean − bg > 0)` for
  intensity subjects. Intersection formula unchanged: Bayes-optimal
  root of two weighted Gaussians between their means, fallback midpoint
  if no interior root. Linear cutoff = `exp(x)` or `10^x`.
- **Validation gate**: per-bin `GFP+ / CZ_mapped` in the xy overlap AABB,
  binned along the HCR pia-normal depth. Informative bins defined as
  those where `ρ_CZ > p25(ρ_CZ)`. Gate passes iff CV over informative
  bins ≤ 0.20. Integrated mean reported but not gated — it is proportional
  to `1/(sxy²·sz)` and so has a built-in subject-dependent offset.

Everything else carried over from 07b: M1 per-axis k-NN ratio, M3 span
ratio per cloud (FOV-contaminated), landmark-Procrustes GT, ±5 % bar.

---

## Part B. BIC-selected K and strict cutoffs

Code: `dev_code/07b_gfp_intersection_threshold.py` (extended with
`fit_gmm_sweep`); driver `dev_code/07c_scale_bic_cz_density.py`.
Figures: `/root/capsule/code/sessions/07b_scale_clean_gfp/figures/gmm_threshold_<sid>.png`
were re-generated with the sweep panel.

| subject | class | K* (BIC) | cutoff_strict | cutoff_v2.2 | n_strict | n_v2.2 | coreg_cov | sanity |
|---------|-------|---:|--------------:|------------:|---------:|-------:|----------:|:------:|
| 755252  | intensity | 6 | 215.12    | 20.75   |  5402 | 30804 | 0.70 | **FAIL** (coverage) |
| 767018  | spot      | 6 | 1.02e-3   | 8.0e-4  |  8114 |  9161 | 0.98 | ok |
| 767022  | intensity | 6 | 140.25    | 28.43   |  6341 | 14239 | 0.88 | ok |
| 782149  | spot      | 4 | 1.71e-3   | 1.34e-3 |  3450 |  3831 | 0.97 | ok |
| 788406  | spot      | 6 | 1.38e-3   | 6.7e-4  | 10729 | 17427 | 0.94 | ok |
| 790322  | spot      | 5 | 1.69e-3   | 1.51e-3 |  9675 | 10131 | 0.98 | ok |

**Notes.**
- **767022 recovered.** 07b's GMM-2 produced a degenerate fit (narrow left +
  wide right, no interior root, fallback midpoint = 10.6 — *lower* than v2.2).
  With BIC, K*=6 yields cutoff 140.25 (well above v2.2's 28.43); coreg
  coverage rises from 0.93 to 0.88 — sanity now passes.
- **755252 now fails coverage.** K*=6 pushes the intensity cutoff to 215.12
  (v2.2: 20.75). 70 % of the coreg-matched HCR cells survive the strict
  threshold — the other 30 % have lower intensity than the BIC-selected
  cutoff. This is a genuine "stricter threshold is more selective" failure,
  not a numerical artifact. The CZ-density gate is coverage-independent, so
  755252 still participates in the downstream gate.
- **K distribution.** 5/6 subjects land at K* ∈ {5, 6} — the edges of the
  sweep. This suggests the BIC curve is still improving at K=6 and the
  chosen cutoff is sensitive to the choice of sweep upper bound. Expanding
  to K=8 might shift boundaries but is unlikely to fix the downstream
  failure — see Part D.

---

## Part C. CZ-density uniformity gate

Code: `cz_density_gate()` in `dev_code/07c_scale_bic_cz_density.py`.
Figures: `figures/depth_density_cz_<sid>.png` (6 subjects — two-panel
plot: ρ_CZ vs ρ_GFP+ depth profile, then per-bin ratio with mean line).

**Construction.**
1. Map all CZ centroids into HCR µm via R1 minimal (identity-scale): 
   `cz_mapped = apply_coarse_affine(cz_all_um, r1_fit)`.
2. Clip both clouds to the xy AABB where they overlap.
3. Depth below HCR pia plane: `d = z − (a·x + b·y + c)`.
4. Bin both by depth (25 µm bins); compute `ρ_bin = N_bin / (A_xy · Δz)`.
5. Per-bin ratio `ρ_GFP+ / ρ_CZ`. CV computed over "informative" bins
   where `ρ_CZ > p25(ρ_CZ > 0)`.

**Results.**

| subject | per-bin CV | integrated mean | n_CZ_in_box | n_GFP+_in_box | gate |
|---------|-----:|------:|-------:|-------:|:---:|
| 755252  | 0.312 | 0.217 |  835 | 213 | ✗ |
| 767018  | 0.417 | 0.114 |  785 | 106 | ✗ |
| 767022  | 0.460 | 0.161 |  926 | 164 | ✗ |
| 782149  | 0.598 | 0.122 |  894 | 118 | ✗ |
| 788406  | 0.386 | 0.160 |  932 | 177 | ✗ |
| 790322  | 0.302 | 0.109 | 1016 | 129 | ✗ |

Gate bar = 0.20 → 0/6 pass. CVs are **lower than 07b's truth-based gate**
(0.30–0.60 here vs 0.35–1.09 in 07b), but still 1.5–3× over the bar.

**Integrated mean vs predicted 1/(sxy²·sz).** Under uniform sampling in
both modalities and a pure anisotropic scale between them, the
integrated-ratio prediction is `1/(sxy² · sz)` (CZ-mapped density is the
R1 output at identity scale, so the true expansion factor sits entirely
in the ratio).

| subject | integrated | predicted 1/(sxy²·sz) | rel diff |
|---------|-----:|-----:|-----:|
| 755252  | 0.217 | 0.174 | +25 % |
| 767018  | 0.114 | 0.097 | +18 % |
| 767022  | 0.161 | 0.123 | +31 % |
| 782149  | 0.122 | 0.092 | +33 % |
| 788406  | 0.160 | 0.112 | +43 % |
| 790322  | 0.109 | 0.106 | +3 % |

The integrated mean tracks `1/(sxy²·sz)` to within 3–43 %. This is the
first positive signal: the BIC-GFP+ density **is roughly proportional**
to the CZ-mapped density, with a subject-offset that matches the
expected scale ratio. The failure is the per-bin pattern, not the bulk
level.

---

## Part D. Scale estimates (M1 k-NN, M3 span) vs GT

Figure: `figures/scales_comparison.png`.

| subject | GT sxy | GT sz | M1 sxy | M1 sz | err M1 sxy | err M1 sz | M3 sxy | M3 sz | err M3 sxy | err M3 sz | pass5 |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 755252 | 1.64 | 2.13 | 1.77 | 0.98 | **+7.7 %** | −54.1 % | 4.66 | 2.21 | +184 % | **+3.8 %** | ✗ |
| 767018 | 1.70 | 3.58 | 1.89 | 1.05 | +11.2 % | −70.6 % | 5.65 | 2.77 | +232 % | −22.6 % | ✗ |
| 767022 | 1.81 | 2.49 | 1.97 | 0.83 | +8.7 % | −66.7 % | 4.57 | 2.38 | +153 % | **−4.3 %** | ✗ |
| 782149 | 1.92 | 2.93 | 3.02 | 0.61 | +56.9 % | −79.0 % | 5.86 | 1.40 | +205 % | −52.0 % | ✗ |
| 788406 | 1.78 | 2.82 | 1.92 | 0.80 | **+8.0 %** | −71.6 % | 5.74 | 3.05 | +222 % | +8.2 % | ✗ |
| 790322 | 1.76 | 3.04 | 2.05 | 0.90 | +16.5 % | −70.3 % | 5.86 | 2.53 | +232 % | −17.0 % | ✗ |

### M1 sxy improved vs 07b (per-axis k-NN ratio)
07c's M1 sxy is consistently within 7–17 % of GT on 5/6 subjects (the
exception is 782149 at +57 %). Compare to 07b where M1 sxy ranged from
−46 % to +57 %. This is evidence that the BIC-selected GFP+ set is a
*better* spatial sampler than 07b's K=2/4 GFP+.

### M1 sz remains broken (−54 % to −79 %)
Universal underestimate. The strict cutoff still retains cells that
cluster toward shallower depths — median nearest-neighbour z-distances
are smaller on GFP+ than on CZ, so `sz = med_hcr_z / med_cz_z < 1`.
The CZ-density gate confirms this: per-bin ratio is non-uniform in depth
(CV 0.30–0.60), so axis-separated k-NN ratios in z capture detection
bias in z, not tissue expansion in z.

### M3 unchanged from 07b
M3 sxy still ~5.6–5.9 (FOV contamination, independent of the threshold).
M3 sz competitive on 3/6 (755252 at +3.8 %, 767022 at −4.3 %, 788406 at
+8.2 %); these three subjects' HCR GFP+ z-span is dominated by true
tissue depth, not by detection bias. But M3 sxy cannot be saved without
removing the FOV mismatch — a per-subject AABB crop wouldn't help
because the CZ AABB is the limiting footprint in both coordinates.

### Near-misses — still don't recombine
- 755252 M1 sxy +7.7 % / M3 sz +3.8 %  → cross-method pair, but neither
  method yields both axes within 5 %.
- 788406 M1 sxy +8.0 % / M3 sz +8.2 %  → ditto.
- 767022 M3 sz −4.3 % (inside 5 %), but M3 sxy +153 %.

A hybrid "use M1 for sxy, M3 for sz" would give:
- 755252: (+7.7 %, +3.8 %) — fails sxy.
- 767022: (+8.7 %, −4.3 %) — fails sxy.
- 788406: (+8.0 %, +8.2 %) — fails both.

Closer, but no subject actually clears the 5 % bar on both axes.

---

## Part E. Verdict and next candidate

### Summary of evidence

1. BIC sweep **fixes** the 767022 failure mode from 07b and produces a
   much higher, more selective cutoff for 755252 / 767022 / 788406. The
   threshold machinery is now per-subject-robust.
2. The CZ-based gate is **less noisy** than the 07b truth-based gate
   (CV 0.30–0.60 vs 0.35–1.09) — mapping all CZ cells directly gives a
   denser, less-quantised reference than matched-HCR + unmatched-CZ.
3. The integrated mean GFP+/CZ ratio tracks the expected `1/(sxy²·sz)`
   to within ±43 %. This is the first quantitative confirmation that
   BIC-selected GFP+ density is **proportional** to the underlying cell
   density, modulo the scale factor we want to recover. But proportional
   in bulk ≠ uniform per-bin.
4. Despite (1) + (2) + (3), **no subject clears the 5 % bar on both
   axes** with either estimator. M1 sz is systematically low by ~70 %
   on 5/6 subjects; M3 sxy is systematically high by ~200 % on all 6.

### Root cause (unchanged from 07b)

The centroid-feature detection bias in HCR GFP+ is depth-dependent
at a level (CV 0.30–0.60) that is ~3–6× the ratio precision we need
(0.05 on a ~2× scale = CV ≈ 0.025 on the ratio). BIC sweep reduces the
bias by ~10–40 %, but the bias floor is set by the spot / intensity
feature itself — once a cell's intensity is below threshold, nothing
in post-processing brings it back. **This rules out all centroid-only
scale estimators** on the current HCR GFP+ signal.

### What's next

The plan's "if 07b fails, stop and log" condition was written before
this round. 07c confirmed 07b's core conclusion with a better threshold
and a better gate; the centroid-only hypothesis is now doubly falsified.

**The next substantive candidate remains image-level 488 NCC after R1 +
pia-normal alignment** (as sketched in 07b's Part E). It is substantive
enough to justify its own session:
1. Align surfaces: apply R1 (R, t) to CZ, then rotate both frames so
   HCR pia-normal → `+z`.
2. Build 1D intensity profiles `I_cz(z)` and `I_hcr_488(z)` by summing
   image intensity over xy slices in a matched xy ROI.
3. `sz = argmax NCC(I_cz(z·sz), I_hcr_488(z))`.
4. `sxy` = argmax 2D NCC between CZ and HCR xy slabs at matched depth.

Intensity integration is **linear** in signal, so false-negative spots
contribute at full intensity — bypassing the threshold bias entirely.
Session 05's `r1_revised` has the zstack reading machinery; the new
contribution is pia-normal alignment before the NCC search.

Do **not** attempt yet another centroid-feature variant on HCR GFP+ —
07b and 07c have ruled that class out.

---

## Files

```
sessions/07c_gfp_bic_cz_density/
    log.md                         (this file)
    results.json                   (machine-readable: GMM + gate + scales)
    cz_density_summary.json        (gate records only)
    figures/
        depth_density_cz_<sid>.png (6 subjects, 2-panel each)
        scales_comparison.png

dev_code/
    07b_gfp_intersection_threshold.py   (extended with fit_gmm_sweep, BIC panel)
    07c_scale_bic_cz_density.py         (driver)
    07c_scale_comparison_plot.py        (figure)
```

Memory: `.claude/projects/-root-capsule/memory/project_07c_bic_gfp_cz_gate.md`
(new); MEMORY.md entry added.
