# Session 06 — Local distance-based scaling estimator

## Goal

Estimate anisotropic scale `(sxy, sz)` between CZ GCaMP+ and HCR GFP+
centroid clouds from **local distance statistics only**, given R1
minimal `(R, t)` for localization. Verify against the anisotropic
expansion table in `docs/01 Data Description.md`.

**Stopping condition (from plan):** per-subject `sxy` **and** `sz`
within ±20 % of GT on **all 6/6 subjects**. If met, the estimator is
promoted into `CoarseAffineV2.scales` for the continued R1 effort.
If not, the method is insufficient — log per-subject bias and stop.

## Method

- **M1 — axis-separated k-NN distance ratio (primary).**
  `sxy = median(2D-kNN_xy)_hcr / median(2D-kNN_xy)_cz`,
  `sz = median(1D-kNN_z)_hcr / median(1D-kNN_z)_cz`.
  Separate xy/z projections are mathematically required: a 3D-kNN
  neighbourhood is isotropic in physical space so per-axis deltas
  within that neighbourhood all scale by `(sxy²·sz)^(1/3)` — confirmed
  on synthetic.
- **M2 — local volume-density ratio (isotropic baseline).**
  `s_iso = (ρ_cz / ρ_hcr)^(1/3)` via `k=10` mean-k-NN sphere density.
- **Engineering.**
  ±0.5 µm uniform jitter on z (break 1-µm quantization ties).
  Iterative HCR crop: start at R1 feasibility bound
  `sxy_upper = L_hcr_xy / L_cz_xy ≈ 5×`, shrink to `1.3 × max(est, 1.2)`
  per iteration until convergence (rel_tol 0.02, max_iter 8).

Synthetic sanity (matched populations): `sxy_est = 1.770`,
`sz_est = 2.765` against truth `(1.77, 2.82)`. Passes to < 2 %.
Synthetic with added uniform "extras" inside HCR bbox (`extras_factor
≥ 1`) breaks the estimate heavily — this signals sensitivity to
detection disparity.

## Benchmark result — 0/6 pass

`python dev_code/06_local_distance_benchmark.py` →
`sessions/06_local_distance_scale/results.json`.

| subject | n_cz | n_hcr_local | f = n_hcr/n_cz | sxy_gt | sxy_est | rel_err_sxy | sz_gt | sz_est | rel_err_sz | pass_xy | pass_z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 788406 | 932  | 2242 | 2.40 | 1.78 | 1.43 | −19.6 % | 2.82 | 0.78 | −72.4 % | ✓ | ✗ |
| 790322 | 1016 | 2201 | 2.17 | 1.76 | 2.00 | +13.2 % | 3.04 | 0.90 | −70.4 % | ✓ | ✗ |
| 767018 | 785  | 1699 | 2.16 | 1.70 | 1.75 |  +2.7 % | 3.58 | 1.05 | −70.7 % | ✓ | ✗ |
| 782149 | 894  | 2099 | 2.35 | 1.92 | 2.83 | +47.0 % | 2.93 | 0.61 | −79.3 % | ✗ | ✗ |
| 755252 | 835  | 5718 | 6.85 | 1.64 | 0.71 | −56.9 % | 2.13 | 0.27 | −87.5 % | ✗ | ✗ |
| 767022 | 926  | 2172 | 2.35 | 1.81 | 1.20 | −33.7 % | 2.49 | 0.79 | −68.3 % | ✗ | ✗ |

**Pass sxy (±20 %): 3/6. Pass sz (±20 %): 0/6. Both: 0/6.** Stopping
condition not met → method rejected per plan.

## Failure-mode diagnosis

The ratio `f = n_hcr_local / n_cz` is ≈ 2.2–2.4 on five subjects and
6.9 on 755252. Under matched-population theory (HCR cells inside the
overlap = CZ cells), `f ≈ 1`; the observed ≥ 2× excess is a direct
detection disparity in the overlap volume. For an ideal 1D-z k-NN
measurement, this enters the ratio as:

- `sxy_est / sxy_gt ≈ 1 / √f` (2D density),  predicted 0.65 at f=2.4.
- `sz_est  / sz_gt  ≈ 1 / f`   (1D density),  predicted 0.42 at f=2.4.

The observed `sxy_est / sxy_gt` on the first four subjects (0.80,
1.13, 1.03, 1.47) is less systematic than `1/√f` would suggest —
sxy is partly rescued because the iterative crop tightens to the
correct xy scale when the density ratio is uniform. But sz comes out
≈ 0.27 on every subject — much worse than the density prediction
1/f ≈ 0.42. The extra loss comes from the z-jitter floor:

- CZ median 1-NN z = 0.13–0.17 µm; median 5-NN z = 0.46–0.56 µm.
  Both **smaller than the ±0.5 µm jitter**. Since z is stored at 1 µm
  and CZ has ~900 cells in ~400 µm, there are ≈ 2 cells per z-plane,
  so k ≤ 5 reaches at most one neighbouring plane and the kNN
  distance is dominated by within-plane jitter, not the z-plane
  separation which carries the sz signal.
- HCR median 1-NN z is similar (0.04–0.15 µm), again dominated by
  within-plane jitter at higher local density.
- The HCR stat is further suppressed by higher within-plane cell
  counts (f × more cells per plane → smaller within-plane jitter NN),
  so the ratio `sz_est = hcr/cz` falls well below `1/f`.

`755252` is the worst outlier on both axes because it has a pathologically
dense HCR GFP+ population (30 804 total vs ~10 k elsewhere) driven
through autofluorescence-limited thresholding (see `03_image_based_surface.md`
memory); the crop contains f ≈ 7× more cells, collapsing both estimates.

The M2 sanity `s_iso_est` is mostly unbiased in xy/z-averaged sense
(rel errors −7 % to −34 % on five subjects, −48 % on 755252), which
confirms that the anisotropic decomposition — not the absolute density
— is what breaks.

## Why the idea fails (general form)

The method assumes CZ and HCR GFP+ sample the same underlying neurons
at equal detection efficiency within the overlap. v2.2 GFP+ defaults
deliver good cover*ratios* per subject (coreg coverage 0.93–0.99) but
the **absolute detection rate** in HCR inside the CZ overlap is still
~2.3× the CZ GCaMP+ rate. Under that regime:

1. 1D-z k-NN at small k is structurally dominated by z-quantization
   + within-plane jitter, not by z-plane separation; the sz signal is
   attenuated beyond the clean density prediction.
2. Density ratio `f = N_hcr_local / N_cz` is close to constant across
   five subjects (≈ 2.3×), so the per-axis scales are multiplied by a
   **common** factor ≈ `1/√f` (xy) and `1/f` (z) which varies subject
   to subject, preventing a simple global correction.
3. The one outlier 755252 has f ≈ 7×, showing the factor is not
   tissue-universal and depends on local GFP+ detection sensitivity.

This rules out local-distance-only scale recovery as a drop-in
replacement for density-map NCC. Any usable variant must either (a)
match CZ and HCR populations before ratioing (e.g., via reciprocal
nearest-neighbour filtering) to remove the `f` factor, or (b) use a
z-statistic that is insensitive to the 1-µm quantization (e.g.,
plane-spacing histogram rather than per-cell kNN).

## Artifacts

- `dev_code/local_distance_scale.py` — estimator module (M1 + M2,
  iterative crop, z-jitter, synthetic sanity in `__main__`).
- `dev_code/06_local_distance_benchmark.py` — 6-subject driver.
- `sessions/06_local_distance_scale/results.json` — per-subject
  estimates, GT, crop diagnostics, axis-separated k-NN summaries.

## Conclusion & handoff

The stopping condition (all 6/6 within ±20 % on both axes) is not
met (0/6). Local ROI-to-ROI distance statistics are **not** a
sufficient scale source for R1 with the current v2.2 GFP+ thresholds,
because HCR GFP+ populations inside the CZ overlap carry a
subject-specific ~2–7× excess over CZ GCaMP+ that biases both sxy
(weakly) and sz (strongly, compounded by z-quantization).

**Next candidate directions (do not attempt here):**

1. **Match-then-measure.** Run a coarse reciprocal-NN matching on the
   R1-mapped clouds, restrict both CZ and HCR to the matched subset,
   and re-run M1. This removes the `f` factor by construction; sz
   still has to cope with 1-µm quantization (use much larger k, or a
   plane-spacing histogram).
2. **Use the HCR image itself.** The density-map NCC was already
   shown to fail in session 05 deviation 2, but a complementary
   signal — e.g., z-spacing of surface-gated ROI peaks vs CZ z-spacing
   — bypasses GFP+ threshold sensitivity.
3. **Pin sxy/sz from a priori calibration.** If anisotropic expansion
   is near-constant per imaging protocol, hard-code a tissue-prior
   range and let R1 refine within it.

Session 06 ends here; session 07 should pick one of the above based
on the current grand-plan priority.
