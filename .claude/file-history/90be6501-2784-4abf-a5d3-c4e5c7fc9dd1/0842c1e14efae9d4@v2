# S49 (= S46-d) — Alt coarse path for 782149

**Status:** negative with bug-fix byproduct. I2 init-scale direction corrected (shipped). C1 wired to use I2 warm-start (shipped, but regresses vs. plain P1 and does not unlock 782149). 782149 remains unreachable by any centroid method; next session must try I3 deformable or M-series.

## Problem (from S48)

782149 is the only subject still at `r@20 = 0` across every tier-1 pipeline stage (P1, C1, B1/B2). S45's top-K ceiling analysis shows `GT_in_top_500 = 0.000` — the GT partner is not even within the 500 nearest HCR-GFP+ centroids of each CZ cell under default warm-start. So no centroid-only refinement can recover 782149; the coarse alignment must be re-built from image data, not centroid data.

Pia tilt on 782149 is 12° (per-subject worst), Z-overlap only 55 % (thin section), and the GT region is ~335 µm off from the HCR-GFP+ centroid (so default centroid-align places CZ in the wrong HCR region).

Hypothesis for S46-d: I2 (SimpleITK MI affine on the 488 volume, which includes non-GFP tissue) gives a better translation than default centroid-align. If I2's residual is small enough to land GT in P1's K=5 putative set, C1 unlocks 782149.

## Finding 1 — I2 was silently broken (init-scale direction reversed)

`_i2_sitk_affine.py` passed `init_scale=(1.8, 1.8, 2.8)` to `mi_affine`, which constructs SITK's forward transform `p_moving = A @ (p_fixed − c) + c + t` with `A_diag = (−1.8, −1.8, 2.8)`. Under SITK's `reg.Execute(hcr, cz)` (fixed=HCR, moving=CZ), A maps HCR→CZ, which for our expansion microscopy (CZ is the unexpanded tissue, HCR is 1.77–2.83× expanded) should have scales ~(1/1.8, 1/1.8, 1/2.8) = (0.56, 0.56, 0.36), i.e. shrinking HCR coordinates to CZ scale.

Fed (1.8, 1.8, 2.8) instead, the forward transform magnified HCR offsets by 1.8×, mapping HCR-centered points to positions outside the CZ volume (where the interpolator returns 0). MI metric became flat → optimizer couldn't descend → A stayed at its wrong init for every subject.

Symptom (probe_i2_residual_v2.log):
```
788406: scales_inv=[0.56, 0.555, 0.358]  # exactly the init values
782149: scales_inv=[0.555, 0.549, 0.349]  # exactly the init values
```
With the fixed init `init_scale=(1/1.8, 1/1.8, 1/2.8)` (probe_i2_residual_v3.log):
```
788406: scales_inv=[2.51, 1.82, 1.75]   # recovered ~real anisotropy
782149: scales_inv=[3.29, 1.84, 1.77]   # Z slightly off (expected)
```

The fix is one line in `_i2_sitk_affine.py`:
```python
init_scale=(1.0 / 1.8, 1.0 / 1.8, 1.0 / 2.8),
```

This is a **real bug fix** that affected every I2/C1/I3 benchmark run in the repo — all prior I2 numbers were the raw init transform, not an optimized fit. The S44 note "C1 ties P1" was a consequence: C1 ran I2 (no-op) + P1 (default), so C1 ≡ P1.

## Finding 2 — C1's hybrid wiring was broken (no warm-start forwarded)

Old C1:
```python
r_i2 = run_i2(s)
r_p1 = run_p1(s)  # never sees I2's affine
return CoregResult(pairs_df=r_p1.pairs_df, ...)
```
This is a no-op hybrid — P1 runs with its own default warm-start regardless of I2's output. Fixed (this session) to apply I2's SITK inverse to CZ centroids and pass them as `cz_init=` to P1.

## Finding 3 — Even with bug fixes, C1 regresses vs. P1

Post-fix benchmark (bench_c1.py, default C1 with K=30, c_bar=40 µm):

| subject | P1 r@20 | P1 med µm | C1 r@20 | C1 med µm |
|---------|---------|-----------|---------|-----------|
| 788406  | 0.234   | 80.2      | 0.000   | 290.4     |
| 755252  | 0.044   | 98.5      | 0.000   | 386.1     |
| 767022  | 0.103   | 110.9     | 0.000   | 193.1     |
| 782149  | 0.000   | 1134.9    | 0.000   | 198.3     |

I2's residual (per probe_i2_residual_v3.log):
- 788406: median 155.7 µm, p90 210.3, GT_in_top_20 = 11.2 %
- 782149: median 546.0 µm, p90 601.7, GT_in_top_20 = 5.4 %

I2's 155 µm residual on 788406 is ~8× the 1-NN spacing (20 µm). P1's default ICP-based warm-start (session 07) fits anisotropic scale and translation on the subject's own centroid data and achieves ≤ 20 µm residual — tight enough for K=5 to contain the GT partner. I2's image-MI fit is objectively **coarser** than ICP on centroid data. Passing it as a warm-start replaces a 20 µm warp with a 155 µm warp, shifting GT out of the K=5 putative list. Widening to K=30 doesn't help because GNC-TLS refits on mostly-wrong putatives and drifts further.

For 782149, I2 is a partial win (5.4 % GT_in_top_20 vs. P1's 0 %), but the residual is 546 µm — still 25× the NN spacing. No K value in [5, 30] recovers GT.

## Decision (S46-d)

**I2 alone does not unlock 782149.** C1 in its fixed form regresses precision on subjects where P1 already works, and does not close the gap on 782149.

Three forward paths, in order of expected ROI:

1. **I3 chained with I2 + point-apply** — run I2 affine as initial, then mi_bspline deformable on top, then apply composed (affine + B-spline) to CZ centroids. The B-spline should absorb the residual nonrigid warp (30-60 µm residual per F5 spec). Non-trivial infrastructure: mi_bspline currently doesn't use `initial_affine` in the SITK transform and doesn't return a point-apply function. Probably 1-2 sessions to build.

2. **M-series (mask NCC)** — M1 uses GFP+ mask density at 20 µm isotropic grid, sweeps over per-axis scales. Infrastructure (F1/F2/F3/F4) is now available (S48 shipped `cz_voronoi_labels`). Expected to behave similarly to R1 on sparse-GFP+ stress subjects — 782149 has only 303 GT cells, so GFP+ mask density signal may be as weak as centroid density. Medium-risk 1-2 sessions.

3. **Accept 782149 as structurally unreachable under the current method list** — document as a known limitation, ship the I2 + C1 fixes, and move to M-series or learned methods (G1 GNN) for the main roadmap.

**Recommend (3) for immediate ship**, then revisit 782149 after F8 (synthetic warps) + G1 (GNN) land — a learned method trained on synthetic warps may be robust to 782149's partial-overlap in a way that hand-crafted coarse alignment is not.

## Shipped

- `bench/candidate_impls/_i2_sitk_affine.py` — `init_scale=(1/1.8, 1/1.8, 1/2.8)`. Previously broken; all I2 benchmarks pre-S49 should be considered invalid.
- `bench/candidate_impls/_c1_image_centroid.py` — now applies I2 inverse to CZ centroids and forwards as `cz_init=` to P1. Also takes `K` and `c_bar` kwargs so callers can widen putative search.
- Per-cell residual probes `probe_i2_residual_v3.log` (corrected) and `bench_c1_K30.log`.

## Files

- `sessions/49_s46d_782149_alt_coarse/probe_i2_residual.py` — I2 per-cell residual probe (validation-only; uses `s.coreg_table`).
- `sessions/49_s46d_782149_alt_coarse/probe_i2_residual_v3.log` — post-bug-fix I2 residual numbers.
- `sessions/49_s46d_782149_alt_coarse/bench_c1.py` — 4-subject P1 vs. C1 sweep.
- `sessions/49_s46d_782149_alt_coarse/bench_c1_K30.log` — showed C1 regresses on all subjects.
