# Session 32 — Trimmed (partial-overlap) ICP

## Why

S30 (rotation seeds) and S31 (M1 widened) both failed to recover
782149. S31's diagnostic established that **782149 is a partial-overlap
problem**: only 34 % of CZ cells have HCR partners and 45 % of CZ (at
true scale) lies outside HCR's Z range. Hypothesis: trimming ICP's
worst-residual pairs aggressively enough will let ICP converge on the
matching subset.

## What was built

`probe_trim.py` — 6 translation seeds × 6 trim quantiles × 6 subjects
= 216 ICP runs. Trim quantile ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; the
trim parameter is `inlier_residual_quantile` passed through
`estimate_scales_icp_multi_start` to `estimate_scales_icp`. Scored each
combination with:
- self-supervised: `recip × unique_frac` (same as S29 default ranker).
- oracle (diagnostic only): `n_gt_lt50`, median GT residual.

## Result — mixed; don't fold

| Subject | SS pick (seed, trim) | SS n_lt50 | Oracle best | OR n_lt50 | S29 prod |
|---------|---------------------|----------:|-------------|----------:|---------:|
| 788406  | dz-100, 0.4 | 191 | same | 191 | ~185 |
| 755252  | q75, 0.8    |   0 | hcr_gfp, 0.6 | 179 | ~40 |
| 767018  | hcr_gfp, 0.9 | 41 | hcr_gfp, 0.6 |  43 |  41 |
| 767022  | hcr_gfp, 0.5 |  0 | hcr_gfp, 0.4 |   0 | ~80 |
| **782149** | q25, 0.7 |  **0** | hcr_gfp, 0.4 | **0** | **0** |
| 790322  | hcr_gfp, 0.8 | 279 | same | 279 | ~260 |

- **788406** small improvement (dz-100, trim=0.4) — SS matches oracle.
- **755252**: oracle shows `hcr_gfp × trim=0.6` would give 179 pairs,
  but SS ranker picks `q75 × trim=0.8` (0 pairs). Same scales
  (sxy=1.50, sz=2.23) on both — the wrong basin has **more**
  reciprocal matches (536 vs 501), so rank_score higher (439 vs 417).
  SS ranker cannot distinguish multi-basin solutions at matched scale.
- **767022** regresses (0 vs 80). Trim=0.5 finds a squeezed-basin
  solution with higher rank than trim=0.9's correct one.
- **782149** remains at 0 across every (seed, trim) combination.

**Conclusion: don't fold trim sweep into default.** SS ranker breaks
down when HCR has multiple density basins that each yield similarly
high reciprocal-NN counts. Kept probe as a reference; no code merged
into production.

## Deeper diagnostic — 782149 has no ICP local minimum at truth

Additional tests on 782149:

1. **Seed ICP directly from the GT landmark-fit transform** (R, S, t
   with rms=38 µm on GT pairs): ICP converges away from truth with
   median=666 µm across every trim level in {0.4…0.95}. The GT
   solution is not a local minimum of the full-cloud reciprocal-NN
   objective.

2. **Dense 180-seed XY translation grid** over HCR bbox (stride
   400 µm XY × 200 µm Z): all 180 seeds give **0 GT pairs within
   50 µm**. The correct basin is not reachable by translation seeding
   alone.

3. **CZ Z-truncation sweep**: keep only the top *frac* of CZ by Z
   (superficial cells only). At frac=0.4 (CZ subset contains 92 % of
   GT-paired cells), ICP recovers scales close to truth
   (sxy=1.96, sz=3.13 vs GT 2.02, 2.91) but translation still lands
   365-453 µm from truth on every frac in {0.3…1.0}.

4. **Non-GT/GT ratio** on 782149 is 11.7 — not particularly high
   vs other subjects (755252 is 47.3; 767018 is 32.6). False-positive
   GFP+ density is not the root cause.

5. **GT-region geometry**: HCR GT-paired cells occupy a compact bbox
   (x: 683-1525, y: 951-1704, z: 270-830); only 608/3831 = 16 % of
   HCR GFP+ cells are within ±50 µm of this region. GT centroid is
   335 µm from HCR GFP+ centroid (vs 130 µm on 788406). The correct
   basin is spatially isolated from the ICP's natural attractor.

**Conclusion: 782149 cannot be solved by any combination of
translation seeds + rotation seeds + trim + CZ truncation within the
standard reciprocal-NN ICP formulation.** The objective has no
local minimum at truth. This is a multi-basin problem where the
wrong basins have more reciprocal matches and tighter residuals than
the correct basin.

## What 782149 actually needs

Requires one of:
1. **Pia-surface-anchored alignment** — use the two pia surfaces
   (independent of GFP+ annotations) to constrain translation, then
   search only over scale and XY offset within the cortical slab.
2. **Image-level coarse alignment** — I2 MI-affine (already
   implemented, previously failed; needs re-test after fixing
   tuple-unwrap bug).
3. **Learned matcher** with feature-based correspondences that
   encode local cortical structure rather than pure geometry (G1 GNN
   with F6 features — requires F8 training data).
4. **Tight GFP+ restriction** — if we can tune the GFP+ threshold
   per-subject to reduce 3831 → ~600, but this requires a
   distribution-driven criterion that happens to be stricter on
   782149.

None of these are in scope for this session. The highest-leverage
next step for the full pipeline is either **I2 rerun with full-stack
bug fix** or **pia-anchored coarse alignment (new)**.

## Files

- `probe_trim.py` — 216-run trim × seed sweep.
- `probe_all.csv` — full per-run results.
- `summary.csv` — per-subject SS pick vs oracle best.

## Status

abandoned (no improvement; SS ranker regresses well-performing
subjects; 782149 unreachable).
