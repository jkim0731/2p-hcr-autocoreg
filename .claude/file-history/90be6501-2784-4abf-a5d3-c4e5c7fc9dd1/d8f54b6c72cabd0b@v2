# S57 — I3 B-spline composition fix + 782149 unlock attempt

## Goal
Fix S49-diagnosed bugs in I3 (`mi_bspline` ignored `initial_affine`; `run_i3`
didn't chain I2 first) and test whether image-level MI registration unlocks
any of the centroid-unreachable subjects (782149, 755252, 767018).

## What shipped

### 1. `lib/sitk_wrapper.py::mi_bspline` rewrite
- Now accepts `initial_affine`, `initial_translation`, `initial_center` (all ZYX).
- Builds a `sitk.AffineTransform(3)` with ZYX→XYZ permutation
  `A_xyz = P @ A_zyx @ P.T`, passes it as `SetMovingInitialTransform(aff)`
  so the B-spline only fits the nonrigid residual on top.
- Recovers fitted B-spline from composite output via
  `out_tx.GetNthTransform(0)` and stacks into a `sitk.CompositeTransform`
  (add aff first, then fitted bspline — SITK is stack / last-in-first-out).
- Added `MIFitResult._sitk_composite` field + `apply_forward_hcr_to_cz(pts)`
  method that applies composite via `TransformPoint` loop with ZYX/XYZ
  permutation. Used for forward-mapping HCR centroids through aff∘bspline.
- Added `MI_BSPLINE_VERBOSE` module flag for per-iter callback (diagnosed
  bspline compute cost: ~2 s/iter at 120 µm grid, ~15 s/iter at 60 µm,
  ~60 s/iter at 30 µm).

### 2. `bench/candidate_impls/_i3_bspline.py` rewrite
- `run_i3` now chains I2 → mi_bspline (with I2's affine as initial) →
  forward-map HCR GFP+ centroids through composite → cKDTree NN in
  predicted-CZ space → emit pairs.
- Emits `pairs_df` with standard columns so it can stand alone or feed
  downstream consumers.
- Supports `skip_bspline=True` baseline (I2 affine alone + NN emission).

## Benchmark result — negative

| subject | n_cz | n_hcr_gfp | I3_skip_bspline r@20 | nn_p50 µm | med_err µm |
|---------|------|-----------|----------------------|-----------|------------|
| 788406  | 932  | 17 427    | **0.001**            | 18.2      | 134        |
| 782149  | 894  | 3 831     | **0.000**            | 35.5      | 277        |
| 755252  | 835  | 30 804    | **0.000**            | 12.5      | 443        |
| 767018  | 785  | 9 161     | **0.000**            | 17.5      | 317        |

Adding B-spline on 788406 (20–30 iter @ 60 µm grid) made it slightly worse
(r@20=0, med=219–238 µm) — the MI metric improved (−0.0074 → −0.0086)
but the geometric accuracy regressed. The B-spline finds image-overlap
minima that move HCR centroids in ways that don't preserve cell identity.

## Why I3-as-direct-pair-emitter fails

The failure mode is the same across all subjects. **nn_p50** (median NN
distance in predicted-CZ space) is small — 12–35 µm — meaning I2's affine
maps HCR centroids close to some CZ cells. But **med_err** (median distance
from predicted-HCR centroid to GT-HCR centroid) is 134–443 µm — meaning
the HCR cell we matched is almost never the correct partner.

This is the **"correct affine, wrong identity"** regime. At HCR 1-NN
spacing ≈ 20 µm, any CZ cell has multiple HCR candidates within the affine
residual. CZ→nearest-predicted-HCR picks one at random within the ambiguity
ball. B-spline doesn't reduce the ambiguity because local nonrigid
deformation is much smaller than the 20 µm inter-cell spacing.

P1 handles this regime by matching on F6 features (feature-NN), not
geometric-NN. I3 has no feature channel.

## Implications for future work

1. **I3-as-pair-emitter is abandoned.** NN in predicted-CZ space is
   structurally the wrong end-point for a 20 µm inter-cell spacing + 16–43
   µm affine residual regime.
2. **I3-as-warm-start is still viable.** The composed (aff ∘ bspline)
   transform could seed P1's putative-correspondence generator with a
   tighter initial crop / better per-axis scale than P1's own multistart.
   S49's C1 variant tested I2-alone as warmstart and found +0.01 r@20 on
   788406 only; I3 with bspline would likely be similar or worse given the
   metric-vs-geometry divergence we observed. Defer unless an image-only
   track is explicitly needed later.
3. **782149 remains unreachable via centroid or I3-NN.** Per S49/S56, the
   true unlock requires (a) G1 GNN trained on F8 synthetic warps (handles
   partial overlap + feature-based matching) or (b) I4 cross-modal image
   descriptors (tier 3). Both blocked by F8.

## Ship decision
- **Ship** the mi_bspline / run_i3 mechanics fix (code is correct and
  usable as infrastructure for future image-level work).
- **Do not** add I3 to the default C5 ensemble — r@20 ≤ 0.001 on all 6
  benchmark subjects makes it net-negative.
- **Status for I3**: `first_pass_done_infrastructure_only` (mechanics
  fixed; no per-pair signal; defer ownership of 782149 to F8/G1 track).

## Files
- `lib/sitk_wrapper.py` — mi_bspline rewrite + apply_forward_hcr_to_cz
- `bench/candidate_impls/_i3_bspline.py` — run_i3 rewrite
- `sessions/57_i3_bspline_compose_fix/probe_i3_788406.py` — 788406 probe
- `sessions/57_i3_bspline_compose_fix/probe_i3_782149.py` — stress probe
- `sessions/57_i3_bspline_compose_fix/minimal_bspline_788406*.py` — timing diag

## Next (per S56 pending + S57 findings)
F8 synthetic-warp pipeline (unblocks G1 retraining + I4). 782149 is the
forcing function — all ensemble work has saturated around r@20=1.08 with
782149 contributing 0. F8+G1 is the tier-1 path to break past that.
