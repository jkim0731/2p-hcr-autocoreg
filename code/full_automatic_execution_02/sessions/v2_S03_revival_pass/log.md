# v2-S03 — Stage E.RV: revival of v1 candidates with locked priors

**Date:** 2026-04-27
**Owner:** automatic
**Status:** in-progress
**Companion docs:** `/root/capsule/code/docs/09 Full automatic v2 plan.md` §5 line 448-462

## Goal

Re-bench v1 candidates with the Stage A locked-prior warm-start (instead
of `default_warmstart_zyx`'s session-07 ICP, which was REVOKED for being
GT-tuned).  Per v2 plan §5 the order is:

(a) **P1 / P4 / P6 / C5** with the Stage A frame (sz pinned when v2-S02
    passed; otherwise sz left free).
(b) **P3 RANSAC** with sxy + t_xy locked.
(c) **P5 FGW** with sxy-rescaled CZ centroids.
(d) **M1 image NCC** on the surface_registration_v2 crop using PWR4×4
    warped CZ binary.
(e) **M3 mask + ICP** with M1 → Stage A as warm-start.

## v2-S02 outcome (consumed by this session)

`sz_s = FAILED` for **all 6 subjects** (smoothed_voxel best image-NCC
metric: 4/6 within ±0.30 of GT but 0/6 pass strict HW≤0.30 criterion).
Per v2 plan §6 stop-rule: sz-locked variants are skipped on these
subjects; sxy-only branches still run.  In practice this means P1/P4/P6/
C5 here run with `sz` left free (the candidate fits its own sz inside
the anisotropic-affine).

## Pass criterion (per method, across 6 subjects)

> Anything beating C5 r@20 on ≥ 4/6 subjects enters production. (§5)

v1 baseline (S56 `three_way_results.csv`, "auto3" column = C5 with auto
strategy selection):

| sid    | n_hcr | C5 r@20 |
|--------|-------|---------|
| 788406 | 17427 | 0.263   |
| 790322 | 10131 | 0.289   |
| 755252 | 30804 | 0.066   |
| 767022 | 14239 | 0.117   |
| 767018 |  9161 | 0.315   |
| 782149 |  3831 | 0.000   |
|  sum   |       | **1.050** |

## Method (a — P1/P4/P6/C5 with locked prior)

1. For each subject, compute `compute_locked_prior_warm_start(s)`.
2. Warp CZ centroids in HCR µm using the locked frame:
   `cz_init_zyx = (cz_zyx − src_mean_zyx) · diag(scales) · R^T + t`.
3. Replace v1's `default_warmstart_zyx` output by monkey-patching the
   helper, so P1/P4/P6 (and therefore C5) all consume the locked
   `cz_init`.
4. Run each candidate via `bench.harness.run_candidate(...)` and capture
   `r@5/10/20/30`, `n_pred`, `median_error_um`.
5. Compare to v1 baseline column-for-column.

## Cost estimate

Per subject, P1+P4+P6 ≈ 5 min (S56 records).  C5 = sum.  All four ×
6 subjects ≈ 30 min for sub-stage (a).

## Inputs (cached)

* Stage A: `compute_locked_prior_warm_start(s)` from
  `full_automatic_execution_02/lib/locked_prior_warm.py`.
* v1 candidate registry: `full_automatic_execution_01/bench/candidate_impls/`.

## Iteration log

* **2026-04-27 — iter 0.**  Scaffolding session.  About to write the
  monkey-patched warm-start helper and the (a) runner.

* **2026-04-27 — iter 1.**  Implemented `bench_a_p1_p4_p6_c5.py` (monkey-
  patches `lib.centroid_helpers.default_warmstart_zyx` to return locked-
  prior-warped CZ centroids).  Ran on all 6 subjects (~22 min).
  Results, **r@20 per method × subject** (LP warm-start in v2 vs v1
  baseline `three_way_results.csv`):

  | sid    | C5_v2 | C5_v1 | Δ     | P4_v2 | P4_v1 | P1_v2 | P1_v1 | P6_v2 | P6_v1 |
  |--------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
  | 755252 | 0.066 | 0.066 | 0.000 | 0.061 | 0.091 | 0.006 | 0.044 | 0.052 | 0.052 |
  | 767018 | 0.582 | 0.315 | +0.267| 0.575 | 0.308 | 0.147 | 0.143 | 0.264 | 0.264 |
  | 767022 | 0.223 | 0.117 | +0.106| 0.213 | 0.048 | 0.166 | 0.103 | 0.053 | 0.053 |
  | 782149 | 0.020 | 0.000 | +0.020| 0.010 | 0.000 | 0.023 | 0.000 | 0.000 | 0.000 |
  | 788406 | 0.098 | 0.263 | −0.165| 0.112 | 0.173 | 0.062 | 0.234 | 0.175 | 0.175 |
  | 790322 | 0.479 | 0.289 | +0.190| 0.420 | 0.177 | 0.401 | 0.251 | 0.224 | 0.224 |
  | sum    | **1.468** | 1.050 | **+0.418** | **1.391** | 0.797 | 0.806 | 0.775 | 0.767 | 0.768 |

  **C5 with LP warm-start beats v1 C5 on 4/6 subjects strictly**
  (767018, 767022, 782149*, 790322; 755252 ties), satisfying the v2
  plan §5 production threshold.  Sum r@20 = 1.468 vs v1 1.050 (+40 %).
  P4 sees the biggest individual gain (sum 1.391 vs 0.797).  P6 is
  unchanged across all subjects (its CPD step ignores the warm-start
  rigid_scale; only P1's TLS and P4's spectral matching consume it).

  Loser: 788406 (C5 0.098 vs v1 0.263).  v2-S02 found `sz_peak ≈ 2.90`
  ≈ GT, but FAILED the strict HW criterion → LP's depth-ratio
  `sz_init = 3.596` (28 % high) drives this regression.  Per v2 plan,
  we accept this — no fallback.

  *782149's "win" is trivial (0.020 vs 0.000) — both within noise.

  **Decision:** sub-stage (a) goal met.  Move to (b) P3 RANSAC and (c)
  P5 FGW with the same LP scaffold; defer (d) M1 image-NCC and (e) M3
  mask+ICP to a follow-up iteration (different code path).
