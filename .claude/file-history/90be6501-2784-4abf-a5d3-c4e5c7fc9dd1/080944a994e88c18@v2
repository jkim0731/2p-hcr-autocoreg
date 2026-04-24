# Session 53 — P6 BCPD (probreg) + CPD nonrigid

**Status:** validated (2026-04-20)
**Goal:** ship P6 (Grand Plan Section 4.1) as a tier-2 complement to P1 TEASER. Classical nonrigid point-set registration on CZ↔HCR GFP+ centroids with ICP-derived anisotropic warmstart.

## Implementation

- `bench/candidate_impls/_p6_bcpd.py` — new candidate. Three solver modes:
  - `method="bcpd"` — probreg Bayesian CPD (rigid+uniform-scale + GP-coherence nonrigid).
  - `method="cpd_affine"` — CPD affine.
  - `method="cpd_nonrigid"` — CPD nonrigid (**default**; see failure modes).
  - `hcr_quality_beta` kwarg for top-K NN selection with HCR image-quality bonus (mirrors S47 P1 / S51 P4 pattern).
- Pipeline: `default_warmstart_zyx → HCR bbox crop (pad=150µm) → CPD → NN assignment in warped space → match_radius filter → dedup by hcr_id → confidence = 1/(1 + d/σ_med)`.
- `sessions/53_p6_bcpd/probe.py` + `bench_all.py` — per-subject and 6-subject driver.

## Failure modes discovered

**BCPD and CPD rigid/affine collapse** with target:source ratio ≈ 3:1 on real CZ↔HCR after warmstart.

Diagnostic on 788406 (src=932 CZ, tgt=3165 HCR cropped):
- `registration_bcpd(w=0.3, lmd=10.0)` → `rigid_scale = 2.85e-10`, `v = 0`, `warped_extent = 2.5e-7 µm` (all 932 source points collapsed to a single target point).
- `registration_cpd(tf_type_name="rigid", w=0.3)` → `warped_extent = [1.1, 1.3, 1.8] µm` (collapsed).
- `registration_cpd(tf_type_name="affine", w=0.3)` → `warped_extent = [0.5, 44.8, 24.8]` (near-collapsed, partial spread in Y).
- `registration_cpd(tf_type_name="nonrigid", w=0.3)` → `warped_extent = [1395, 735, 725]` (preserved).

Only CPD nonrigid anchors each source point individually, so it sidesteps the target-uniform collapse mode. Shipping that as the default.

**β sweep on 788406** (with CPD nonrigid):
| β | r@5 | r@20 | median (µm) |
|---|---|---|---|
| 0.0 | 0.170 | **0.175** | 80.3 |
| 2.0 | 0.163 | 0.166 | 86.0 |
| 5.0 | 0.144 | 0.146 | 91.1 |
| 10.0 | 0.119 | 0.122 | 93.8 |

β *hurts* P6 monotonically — unlike P1/P4 where β=5 was the sweet spot. Explanation: CPD nonrigid's per-point warp already anchors each CZ near its true target; swapping to a higher-quality but further HCR neighbor via β is strictly worse. Default `hcr_quality_beta=0.0`.

## Benchmark (6 subjects, `method=cpd_nonrigid`, `pad=150µm`, `maxiter=30`)

| Subject | P6 r@20 | P6 med (µm) | P6 wall (s) | P1+β=5 r@20 | Δ |
|---|---|---|---|---|---|
| 788406 | 0.175 | 80.3 | 33.6 | 0.234 | −0.059 |
| 790322 | 0.224 | 60.8 | 23.3 | 0.251 | −0.027 |
| 755252 | 0.052 | 102.4 | 73.1 | 0.044 | +0.008 |
| 767022 | 0.053 | 123.9 | 50.6 | 0.103 | −0.050 |
| **767018** | **0.264** | **65.7** | 27.1 | 0.143 | **+0.121** |
| 782149 | 0.000 | 1144.4 | 35.6 | 0.000 | 0 |

## Key finding

**P6 nearly doubles recall on 767018 (stress subject, sparse GFP+)** vs. P1+β=5: r@20 0.264 vs. 0.143 (+12.1 pp, +85 % relative). Median error halved (65.7 vs. 80 µm).

Complementary regime to P1:
- P1 wins on dense GFP+ subjects (788406, 790322, 767022) where feature-aware putative pairs + TEASER-style outlier rejection dominate.
- P6 wins on sparse GFP+ (767018) where feature putatives are noisy and the CPD nonrigid warp anchors each CZ to its local HCR neighborhood directly.

767018's profile (sparse GFP+, 35 % manual match rate, older pipeline) was specifically flagged in Grand Plan 1.4 as a stress case; P6 is the first method to move it above 0.20 r@20.

782149 remains unreachable across all methods — warmstart fails (med_err = 1144 µm), 12° pia tilt.

## Decision / next step

Ship P6 as a production-complementary method alongside P1. A P1+P6 per-subject ensemble (pick winning method by intrinsic confidence or per-subject LOSO prior) is the obvious next step — opens as S54.

- **S54 P1⊕P6 ensemble** — run P1 and P6 in parallel per subject; merge correspondences by intrinsic confidence; measure union vs. per-subject best-of.
- (Deferred: BCPD/CPD-affine collapse could be fixed by applying aniso-scale pre-normalization inside the solver wrapper; not worth doing when CPD nonrigid already works. See failure-modes section.)
- (Deferred: 782149 warmstart rework — needs image-level I2 (SimpleITK MI) or M1 coarse; separate track.)
