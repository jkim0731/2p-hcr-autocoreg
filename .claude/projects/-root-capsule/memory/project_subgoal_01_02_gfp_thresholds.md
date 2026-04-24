---
name: Subgoal 04/R1/02 — intensity GFP+ threshold landed
description: Intensity subjects 755252/767022 now use peakgauss3_mean_bg_p1; spot subjects use peakgauss3_density_p0.1. Both promoted to loader defaults on 2026-04-17 (v2.2).
type: project
originSessionId: caec2b61-c841-4802-8466-6112cb68e89f
---

Session 04/R1 subgoals 01 + 02 closed out in v2.2 (2026-04-17).

**Spot default (`DEFAULT_GFP_THRESHOLD_METHOD`):**
`peakgauss3_density_p0.1` — 3-comp GMM on `log(density>0)`, rightmost
component, `exp(μ + z_{0.001} · σ)` threshold. Tighter than the prior
`yen_log_joint` on every spot subject (9.5–13.7 % GFP+ vs 9.5–16.1 %)
while raising coreg coverage from 0.96 to 0.99.

**Intensity default (`DEFAULT_GFP_INTENSITY_METHOD`):**
`peakgauss3_mean_bg_p1` — 3-comp GMM on `log(mean − background)` for
cells with `mean > bg`, rightmost component, `exp(μ + z_{0.01} · σ)`
threshold. Drops GFP+ from 100 % to 37 % (755252) / 19 % (767022).
Coverage slips to 0.93 / 0.95 (below the 0.95 bar on 755252) — user
accepted this trade-off on 2026-04-17 in exchange for a sensible GFP+
fraction on the two intensity-only subjects.

**Why:** Raw `log(mean)` is dominated by the autofluorescence bulk;
`mean − background` reveals a clean rightmost signal peak that matches
the coreg distribution. PeakGauss3 with `p=1` (not `p=0.1` as used on
the density side) because the signal component is broader on 767022.

**How to apply:** `load_subject(sid)` now returns GFP+-thresholded
frames for all 6 benchmark subjects. Use `gfp_threshold_method='yen_log_joint'`
or `gfp_intensity_method='none'` for rollback. R1 origin error on the
intensity subjects regresses slightly (755252: 71→124 µm; 767022:
158→181 µm) as a known side-effect of the GFP+ centroid shifting when
the candidate set shrinks — this is documented in `log.md` §Subgoal
01+02 v2.2, not something to tune against. R2 is the right place to
correct residual translation.

Artefacts: `sessions/04_R1_coarse_align/log.md` §Subgoal 01+02 v2.2,
`notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb` §9 addendum,
`notebooks/04_R1_subgoal_02_intensity_threshold.ipynb`,
`r1_results.json` (v2.2 current).
