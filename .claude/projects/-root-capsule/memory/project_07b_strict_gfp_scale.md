---
name: Session 07b strict-GFP+ scale estimator FAILED
description: Stricter GFP+ threshold (GMM-intersection between rightmost and next component) does NOT make HCR GFP+ a uniform subsample of truth; M1/M3 scale estimates still blocked by detection bias.
type: project
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
Session 07b tested: "if we cut HCR GFP+ at the Bayes-optimal intersection
between the rightmost and second-rightmost GMM components on log(feature),
does the stricter set become a subject- and depth-independent subsample
of the truth population (matched-HCR + unmatched-CZ)?" **No.**

**Result at 5 % bar, 6/6 required:**
- GMM sanity: 5/6 (767022 degenerate — GMM-2 yields narrow+wide
  components with no interior root; strict cutoff *smaller* than v2.2).
- Depth-density uniformity gate (CV ≤ 0.20 AND integrated ∈ [0.8, 1.25]):
  **0/5**. Per-bin CV 0.35–1.09; integrated ratio 0.41–4.75. Modest
  reduction from v2.2 (5–17 % on CV) but nowhere near the 5 % scale bar's
  implied uniformity requirement.
- Scale both-axes ±5 % on M1 or M3: **0/5**. Near-misses: 788406 M1 sxy
  at −1.5 %; 755252 M3 sz at −2.4 %. Nothing that recombines into a
  both-axes pass on a single subject.

**Failure patterns (diagnostic):**
- M1 (per-axis k-NN ratio) systematically underestimates sz by ~70 %
  on all 5 real subjects. Strict-GFP+ retains high-feature cells which
  cluster shallower → z-spread narrower than CZ-mapped → median z-NN
  distances too small.
- M3 (span ratio per cloud) systematically overestimates sxy by ~3.3×.
  HCR GFP+ covers full HCR FOV in xy, wider than R1-rotated CZ footprint
  → span ratio measures FOV coverage, not tissue expansion.
- Synthetic check passes at <2 % on both M1 and M3 — estimators are
  mathematically sound; failure is data-quality.

**Rules out:** all centroid-only scale estimators on HCR GFP+ (sessions
06 k-NN, 07 ICP, 07b strict-GFP+ M1/M3). The residual per-subject
depth-dependent detection bias is larger than the scale ratio we want
to measure. Further threshold tweaking is unlikely to help — the bias
is baked into the image-level spot/intensity feature itself, not just
the threshold.

**Also ruled out:** surface-only (07c) — pia-plane tilt erased by R1;
z-extent from pia-normal contaminated by modality-specific imaging depth.

**Next candidate (new session, 07c' — NOT implemented):** image-level
488 NCC after R1 + pia-normal alignment. Match raw intensity profiles
along z and in xy-slabs, bypassing the GFP+/GFP− binary. Intensity
integration is linear in signal, so false-negative spots still contribute
at full weight.

**Reusable from 07b:**
- `dev_code/07b_gfp_intersection_threshold.py` — GMM-intersection
  threshold machinery + 767022-degenerate detector, usable as a diagnostic.
- `dev_code/07b_scale_from_clean_gfp.py` — depth-density monkey-patch
  pattern (`_patched_loader` overrides `load_subject` binding in the
  diagnostic module's namespace).
- **Do NOT** reuse M3 span-ratio as a scale estimator — FOV-contaminated.

**Where:** `sessions/07b_scale_clean_gfp/log.md`, `results.json`,
`notebook.ipynb`; figures `gmm_threshold_<sid>.png`,
`depth_density_strict_<sid>.png`, `scales_comparison.png`.
