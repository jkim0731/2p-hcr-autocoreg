---
name: HCR pia surface fitter — default is quantile_ceiling (N22)
description: Canonical HCR surface fitting method in analyze_subject is quantile_ceiling (N22), replacing image_ceiling.
type: project
originSessionId: 69562ade-e31a-49ae-a77b-e0ac403e51de
---
`analyze_subject` in `dev_code/benchmark_analysis.py` defaults to
`hcr_surface_method="quantile_ceiling"` (function
`estimate_pia_surface_quantile_ceiling`). The returned
`hcr_surface` dict is the canonical fitted pia surface — downstream
code should treat it as a **feature**, not a diagnostic.

Recipe (N22, promoted from `dev_code/03_surface_790322_explore.py`):
  1. Anchor = `estimate_pia_surface_image_ceiling(..., safety_offset_um=0)`.
  2. Per-column top-of-signal inside anchor_z ± 150 µm
     (`relative_margin=0.25`, `min_signal_abs_frac=0.20`, `min_thick_um=10`).
  3. IRLS quadratic quantile regression at q = 0.70
     (puts 70 % of tops above the surface → pulls the fit deeper than
     a symmetric Huber fit).
  4. ROI-envelope clamp with `within_tile_q=0.10`, `safety_offset_um=3`
     (N22 fix: ignores shallowest 10 % of cells per 120 µm tile, so a
     handful of shallow-outlier cells per tile can't lift the surface).

**Why:** under the strict 2nd-percentile clamp (old `image_ceiling`),
subject 790322 had `onset_depth_um = 77.5 µm` while other benchmark
subjects were ≤ 17.5 µm. N22 collapses all 6 benchmark subjects to
2.5–7.5 µm onset with `above_frac ≤ 0.08 %`. Verified 2026-04-16
on 790322/755252/767018/782149/788406 after promotion.

**How to apply:** when loading or comparing benchmark results, assume
HCR surfaces under the new default (`quantile_ceiling`). The old
`image_ceiling` surface is still produced and returned under
`hcr_surface_image_ceiling` for regression/ablation.

Reference session: `sessions/03_surface_estimation_v2/log.md`
(N22 addendum); notebook `notebooks/03_surface_estimation_iteration_v3.ipynb`.
