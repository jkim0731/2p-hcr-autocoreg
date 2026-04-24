---
name: Session 07c BIC-GFP+ + GT-mapped density gate FAILED (corrected)
description: BIC-selected GMM K∈[2,6] per subject + GT-Procrustes CZ-density gate; initial R1-identity gate had a volume bug; corrected CV 0.21–0.38 (not 0.30–0.60); 0/6 pass 0.20 bar but 790322 is 7% over; integrated detection fraction ≈ 1.0 in true overlap; M1 sz still −70.8 % ± 0.7 %.
type: project
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
Session 07c tested two fixes to 07b, then found and corrected a
volume-selection bug in its own diagnostic.

**Two 07b fixes:**
1. **K was wrong.** Production R1 (Session 04) uses GMM-3 for intensity;
   07b used GMM-2. 07c sweeps K∈[2,6] per subject, selects K* by BIC.
2. **Truth proxy replaced.** 07b compared GFP+ to matched-HCR ∪ unmatched-CZ
   (coreg-derived proxy). 07c compares GFP+ directly to all CZ centroids
   mapped into HCR (coreg-free, dense reference).

**K* distribution.** 5/6 subjects at K* ∈ {5, 6}; 782149 at K*=4.
767022's 07b GMM-2 degeneracy fixed (K*=6, cutoff 140.25 vs v2.2's
28.43). 755252 now fails coreg coverage (0.70 vs 0.80 bar) because
K*=6 produces cutoff 215.12 (v2.2: 20.75); CZ-density gate is
coverage-independent so 755252 still ran.

**Volume bug and fix.** Initial 07c `cz_density_gate` used
`apply_coarse_affine(cz, r1_fit)` with `r1_fit.scales = identity` —
CZ cells were NOT expanded to their true coregistered volume. The
"overlap AABB" was dominated by CZ-native xy footprint (~0.25 mm²
vs the true ~0.5–0.6 mm²) and by CZ-native z-span (~500 µm vs the
true ~1000 µm on most subjects). User noticed: "depth from pia
capped below 700 µm when HCR is thicker; 782149 shown with same
x-axis as others — suspicious." Confirmed by re-run with
GT-anisotropic Procrustes map in `dev_code/07c_gate_gt_recheck.py`;
figures `depth_density_gt_<sid>.png`, summary
`cz_density_gt_summary.json`. GT is diagnostic-only — no leak into
M1/M3/threshold.

**Corrected CZ-density gate results (GT-Procrustes):**
- Per-bin CV of ρ_GFP+/ρ_CZ on informative bins: **0.21–0.38**
  (initial buggy pass had reported 0.30–0.60). 790322 is 0.214
  (7 % over the 0.20 bar); 767022 is 0.229; 767018 is 0.285.
  **0/6 pass** but the margin is small.
- **Integrated mean ≈ 1.0** (range 0.92–1.33) — BIC-GFP+ detects
  essentially the same total density as CZ inside the true overlap.
  This *replaces* the initial finding that "integrated tracks
  1/(sxy²·sz) within ±43 %", which was an artifact of the R1-identity
  volume compression. Under GT mapping the ratio is simply the
  detection fraction f ≈ 1.
- Scale-volume consistency (byproduct): the ratio of the two gate
  framings (R1-identity integrated / GT integrated) implies `sxy²·sz`
  within −30 % → +12 % of the GT value. Order-of-magnitude correct;
  not a scale estimator.

**Scale results (unchanged from original 07c run — no GT leak):**
- M1 sxy: 5/6 within 7–17 %; 782149 is +57 %.
- M1 sz: **universal underestimate 54–79 %**. Good-subset (767018,
  788406, 790322) M1 sz is tightly clustered at **−70.8 % ± 0.7 %** —
  a sharp, low-variance systematic tied to depth-dependent detection
  shape, not sampling noise.
- M3 sxy: 4.6–5.9× (FOV-contaminated, threshold-independent).
- M3 sz: 3/6 inside 10 % but split across good/bad subsets.

**Conclusion (moderated vs original).** The corrected diagnostic says:
BIC-GFP+ captures bulk density at near-unity detection fraction, but
the per-bin depth profile is non-uniform enough (CV 0.21–0.38) to
shift M1 sz by ~70 %. Centroid post-processing has not closed the gap
on any threshold family we've tried; the CV floor appears to be set by
the feature itself, not the threshold. The 07b+07c verdict is still
**failure at the 5 %/6-of-6 bar**, but the framing "doubly falsified"
over-states it — on the corrected diagnostic, 3/6 subjects are under
CV 0.30 and 790322 is within 7 % of passing. A substantial reduction
in per-bin bias (e.g., from a fundamentally different feature) would
have a reasonable chance of pushing the CV below 0.20.

**Reusable from 07c:**
- `dev_code/07b_gfp_intersection_threshold.py` — has
  `fit_gmm_sweep(log_values, k_min=2, k_max=6)` + BIC-selection panel.
- `dev_code/07c_gate_gt_recheck.py::gt_density_gate(...)` — corrected
  GT-Procrustes CZ→HCR mapping for depth-density diagnostics. Prefer
  this over the R1-identity version in `07c_scale_bic_cz_density.py`.

**Do NOT** try another centroid-only HCR GFP+ scale estimator.

**Next candidate (new session, 07c' — NOT implemented):** image-level
488 NCC after R1 + pia-normal alignment. Intensity is linear in signal
so threshold bias is bypassed. Session 05 has zstack reading machinery;
new contribution is pia-normal alignment before NCC.

**Where:** `sessions/07c_gfp_bic_cz_density/log.md`, `results.json`,
`cz_density_summary.json` (buggy), `cz_density_gt_summary.json`
(corrected), `summary.ipynb` (updated to GT diagnostic), figures
`depth_density_cz_<sid>.png` (6, buggy) and `depth_density_gt_<sid>.png`
(6, corrected), `scales_comparison.png`.
