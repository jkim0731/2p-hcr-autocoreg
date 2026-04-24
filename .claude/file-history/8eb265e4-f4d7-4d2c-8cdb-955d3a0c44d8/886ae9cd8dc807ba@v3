---
name: Session 07 anisotropic-ICP scale estimator REVOKED
description: Session 07 ICP (sxy, sz) estimator is NOT valid — its 6/6 pass was against a GT-tuned boundary penalty, and the depth-density diagnostic shows centroid features can't recover scale.
type: project
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
Do not use `estimate_scales_icp_multi_start` as the R1 `(sxy, sz)` source. The Part C "6/6 pass" is benchmark-overfit, not a validated method.

**Why revoked:**
- The passing score includes `−1e6·1[at_bound]` and `+10·sz` — both tuned against landmark-Procrustes GT. Without those, 4/6 sxy values clip to the feasibility upper bound (2.0) because the ICP reciprocal-NN basin there fits the HCR GFP+ density better than the true scale does.
- Depth-density diagnostic (`dev_code/07_depth_density_diagnosis.py`, Part D of the session log) shows HCR GFP+ has subject-specific, depth-dependent detection bias vs the truth baseline (matched-HCR + unmatched-CZ mapped). Integrated GFP+/truth spans 0.46→7.31 across 6 subjects; per-bin CV is 0.33–1.12. CZ/truth ≈ 1 everywhere → bias is on HCR side.
- 782149: GFP+ = 0 beyond ~600 µm depth even though matched-HCR + CZ-unmatched exist to 1362 µm — severe under-detection at depth.
- 755252: GFP+ is ~16× truth at middle depths, dropping to 1× at edges — severe over-detection.

**How to apply (R1 work):**
- Neither `local_distance_scale.py` (session 06 k-NN) nor `estimate_scales_icp_multi_start` (session 07 ICP) give trustworthy per-axis scales — both assume same-population sampling that the data violates.
- For scales, use a non-centroid signal: (a) surface-to-surface Procrustes (z-scale from pia-plane offset, sxy from xy spread of fit points), (b) image-level CZ↔HCR-488 correlation after surface alignment, or (c) expansion-table prior (sxy ∈ [1.64, 1.92], sz ∈ [2.13, 3.58]) as a fixed value rather than an estimate.
- Scale-estimation from GFP+/CZ centroids is blocked on re-characterising the GFP+ threshold to at least track matched-HCR up to a uniform (subject-independent, depth-independent) scalar. Until then, do not treat GFP+ as a subsample of the true neuron population.

**Where:**
- Part C (original promotion, revoked) + Part D (diagnostic) in `sessions/07_scale_failure_diagnosis/log.md`.
- Figures: `sessions/07_scale_failure_diagnosis/figures/depth_density_<sid>.png`.
- Summary JSON: `sessions/07_scale_failure_diagnosis/depth_density_summary.json`.
