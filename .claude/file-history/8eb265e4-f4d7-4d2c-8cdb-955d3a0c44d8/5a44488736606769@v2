---
name: Session 07e — σ_z(d) profile NCC for sz FAILED
description: σ_z(d) of centroids is near-constant over cortex — 1D profile NCC framework exhausted across 06/07d/07e.
type: project
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
**Fact.** Session 07e (sz from σ_z(d) sliding-window NCC + sxy from
xy k-NN, iterated with s0=[2,2,2] R1 prior, half-step damping) failed
the 5 %/both-axes bar on the primary pair 788406 + 790322 (0/2).
788406: sxy −24 %, sz −31 %. 790322: sxy +20 %, sz −42 %.

**Why:** σ_z(d) of centroid positions is approximately constant
across the cortical column (CZ CV ~6 %, HCR CV ~10 %). The only
structured feature in HCR σ_z is a tissue-bottom-edge drop from
~30 µm → ~14 µm that has no CZ equivalent (CZ terminates shallower).
NCC on near-flat profiles across modalities has no real alignment
signal — peak-to-sd ratio on the NCC curve is just 2.5 (noise level)
vs 8+ on synthetic control. Synthetic recovery (unstretched HCR as
synthetic CZ) works cleanly: c_true recovered within 1.3 % at NCC
0.88–0.99. The fitter is correct; the assumption that σ_z(d) carries
stretch information is wrong.

**Why:** Biologically expected — uniform cortical-column lateral
density gives σ_z ≈ W/√12 inside the sliding window. The only
variability comes from edge sampling, which is modality-specific.

**How to apply:** Do not revisit 1D profile NCC along pia-normal.
Three sessions confirm the ceiling: 06 (density, 0/6), 07d (intensity,
2/6 best with per-subject oracle 3/6), 07e (σ_z, 0/2 primary). The
information required to distinguish stretch is lateral, not
depth-radial. Open candidates (not started): σ_x(d)/σ_y(d) 2D
profiles, layer-boundary depth matching with independent layer
segmentation, N-conservation `sz = N/sxy²` given independent sxy,
2D `I(x,d)`/`I(y,d)` intensity NCC.

**How to apply:** Always run a 10-line pre-check on the feature
itself before designing a matcher. Plotting raw HCR σ_z(d) first
would have rejected the hypothesis before 400 lines of estimator
code. Files: `dev_code/07e_sz_from_zvar_profile.py`,
`dev_code/07e_synthetic_check.py`,
`sessions/07e_sz_from_zvar_profile/`.
