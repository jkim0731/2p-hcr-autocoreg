---
name: C5 plateau — §9.6 post-plateau priority queue exhausted (all 3 retrials negative)
description: As of 2026-04-22, C5 (P1⊕P4⊕P6 ensemble) sum r@20 = 1.080 (6-subj) / 0.866 (3-subj) is the empirical ceiling; all three post-C5 retrials (B-series, G-learned-centroid, C2 image-conditioned) have landed strict negative. Any further autonomous work on centroid/patch methods requires explicit user authorisation.
type: project
originSessionId: 90be6501-2784-4abf-a5d3-c4e5c7fc9dd1
---
**Grand Plan §9.6 priority queue status (2026-04-22):**

| # | Retrial | Session | Sum Δ r@20 vs C5 (3-subj, primary metric) | Status |
|---|---------|---------|-------------------------------------------:|--------|
| 1 | B-series (C5-seed TPS expansion)       | S62 | −0.502 (B3) / +0.001 (B3b)                | abandoned |
| 2 | G1-LOSO (real-supervised centroid-only) | S63 | −0.686                                     | abandoned |
| 3 | C2 image-conditioned GNN               | S64 | −0.854 (also 15× worse than G1-LOSO)      | abandoned_strict_negative |

**Why:** Each retrial targeted a distinct hypothesised bottleneck (seeding, training signal, feature modality); each landed negative. S64's negative is particularly diagnostic — adding 3D-CNN image patches makes things *worse* than centroid-only G1-LOSO, because the patch encoder memorises per-subject image texture as a shortcut rather than learning modality-invariant features.

**How to apply:** Before proposing new work on the G-series, B-series, or image-conditioned-GNN family on the current feature set, recognise this family is closed. C5 sum r@20 = 1.080 (6-subj) / 0.866 (3-subj) is the current production ceiling. Any further autonomous work on this track requires explicit user authorisation and should be framed as a tier-3 multi-session investment with uncertain ROI (especially since 782149 remains r@20=0 on every method tried to date).

**Remaining unblocked directions (all require user authorisation):**
1. Cross-modal self-supervision with subject-ID-invariance loss (new F8 variant; addresses S64 root cause).
2. Mask-based M-series after per-cell CZ segmentation (cellpose-on-z-stack preprocessing).
3. QF1 fallback classifier on C5 pairs (ROI-capped per S55's 0 r@20 calibration lift).
4. I3-composed image-level MI/B-spline with C5-conditioned pair emitter.
