# S59 — F8 CZ-aware domain randomization probe (abandoned)

## Goal

S58 diagnosed a synthetic→real transfer gap: training on F8 HCR→HCR
asymmetric warps + InfoNCE loss descends to 2.0 on small clouds but
inference r@20 ≤ 0.04 on real CZ↔HCR. S58's own next-step list said
"F8 CZ-aware extension" might close the gap — S59 tests that directly.

## Approach

Added `sample_cz_aware` wrapping F8's asymmetric sampler with three
CZ-like source-side perturbations:

1. **Position noise** — Gaussian σ=8 µm on source centroids (CZ segmentation drift proxy).
2. **Spatial gradient dropout** — keep prob 1.0→0.6 top-to-bottom in Z
   (CZ partial-depth coverage proxy).
3. **Higher source drop** — `source_drop_rate=0.5` (matches CZ match-rate ~0.5).

Training: 500 iter on 788406 HCR GFP+ with LR=1e-4, 20-dim features.
Fail-fast criterion: r@20 ≤ 0.02 → abandon; > 0.10 → escalate to
full 6-subject bench.

## Result — NEGATIVE

```
initial50_mean=6.390  final50_mean=5.498  descent=0.892
n_pred=932  n_in_gt=787  hits@20=1  r@20=0.001
```

**r@20=0.001** — strictly worse than S58's 0.003 on the same subject
with the same model class and no CZ-aware perturbations. Training
descent (0.892/500 iter) is slower than S58 baseline (1.467/500 iter);
the extra perturbations made the synthetic matching *harder* to learn
without making the learned features *more transferable*.

## Why CZ-aware randomization doesn't bridge the gap

Three perturbations proposed in S58 as the "low-cost" fix for the
transfer gap are:

- **Position noise 8 µm**: CZ centroid noise is structurally different
  from Gaussian — CZ segmentation errors are topological (missing cells,
  merged cells, phantom cells), not jittery positions.
- **Spatial gradient dropout**: approximates CZ's partial-Z coverage
  but doesn't reproduce CZ's sparser *radial* density at depth or the
  image-quality-dependent density gradient.
- **Source drop rate 0.5**: target-drop + source-drop already covered
  correspondence partiality in S58; adding more drop just starves
  training signal.

Root cause reaffirmed: **real CZ is a different modality, not a noisy
HCR subsample**. Simulating modality gap from centroids-only is not
feasible without either (a) real CZ centroids as training source with
a known HCR pairing (requires manual labels we don't have at scale), or
(b) image-conditioned features that encode the modality directly
(C2 per Grand Plan §4.4).

## Ship decision

**Do not ship CZ-aware F8 extension.** Strictly negative on 788406 in
500 iter; extrapolation to 6-subject bench is strongly negative.
Keep `lib/synthetic_warps.py` as shipped in S58.

## Status: `abandoned`

C5 centroid-only plateau at sum r@20 = **1.080** is confirmed as the
ceiling reachable by centroid + synthetic-warp self-supervision alone.
Further gains require either image-conditioned features (C2, tier 2)
or cross-modal descriptor learning (I4, tier 3).

## Files

- `probe_cz_aware.py` + `.log` — 500-iter retrain + real-inference probe.

## Next session (per convergent evidence from S56/S57/S58/S59)

Honest stopping point for autonomous centroid-only work. The next
tier requires substantial new infrastructure:

- **C2 image-conditioned GNN** — 3D CNN patch encoder (16³ voxels)
  concatenated with F6 features, fed into G1's graph-attention +
  Sinkhorn head. Needs GPU, patch-extraction pipeline, training infra.
- **I4 cross-modal descriptor** — dual-branch CNN on CZ and HCR image
  patches with metric learning. More infrastructure than C2.
- **QF1 fallback classifier** — learned gate on candidate pairs using
  F6 + residuals + intrinsic confidence. Does not add new matches but
  re-ranks union_conf collisions. Conditional per Grand Plan §9.6
  (only activate if intrinsic confidence Brier > 0.15 post-calibration;
  S55 showed calibrated Brier 0.08–0.27 → not triggering the
  conditional gate).

All three are multi-session efforts with uncertain ROI given 782149's
structural unreachability (partial overlap + 12° tilt + thin Z).
Reaching ≥ 0.60 per subject (Grand Plan §9.6 ship target) on 782149
specifically requires image-level modality bridging; this is the gating
question the user should prioritize.
