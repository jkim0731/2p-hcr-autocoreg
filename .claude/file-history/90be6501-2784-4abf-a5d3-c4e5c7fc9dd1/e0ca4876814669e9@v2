# S58 — G1 loss rewrite + feature enrichment + LR fix

## Goal

S52 diagnosed that G1's original Sinkhorn-NLL loss plateaued at ~4.95
because the 4200 unmatched-tgt cells dominated the sum-normalised
denominator. S58 set out to fix training and produce a G1 that can emit
useful pairs — in particular to unlock 782149 (currently r@20=0 everywhere).

## What shipped in code

### 1. `bench/candidate_impls/_g1_gnn_matcher.py::_pair_loss`
- Rewritten from Sinkhorn-with-dustbin NLL → symmetric InfoNCE over
  matched pairs only. Unmatched cells contribute to the partition denom
  but are not explicit targets. Sinkhorn+dustbin retained at inference
  for consistent one-to-one outputs.

### 2. `bench/candidate_impls/_g1_gnn_matcher.py::_simple_features`
- Expanded from 16-dim (k-NN distances + elevations) to 20-dim:
  added per-cloud min/max-normalised position (3-dim coarse spatial
  anchor preserved across asymmetric scale+TPS warps) and local density
  within 30 µm / median (1-dim).
- Self-contained (no pia surface); matches training+inference feature
  distributions.

### 3. `run_g1` default — `use_f6=False`
- Was `use_f6=True`; inference pulled 200-dim F6 features projected to
  first 16 cols — different distribution from training. Switch keeps
  train+inference on the same 20-dim features.

### 4. `_train_self_supervised` LR — 1e-3 → 1e-4
- LR=1e-3 descended briefly then overshot to uniform baseline by iter
  500 (oscillation around 6.75). LR=1e-4 gives stable monotonic descent.

## Diagnostic sanity progression

| Probe | Loss | Feat | LR | Iter | Descent |
|-------|------|------|----|------|---------|
| S52 orig | Sinkhorn NLL (sum-norm) | 16-dim | 1e-3 | 300 | ~0.1 (plateau at 4.95) |
| S58 v1 | Per-side-avg Sinkhorn | 16-dim | 1e-3 | 200 | 0.545 |
| S58 v2 | InfoNCE | 16-dim | 1e-3 | 200 | 0.021 |
| S58 v3 | InfoNCE | 20-dim | 1e-3 | 500 | oscillates 6.5→6.0→6.7 |
| S58 v4 | InfoNCE | 20-dim | 1e-4 | 500 | **1.467 (6.64 → 5.17 monotone)** |

## Full 6-subject bench — NEGATIVE

`sessions/58_g1_loss_rewrite/bench_g1.py`, 1000 iter, LR=1e-4, use_f6=False.

| sid | r@20 | med_err µm | n_pred | train_loss_final | wall |
|-----|-----:|-----------:|-------:|-----------------:|-----:|
| 788406 | 0.003 | 858.8 | 714 | 6.053 | 467s |
| 790322 | 0.040 | 902.5 | 735 | 2.028 | 49s |
| 755252 | 0.000 | 656.2 | 609 | 5.433 | 936s |
| 767022 | 0.003 | 764.8 | 742 | 2.149 | 68s |
| 767018 | 0.000 | 719.0 | 198 | 4.409 | 258s |
| 782149 | 0.000 | 978.8 | 258 | 2.182 | 43s |
| **sum r@20** | **0.045** | | | | |

**C5 baseline sum r@20 = 1.080.** G1 is ~24× worse.

## Why training converges but inference fails

Pattern: subjects with small HCR GFP+ clouds (790322, 767022, 782149)
reach training loss ~2.0 (very low — model clearly learned something on
synthetic warps). Subjects with large HCR clouds (788406, 755252) don't
converge in 1000 iter. **But r@20 is near-zero across all 6 regardless
of training loss.**

The learned matcher does not transfer from synthetic HCR→HCR warps to
real CZ↔HCR. Root causes (diagnosed but not fixed):

1. **Domain shift.** Synthetic training source is a random subsample of
   HCR GFP+; target is the same cells after an asymmetric anisotropic
   scale + TPS warp. Real CZ is a *different modality* with its own
   segmentation noise, depth distribution, and density profile. The
   model never sees real CZ features during training.
2. **Overlap pattern.** Real CZ ⊂ HCR in XY and covers ~500 µm Z of a
   ~1300 µm HCR stack; synthetic source is a uniform random subsample
   of the HCR bulk. The partial-overlap topology differs.
3. **Scale mismatch.** Training cubes are ≈ 400 µm containing ~150–250
   points; inference clouds are 1000 µm containing 800–17 000 points.
   pos_norm is per-cloud min-max — its meaning differs at these scales.
4. **Cross-attention softmax sharpness.** Trained on ~4200 target
   candidates; inference has 3 800–17 000, making softmax denominator
   4–5× larger and effective similarity magnitudes different than the
   model was tuned for.

## Ship decision — keep mechanics, don't add G1 to ensemble

**Keep (shipped):**
- `_pair_loss` InfoNCE (cleaner and doesn't plateau like Sinkhorn-sum-NLL).
- `_simple_features` 20-dim (richer spatial anchoring, identical
  train+inference).
- `use_f6=False` default, LR=1e-4.

**Do not:** add G1 to C5 ensemble — sum r@20 would drop.

**Status:** `first_pass_done_infrastructure_only`. G1 mechanics (GNN +
cross-attn + Sinkhorn+InfoNCE) are sound; real blocker is the
synthetic-to-real transfer gap.

## Implications for future work

1. **F8 extension needed** to bridge synthetic→real gap. Options:
   - Train on (CZ centroids from actual subject) → (HCR-warped-to-fit).
     Requires a rough I2/R1 warm-start to place CZ cells in HCR space
     + a simulated "which HCR cell generated each CZ cell" assignment.
   - Add domain randomisation: simulate CZ-like dropout (keep ~50% of
     cells), add Gaussian noise to positions, downsample density.
2. **C2 image-conditioned GNN is the natural successor.** Per Grand
   Plan Section 4.4, C2 augments F6 hand-features with CNN embeddings
   from 16³ image patches. The image gives modality-bridging signal
   that pure centroid features can't.
3. **782149 remains unreachable** by centroid-only methods. Sum r@20
   plateau at 1.080 is structurally bound by 782149 contributing 0.

## Files

- `bench/candidate_impls/_g1_gnn_matcher.py` — core edits (loss, features, LR).
- `sessions/58_g1_loss_rewrite/probe_loss_sanity.py` + `.log` — v1 per-side-avg.
- `sessions/58_g1_loss_rewrite/probe_loss_sanity_v2.log` — v2 InfoNCE baseline.
- `sessions/58_g1_loss_rewrite/probe_rich_features.py` + `.log` — v3 rich feat.
- `sessions/58_g1_loss_rewrite/probe_long_descent.py` + `.log` — v3 overshoot.
- `sessions/58_g1_loss_rewrite/probe_lower_lr.py` + `.log` — v4 LR=1e-4 stable.
- `sessions/58_g1_loss_rewrite/bench_g1.py` + `.log` — 6-subject bench.

## Next session

F8 CZ-aware extension or C2 image-conditioned GNN.
