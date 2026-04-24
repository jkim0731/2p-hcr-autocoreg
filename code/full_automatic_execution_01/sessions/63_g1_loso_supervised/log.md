# S63 — G1 LOSO-supervised retraining (real `coreg_table`)

## Goal

S58/S59 showed G1 trained on F8 synthetic HCR→HCR warps does not transfer to
real CZ↔HCR inference (r@20 ≤ 0.04 on 788406). Root cause hypothesis: real CZ
is a different modality (segmentation noise, density profile, depth
distribution) from any centroid-only simulation of an HCR subsample. S63
tests a direct alternative: bypass synthetic warps entirely and train G1
supervised on real `coreg_table` pairs with leave-one-subject-out across all
six benchmark subjects (the three training-only subjects contribute
supervision but are not validated against).

Per 2026-04-21 priority queue: **G1-LOSO is experiment (2) in the cheap
retrial pair**; experiment (1) was S62 B-retrial (`abandoned`). Only if both
land negative does C2 image-conditioned GNN become the next authorized
infrastructure step.

## Method

`bench/candidate_impls/_g1_loso_matcher.py`: per iter, sample one training
subject (of the 5 non-held-out), jitter its warm-started CZ centroids ±4 µm,
crop HCR to CZ bbox + 200 µm pad, extract 20-dim `_simple_features` on both
clouds, build k-NN graphs (k=8), forward the existing GNNMatcher (hidden=96,
n_layers=4, cross_layers=3), compute InfoNCE `_pair_loss` on the
`coreg_table` pairs that survive the crop, and Adam-step (lr=1e-4). 2000
training iter per held-out subject. Inference: same pipeline on the
held-out subject, Sinkhorn + dustbin assignment, greedy one-to-one by
confidence.

## Results — refuted across all 3 validation subjects

| Subject | C5 r@20 | G1-LOSO r@20 | med (µm) | hits / denom | train loss (last 50) | Δ (G1-LOSO − C5) |
|---------|--------:|-------------:|---------:|--------------:|---------------------:|-----------------:|
| 788406  | 0.262   | **0.013**    | 219.5    | 10 / 764      | 2.34                 | **−0.249**       |
| 790322  | 0.289   | **0.113**    | 133.5    | 88 / 781      | 2.57                 | **−0.177**       |
| 767018  | 0.315   | **0.055**    | 198.2    | 15 / 273      | 2.62                 | **−0.260**       |
| **SUM** | **0.867** | **0.181**  | —        | —             | —                    | **−0.686**       |

Training pool per held-out: 2,684 GT pairs total across the 5 non-held-out
subjects — orders of magnitude more supervision than F8 synthetic warps
provided. All three runs descended the InfoNCE loss from ~6.9 → ~2.5 (80 %
of the descent from random init), confirming the model did learn *something*
on the training distribution.

## Why real-supervised G1 still fails

1. **Loss descent does not imply generalizable matching.** Final training
   loss ≈ 2.5 means the correct HCR gets about `exp(-2.5) ≈ 8 %` of the
   Sinkhorn mass *on the training subjects after overfitting*. The
   held-out inference shows median positional error 133–220 µm — an order
   of magnitude above the 20 µm recall threshold. The model learned
   training-distribution patterns that do not transfer to a held-out
   subject's local geometry.

2. **Per-subject variance is enormous.** 790322 yields 0.113 recall (real
   signal present); 788406 yields 0.013 (near-random). This is the
   signature of a model that has overfit to training subjects whose local
   geometry happens to resemble 790322's, with no systematic generalization.

3. **Centroid-only features are insufficient.** 20-dim `_simple_features`
   (k-NN distances + elevations + position_norm + density) are
   rotation/scale-insensitive but do not encode modality-specific cues
   that distinguish CZ from HCR. A learned matcher with only these inputs
   cannot overcome the CZ↔HCR modality gap even with real supervision,
   because the shared feature space loses the very cues that differentiate
   a correct cross-modal pair from a nearby wrong one.

4. **The root cause is modality, not training signal.** S58/S59 argued
   F8's failure was a synthetic→real gap; S63 shows that swapping synthetic
   for real supervision does not fix it. The constraint is on what a
   GNN with centroid-only inputs can learn, not on what training data
   was used.

## Implication — G-learned family is closed for centroid-only inputs

- G1 trained on F8 synthetic warps: **rejected S58/S59** (r@20 ≤ 0.04).
- G1 trained on real `coreg_table` LOSO: **rejected S63** (sum Δ = −0.686).
- G2 (contrastive embedding, same features) and G3 (learned edge predictor,
  same features) are expected to fail identically — they share the
  centroid-only feature ceiling.

Centroid-only autonomous work is now fully saturated across three families:

- **Hand-crafted matchers (P/M/C):** C5 sum r@20 = 0.867 (1.080 on the
  6-subject roster); no combination improves (S60).
- **Classical greedy expansion (B):** sparse-seed TPS cannot bridge the
  nonrigid warp (S23/S24, S62 with smart seeds).
- **Learned matchers (G):** both synthetic-supervised (S58/S59) and
  real-supervised (S63) fail below C5.

## Decision — both retrials negative; C2 is the next authorized step

Per the 2026-04-21 directive in `grand_plan_working.md` §9.6:

> "Only if both retrials [B-retrial, G1-LOSO] land negative does C2
> image-conditioned GNN become the next authorized infrastructure step."

Both are negative. **C2 is authorized.** C2 adds a 3D-CNN patch encoder
(small ResNet on 16³-voxel patches at 4 µm spacing from CZ z-stack / HCR
488 volume around each centroid) whose output concatenates with F6 /
`_simple_features` before the G1 cross-attention head. The added image
channel encodes the modality-specific cues (intensity profile, local
texture, segmentation boundary) that centroid-only features structurally
cannot capture — directly targeting the S63 failure mode.

Cost: multi-session, requires GPU training, and depends on F8-style
synthetic-image-warp pipeline for self-supervised pretraining (the
`coreg_table` pairs alone are probably too few to train a CNN from
scratch). Still the cheapest next step per §9.6 priority queue.

## Budget note

Main 3-subject bench (`bench_g1_loso.py`) was SIGTERM'd by its 3500 s
external timeout at iter 1400 on 767018. 788406 and 790322 results were
captured in `bench_g1_loso.log`; 767018 re-run in isolation
(`bench_g1_loso_767018.py`, same rng_seed=0 → deterministic reproduction of
the cut training trajectory), finished in 1150 s. Consolidated CSV at
`bench_g1_loso.csv`.

## Files

- `bench/candidate_impls/_g1_loso_matcher.py` — LOSO supervised trainer +
  `G1_LOSO` candidate registration.
- `bench_g1_loso.py` + `.log` — main 3-subject bench (SIGTERM at 767018
  iter 1400; 788406/790322 clean).
- `bench_g1_loso_767018.py` + `.log` + `.csv` — 767018 isolated rerun.
- `bench_g1_loso.csv` — consolidated 3-subject results.

## Status: `abandoned` (G1-LOSO refuted; G-learned centroid-only family closed)
