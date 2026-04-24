# S64 — C2 image-conditioned GNN (multi-session infrastructure)

## Why now

Per `grand_plan_working.md` §9.6 priority queue (2026-04-21):

1. ~~B-retrial~~ — S62 `abandoned` (sum Δ r@20 = −0.502 for B3; B3b flat).
2. ~~G1-LOSO~~ — S63 `abandoned` (sum Δ r@20 = −0.686 even with real
   supervision; refutes S58/S59 "synthetic→real gap" as root cause).
3. **C2 image-conditioned GNN** — authorized: G1-LOSO's failure with
   real supervision plus the catastrophic held-out median positional
   error (133–220 µm vs 20 µm threshold) shows architecture isn't the
   bottleneck — **centroid-only features are the bottleneck**. C2 adds
   image patches around each centroid so the network has access to the
   modality-specific cues (intensity profile, local texture, segmentation
   boundary, soma-morphology) that 20-dim `_simple_features` structurally
   cannot encode.

## C2 design (from grand plan §4.4)

Dual encoder: small 3D-CNN on 16³-voxel patches at 4 µm spacing (local
field-of-view 64 µm cube) around each centroid from CZ z-stack and HCR
488 zarr. CNN output concatenates with F6 / `_simple_features` before the
existing G1 cross-attention head.

Two-stage training:
- **Stage 1**: self-supervised on F8-style synthetic image warps (new —
  F8 currently only handles centroid warps, not image volumes).
- **Stage 2**: fine-tune on real `coreg_table` pairs with LOSO
  (piggybacks on S63's infrastructure).

## Subgoal decomposition (multi-session)

| # | Subgoal | Deliverable | Session scope |
|---|---------|-------------|---------------|
| 1 | Image patch extractor | `lib/image_patches.py::extract_cz_patches(s, centroids_um, size=16, spacing_um=4)` + HCR twin | This session (one step) |
| 2 | F8-image synthetic warp | Extend F8 from point-only to voxel-level (CZ z-stack crop → anisotropic scale + TPS → warped CZ cube). | Next session |
| 3 | 3D-CNN patch encoder | `lib/patch_encoder.py::PatchEncoder` — ResNet-10-style small CNN (3 down-blocks, 128-dim output). | Next session |
| 4 | C2 matcher | `bench/candidate_impls/_c2_image_gnn.py` — encode patches, concat with `_simple_features`, feed to existing GNNMatcher. | After 2+3 land |
| 5 | Training loop | Stage 1 self-supervised on F8-image warps; Stage 2 LOSO on real pairs (piggyback S63). | Separate session |
| 6 | 3-subject bench | F9 on 788406/790322/767018; compare to C5 = 0.263/0.289/0.315 and G1-LOSO = 0.013/0.113/0.055. | Separate session |

## This session — subgoal 1: image patch extractor

Small, testable, CPU-only. No GPU or training dependencies yet.

Goal: given a `SubjectData` and an `(N, 3)` array of centroids in physical
µm, return `(N, 1, D, H, W)` float32 patches from each modality at a
chosen voxel spacing. Out-of-volume centroids pad with zeros. Handles
CZ's native anisotropy (CZ z=2 µm, xy=0.494 µm) and HCR's (level-2:
xy=0.988 µm, z=2 µm).

Implementation plan:
- CZ: `benchmark_analysis.py::load_cz_y_slab()` already has whole-volume
  load; reuse but keep the whole `(Z, Y, X)` array in memory once
  (~450 MB at uint16).
- HCR: `benchmark_analysis.py::load_hcr_volume(s, '488', level=2)`.
- Per-centroid: `scipy.ndimage.map_coordinates` on the voxel grid at the
  target isotropic µm sampling lattice. Linear interpolation.
- Normalize: per-patch z-score (subtract mean, divide by std + 1e-3);
  clip at ±6 σ.

Success criterion: on 788406, extracting patches around all CZ centroids
produces no NaN/Inf, finite std > 0 for ≥99 % of patches, and runtime
≤ 30 s for 1000 CZ cells.

## Subgoal 1 result — `validated`

Smoke test on 788406 (`smoke_test_patches.py`):

| Modality | N | Shape | Runtime | % finite-std | Value range |
|----------|--:|-------|--------:|-------------:|-------------|
| CZ       | 932 | `(N, 1, 16, 16, 16)` | 1.58 s | 100 % | [−6.00, 6.00] (clipped) |
| HCR 488 L2 | 1000 | `(N, 1, 16, 16, 16)` | 89.85 s* | 100 % | [−2.14, 6.00] |

*HCR runtime dominated by one-time zarr level-2 volume load (~3 GB
uint16). Cached per-subject on `s._hcr_volume_cache_488_L2`; subsequent
calls reuse the cached volume. Actual sampling with
`map_coordinates` is sub-second for 1000 patches.

All success criteria satisfied:
- ✓ No NaN/Inf in either modality.
- ✓ ≥ 99 % of patches have finite std > 0.
- ✓ CZ runtime well under 30 s for 1000 cells.

Files shipped:
- `lib/image_patches.py` — `extract_cz_patches`, `extract_hcr_patches`,
  `PatchExtractConfig`. Per-patch z-score + ±6 σ clip.
- `sessions/64_c2_image_conditioned_gnn/smoke_test_patches.py`.

## Subgoal 2 result — `validated`

F8-image extension shipped as `lib/synthetic_warps.py::sample_voxel_warp_sample`.
Smoke test on 788406 HCR 488 L2 volume (`smoke_test_voxel_warp.py`):

| Sample | Ns | Nw | corr | scales (z, xy) | TPS residual RMS | corr-pair |src−warped| (z-norm) |
|-------:|---:|---:|-----:|----------------|-----------------:|--------------------------------:|
| 0 | 201 | 1694 | 168 | (2.84, 1.65, 1.65) | 52.4 µm | min 0.37 / med 0.70 / max 0.93 |
| 1 | 259 | 1696 | 223 | (2.89, 1.55, 1.55) | 60.1 µm | min 0.36 / med 0.67 / max 0.93 |
| 2 | 157 | 1693 | 129 | (3.34, 1.52, 1.52) | 18.0 µm | min 0.41 / med 0.64 / max 0.89 |

Checks passed:
- ✓ Ns = 150–260 (CZ-like count after 30 % drop from ~900-cap).
- ✓ Nw ≈ 1700 (target cap 2000 minus 15 % drop).
- ✓ Correspondence count 129–223 → real GT intersection.
- ✓ Sampled scales inside benchmark bounds (XY ∈ [1.5, 2.0], Z ∈ [2.0, 3.5]).
- ✓ Yaw clustered at 180° ± 10° (structural prior).
- ✓ TPS residual RMS 18–60 µm — consistent with 25 µm control-point jitter.
- ✓ **Non-degenerate GT pairs:** mean |src − warped| in z-score units
  = 0.36–0.93 → patches differ by ~0.7 σ around matched cells. The
  CNN encoder will have meaningful warp-invariance signal to learn;
  matching is not identity-trivial.

Runtime: ~30 s per sample (≈ 17 ms per patch sampled × 1700 + TPS RBF).
One-time HCR volume load: 55 s. For stage-1 pretraining with 20k
samples on CPU: ~7 hours data generation; will move sampling to GPU or
cache a pool of pre-generated samples if budget permits.

Files shipped:
- `lib/image_patches.py::sample_patches_oriented` — patch extractor with
  rotatable lattice (enables `(R·S)^{-1}` sampling for warped patches).
- `lib/synthetic_warps.py::sample_voxel_warp_sample` + `VoxelWarpSample`
  dataclass.
- `sessions/64_c2_image_conditioned_gnn/smoke_test_voxel_warp.py`.

## Subgoal 3 result — `validated`

`lib/patch_encoder.py::PatchEncoder` shipped — small 3D-ResNet with a
stem + 3 residual blocks + global average pool + linear projection.
Each residual block = 2× (Conv3d k=3, stride-s, InstanceNorm, ReLU)
with a 1×1 stride-s shortcut where channels or stride change.

Smoke test on CPU (`smoke_test_patch_encoder.py`):

| Metric | Value | Target |
|--------|------:|-------:|
| Param count | 229 296 | ≤ 250 000 |
| Forward (32, 1, 16³) → (32, 64) | 8.9 ms | < 1 000 ms |
| Loss backward | 146 ms | — |
| Grads finite | ✓ | ✓ |
| max \|grad\| | 0.035 | > 0 |
| min cos-sim across random inputs | 0.948 | < 0.99 |

All success criteria met. Cosine similarity across random inputs is
0.95±0.01 at initialization — expected for an untrained GAP+linear head
before training differentiates the embedding space; the test bar was
the ≠-identity check (< 0.99).

Files shipped:
- `lib/patch_encoder.py` — `PatchEncoder`, `ResBlock3D`.
- `sessions/64_c2_image_conditioned_gnn/smoke_test_patch_encoder.py`.

## Subgoal 4 result — `validated`

`bench/candidate_impls/_c2_image_gnn.py::C2Matcher` shipped — wraps a
`PatchEncoder` with a widened `GNNMatcher` (in_dim = hand_dim + patch_dim
= 20 + 64 = 84). Forward takes `(f, patches, edges)` on both sides; the
CNN embeds each cell's 16³ patch, concatenates with the hand feature,
and feeds the result through the existing G1 GNN+cross-attention+
Sinkhorn head.

Smoke test on CPU (`smoke_test_c2_matcher.py`) with synthetic inputs
(CZ N=100, HCR N=300 random centroids, random 16³ patches):

| Metric | Value |
|--------|------:|
| Total params | 443 378 (CNN 229 296 + GNN 214 082) |
| Forward (100 × 300) | 178 ms |
| Backward | 507 ms |
| Sinkhorn col-sums | 1.000 (rect. matrix — rows→2.98, OK) |
| Pair loss (random corr 10 pairs) | 5.14 |
| CNN grads finite / max \|g\| | ✓ / 0.010 |
| GNN grads finite / max \|g\| | ✓ / 0.182 |

All success criteria met. Gradients flow end-to-end through both the
CNN patch encoder and the GNN matcher under `_pair_loss` InfoNCE.

Files shipped:
- `bench/candidate_impls/_c2_image_gnn.py` — `C2Matcher`.
- `sessions/64_c2_image_conditioned_gnn/smoke_test_c2_matcher.py`.

Timing projection for training: at N=(~1000 CZ, ~5000 HCR crop) the
forward-backward should scale ≈ 15× (patch encoder is O(N)), giving
~10 s per iteration. 2000 iterations ≈ 5–6 h on CPU. Patch caching
per subject (one pass per modality per subject, reused across the
LOSO loop) will be required in subgoal 5 to hit a reasonable budget.

## Status: `subgoal_4_validated; subgoal_5_pending (training loop, separate session per plan)`

## Session boundary — end of infrastructure subgoals

Subgoals 1–4 (image patch extractor, F8-image voxel warps, 3D-CNN patch
encoder, C2 matcher architecture) are all shipped and smoke-tested.
Subgoal 5 (training loop) is explicitly a "Separate session" in the S64
decomposition table — it requires multi-subject patch preprocessing
(~5 min/subject; ~500 MB RAM/subject for HCR patches cached across
training iterations) plus a non-trivial two-stage training run.
Handing off to a fresh session with full context budget for that work.

### Handoff summary — what a future session needs

Ready-to-import:
- `lib.image_patches.extract_cz_patches(s, centroids_um)` — CZ patches.
- `lib.image_patches.extract_hcr_patches(s, centroids_um, channel="488", level=2)`.
- `lib.image_patches.sample_patches_oriented(vol, centroids_um, voxel_spacing_um, orient=...)`.
- `lib.synthetic_warps.sample_voxel_warp_sample(vol, voxel_spacing_um, centroids_um, ...)` — Stage-1 pretraining data.
- `bench.candidate_impls._c2_image_gnn.C2Matcher(hand_dim, patch_dim, hidden, n_layers, cross_layers)`.

Reuse from S63:
- `bench.candidate_impls._g1_loso_matcher._preload_subject(sid)` — load CZ+HCR centroids+coreg pairs.
- `bench.candidate_impls._g1_loso_matcher._crop_hcr(cz_init, hcr_um, pad_um)` — bbox crop.
- `bench.candidate_impls._g1_gnn_matcher._simple_features(pts_um, k=8)` — 20-dim hand feature.
- `bench.candidate_impls._g1_gnn_matcher._build_knn_graph(pts_um, k=8)` — edge index.
- `bench.candidate_impls._g1_gnn_matcher._pair_loss(sim, dustbin, corr, Na, Nb)` — symmetric InfoNCE.
- `bench.candidate_impls._g1_gnn_matcher._assignment(sim, dustbin, n_iter=30)` — Sinkhorn+dustbin.

Outstanding implementation work for subgoal 5:
1. **Per-subject patch cache.** For every training subject, extract (a)
   CZ patches for all ~1000 CZ cells, (b) HCR patches for the cropped
   cells around `cz_init` (typically ≤ 5000 GFP+ cells within 200 µm
   of the warm-started CZ bbox). Persist on `SubjectData` like the
   centroid cache in S63.
2. **Stage 1 pretraining (optional per subject).** Sample voxel warps
   via `sample_voxel_warp_sample` using the subject's cached HCR 488 L2
   volume. Train C2 on InfoNCE over warp-GT correspondences.
   Generation is ~30 s/sample; pre-generate a pool (e.g. 200 samples)
   to amortise cost across ~2000 training iterations.
3. **Stage 2 LOSO finetune.** Mirror `run_g1_loso` structure — sample
   a training subject per iteration, compute features + edges from the
   cached CZ+HCR centroids/patches, run forward, InfoNCE on the real
   `coreg_table` pairs (already in local indices).
4. **Inference.** Extract patches for the held-out subject's CZ +
   HCR-crop, forward C2, Sinkhorn → one-to-one matches, confidence =
   matched / (matched + dust). Emit a `CoregResult` identical in
   structure to `run_g1_loso`.
5. **Subgoal-6 bench.** F9 run on 788406/790322/767018; compare r@20
   to C5 (0.263/0.289/0.315) and G1-LOSO (0.013/0.113/0.055). Decision
   rule (per §9.6): sum Δ r@20 > 0 across 3 subjects → promote; ≤ 0 →
   document the negative and close G-series on centroid+patch data.

### Timing budget estimate for subgoal 5 (based on subgoal 1-4 benches)

- Preprocessing per subject: ~60 s HCR L2 load + ~1 s CZ patches + ~5 s
  HCR crop patches + ~100 MB cache per subject (CZ patches) + ~500 MB
  (HCR crop patches). 6 subjects × ~70 s = ~7 min setup.
- Stage 1 pretrain (optional): 200-sample cache × 30 s each = 100 min
  up-front. 2000 iter × ~10 s/iter = ~5–6 h. Defer/skip if Stage 2
  alone shows descent.
- Stage 2 LOSO: no per-iter re-extraction (patches cached), ~10 s/iter
  × 2000 iter = ~5–6 h per held-out-subject.
- Three held-out subjects × ~6 h = ~18 h. Consider GPU once available
  or reduce held-out pool.

## Subgoal 5 result — `infrastructure_validated`

`bench/candidate_impls/_c2_image_gnn.py::run_c2_loso` shipped — mirrors
`run_g1_loso` but with Stage-2-only supervised training (Stage 1
self-supervised pretrain deferred; the dry run below shows descent is
already meaningful from real pairs alone, and pretrain doubles the
budget).

Dry run (`sessions/64_c2_image_conditioned_gnn/dry_run_c2_loso.py`):
held=788406, train=["790322"], n_iter=100.

| Phase | Result |
|-------|--------|
| Preload 790322 | 84.6 s — CZ=1016, HCR crop=4447/10131, 769 GT pairs |
| Preload 788406 | 93.6 s — CZ=932, HCR crop=8818/17427, 784 GT pairs |
| Model build | hand_dim=20, patch_dim=64, **443 378 params** |
| Iter 50/100 | loss 6.283 (t=310 s) |
| Iter 100/100 | loss 3.646 (t=632 s) |
| Inference | 862 pairs emitted, confidence median 0.996 |
| r@20 held-out | **2 / 730 = 0.003**, median err 442.3 µm |
| Total wall | 822 s (13.7 min) |

Loss descent validates the full pipeline — `log(4447) ≈ 8.40` is the
uniform-baseline InfoNCE; final `mean50 = 3.65` is 4.75 nats below, so
the CNN+GNN is learning. Held-out r@20 is near-chance because 100 iter
on 1 training subject is severely undertrained (vs. the G1-LOSO
baseline of 2000 iter × 5 subjects).

Architecture + training + inference + evaluation plumbing all work end
to end. Subgoal-6 full bench ready to launch.

Files shipped:
- `bench/candidate_impls/_c2_image_gnn.py` — `run_c2_loso`,
  `_preload_c2_subject`, `_train_c2_stage2`, `_infer_c2`, registered as
  F9 candidate `"C2_LOSO"`.
- `sessions/64_c2_image_conditioned_gnn/dry_run_c2_loso.py`.

## Subgoal 6 result — `abandoned_strict_negative`

Full 3-held-out bench (`bench_c2_loso.py`, 8.5 min preload + 3 × ~8 h
training at 2000 iter each, 24 h total wall):

| Held-out | C2 r@20 | C5 r@20 | G1-LOSO r@20 | Δ vs C5 | Δ vs G1-LOSO | train loss mean50 | med err (µm) |
|----------|--------:|--------:|-------------:|--------:|-------------:|-------------------:|-------------:|
| 788406   | 0.005   | 0.262   | 0.013        | −0.257  | −0.008       | 0.133              | 315.3 |
| 790322   | 0.006   | 0.289   | 0.113        | −0.283  | −0.107       | 0.216              | 383.5 |
| 767018   | 0.000   | 0.315   | 0.055        | −0.315  | −0.055       | 0.114              | 362.3 |
| **Sum**  | **0.012** | **0.866** | **0.181** | **−0.854** | **−0.169** | — | — |

**Decision (§9.6): CLOSE G-series on centroid+patch.**

### What happened

Training descended cleanly on all 3 runs (mean50 loss 0.11–0.22, far
below uniform `log(Nb) ≈ 8`), so the C2 network *learned* the training
pairs. But the learned representation does not transfer to held-out
subjects — **C2 is strictly worse than centroid-only G1-LOSO**. Sum
r@20 went backwards by 0.169 when CNN patches were added.

Median held-out position error 315–384 µm — essentially random
placements within the cropped HCR volume. Compare to G1-LOSO's
133–220 µm (still poor but ~2× better than C2).

### Root cause — patch encoder overfits to subject-specific texture

The patch encoder memorises per-subject HCR-488 and CZ intensity
statistics (illumination bias, staining variance, acquisition noise)
that differ between subjects. With 5 training subjects and no
explicit domain-invariance loss, the CNN treats those subject-ID
signatures as the strongest cue — training loss plummets while
held-out matching uses features the new subject does not have.

This refutes the working hypothesis from S63's closure:

> "centroid-only feature ceiling" is the bottleneck; C2 adds
> modality-specific cues that differentiate CZ from HCR.

The CNN *does* add modality-specific cues, but they are the *wrong*
modality-specific cues — per-subject rather than per-modality. The
existing F8 synthetic warp generator samples source+target from the
same subject, so Stage-1 pretraining would have the same flaw (every
warp-pair shares subject-identity texture). Bridging the gap requires
one of:

1. **Cross-subject patch augmentation** — e.g., random per-patch
   histogram matching to a canonical reference, or learned instance
   normalisation that strips subject-bias at inference.
2. **Explicit domain-invariance loss** — adversarial head predicting
   subject-ID on patch embeddings, trained against.
3. **Modality-transfer self-supervision** — Stage-1 trains on
   HCR-patch → (synthesised CZ-patch) pairs learned from within the
   6-subject corpus; removes subject-ID as a shortcut.

All three are multi-session tier-3 work whose ROI is uncertain given
this result (and given 782149 remains r@20=0 on every method).

### Explicit close-out

Per 2026-04-21 §9.6 priority queue:
1. B-retrial (S62) — closed negative.
2. G1-LOSO (S63) — closed negative.
3. C2 image-conditioned GNN (S64) — **closed negative**.

All three post-C5-plateau retrials have landed. **No further
autonomous work is authorised on the G-series / B-series /
image-conditioned-GNN family on the current feature set.** The
Grand Plan centroid+patch track is exhausted.

Next possible directions (all require explicit user authorisation,
each with uncertain ROI):

- Cross-modal self-supervision that removes subject-ID shortcut (new F8
  variant; tier 3).
- Mask-based M-series unblocked by a per-cell CZ segmentation
  (cellpose-on-z-stack; tier 3, heavy preprocessing).
- QF1 fallback classifier on C5's existing pairs (conditional; ROI
  capped since S55 calibration gave 0 r@20 lift).
- Image-level MI/B-spline pipeline with C5-conditioned pair emitter
  (tier 3, uncertain on 782149).

Files shipped:
- `sessions/64_c2_image_conditioned_gnn/bench_c2_loso.py`.
- `sessions/64_c2_image_conditioned_gnn/bench_c2_loso.csv`.
- `sessions/64_c2_image_conditioned_gnn/logs/bench_c2_loso.out`.

## Final status: `abandoned_strict_negative`


