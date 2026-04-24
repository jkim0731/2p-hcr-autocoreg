# S45 — M4-IoU-augmented P1 per-pair ranking

**Status.** `validated_negative`. Three findings, in order of
importance:

1. **M4 per-cell Dice is structurally blocked by CZ-side data.**
   `*_seg-mask-outline.tif` is binary uint8 {0,1} on all 4 subjects —
   no per-cell CZ labels exist. F2 `load_cz_seg_mask` returns
   `n_labels=1`, so per-cell Dice cannot be computed without first
   regenerating a CZ per-cell segmentation (e.g. cellpose on the CZ
   z-stack).

2. **HCR-only re-ranker signal is marginal and 3/4-subject blocked.**
   Only 788406 has HCR `metrics.pickle` (counts/volume/density). On
   788406, a within-CZ z-scored logistic regression on
   `(z_dist, z_cos, z_log_counts, z_log_volume, z_log_density)` lifts
   AUC by +0.027 in-sample over the `(z_dist, z_cos)` baseline
   (0.794 vs 0.767). This is not enough to justify candidate-registry
   churn, particularly when it cannot apply to stress subjects.

3. **The true bottleneck is putative generation, not ranking.** Top-K
   ceiling on 4 subjects shows wildly divergent structure: 788406
   (K=100 → 95.4 %) is ranker-bounded; 755252 (p95 rank 11 998) and
   767022 (p95 rank 5 034) are partly generator-broken; **782149 has
   GT in top-500 on 0/303 CZ cells** — F6+Euclidean provides zero
   signal there. No top-K re-ranker can fix 782149.

## Setup

S44 concluded: coarse localization is not the bottleneck on 755252 /
767022 (P1's basin is correct); the bottleneck is per-pair ranking. S45
aimed to add per-cell volumetric IoU (via F1/F2/F4 seg-mask helpers) as a
third putative-ranking feature alongside F6 cosine + Euclidean distance.

Smoke-test plan: verify per-cell volumes are identity-informative before
investing in F3 cross-resolution resampling.

## Blocker 1 — CZ seg-mask is binary (uint8 {0,1}) on all 4 subjects

Direct inspection of `*_seg-mask-outline.tif`:

| Subject | shape            | dtype | n_unique | max |
|---------|------------------|-------|---------:|----:|
| 788406  | (450, 512, 512)  | uint8 |        2 |   1 |
| 755252  | (450, 512, 512)  | uint8 |        2 |   1 |
| 767022  | (450, 512, 512)  | uint8 |        2 |   1 |
| 782149  | (450, 512, 512)  | uint8 |        2 |   1 |

These are membrane/outline rasters with no per-cell label embedding. F2
`load_cz_seg_mask` cannot return per-cell volumes without a genuine
per-cell segmentation. **The Grand Plan §4.2 M4 candidate as specified
(per-cell Dice/Jaccard) requires a CZ-side per-cell segmentation that
does not exist in the current dataset.** To unlock M4, a new preprocessing
step is needed — e.g. run cellpose on `reg-dim-swapped.ome.tif` (the CZ
z-stack is single-channel float64 at 450×512×512) to produce per-cz_id
labels.

Inspection script preserved as `probe_vol_ratio.py` (runs but reports
0/787 GT pairs with volume info — `np.unique(cz_mask) = {0, 1}` leaves
`n_labels=1` and no coverage of `coreg_table.cz_id`).

## Blocker 2 — HCR volume metadata exists only on 788406

`hcr_gfp_df['volume']` column depends on `metrics.pickle` under
`cell_body_segmentation/`. On 755252 and 767022, the file is absent, so
`hcr_gfp_df` has no `volume` column (only `counts`, and `density` via
`counts/volume` is also missing). This eliminates per-cell HCR volume as a
cross-subject feature. The HCR `segmentation_mask.zarr` IS labelled (uint32
at shape 1518×9282×9269 on 788406, ~17 unique IDs in a 32×256×256 tile), so
F1 works where the data exists — but for stress subjects there is no
per-cell HCR volume at all.

## Pivot — HCR-only feature re-ranker diagnostic on 788406

Since M4 is blocked, test the subsidiary question: **among P1's K=5
putative HCR partners for a CZ cell, does HCR-side per-cell
`{counts, volume, density}` discriminate correct from wrong partners?**
If yes, an HCR-only re-ranker is still a path to lift on 788406 alone.

Probe: `probe_hcr_feat_rerank.py`.
- Rebuild P1's putative list with same score `D − 25·cos` (identical to
  `_p1_teaser.py:_seed_putative`).
- Label each top-5 HCR partner as correct or wrong per `coreg_table`.
- Compare feature distributions and fit a within-CZ-normalised logistic
  regression on (z_dist, z_cos, z_log_counts, z_log_volume, z_log_density).

### Results on 788406

| Metric | Value |
|--------|-------|
| GT in P1's top-5 | **383/784 (48.9 %)** |
| n correct putatives | 383 |
| n wrong putatives | 1 532 |

Per-feature within-putative AUC (correct > wrong?):

| Feature    | AUC   | Interpretation |
|-----------:|:-----:|----------------|
| `density`  | 0.619 | Modest positive |
| `counts`   | 0.554 | Weak positive |
| `volume`   | 0.437 | Slightly inverted |
| `dist_um`  | 0.250 | Inverted (expected — small dist = better) |
| `cos`      | 0.449 | Near-random |

Logistic regression on within-CZ z-features:
- Full model (incl. HCR features): AUC = **0.794** in-sample
- Baseline (z_dist + z_cos only): AUC = **0.767** in-sample
- **Lift from HCR features: +0.027 (in-sample, likely <0.01 cross-validated)**

Strongest coefficients: `z_log_counts +0.754`, `z_density +0.651`,
`z_log_counts` and `z_density` disagree (density = counts/volume, so they
partially cancel) — weak evidence that the model is stable.

### Interpretation

1. **The hard ceiling is putative generation, not ranking.** With GT in
   top-5 at only 48.9 %, a perfect HCR-only re-ranker caps P1's r@20 at
   ~0.244 on 788406 (= 383/787 × assignment-success-rate) — not the 0.377
   centroid-only oracle from S41, and not even close to closing the gap
   to the per-subject recall that we'd need on stress subjects.

2. **HCR features are only marginally informative.** Density is the best
   single feature (AUC 0.619) and survives the LR; counts is weak;
   volume is near-random. The +0.027 LR-AUC lift is optimistic
   (in-sample) and will likely drop to near-zero under cross-validation.

3. **On stress subjects (755252, 767022) no HCR volume is available at
   all** — so this re-ranker pivot wouldn't apply.

## topk-ceiling sweep (final)

`probe_topk_ceiling.py` ran across 4 subjects at K ∈ {1, 3, 5, 10, 20,
50, 100, 500}. GT-in-top-K, per subject:

| subject | K=1   | K=3   | K=5   | K=10  | K=20  | K=50  | K=100 | K=500 |
|---------|-------|-------|-------|-------|-------|-------|-------|-------|
| 788406  | 0.225 | 0.405 | 0.487 | 0.540 | 0.615 | 0.757 | **0.954** | 0.996 |
| 755252  | 0.066 | 0.125 | 0.160 | 0.219 | 0.280 | 0.473 | 0.673 | 0.844 |
| 767022  | 0.059 | 0.156 | 0.241 | 0.386 | 0.480 | 0.591 | 0.762 | 0.858 |
| 782149  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

GT-rank quantiles (when GT is present in HCR GFP+ list):

| subject | n    | p25  | p50  | p75   | p95    | max    |
|---------|-----:|-----:|-----:|------:|-------:|-------:|
| 788406  |  784 |    1 |    5 |    48 |     97 |    154 |
| 755252  |  596 |   12 |   48 |   109 |  11998 |  30614 |
| 767022  |  750 |    4 |   18 |    87 |   5034 |  14122 |
| 782149  |  299 | 1635 | 1878 | 2182 |   2561 |   2810 |

GT missing from HCR GFP+ list entirely: 788406=3/787, 755252=43/639,
767022=43/793, 782149=4/303.

### Interpretation of the ceiling

**788406 — ranker-bounded.** GT is in top-100 for 95 % of CZ cells and
the tail is tight (max rank 154). Widening K to 100 and installing a
stronger ranker could meaningfully lift recall. This is the only
subject where the "top-K re-ranker" playbook still applies.

**755252 and 767022 — generator partly broken.** GT-in-top-20 is
28–48 %; GT-in-top-100 is 67–76 %. The upper tails are catastrophic —
p95 rank 5034 on 767022 and 11998 on 755252, max 14 k / 30 k. That
means for ≥ 5 % of CZ cells the F6+Euclidean score places GT near the
middle of a 10–30 k HCR GFP+ list — the generator has lost those cells
entirely. Even K=500 only reaches 84–86 % because the tail extends
beyond K=500.

**782149 — generator structurally broken.** GT is not in top-500 on
*any* CZ cell (0/303). GT ranks cluster at p50=1878 out of ~2810 HCR
GFP+ — essentially random with respect to the F6+dist score. F6+dist
provides **no signal** on this subject; widening K is hopeless.
Different coarse-to-fine path required (image/mask modalities, or
synthetic-warp-trained GNN).

### Upshot for P1 redesign

- Top-K re-ranking on centroids is worthwhile on 788406 (K=100 instead
  of K=5, + a smarter ranker than `D − 25·cos`).
- Stress-subject path is **not** a ranker upgrade — it needs a
  different putative generator. Centroid-only methods are exhausted
  for 782149; the next tier must bring per-centroid image / mask
  features.

## Decision

- **Close S45 as `validated_negative`.** M4 per-cell Dice is infeasible
  with current data; HCR-only re-ranking lift is marginal; top-K
  ceiling demonstrates the generator (not the ranker) is the real
  limit on 3/4 subjects.
- **Do NOT ship an HCR-feature-augmented P1.** +0.027 in-sample AUC is
  not worth candidate-registry churn, particularly when 3/4 subjects
  lack HCR volume metadata.
- **Update Grand Plan §4.2 M4** to `blocked_by_data` and document the
  missing prerequisite — per-cell CZ segmentation (cellpose on the CZ
  z-stack, tracked under S46-c below).
- **S46 candidates (ordered by cost):**
  - **S46-b — per-centroid CZ z-stack image features** (cheapest). F2
    is unusable for per-cell labels, but `reg-dim-swapped.ome.tif`
    gives the raw CZ z-stack at 450×512×512 voxels. Extract
    per-centroid image features: local intensity mean, variance in a
    3-voxel bbox, depth-normalised intensity, local Laplacian, soft
    cluster membership from a small unsupervised k-means on texture
    patches. Feed into F6 as an additional invariant block. Tests
    whether CZ-side morphology can provide the orthogonal signal
    F6+dist lacks on 755252/767022/782149 — and whether the 782149
    generator can recover any signal at all from image data.
  - **S46-c — cellpose-on-CZ-zstack** (medium cost). Produces true
    per-cell CZ labels; unlocks F2/M4 on all subjects. ~1–2 sessions
    of preprocessing work; high downstream value. Gated on S46-b
    outcome: if per-centroid image features alone lift stress-subject
    recall materially, defer S46-c; otherwise S46-c is the path to
    M-series methods.
  - **S46-d — cross-modal (image or mask) coarse alignment for 782149**
    (higher cost). Even with perfect centroid features, 782149 may
    require I-series (SimpleITK MI) or M-series (mask NCC) coarse
    alignment before any centroid-level ranker can engage.

**S46-b is the cheapest next step and reuses existing F6 plumbing.**

## Artifacts

- `probe_vol_ratio.py` — volume-ratio smoke test (blocked by binary CZ mask).
- `probe_vol_ratio.log` — shows the blocker (0/787 GT pairs with volume info).
- `probe_hcr_feat_rerank.py` — HCR-only re-ranker diagnostic (this session's primary experiment).
- `probe_hcr_feat_rerank.log` — per-feature AUCs + LR fit.
- `probe_topk_ceiling.py` — K-ceiling sweep (running now).
- `probe_topk_ceiling.log` — GT-in-top-K per subject.
- `topk_ceiling.csv` — machine-readable.
