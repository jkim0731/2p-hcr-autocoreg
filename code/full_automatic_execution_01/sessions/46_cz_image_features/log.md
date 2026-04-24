# S46-b — per-centroid CZ/HCR z-stack image features

**Status.** `validated_positive`. HCR-side image features provide a
real, sizeable top-K lift on stress subjects when used as a global
quality bonus in P1's putative scoring. CZ-side image features are
**not** discriminative in the within-putative setting (AUC = 0.5), and
cross-modal abs-difference is **anti-correlated** (AUC 0.25–0.42), so
they cannot feed a symmetric cosine term.

Recommended follow-up (S47): ship the HCR-quality bonus as an HCR-side
scalar in F6 (or a separate module consumed by the P1 scorer), and
re-benchmark the full P1 pipeline cross-subject.

## Setup

S45 closed `validated_negative` for M4-IoU (CZ seg-mask is binary, no
per-cell labels) and HCR-only categorical features (counts/volume/
density exist on 1/4 subjects). The top-K ceiling showed the real
bottleneck is putative generation on 755252/767022 (p95 rank 5 034 /
11 998) and utterly missing on 782149 (GT in top-500 = 0/303).

S46-b tests whether per-centroid image features — `mean`, `std`, `p90`,
and `|laplacian|_mean` in a small µm bbox (±2 µm Z, ±3 µm XY) from the
CZ `reg-dim-swapped.ome.tif` z-stack and the HCR channel-488 fused
zarr (level 2) — provide orthogonal signal to F6 + distance.

## Probe 1 — within-putative AUC (`probe_image_feat_signal.py`)

For each CZ with GT in P1's top-5 (score `D − 25·cos(F6_CZ, F6_HCR)`),
label each of the K=5 HCR putatives as correct (GT) vs. wrong and
compare per-feature means within the CZ group. AUC > 0.5 means the
feature separates correct from wrong among GFP+ HCR candidates already
geometrically close to the CZ cell.

### Per-feature AUC (correct > wrong within a CZ group)

| Feature        | 788406 | 755252 | 767022 | 782149 |
|----------------|:------:|:------:|:------:|:------:|
| dist_um        | 0.250  | 0.353  | 0.449  |  n/a   |
| cos(F6)        | 0.449  | 0.600  | 0.579  |  n/a   |
| **hcr_mean**   | 0.585  | 0.748  | 0.721  |  n/a   |
| **hcr_std**    | 0.587  | 0.735  | 0.736  |  n/a   |
| **hcr_p90**    | 0.596  | 0.740  | 0.726  |  n/a   |
| **hcr_lap**    | 0.579  | 0.735  | 0.734  |  n/a   |
| cz_mean        | 0.500  | 0.500  | 0.500  |  n/a   |
| cz_std         | 0.500  | 0.500  | 0.500  |  n/a   |
| cz_p90         | 0.500  | 0.500  | 0.500  |  n/a   |
| cz_lap         | 0.500  | 0.500  | 0.500  |  n/a   |
| abs_diff_mean  | 0.402  | 0.252  | 0.274  |  n/a   |
| abs_diff_std   | 0.419  | 0.275  | 0.283  |  n/a   |
| abs_diff_p90   | 0.412  | 0.260  | 0.275  |  n/a   |
| abs_diff_lap   | 0.394  | 0.265  | 0.279  |  n/a   |

`782149`: GT is never in P1's top-5 → no correct putatives to score.

### Within-CZ z-scored LR (baseline vs. full)

| Subject | LR baseline (`z_dist + z_cos`) | LR full (`+ image feats`) | Δ AUC |
|---------|:---:|:---:|:-----:|
| 788406  | 0.767 | 0.785 | +0.018 |
| 755252  | 0.684 | 0.808 | **+0.124** |
| 767022  | 0.565 | 0.758 | **+0.193** |
| 782149  |  —    |  —    |  —     |

### Interpretation of probe 1

1. **CZ-side image features are constant within a CZ putative group**
   (the CZ cell is the same across its 5 putatives), so their AUC is
   mechanically 0.5. Useful across CZ cells, but not for within-CZ
   re-ranking.
2. **HCR-side image features discriminate correct from wrong
   putatives**: AUC 0.58 on 788406 (modest), **0.72–0.75 on stress
   subjects**. Strong HCR-side signal exactly where F6+dist is weakest.
3. **Cross-modal `abs_diff` is anti-correlated** (AUC 0.25–0.42): a
   CZ-cell's image features are *further* from its true GFP+ partner's
   than from random GFP+ cells. A cross-modal cosine term would be
   worse than nothing. The two modalities' image statistics live on
   different scales.

## Probe 2 — full-argsort top-K shift (`probe_topk_shift.py`)

Because cross-modal cosine is a non-starter and CZ-side features don't
re-rank, the only viable integration is a **global HCR quality bonus**:

    hcr_quality(j) = sum_f z_HCR(f)[j]   for f ∈ {mean, std, p90, |lap|}
    score(i, j)    = D(i, j) − 25·cos_F6(i, j) − beta · hcr_quality(j)

with `z_HCR` computed by within-HCR z-scoring each image feature.
Higher-quality HCR cells become globally preferred targets; this
re-ranks noisy/spurious HCR candidates downward for every CZ cell.

Sweep `beta ∈ {0, 5, 15, 30, 60}` on full argsort (not truncated to
K=5); measure GT-in-top-K for `K ∈ {5, 20, 50, 100, 500}` across 4
subjects.

### GT-in-top-20 as a function of beta

| beta | 788406 | 755252 | 767022 | 782149 |
|:---:|:-----:|:-----:|:-----:|:-----:|
| 0.0 | 0.615 | 0.280 | 0.480 | 0.000 |
| **5.0** | **0.638** | **0.429** | **0.507** | 0.000 |
| 15.0 | 0.412 | 0.147 | 0.140 | 0.000 |
| 30.0 | 0.085 | 0.019 | 0.023 | 0.000 |
| 60.0 | 0.019 | 0.005 | 0.011 | 0.000 |

### GT-in-top-5 as a function of beta

| beta | 788406 | 755252 | 767022 | 782149 |
|:---:|:-----:|:-----:|:-----:|:-----:|
| 0.0 | 0.487 | 0.160 | 0.241 | 0.000 |
| 5.0 | 0.465 | 0.266 | 0.211 | 0.000 |
| 15.0 | 0.136 | 0.036 | 0.025 | 0.000 |
| 30.0 | 0.024 | 0.005 | 0.009 | 0.000 |
| 60.0 | 0.005 | 0.000 | 0.004 | 0.000 |

### Per-subject delta at beta=5 vs. baseline

|   | 788406 | 755252 | 767022 | 782149 |
|---|:--:|:--:|:--:|:--:|
| ΔK=5  | −2.2 pp | **+10.6 pp** | −3.0 pp  | 0 |
| ΔK=20 | **+2.3 pp** | **+14.9 pp** | **+2.7 pp** | 0 |
| ΔK=50 | **+2.7 pp** | **+12.0 pp** | **+1.6 pp** | 0 |
| ΔK=100 | +0.7 pp | **+5.8 pp** | **+1.5 pp** | 0 |

### Interpretation of probe 2

1. **beta = 5 is a consistent sweet spot** at K=20 and K=50 on all
   three subjects where the generator is not structurally broken. On
   755252 the lift is very large (+15 pp at K=20, +12 pp at K=50, +6 pp
   at K=100) — the HCR quality bonus is specifically good at lifting
   the mid-rank / tail cases on stress subjects.
2. **Larger betas are catastrophic.** At beta=15 even 788406 loses
   20 pp at K=20; by beta=30 the argsort is dominated by quality, not
   geometry. The scaling between `D` (µm distances, O(1–100)) and the
   z-scored HCR-quality sum (O(−2, 70) — see log for min/max) means
   geometry loses to quality past ~5.
3. **K=5 lifts are less consistent.** 788406 and 767022 lose 2–3 pp
   at K=5 while K=20+ gains. The bonus moves low-confidence GT pairs
   *into* the K=20 window, but also re-orders the K=5 head enough to
   push some easy cases out. For a P1 pipeline that passes ≥ top-20 to
   downstream (TPS / RANSAC), the net effect is positive on every
   subject.
4. **782149 is untouched**, as expected: the generator places GT ranks
   at p50=1878 / p95=2561 out of ~2810 HCR GFP+ — no rank shift
   recoverable by a global bonus at any beta.

## Decision

- **Close S46-b as `validated_positive`.** HCR-quality-as-global-bonus
  at beta ≈ 5 is a consistent lift at K=20+ on 3/4 subjects (+2 to +15
  pp), zero cost on the structurally-broken 4th. This is the first
  candidate-level improvement we've found for stress subjects since
  S35.
- **Queue S47 — production integration.** Ship the HCR-quality bonus
  as an HCR-side scalar accessible to `_p1_teaser._seed_putative`.
  Candidates for where to put it:
  - `lib/image_quality.py::hcr_quality(s) -> (N_hcr,)` scalar, keyed by
    `hcr_id`. Cache results per subject to avoid repeated zarr IO.
  - Add an optional `hcr_quality_bonus_beta` parameter to the P1
    putative scorer (default 0.0 so existing harness continues to
    reproduce).
- **Leave CZ-side image features out.** Within-putative AUC 0.5 and
  constant-within-group property make them unusable for re-ranking.
  They may resurface as input to a downstream feature-based classifier
  (QF1) if that path is ever reopened.
- **S46-c (cellpose-on-CZ-zstack) is still the correct next big ticket**
  to unlock M-series per-cell Dice on stress subjects. Re-open after
  S47 lands.
- **S46-d (I/M-series coarse alignment for 782149) remains separate.**
  No centroid-level tweak can recover 782149; a different coarse-to-
  fine route is required.

## Artifacts

- `image_features.py` — per-centroid image-feature helpers
  (`cz_image_features`, `hcr_image_features`, `_extract_bbox_features`)
  used by both probes. Intended to migrate to `lib/image_quality.py`
  in S47.
- `probe_image_feat_signal.py` — within-putative AUC + LR diagnostic.
- `probe_788406.log`, `probe_755252.log`, `probe_767022.log`,
  `probe_782149.log` — per-subject within-putative logs.
- `image_feat_auc.csv` — machine-readable within-putative AUC table
  (partial — only 755252; full AUC table in-lined in per-subject logs).
- `probe_topk_shift.py` — full-argsort shift probe with the HCR-
  quality bonus formulation.
- `probe_topk_shift.log` — raw per-subject + beta log.
- `topk_shift.csv` — machine-readable (subject × beta × K) grid for the
  top-K shift probe.
