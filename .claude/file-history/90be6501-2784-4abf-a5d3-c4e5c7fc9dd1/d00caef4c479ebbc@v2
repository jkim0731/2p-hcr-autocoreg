# S38 — G1 rewrite: unified feature space

**Status.** validated_negative (real-benchmark transfer fails even with feature-space fix).

## Motivation

S37 diagnosed G1 rec=0 on all 4 real subjects with a root cause: train/inference
feature-space mismatch. Training in `_g1_gnn_matcher._train_self_supervised`
uses the 16-dim `_simple_features` (8 k-NN distances + 8 sorted elevations),
but inference with the default `use_f6=True` feeds F6 truncated to the first
16 dims — i.e. the encoder sees knn_elev + knn_azdiff at inference, never the
distance features it was trained on. S38 tests whether unifying the feature
space at train and inference rescues the matcher.

## Method

Three nested probes:

1. **probe_train_val.py** — single-subject synthetic-warp train/val, 1200 iter,
   48-hidden × 3+2 layers. Does the matcher converge on synthetic warps when
   features are unified?

2. **probe_train_long.py** — two configs at 5000 iter each on 788406:
   - A: hidden=48, 3+2 layers (baseline)
   - B: hidden=96, 4+3 layers (wider)
   Does longer training / more capacity saturate?

3. **probe_real.py** — trains the B config (96×4+3, 5000 iter) per subject on
   the subject's own HCR cloud via synthetic warps, then applies it to real
   CZ↔HCR using `_simple_features` at inference (unified with training).
   All 4 subjects.

## Results

### Synthetic-warp training (probe_train_val.py, 788406)

```
 iter  train_loss  val_hit  p<10   p<50   med
  120      2.088    0.187   0.187  0.206  306
  300      1.758    0.281   0.281  0.304  231
  600      1.465    0.344   0.345  0.367  182
 1200      1.530    0.412   0.412  0.431  118
```

Wall 67 s. Matcher learns — monotonic rise in val hit-fraction from 18.7 %
(120 iter) to 41.2 % (1200 iter).

### Long / wide (probe_train_long.py, 788406, 5000 iter)

| Config              | val_hit @ 5000 | median err |
|---------------------|----------------|------------|
| A: 48 hidden, 3+2   | 51.1 %         | ~50 µm     |
| B: 96 hidden, 4+3   | 55.9 %         | ~35 µm     |

Both saturate 55-56 % on synthetic warps; wider gets a small edge. The
training signal is real and the matcher is capacity-limited around this size.

### Real CZ↔HCR transfer (probe_real.py, B config, 5000 iter per subject)

| Subject | n_hcr | train s | loss  | n_pred | n_gt | exact_id | n_lt10 | n_lt50 | median µm |
|---------|-------|---------|-------|--------|------|----------|--------|--------|-----------|
| 788406  | 17427 |   365   | 2.88  |  932   | 787  |   5      |   5    |   9    |    901    |
| 755252  | 30804 |  1039   | 0.84  |  835   | 639  |   0      |   0    |   0    |    743    |
| 767022  | 14239 |   385   | 1.27  |  926   | 793  |   1      |   1    |   1    |    781    |
| 782149  |  3831 |   137   | 1.00  |  894   | 303  |   3      |   3    |   3    |    992    |

788406: 5/787 (0.6 %) exact, 9/787 lt50, median 901 µm. 755252: 0/639 exact,
0 lt50, median 743 µm. 767022: 1/793 exact, 1 lt50, median 781 µm. 782149:
3/303 exact, 3 lt50, median 992 µm. Every subject — primary, secondary, and
stress — lands at essentially zero real-match rate despite 56 % synthetic val
accuracy during training.

## Diagnosis

The matcher learns to solve synthetic HCR→warped-HCR matching but the
representation does not transfer to real CZ↔HCR. The domain gap sources:

- **Asymmetry.** F8 synthetic warps draw both source and target from the same
  HCR cloud (same density, same depth profile, same GFP+ segmentation
  statistics). Real CZ has ~785-1016 cells inside a ~500 µm × 500 µm × 450 µm
  slab; HCR has 3.8k-31k GFP+ cells over a 2-3× larger volume. The matcher
  never saw this cardinality asymmetry or partial-overlap regime at train.
- **Scale discrepancy.** Synthetic warps use anisotropic scales in
  [1.5, 2.0] × [2.0, 3.5] drawn uniformly. Benchmark scales cluster tighter
  (1.64-1.92 × 2.13-3.58) and differ per-subject; the trained model has no
  per-subject calibration.
- **CZ-only feature distribution.** CZ segmentations are sparser and
  longer-tailed in local-density distribution than any sub-cube of HCR. The
  16-dim `_simple_features` (k-NN distances + elevations) captures *shape*
  but not this statistical skew.

## Decision

G1 as a self-supervised synthetic-warp trainer is a validated dead end on the
current F8 pipeline. Options to unblock:

1. **Modality-asymmetric F8.** Sample source from CZ-like sub-region (smaller
   cube, tighter density), target from HCR-like super-region (larger cube,
   higher density, more distractors). Preserve subject-specific scale
   distribution. Keep `_simple_features` but widen the training distribution
   to include the actual asymmetry.
2. **Weak supervision on `coreg_table.csv`.** Abandon fully self-supervised.
   Train on real pairs from 4-5 subjects LOSO; evaluate on held-out subject.
   Breaks the CLAUDE.md binding rule only if used for design — using it for
   *training data* of a learned model is allowed under the Grand Plan's
   Stage-2/3 regime.
3. **Pivot to P1 TEASER with F6-weighted putatives.** Certifiable robustness
   matches the 99 % outlier regime; doesn't require any training at all.
   Route recommended by S37 log if G1 rewrite also failed.

Given S38's confirmed negative, the next session should pick option 3 (P1
TEASER) as the shortest path to a real-working candidate, and defer G1
rewrites to after an asymmetric F8 exists (option 1) — both options preserve
the GNN favor without spending further cycles on a broken synthetic target.

## Artifacts

- `probe_train_val.py` + `probe_train.log` — initial 1200-iter probe
- `probe_train_long.py` + `probe_train_long.log` — 5000-iter A/B configs
- `probe_real.py` + `probe_real.log` — real-benchmark transfer test
- `real.csv` — machine-readable per-subject metrics
