# Session 37 — G1 GNN matcher on stress subjects (negative, diagnosed)

## Why

S36 (I2+NN) established I2 coarse alignment is too loose for direct
centroid matching. G1 (hand-feature GNN) is the Grand Plan's favored
learned method and the theoretical fix for 782149's wrong-basin
failure: k-NN angle signatures + inter-ROI distance ranks are
rotation-invariant, scale-invariant (via local 1-NN median), and
density-insensitive, so the 3831 sparse HCR-GFP+ cells shouldn't
confuse the matcher.

**G1** (`bench/candidate_impls/_g1_gnn_matcher.py`) and **F6**
(`lib/cell_features.py`) were implemented before S29. They were never
exercised on stress subjects with the S28-fixed warm-start / harness.

## What was built

`probe_g1.py` runs G1 via the F9 harness on `{788406, 755252, 767022,
782149}`. G1's default: self-supervised training on 120 synthetic-warp
samples of the subject's HCR GFP+ cloud (via F8 `sample_warped_pair`),
then cross-graph attention + Sinkhorn + dustbin at inference.

## Results

| Subject | n_pred | n_gt | recall | rec@5 | rec@10 | rec@20 | median | wall |
|---------|------:|-----:|-------:|------:|-------:|-------:|-------:|------|
| 788406  | 297   | 787  | 0.000  | 0.000 | 0.000  | 0.000  | 1000 µm | 70 s |
| 755252  | 472   | 639  | 0.000  | 0.000 | 0.000  | 0.000  |  734 µm | 79 s |
| 767022  | 288   | 793  | 0.000  | 0.000 | 0.000  | 0.000  |  838 µm | 49 s |
| 782149  |  78   | 303  | 0.000  | 0.000 | 0.000  | 0.000  | 1019 µm | 41 s |

**All four: recall = 0 at every threshold**; median error 734–1019 µm
(≈ half the tissue extent — essentially uniform-random assignment).

## Root cause (diagnosed)

G1 has a **feature-space mismatch between training and inference** that
makes the learned encoder useless at inference:

1. **Training** (line 210, `_train_self_supervised`) calls
   `_simple_features` per sample: 16-dim vector of k-NN distances (8)
   + sorted k-NN elevations (8) on points in (z,y,x), with the "up"
   axis fixed to `[1, 0, 0]` (a z-axis unit vector).
2. **Inference** (line 214, `use_f6=True` default) calls
   `extract_cell_features` → F6 produces ~60-dim vectors: k-NN
   elevations + k-NN azimuth-diffs + depth rank + density ratio +
   dist-knn-z + volume ratio + GFP intensity + layer histogram.
3. **Dim reconciliation** (line 224): `fc_d = fc[:, :target_dim]` —
   takes the *first 16* F6 columns to match the encoder's input width.
   These are `knn_elev_0..5` and `knn_azdiff_0..1`, i.e. *angles* on
   both axes, while the encoder was trained to consume *distances in
   the first 8 dims and elevations in the next 8*. The encoder is
   receiving garbage.

Additional factors that amplify the mismatch:

- `_simple_features` uses `up = [1, 0, 0]` (= `+z` in (z,y,x) notation).
  F6's k-NN angles are computed with `_pia_normal` derived from the
  pia-plane tilt, which is never zero — so even if we aligned the
  feature indices, the "local up" reference frames differ.
- Training samples 400 µm cubes via `sample_warped_pair`; at inference
  the CZ side spans ~1.6 mm × 1.2 mm × 0.4 mm and HCR spans > 2 mm in
  XY. The graph-scale distribution the model saw at training is a
  small slice of what it sees at inference.
- 120 iterations of a 48-hidden GNN is well under what a transformer
  matcher with ~50 k params typically needs.

The `else` branch (`use_f6=False`) calls `_simple_features` at
inference too, which would fix the feature-space mismatch — but
`use_f6=True` is the default, and the branch was not tested on
benchmark subjects.

## Conclusion

G1 in its current form is **structurally broken**, not just
under-trained. The negative result is not informative about whether
feature-based matching can solve 782149 — it says only that *this
particular implementation* cannot solve any subject.

**Do NOT discard G1.** The architecture (cross-graph attention +
Sinkhorn+dustbin, self-supervised on F8 warps) is sound. The fix is
~0.5 session of work:

1. Unify train/inference features. Either (a) `_simple_features` at
   both (simpler), or (b) F6 at both (richer — requires an
   `extract_cell_features`-style extractor that works on a warped
   cloud with no surface fit, e.g. using the sample's principal-axis
   `up` estimate instead of `pia_normal`).
2. Train for ≥ 1000 iterations with ≥ 200-cell cubes (closer to the
   per-subject graph density at inference).
3. Before touching benchmark, validate on held-out synthetic warps:
   pair recall at 10 µm on the warped target ≥ 0.8 is a prerequisite
   for running on real data.
4. Normalize features with train-set statistics, not subject-specific
   mean/std (current inference computes μ/σ from F6 on the actual HCR,
   which is a different distribution than training).

## Next steps (session 38)

Refactor `_g1_gnn_matcher.py` per the 4-point fix above; re-run on all
4 subjects; report. If after the fix G1 still scores 0 on 782149, the
bottleneck is *not* the feature-space but the matching objective, and
we can justify moving to P1 TEASER with F6 feature-weighted putative
correspondences (which reuses F6 without the learned-matcher machinery).

## Files

- `probe_g1.py` — benchmark-harness invocation.
- `probe.log` — stdout.
- `g1_stress.json` — full row dump from the harness.

## Status

validated_negative — G1 recall = 0 on all 4 subjects due to a
train/inference feature-space mismatch. Root cause identified; fix
queued as S38.
