# Session 02 — F6 per-cell feature extractor

**Subgoal ID:** F6
**Plan reference:** `07 Grand Plan.md §3 F6`
**Status:** completed

## Goal

Compute an interpretable, rotation-invariant-after-180°, **(N, D)** per-cell
feature matrix for each modality.  Feeds G-series GNN, P-series putative
generators, and any QF classifier.

## Method

`lib/cell_features.py::extract_cell_features(s, modality, ...)` returns
`(F, names, ids)`.

Feature groups (D = 41):

| Group | Cols | Notes |
|-------|-----|-------|
| k-NN elevations `knn_elev_0..5` | 6 | asin(dot with up) of 6 NNs, sorted |
| k-NN azimuth-diffs `knn_azdiff_0..5` | 6 | sorted cyclic-gap signature → rotation-invariant around up |
| `depth_um` | 1 | raw depth-from-pia (scale-sensitive; masked out by `invariant_feature_mask`) |
| `depth_rank_ring`, `depth_rank_fullpop` | 2 | rank within 50 µm XY neighbourhood |
| `count_gfp_30um`, `count_all_30um`, `density_ratio_30um` | 3 | local density within 30 µm |
| `dist_knn_z_{1..10}` | 10 | k-NN distances / global 1-NN median |
| `vol_ratio` | 1 | cell volume / local median volume (HCR only) |
| `gfp_density`, `gfp_count`, `gfp_log1p_count`, `gfp_mean_minus_bg` | 4 | HCR GFP+ intensity features; NaN on CZ |
| `layer_hist_{0..7}` | 8 | 8-bin normalised histogram of neighbourhood depths |

`invariant_feature_mask(names)` → 37/41 columns are safe to match across
modalities without knowing anisotropic scale (excludes `depth_um`,
`count_gfp_30um`, `count_all_30um`, `vol_ratio` as raw counts).

The pia surface is fetched via `benchmark_analysis.analyze_subject` and
cached per-subject (`_SURFACE_CACHE`) — first call ~40 s (HCR zarr
loads), subsequent calls ~6 s for HCR_GFP (N ≈ 17 k).

## Sanity check on 788406

After HCR-z-scoring shared-invariant features (32/37 columns finite on
both sides):

| Quantity | Value |
|----------|-------|
| GT-pair cosine (median) | 0.087 |
| Random cosine (median) | −0.005 |
| Pairs tested | 784 |

The signal is real but weak.  The plan's threshold of **median ≥ 0.6**
on raw cosine appears to have been optimistic — the plan explicitly
notes the features are *inputs* for learned matchers rather than
stand-alone matching cues, so this is a **lower bound** on their
utility.  Downstream candidates (G1 Sinkhorn matcher, P1 TEASER
putative scorer, QF1 GBT) will use these features in combination with
geometric consistency, where the discriminative power is much larger
than raw cosine can measure.

## Deviations

- Accept both `counts` (spot subjects) and `count` (intensity subjects)
  schemas in `hcr_gfp_df`.
- Pia normal computed from surface-fit coefficients `(a, b, c)` assuming
  planar fit; more complex (quadratic, hybrid, grid) surface types fall
  back to planar coefficients since that's a reasonable local
  approximation for feature extraction.

## Next

S03 — F1 HCR mask loader, then F2–F4, then F5 (SimpleITK) once foundations
land.
