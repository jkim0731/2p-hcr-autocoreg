# S21 — G1 Hand-crafted-feature GNN matcher (favored)

## Goal
End-to-end CZ ↔ HCR cell matcher with cross-graph attention + Sinkhorn +
dustbin, trained self-supervised on F8 synthetic warps.

## API
`bench/candidate_impls/_g1_gnn_matcher.py::run_g1(s, n_train_iter=120,
k=8, use_f6=True)`

## Architecture
- Node encoder: `MLP(D_F6 → 128)` → `TransformerConv(128 → 128) × 4`
  with relative-position edge features (`dx, dy, dz, |d|, angle_xy,
  angle_xz`).
- Cross-graph attention: 4 alternating self/cross blocks.
- Dustbin-augmented Sinkhorn over `[S | dustbin_col; dustbin_row | 0]`.
- Training loss: Sinkhorn-with-dustbin NLL on F8 correspondences +
  matchability BCE + InfoNCE on the embedding.

## Issue & fix
Initial run crashed with `NameError: cz_px_to_um is not defined` (the
module imported `centroids_um` but still called the old helper in one
remaining line).  Patched to use `lib/centroid_helpers.centroids_um`.

## Benchmark (788406, after fix)
- Pending re-run — initial attempt failed; the standalone debug run
  progresses past the centroid loader into the training loop.
- Expected behaviour: same order of recall as G2 since the warm-start
  is still 180°+centroid without scale recovery.  The architectural
  upgrade pays off only when a scale-carrying warm-start is available
  (see sessions/20_C1_image_centroid/).

## Files
- `bench/candidate_impls/_g1_gnn_matcher.py`
