# S22 — G2 Contrastive node embedding (GNN-lite)

## Goal
Lighter-weight variant of G1 — same node encoder + graph conv, but
skip cross-attention and Sinkhorn.  Inference uses nearest-neighbour
in a shared projected embedding space.

## API
`bench/candidate_impls/_g2_contrastive.py::run_g2(s)`

## Method
1. Per-modality node encoder: F6 features → 2-layer MLP → 64-dim
   embedding, k-NN graph conv (k=8) with relative-position edge feats.
2. InfoNCE training on F8 synthetic warps (source ↔ warp cube with
   GT index).
3. At inference, L2-normalise both modality embeddings and FLANN-match.

## Benchmark (788406)
- n_pred = 787, recall = 0.000, runtime 31.8 s.
- All CZ cells get a match, but again at the wrong global pose.

## Files
- `bench/candidate_impls/_g2_contrastive.py`
