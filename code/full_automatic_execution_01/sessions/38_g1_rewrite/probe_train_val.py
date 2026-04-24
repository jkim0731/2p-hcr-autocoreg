"""S38 probe: is G1's matcher actually trainable on synthetic warps?

S37 established G1 rec=0 on all real subjects. Root cause: feature-space
mismatch between train and inference. This probe validates training in
isolation — does the matcher converge on synthetic-warp pairs from an HCR
cloud, using the SAME features at train and inference?

Plan:
  1. Load 788406's HCR GFP+ cloud.
  2. Build a train set of N=200 synthetic warps + val set of 40.
  3. Train for iters ∈ {120, 300, 600, 1200}.
  4. At each checkpoint, measure on val:
     - Pair-recall @ 10 µm position error (under oracle correspondence)
     - Pair-recall by assignment: does argmax[P] pick the GT pair?
  5. Report learning curve + final val metrics.

If the matcher converges to ≥ 0.8 argmax-pair-recall on val synthetics,
the matcher is trainable and S37's rec=0 on benchmark is purely the
feature-space mismatch + too-few iterations.

If val recall stays near zero even after 1200 iter, the bottleneck is the
architecture or loss, not just training length.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject  # noqa
from lib.centroid_helpers import centroids_um  # noqa
from lib.synthetic_warps import sample_warped_pair  # noqa
from bench.candidate_impls._g1_gnn_matcher import (  # noqa
    GNNMatcher, _build_knn_graph, _simple_features, _assignment, _pair_loss
)


def val_metrics(model, val_samples, k=8):
    """Compute argmax-pair recall on held-out synthetic warps."""
    model.eval()
    total_hits = 0
    total_pairs = 0
    total_pos_err = []
    with torch.no_grad():
        for w in val_samples:
            if len(w.correspondence) < 5:
                continue
            f_s = _simple_features(w.source_um, k=k)
            f_d = _simple_features(w.warped_um, k=k)
            e_s = _build_knn_graph(w.source_um, k=k)
            e_d = _build_knn_graph(w.warped_um, k=k)
            sim, _, _ = model(
                torch.as_tensor(f_s, dtype=torch.float32),
                torch.as_tensor(e_s, dtype=torch.long),
                torch.as_tensor(f_d, dtype=torch.float32),
                torch.as_tensor(e_d, dtype=torch.long),
            )
            P = _assignment(sim, model.dustbin, n_iter=30)
            P_core = P[:-1, :-1].numpy()
            pred_argmax = P_core.argmax(axis=1)
            for i, j_true in w.correspondence:
                total_pairs += 1
                j_pred = pred_argmax[i]
                if j_pred == j_true:
                    total_hits += 1
                    total_pos_err.append(0.0)
                else:
                    err = float(np.linalg.norm(w.warped_um[j_pred] - w.warped_um[j_true]))
                    total_pos_err.append(err)
    model.train()
    if total_pairs == 0:
        return dict(hit_frac=0.0, median_err=None, pairs=0)
    total_pos_err = np.array(total_pos_err)
    return dict(
        hit_frac=total_hits / total_pairs,
        median_err=float(np.median(total_pos_err)),
        p_lt_10=float((total_pos_err < 10).mean()),
        p_lt_50=float((total_pos_err < 50).mean()),
        pairs=total_pairs,
    )


def main():
    print("=== S38 G1 training diagnostic — subject 788406 HCR GFP+ ===", flush=True)
    s = load_subject("788406")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    print(f"  n_hcr_gfp={len(hcr_um)}", flush=True)

    rng_val = np.random.default_rng(42)
    val_samples = []
    for _ in range(40):
        w = sample_warped_pair(hcr_um, rng=rng_val, cube_um=400.0)
        val_samples.append(w)
    print(f"  validation set: {len(val_samples)} synthetic warps, "
          f"median corr={np.median([len(w.correspondence) for w in val_samples]):.0f}",
          flush=True)

    # Build the matcher — infer dims from first sample
    f0 = _simple_features(val_samples[0].source_um)
    in_dim = f0.shape[1]
    print(f"  feature dim (_simple_features) = {in_dim}", flush=True)

    rng_train = np.random.default_rng(1)
    torch.manual_seed(1)
    model = GNNMatcher(in_dim=in_dim, hidden=48, n_layers=3, cross_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoints = [120, 300, 600, 1200]
    losses = []

    print("\n  iter   train_loss   val_hit_frac  val_p_lt_10  val_p_lt_50  val_med_err")
    t_start = time.time()
    for it in range(1, max(checkpoints) + 1):
        w = sample_warped_pair(hcr_um, rng=rng_train, cube_um=400.0)
        if len(w.correspondence) < 5:
            continue
        f_s = _simple_features(w.source_um)
        f_d = _simple_features(w.warped_um)
        e_s = _build_knn_graph(w.source_um, k=8)
        e_d = _build_knn_graph(w.warped_um, k=8)
        f_s_t = torch.as_tensor(f_s, dtype=torch.float32)
        f_d_t = torch.as_tensor(f_d, dtype=torch.float32)
        e_s_t = torch.as_tensor(e_s, dtype=torch.long)
        e_d_t = torch.as_tensor(e_d, dtype=torch.long)
        corr_t = torch.as_tensor(w.correspondence, dtype=torch.long)
        sim, _, _ = model(f_s_t, e_s_t, f_d_t, e_d_t)
        loss = _pair_loss(sim, model.dustbin, corr_t, len(w.source_um), len(w.warped_um))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))

        if it in checkpoints:
            v = val_metrics(model, val_samples)
            tr_mean = float(np.mean(losses[-20:]))
            me = v["median_err"] if v["median_err"] is not None else float("nan")
            print(f"  {it:4d}   {tr_mean:10.3f}   {v['hit_frac']:12.3f}  "
                  f"{v['p_lt_10']:11.3f}  {v['p_lt_50']:11.3f}  {me:10.1f}",
                  flush=True)

    print(f"\ntotal train wall: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
