"""S58 — sanity probe for new _pair_loss.

Loads 788406 HCR GFP+ centroids, samples one asymmetric warp (n_src≈150,
n_tgt≈4200), runs 200 iterations of the training loop, prints loss every
20 iter. The prior symmetric-sum-normalized loss plateaued at ~4.95 (S52
diagnosis). The new per-side-averaged loss should start at ~2*log(N) and
descend as the model learns matching.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._g1_gnn_matcher import (  # noqa: E402
    GNNMatcher, _build_knn_graph, _pair_loss, _simple_features)
from lib.synthetic_warps import sample_asymmetric_warped_pair  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402


def main():
    s = load_subject("788406")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    print(f"n_hcr_gfp={len(hcr_um)}", flush=True)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    def _sample():
        return sample_asymmetric_warped_pair(
            hcr_um, rng=rng,
            source_cube_um=400.0, target_margin_um=400.0,
            source_n_target=900, target_n_cap=12000)

    w = _sample()
    f_src = _simple_features(w.source_um)
    in_dim = f_src.shape[1]
    model = GNNMatcher(in_dim=in_dim, hidden=96, n_layers=4, cross_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_iter = 200
    losses = []
    t0 = time.time()
    for it in range(n_iter):
        w = _sample()
        if len(w.correspondence) < 5 or len(w.source_um) < 10 or len(w.warped_um) < 10:
            continue
        f_src = _simple_features(w.source_um)
        f_dst = _simple_features(w.warped_um)
        e_src = _build_knn_graph(w.source_um, k=8)
        e_dst = _build_knn_graph(w.warped_um, k=8)
        f_src_t = torch.as_tensor(f_src, dtype=torch.float32)
        f_dst_t = torch.as_tensor(f_dst, dtype=torch.float32)
        e_src_t = torch.as_tensor(e_src, dtype=torch.long)
        e_dst_t = torch.as_tensor(e_dst, dtype=torch.long)
        corr_t = torch.as_tensor(w.correspondence, dtype=torch.long)
        sim, _, _ = model(f_src_t, e_src_t, f_dst_t, e_dst_t)
        loss = _pair_loss(sim, model.dustbin, corr_t,
                          len(w.source_um), len(w.warped_um))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (it + 1) % 20 == 0:
            mean_recent = float(np.mean(losses[-20:]))
            print(f"  it={it+1:3d}  loss_recent_mean={mean_recent:.3f}  "
                  f"n_src={len(w.source_um)} n_tgt={len(w.warped_um)} "
                  f"n_corr={len(w.correspondence)} "
                  f"t={time.time()-t0:.1f}s", flush=True)

    print(f"\nInitial 20 mean = {np.mean(losses[:20]):.3f}", flush=True)
    print(f"Final 20 mean   = {np.mean(losses[-20:]):.3f}", flush=True)
    print(f"Descent = {np.mean(losses[:20]) - np.mean(losses[-20:]):.3f}", flush=True)


if __name__ == "__main__":
    main()
