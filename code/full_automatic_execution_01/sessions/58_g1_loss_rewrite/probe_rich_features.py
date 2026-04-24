"""S58 — enriched-feature probe.

Hypothesis: `_simple_features` (16-dim k-NN distances + elevation angles)
is too impoverished for 1-of-4200 asymmetric matching. Add normalized
position within cube + local-density counts as coarse spatial anchors.

If this probe shows descent > 1.0 with same InfoNCE loss, confirms that
feature-richness was the blocker (not loss formulation).
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
    GNNMatcher, _build_knn_graph, _pair_loss)
from lib.synthetic_warps import sample_asymmetric_warped_pair  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402


def _rich_features(pts_um, ref_pts_um=None, k=8):
    """Enriched features for synthetic-warp training.

    Features per cell:
      [0:k]     sorted normalised k-NN distances
      [k:2k]    sorted elevation angles (up = +z)
      [2k:2k+3] normalised position within cube [0,1]^3
      [2k+3]    local density (count within 30µm) / median
    """
    from sklearn.neighbors import NearestNeighbors
    n = len(pts_um)
    knn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(pts_um)
    dists, idx = knn.kneighbors(pts_um)
    dists = dists[:, 1:]; idx = idx[:, 1:]
    med = float(np.median(dists[:, 0])) + 1e-9
    norm_d = dists / med
    up = np.array([1.0, 0.0, 0.0])
    vec = pts_um[idx] - pts_um[:, None, :]
    vec_n = vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)
    elev = np.arcsin(np.clip(vec_n @ up, -1, 1))
    elev_sorted = np.sort(elev, axis=1)

    # Normalised position within cube (assumes source/target cubes share scale
    # range after the asymmetric warp is applied; uses per-cloud min/max)
    pmin = pts_um.min(0); pmax = pts_um.max(0)
    pos_norm = (pts_um - pmin) / (pmax - pmin + 1e-9)

    # Local density within 30 µm
    nn30 = NearestNeighbors(radius=30.0).fit(pts_um)
    counts = np.array([len(nbrs) for nbrs in nn30.radius_neighbors(pts_um, return_distance=False)])
    counts_norm = counts / (np.median(counts) + 1e-9)

    F = np.concatenate([norm_d, elev_sorted, pos_norm, counts_norm[:, None]], axis=1)
    return F.astype(np.float32)


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
    f_src = _rich_features(w.source_um)
    in_dim = f_src.shape[1]
    print(f"in_dim={in_dim}", flush=True)
    model = GNNMatcher(in_dim=in_dim, hidden=96, n_layers=4, cross_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_iter = 200
    losses = []
    t0 = time.time()
    for it in range(n_iter):
        w = _sample()
        if len(w.correspondence) < 5 or len(w.source_um) < 10 or len(w.warped_um) < 10:
            continue
        f_src = _rich_features(w.source_um)
        f_dst = _rich_features(w.warped_um)
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
