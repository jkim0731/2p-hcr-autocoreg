"""S38 probe v2: longer training + wider model — does val hit-rate saturate?

v1 showed 1200 iter reaches 41% hit-frac on 788406 synthetic warps
(hidden=48, 3+2 layers). Before committing to this capacity, check how
val scales with iterations and model width:

  * run A: hidden=48, 3+2 layers, 5000 iter — does val saturate?
  * run B: hidden=96, 4+3 layers, 5000 iter — does capacity matter?
"""
from __future__ import annotations

import sys
import time
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
    model.eval()
    hits = 0; pairs = 0; pos_err = []
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
            pred = P_core.argmax(axis=1)
            for i, j_t in w.correspondence:
                pairs += 1
                if pred[i] == j_t:
                    hits += 1; pos_err.append(0.0)
                else:
                    pos_err.append(float(np.linalg.norm(w.warped_um[pred[i]] - w.warped_um[j_t])))
    model.train()
    if pairs == 0:
        return dict(hit_frac=0.0, p_lt_10=0.0, p_lt_50=0.0, med=None)
    pos_err = np.array(pos_err)
    return dict(hit_frac=hits / pairs, p_lt_10=float((pos_err < 10).mean()),
                p_lt_50=float((pos_err < 50).mean()), med=float(np.median(pos_err)))


def run_config(name, hidden, n_layers, cross_layers, n_iter, val_samples, hcr_um,
               k=8, lr=1e-3, seed=1):
    print(f"\n--- {name}: hidden={hidden}, n_layers={n_layers}, cross={cross_layers}, iter={n_iter}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    f0 = _simple_features(val_samples[0].source_um)
    model = GNNMatcher(in_dim=f0.shape[1], hidden=hidden, n_layers=n_layers,
                       cross_layers=cross_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    checkpoints = [200, 500, 1000, 2000, 3500, 5000]
    print(f"  iter   loss     hit_frac  p<10  p<50  med")
    t0 = time.time()
    for it in range(1, n_iter + 1):
        w = sample_warped_pair(hcr_um, rng=rng, cube_um=400.0)
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
        loss = _pair_loss(sim, model.dustbin,
                          torch.as_tensor(w.correspondence, dtype=torch.long),
                          len(w.source_um), len(w.warped_um))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if it in checkpoints:
            v = val_metrics(model, val_samples)
            me = v["med"] if v["med"] is not None else float("nan")
            print(f"  {it:4d}   {float(np.mean(losses[-20:])):5.2f}   "
                  f"{v['hit_frac']:.3f}    {v['p_lt_10']:.3f} {v['p_lt_50']:.3f} {me:.0f}",
                  flush=True)
    print(f"  total: {time.time()-t0:.0f}s")
    return model, losses


def main():
    print("=== S38 v2 — G1 capacity × training length ===", flush=True)
    s = load_subject("788406")
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    rng_val = np.random.default_rng(42)
    val_samples = [sample_warped_pair(hcr_um, rng=rng_val, cube_um=400.0) for _ in range(40)]
    print(f"  n_hcr={len(hcr_um)}  val_samples={len(val_samples)}  "
          f"median_corr={np.median([len(w.correspondence) for w in val_samples]):.0f}",
          flush=True)

    run_config("A: baseline 48×3+2", hidden=48, n_layers=3, cross_layers=2,
               n_iter=5000, val_samples=val_samples, hcr_um=hcr_um)
    run_config("B: wider 96×4+3", hidden=96, n_layers=4, cross_layers=3,
               n_iter=5000, val_samples=val_samples, hcr_um=hcr_um)


if __name__ == "__main__":
    main()
