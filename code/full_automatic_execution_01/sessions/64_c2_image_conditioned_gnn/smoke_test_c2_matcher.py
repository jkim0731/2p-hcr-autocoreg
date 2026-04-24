"""S64 subgoal 4 smoke test — C2Matcher forward/backward on synthetic inputs."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from bench.candidate_impls._c2_image_gnn import C2Matcher  # noqa: E402
from bench.candidate_impls._g1_gnn_matcher import (  # noqa: E402
    _assignment, _build_knn_graph, _pair_loss,
)


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    hand_dim = 20            # matches _simple_features with k=8
    patch_dim = 64
    N_a, N_b = 100, 300      # CZ 100 cells, HCR crop 300

    model = C2Matcher(hand_dim=hand_dim, patch_dim=patch_dim,
                      hidden=96, n_layers=4, cross_layers=3)
    print(f"C2Matcher total params: {model.param_count:,}", flush=True)
    print(f"  patch_enc : {model.patch_enc.param_count:,}", flush=True)
    print(f"  matcher   : "
          f"{sum(p.numel() for p in model.matcher.parameters()):,}", flush=True)

    # Fake centroids → k-NN graphs (same construction the real pipeline uses).
    pts_a = rng.normal(0, 100, size=(N_a, 3)).astype(np.float32)
    pts_b = rng.normal(0, 100, size=(N_b, 3)).astype(np.float32)
    e_a = _build_knn_graph(pts_a, k=8)
    e_b = _build_knn_graph(pts_b, k=8)

    f_a = torch.randn(N_a, hand_dim)
    f_b = torch.randn(N_b, hand_dim)
    p_a = torch.randn(N_a, 1, 16, 16, 16)
    p_b = torch.randn(N_b, 1, 16, 16, 16)
    e_a_t = torch.as_tensor(e_a, dtype=torch.long)
    e_b_t = torch.as_tensor(e_b, dtype=torch.long)

    # Forward timing
    t0 = time.time()
    sim, mat_a, mat_b = model(f_a, p_a, e_a_t, f_b, p_b, e_b_t)
    t_fw = time.time() - t0
    print(f"forward in {t_fw*1000:.1f} ms", flush=True)
    print(f"  sim shape {tuple(sim.shape)} mat_a {tuple(mat_a.shape)} "
          f"mat_b {tuple(mat_b.shape)}", flush=True)
    assert sim.shape == (N_a, N_b), f"bad sim shape {sim.shape}"
    assert mat_a.shape == (N_a,), f"bad mat_a shape {mat_a.shape}"
    assert mat_b.shape == (N_b,), f"bad mat_b shape {mat_b.shape}"
    assert torch.isfinite(sim).all(), "non-finite sim"
    assert torch.isfinite(mat_a).all() and torch.isfinite(mat_b).all(), "non-finite matchability"

    # Sinkhorn assignment on rectangular matrix: final step normalises cols → 1.
    # Row sums converge to (Nb+1)/(Na+1); that's expected — argmax of P[i, :]
    # is invariant to row scaling, so one-to-one extraction still works.
    with torch.no_grad():
        P = _assignment(sim, model.dustbin, n_iter=20)
    expected_row_sum = (N_b + 1) / (N_a + 1)
    print(f"assignment P shape {tuple(P.shape)} "
          f"row-sum range [{P.sum(1).min():.3f}, {P.sum(1).max():.3f}] "
          f"(expect ≈ {expected_row_sum:.3f}) "
          f"col-sum range [{P.sum(0).min():.3f}, {P.sum(0).max():.3f}]", flush=True)
    assert P.shape == (N_a + 1, N_b + 1)
    assert torch.allclose(P.sum(0), torch.ones(N_b + 1), atol=1e-3)
    assert torch.allclose(P.sum(1), torch.full((N_a + 1,), expected_row_sum), atol=5e-2)
    assert (P >= 0).all() and (P <= 1).all()

    # Backward: pair loss on fake correspondence.
    n_pairs = 10
    corr = torch.stack([
        torch.arange(n_pairs),
        torch.tensor(rng.choice(N_b, size=n_pairs, replace=False), dtype=torch.long),
    ], dim=1)

    model.zero_grad()
    sim2, _, _ = model(f_a, p_a, e_a_t, f_b, p_b, e_b_t)
    loss = _pair_loss(sim2, model.dustbin, corr, N_a, N_b)
    print(f"pair_loss = {loss.item():.4f}", flush=True)
    t0 = time.time()
    loss.backward()
    t_bw = time.time() - t0
    print(f"backward in {t_bw*1000:.1f} ms", flush=True)

    cnn_params = list(model.patch_enc.parameters())
    gnn_params = list(model.matcher.parameters())

    def _has_grad(params):
        gs = [p.grad for p in params if p.grad is not None]
        if not gs:
            return False, 0, 0.0
        finite = all(torch.isfinite(g).all() for g in gs)
        max_abs = max(float(g.abs().max()) for g in gs)
        return finite, len(gs), max_abs

    cnn_finite, cnn_n, cnn_max = _has_grad(cnn_params)
    gnn_finite, gnn_n, gnn_max = _has_grad(gnn_params)
    print(f"CNN grads: finite={cnn_finite} n={cnn_n} max|g|={cnn_max:.6f}", flush=True)
    print(f"GNN grads: finite={gnn_finite} n={gnn_n} max|g|={gnn_max:.6f}", flush=True)
    assert cnn_finite and cnn_max > 0, "CNN gradients missing or degenerate"
    assert gnn_finite and gnn_max > 0, "GNN gradients missing or degenerate"

    print("\n=== PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
