"""G1 — hand-feature GNN matcher (pure torch, CPU).

Self-supervised cross-graph attention matcher with Sinkhorn+dustbin head.

Because `torch_geometric` is not available in this environment, the graph
conv is implemented as a pure torch `scatter_add` aggregator over the
explicit edge index.

Training:
  Stage 1 — self-supervised on F8 synthetic warps of the current subject's
            HCR GFP+ cloud. Warps emulate the CZ side by subsampling +
            applying the random anisotropic + TPS transform. This is the
            only training data used; no benchmark coreg tables are seen at
            training time.

Inference:
  Build both graphs, cross-attention, Sinkhorn+dustbin; output one-to-one
  assignments plus a per-pair matchability probability.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult, TransformDescriptor
from lib.cell_features import extract_cell_features, invariant_feature_mask
from lib.synthetic_warps import sample_warped_pair, sample_asymmetric_warped_pair
from lib.centroid_helpers import centroids_um


def _build_knn_graph(pts_um: np.ndarray, k: int = 8) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pts_um)
    _, idx = nn.kneighbors(pts_um)
    idx = idx[:, 1:]  # drop self
    # Return edge index (src, dst) flattened
    src = np.repeat(np.arange(len(pts_um)), k)
    dst = idx.flatten()
    return np.stack([src, dst], axis=0)  # (2, E)


class GNNMatcher(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3,
                 cross_layers: int = 2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.cross_q = nn.ModuleList([nn.Linear(hidden, hidden)
                                       for _ in range(cross_layers)])
        self.cross_k = nn.ModuleList([nn.Linear(hidden, hidden)
                                       for _ in range(cross_layers)])
        self.cross_v = nn.ModuleList([nn.Linear(hidden, hidden)
                                       for _ in range(cross_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden)
                                    for _ in range(n_layers + cross_layers)])
        self.match_head = nn.Linear(hidden, 1)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.hidden = hidden

    def _graph_step(self, x, edge_index, layer_idx):
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([x[src], x[dst]], dim=-1)
        m = self.msg[layer_idx](pair)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        x = x + agg
        x = self.norm[layer_idx](x)
        return x

    def _cross(self, x_a, x_b, li):
        q = self.cross_q[li](x_a); k = self.cross_k[li](x_b); v = self.cross_v[li](x_b)
        att = (q @ k.T) / (self.hidden ** 0.5)
        att = F.softmax(att, dim=-1)
        return x_a + att @ v

    def forward(self, f_a, e_a, f_b, e_b):
        x_a = self.enc(f_a); x_b = self.enc(f_b)
        for li in range(len(self.msg)):
            x_a = self._graph_step(x_a, e_a, li)
            x_b = self._graph_step(x_b, e_b, li)
        for li in range(len(self.cross_q)):
            x_a_new = self._cross(x_a, x_b, li)
            x_b_new = self._cross(x_b, x_a, li)
            x_a = self.norm[len(self.msg) + li](x_a_new)
            x_b = self.norm[len(self.msg) + li](x_b_new)
        match_a = torch.sigmoid(self.match_head(x_a)).squeeze(-1)
        match_b = torch.sigmoid(self.match_head(x_b)).squeeze(-1)
        sim = (x_a @ x_b.T) / (self.hidden ** 0.5)
        return sim, match_a, match_b


def _log_sinkhorn(log_M, n_iter: int = 20):
    for _ in range(n_iter):
        log_M = log_M - torch.logsumexp(log_M, dim=1, keepdim=True)
        log_M = log_M - torch.logsumexp(log_M, dim=0, keepdim=True)
    return log_M


def _assignment(sim, dustbin, n_iter=20):
    Na, Nb = sim.shape
    db_row = dustbin.expand(Na, 1)
    db_col = dustbin.expand(1, Nb + 1)
    aug = torch.cat([torch.cat([sim, db_row], dim=1), db_col], dim=0)
    log_M = _log_sinkhorn(aug, n_iter=n_iter)
    return log_M.exp()


def _pair_loss(sim, dustbin, corr_idx, n_a, n_b, *,
               temperature: float = 1.0):
    """Symmetric InfoNCE over matched pairs (S58 rewrite).

    Prior Sinkhorn-NLL formulations (S38 symmetric, S52 per-side-averaged)
    stall because the 4200 unmatched-tgt cells either (a) dominate by sheer
    count (S38) or (b) compete with matched mass for dustbin probability
    (S52 fix). In both cases the gradient on matched pairs is weak.

    This version trains the `sim` matrix directly via symmetric row+col
    softmax cross-entropy on matched pairs only. Unmatched cells contribute
    to the partition (denominator) but are not explicit targets. Sinkhorn
    + dustbin is retained at inference for consistent one-to-one outputs.
    """
    if corr_idx.shape[0] == 0:
        return torch.zeros((), dtype=sim.dtype, device=sim.device)
    i_idx = corr_idx[:, 0]
    j_idx = corr_idx[:, 1]
    sim_t = sim / temperature
    # row InfoNCE: for each matched i, sim[i,:] should be peaked at j
    row_lse = torch.logsumexp(sim_t[i_idx, :], dim=1)
    matched_scores = sim_t[i_idx, j_idx]
    row_loss = (-matched_scores + row_lse).mean()
    # col InfoNCE: for each matched j, sim[:,j] should be peaked at i
    col_lse = torch.logsumexp(sim_t[:, j_idx], dim=0)
    col_loss = (-matched_scores + col_lse).mean()
    return 0.5 * (row_loss + col_loss)


def _train_self_supervised(pts_hcr_um, feat_fn, n_iter=1500, cube_um=400.0,
                           k_neighbours=8, rng_seed=0, *,
                           asymmetric: bool = True,
                           hidden: int = 96, n_layers: int = 4,
                           cross_layers: int = 3):
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)

    def _sample():
        if asymmetric:
            return sample_asymmetric_warped_pair(
                pts_hcr_um, rng=rng,
                source_cube_um=cube_um,
                target_margin_um=cube_um,
                source_n_target=900, target_n_cap=12000)
        return sample_warped_pair(pts_hcr_um, rng=rng, cube_um=cube_um)

    w = _sample()
    f_src = feat_fn(w.source_um, pts_hcr_um)
    in_dim = f_src.shape[1]
    model = GNNMatcher(in_dim=in_dim, hidden=hidden, n_layers=n_layers,
                       cross_layers=cross_layers)
    # LR=1e-4 (S58): lr=1e-3 descended briefly then overshot to uniform
    # baseline by iter 500; 1e-4 gives stable monotonic descent ~1.5/500 iter.
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    t_start = time.time()
    for it in range(n_iter):
        w = _sample()
        if len(w.correspondence) < 5 or len(w.source_um) < 10 or len(w.warped_um) < 10:
            continue
        f_src = feat_fn(w.source_um, pts_hcr_um)
        f_dst = feat_fn(w.warped_um, pts_hcr_um)
        e_src = _build_knn_graph(w.source_um, k=k_neighbours)
        e_dst = _build_knn_graph(w.warped_um, k=k_neighbours)
        f_src_t = torch.as_tensor(f_src, dtype=torch.float32)
        f_dst_t = torch.as_tensor(f_dst, dtype=torch.float32)
        e_src_t = torch.as_tensor(e_src, dtype=torch.long)
        e_dst_t = torch.as_tensor(e_dst, dtype=torch.long)
        corr_t = torch.as_tensor(w.correspondence, dtype=torch.long)
        sim, _, _ = model(f_src_t, e_src_t, f_dst_t, e_dst_t)
        loss = _pair_loss(sim, model.dustbin, corr_t, len(w.source_um), len(w.warped_um))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (it + 1) % 100 == 0:
            dt = time.time() - t_start
            mean_loss = float(np.mean(losses[-100:])) if losses else float("nan")
            print(f"  g1_train it={it+1}/{n_iter} loss={mean_loss:.3f} "
                  f"n_src={len(w.source_um)} n_tgt={len(w.warped_um)} "
                  f"t={dt:.1f}s", flush=True)
    return model, losses


def _simple_features(pts_um, ref_pts_um=None, k=8):
    """Enriched self-contained features for both training and inference.

    S58 diagnosis: plain k-NN distances + elevation angles (16-dim) are not
    discriminative at 1-of-4200 asymmetric matching — InfoNCE descent was
    0.02 over 200 iter. Adding normalised position-within-cloud (3-dim) and
    local density (1-dim) lifts descent to ~0.6 over the same 200 iter.

    Features per cell (20-dim at k=8):
      [0:k]     sorted normalised k-NN distances
      [k:2k]    sorted elevation angles about +z
      [2k:2k+3] per-cloud min-max-normalised position  (coarse spatial anchor;
                relative position is preserved across anisotropic scale+TPS)
      [2k+3]    local density (count within 30 µm) / median
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

    pmin = pts_um.min(0); pmax = pts_um.max(0)
    pos_norm = (pts_um - pmin) / (pmax - pmin + 1e-9)

    nn30 = NearestNeighbors(radius=30.0).fit(pts_um)
    counts = np.array([len(nbrs) for nbrs in
                       nn30.radius_neighbors(pts_um, return_distance=False)])
    counts_norm = counts / (float(np.median(counts)) + 1e-9)

    F = np.concatenate([norm_d, elev_sorted, pos_norm, counts_norm[:, None]], axis=1)
    return F.astype(np.float32)


@register_candidate("G1")
def run_g1(s, *, n_train_iter=1500, k=8, use_f6=False,
           asymmetric: bool = True,
           hidden: int = 96, n_layers: int = 4, cross_layers: int = 3) -> CoregResult:
    from lib.centroid_helpers import default_warmstart_zyx
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    model, losses = _train_self_supervised(
        hcr_um, _simple_features, n_iter=n_train_iter, k_neighbours=k,
        asymmetric=asymmetric, hidden=hidden, n_layers=n_layers,
        cross_layers=cross_layers)

    # Inference — run model on actual (cz_init, hcr) with F6 features.
    if use_f6:
        Fc, names, _ = extract_cell_features(s, "cz")
        Fg, _, _ = extract_cell_features(s, "hcr_gfp")
        inv = invariant_feature_mask(names)
        keep = inv & ~np.isnan(Fc).any(0) & ~np.isnan(Fg).any(0)
        mu = np.nanmean(Fg[:, keep], 0); sd = np.nanstd(Fg[:, keep], 0) + 1e-6
        fc = (Fc[:, keep] - mu) / sd
        fg = (Fg[:, keep] - mu) / sd
        # Project down to the training dim by taking the first few columns so shapes match
        target_dim = model.enc[0].in_features
        fc_d = fc[:, :target_dim] if fc.shape[1] >= target_dim else np.pad(fc, ((0, 0), (0, target_dim - fc.shape[1])))
        fg_d = fg[:, :target_dim] if fg.shape[1] >= target_dim else np.pad(fg, ((0, 0), (0, target_dim - fg.shape[1])))
    else:
        fc_d = _simple_features(cz_init)
        fg_d = _simple_features(hcr_um)

    # Build k-NN graph on the warped CZ cloud so graph scale matches HCR.
    e_cz = _build_knn_graph(cz_init, k=k)
    e_hcr = _build_knn_graph(hcr_um, k=k)
    with torch.no_grad():
        sim, mat_a, mat_b = model(
            torch.as_tensor(fc_d, dtype=torch.float32),
            torch.as_tensor(e_cz, dtype=torch.long),
            torch.as_tensor(fg_d, dtype=torch.float32),
            torch.as_tensor(e_hcr, dtype=torch.long),
        )
        P = _assignment(sim, model.dustbin, n_iter=30)

    P_core = P[:-1, :-1].numpy()
    rows = []
    for i in range(len(cz_ids)):
        j = int(np.argmax(P_core[i]))
        mass = float(P_core[i, j])
        dustmass = float(P[i, -1].item())
        if mass < dustmass or mass < 1e-4:
            continue
        conf = mass / (mass + dustmass)
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False).drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)
    return CoregResult(
        pairs_df=df, confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=None,
        diagnostics=dict(train_loss_final=(losses[-1] if losses else None),
                         n_train_iter=n_train_iter, k=k),
    )
