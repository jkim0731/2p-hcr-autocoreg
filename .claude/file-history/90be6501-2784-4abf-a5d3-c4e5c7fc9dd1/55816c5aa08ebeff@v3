"""G2 — contrastive node embedding + nearest-neighbour match.

Light-weight sibling of G1: same node encoder + graph conv layers, but no
cross-attention or Sinkhorn.  Trained with InfoNCE on F8 synthetic warps;
inference = nearest neighbour in shared embedding space + cosine threshold.
"""
from __future__ import annotations

import sys
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

from bench.harness import register_candidate, CoregResult
from lib.synthetic_warps import sample_warped_pair
from bench.candidate_impls._g1_gnn_matcher import (
    _build_knn_graph, _simple_features,
)
from lib.centroid_helpers import centroids_um


class ContrastiveEmbed(nn.Module):
    def __init__(self, in_dim, hidden=48, n_layers=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, f, e):
        x = self.enc(f)
        for li, m in enumerate(self.msg):
            src, dst = e[0], e[1]
            pair = torch.cat([x[src], x[dst]], dim=-1)
            mm = m(pair)
            agg = torch.zeros_like(x); agg.index_add_(0, dst, mm)
            x = self.norm[li](x + agg)
        return F.normalize(self.proj(x), dim=-1)


def _infonce(z_a, z_b, corr, tau=0.1):
    # z_a: (Na, D); z_b: (Nb, D); corr: (N_match, 2) pairs
    ia = corr[:, 0]; ib = corr[:, 1]
    sim = (z_a[ia] @ z_b.T) / tau
    # positives are at columns ib
    target = ib
    return F.cross_entropy(sim, target)


@register_candidate("G2")
def run_g2(s, *, n_train_iter=120, k=8, tau=0.1) -> CoregResult:
    from lib.centroid_helpers import default_warmstart_zyx
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # Train
    w0 = sample_warped_pair(hcr_um, rng=rng)
    f0 = _simple_features(w0.source_um)
    model = ContrastiveEmbed(in_dim=f0.shape[1], hidden=48, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for it in range(n_train_iter):
        w = sample_warped_pair(hcr_um, rng=rng)
        if len(w.correspondence) < 5:
            continue
        f_src = _simple_features(w.source_um)
        f_dst = _simple_features(w.warped_um)
        e_src = _build_knn_graph(w.source_um, k=k)
        e_dst = _build_knn_graph(w.warped_um, k=k)
        z_s = model(torch.as_tensor(f_src, dtype=torch.float32),
                     torch.as_tensor(e_src, dtype=torch.long))
        z_d = model(torch.as_tensor(f_dst, dtype=torch.float32),
                     torch.as_tensor(e_dst, dtype=torch.long))
        corr = torch.as_tensor(w.correspondence, dtype=torch.long)
        l = _infonce(z_s, z_d, corr, tau=tau)
        opt.zero_grad(); l.backward(); opt.step()
        losses.append(float(l.item()))

    # Inference on warped CZ so graph scale matches training (HCR-scale).
    f_cz = _simple_features(cz_init)
    f_hr = _simple_features(hcr_um)
    e_cz = _build_knn_graph(cz_init, k=k)
    e_hr = _build_knn_graph(hcr_um, k=k)
    with torch.no_grad():
        z_cz = model(torch.as_tensor(f_cz, dtype=torch.float32),
                     torch.as_tensor(e_cz, dtype=torch.long)).numpy()
        z_hr = model(torch.as_tensor(f_hr, dtype=torch.float32),
                     torch.as_tensor(e_hr, dtype=torch.long)).numpy()

    sims = z_cz @ z_hr.T
    # One-to-one greedy
    order = np.argsort(-sims.flatten())
    used_cz = set(); used_hr = set()
    rows = []
    for flat in order:
        i = flat // len(hcr_um); j = flat % len(hcr_um)
        if sims[i, j] < 0.1:
            break
        if i in used_cz or j in used_hr:
            continue
        used_cz.add(i); used_hr.add(j)
        conf = float(sims[i, j])
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
            confidence=conf,
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]), cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]), hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    return CoregResult(
        pairs_df=df, confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=None,
        diagnostics=dict(train_loss_final=(losses[-1] if losses else None),
                         n_train_iter=n_train_iter, k=k),
    )
