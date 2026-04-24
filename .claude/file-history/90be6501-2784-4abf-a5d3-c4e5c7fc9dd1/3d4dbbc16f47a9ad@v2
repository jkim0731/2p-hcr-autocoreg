"""G1-LOSO — real-supervised G1 with leave-one-subject-out.

S58/S59 showed G1 trained on F8 synthetic HCR→HCR warps does not transfer
to real CZ↔HCR inference (r@20 ≤ 0.04).  Root cause: real CZ is a different
modality — segmentation noise, density profile, and depth distribution
differ from any centroid-only simulation of HCR subsample.

S63 bypasses F8 entirely: train G1 directly on real coreg_table pairs from
the five *other* benchmark subjects (LOSO), then run inference on the
held-out subject.  Each training iteration:

  1. sample one training subject
  2. load its CZ and HCR GFP+ centroids; warm-start CZ into HCR space
  3. crop HCR to 2x CZ AABB (keeps memory bounded on 755252's 30k cells)
  4. extract 20-dim _simple_features on both clouds
  5. build k-NN graphs; forward GNNMatcher
  6. InfoNCE loss on the real coreg_table pairs that survived the crop

Inference is unchanged from the F8-trained G1: no test-time retraining.
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

from bench.harness import register_candidate, CoregResult
from lib.centroid_helpers import centroids_um, default_warmstart_zyx
from bench.candidate_impls._g1_gnn_matcher import (
    GNNMatcher, _build_knn_graph, _simple_features, _assignment, _pair_loss,
)

SIX_SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]


def _preload_subject(sid: str, keep_subject: bool = False):
    """Load subject data once; return dict for cheap training access."""
    from benchmark_data_loader import load_subject
    s = load_subject(sid)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)
    cz_pos = {int(v): i for i, v in enumerate(cz_ids)}
    hcr_pos = {int(v): i for i, v in enumerate(hcr_ids)}
    pairs = []
    for _, r in s.coreg_table.iterrows():
        ci = cz_pos.get(int(r["cz_id"]))
        hi = hcr_pos.get(int(r["hcr_id"]))
        if ci is not None and hi is not None:
            pairs.append((ci, hi))
    d = dict(
        sid=sid,
        cz_um=cz_um,
        cz_init=cz_init,  # CZ warm-started into HCR space
        hcr_um=hcr_um,
        cz_ids=cz_ids,
        hcr_ids=hcr_ids,
        pairs=np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64),
    )
    if keep_subject:
        d["subject"] = s
    return d


def _crop_hcr(cz_init, hcr_um, pad_um: float = 200.0):
    lo = cz_init.min(0) - pad_um
    hi = cz_init.max(0) + pad_um
    keep = np.all((hcr_um >= lo) & (hcr_um <= hi), axis=1)
    return np.where(keep)[0], hcr_um[keep]


def _train_supervised_loso(train_data: list[dict],
                           *,
                           n_iter: int = 2000,
                           k_neighbours: int = 8,
                           hidden: int = 96,
                           n_layers: int = 4,
                           cross_layers: int = 3,
                           lr: float = 1e-4,
                           jitter_um: float = 4.0,
                           rng_seed: int = 0,
                           ):
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)

    # in_dim from _simple_features on first subject's CZ
    f0 = _simple_features(train_data[0]["cz_init"])
    in_dim = f0.shape[1]
    model = GNNMatcher(in_dim=in_dim, hidden=hidden, n_layers=n_layers,
                       cross_layers=cross_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    per_iter_n_pairs: list[int] = []
    t_start = time.time()

    for it in range(n_iter):
        d = train_data[rng.integers(len(train_data))]
        cz_init = d["cz_init"].copy()
        if jitter_um > 0:
            cz_init = cz_init + rng.normal(0, jitter_um, size=cz_init.shape)

        hcr_keep_idx, hcr_crop = _crop_hcr(cz_init, d["hcr_um"], pad_um=200.0)
        if len(hcr_crop) < 30 or len(cz_init) < 30:
            continue

        hcr_remap = {int(g): i for i, g in enumerate(hcr_keep_idx)}
        pairs = d["pairs"]
        if len(pairs) == 0:
            continue
        mask = np.array([int(hi) in hcr_remap for (_, hi) in pairs], dtype=bool)
        pairs_local = pairs[mask].copy()
        if len(pairs_local) < 5:
            continue
        pairs_local[:, 1] = np.array([hcr_remap[int(hi)] for hi in pairs_local[:, 1]])

        f_cz = _simple_features(cz_init, k=k_neighbours)
        f_hc = _simple_features(hcr_crop, k=k_neighbours)
        e_cz = _build_knn_graph(cz_init, k=k_neighbours)
        e_hc = _build_knn_graph(hcr_crop, k=k_neighbours)

        fc = torch.as_tensor(f_cz, dtype=torch.float32)
        fh = torch.as_tensor(f_hc, dtype=torch.float32)
        ec = torch.as_tensor(e_cz, dtype=torch.long)
        eh = torch.as_tensor(e_hc, dtype=torch.long)
        corr = torch.as_tensor(pairs_local, dtype=torch.long)

        sim, _, _ = model(fc, ec, fh, eh)
        loss = _pair_loss(sim, model.dustbin, corr, len(cz_init), len(hcr_crop))
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        per_iter_n_pairs.append(int(len(pairs_local)))

        if (it + 1) % 100 == 0:
            dt = time.time() - t_start
            mean_loss = float(np.mean(losses[-100:])) if losses else float("nan")
            mean_np = float(np.mean(per_iter_n_pairs[-100:])) if per_iter_n_pairs else 0.0
            print(f"  g1loso_train it={it+1}/{n_iter} loss={mean_loss:.3f} "
                  f"n_pairs={mean_np:.0f} t={dt:.1f}s", flush=True)
    return model, losses


@register_candidate("G1_LOSO")
def run_g1_loso(s, *,
                n_train_iter: int = 2000,
                k: int = 8,
                hidden: int = 96,
                n_layers: int = 4,
                cross_layers: int = 3,
                lr: float = 1e-4,
                jitter_um: float = 4.0,
                train_subjects: list[str] | None = None,
                rng_seed: int = 0,
                verbose: bool = True,
                ) -> CoregResult:
    held = str(s.subject_id)
    if train_subjects is None:
        train_subjects = [sid for sid in SIX_SUBJECTS if sid != held]
    else:
        train_subjects = [sid for sid in train_subjects if sid != held]

    if verbose:
        print(f"  G1-LOSO: held-out={held} train_pool={train_subjects}", flush=True)

    t0 = time.time()
    train_data = [_preload_subject(sid) for sid in train_subjects]
    if verbose:
        total_pairs = sum(len(d["pairs"]) for d in train_data)
        print(f"  G1-LOSO: preload {time.time()-t0:.1f}s, total_gt_pairs={total_pairs}",
              flush=True)

    model, losses = _train_supervised_loso(
        train_data, n_iter=n_train_iter, k_neighbours=k, hidden=hidden,
        n_layers=n_layers, cross_layers=cross_layers, lr=lr, jitter_um=jitter_um,
        rng_seed=rng_seed,
    )

    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ws = default_warmstart_zyx(cz_um, hcr_um)

    hcr_keep_idx, hcr_crop = _crop_hcr(cz_init, hcr_um, pad_um=200.0)
    fc_d = _simple_features(cz_init, k=k)
    fg_d = _simple_features(hcr_crop, k=k)
    e_cz = _build_knn_graph(cz_init, k=k)
    e_hc = _build_knn_graph(hcr_crop, k=k)

    with torch.no_grad():
        sim, _, _ = model(
            torch.as_tensor(fc_d, dtype=torch.float32),
            torch.as_tensor(e_cz, dtype=torch.long),
            torch.as_tensor(fg_d, dtype=torch.float32),
            torch.as_tensor(e_hc, dtype=torch.long),
        )
        P = _assignment(sim, model.dustbin, n_iter=30)

    P_core = P[:-1, :-1].numpy()
    rows = []
    for i in range(len(cz_ids)):
        j_local = int(np.argmax(P_core[i]))
        mass = float(P_core[i, j_local])
        dustmass = float(P[i, -1].item())
        if mass < dustmass or mass < 1e-4:
            continue
        j_abs = int(hcr_keep_idx[j_local])
        conf = mass / (mass + dustmass)
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j_abs]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]), cz_y_um=float(cz_um[i, 1]),
            cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j_abs, 2]), hcr_y_um=float(hcr_um[j_abs, 1]),
            hcr_z_um=float(hcr_um[j_abs, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False).drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)
    if verbose:
        print(f"  G1-LOSO: emitted {len(df)} pairs", flush=True)
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=None,
        diagnostics=dict(
            held_out=held,
            train_subjects=train_subjects,
            n_train_iter=n_train_iter,
            train_loss_final=(float(losses[-1]) if losses else None),
            train_loss_mean50=(float(np.mean(losses[-50:])) if len(losses) >= 50 else None),
            n_hcr_kept=int(len(hcr_crop)),
            n_emitted=int(len(df)),
        ),
    )
