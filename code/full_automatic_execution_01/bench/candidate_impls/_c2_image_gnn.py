"""C2 — image-conditioned GNN matcher (S64 subgoals 4+5).

Extends G1 (`bench/candidate_impls/_g1_gnn_matcher.py::GNNMatcher`) by
concatenating a per-cell 3D-CNN patch embedding onto the 20-dim
`_simple_features` handcrafted descriptor before the matcher's input
MLP. Everything downstream of the input projection — GNN layers, cross
attention, Sinkhorn+dustbin — is shared with G1.

Rationale (from S63 refutation, §9.6):
  G1-LOSO showed that training architecture on real supervision still
  fails (sum Δ r@20 = −0.686 across 3-subject bench). The 20-dim hand
  features are the bottleneck — they structurally cannot encode
  modality-specific cues (intensity profile, local texture,
  segmentation boundary, soma-morphology). C2 adds image patches around
  each centroid so the network has access to those cues.

Training (`run_c2_loso`): real-pair supervised LOSO. Per training subject
we pre-extract + cache patches around cz_init-warped CZ centroids and
around HCR-crop GFP+ centroids. Per iteration: sample one subject,
forward C2, InfoNCE on real `coreg_table` pairs remapped to local
HCR-crop indices. Stage-1 voxel-warp pretraining is available as an
optional prefix but defaults off since sample generation is ~30 s/sample
on CPU — can be turned on once a pre-generated pool exists.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.candidate_impls._g1_gnn_matcher import (  # noqa: E402
    GNNMatcher, _assignment, _build_knn_graph, _pair_loss, _simple_features,
)
from bench.candidate_impls._g1_loso_matcher import _crop_hcr  # noqa: E402
from bench.harness import register_candidate, CoregResult  # noqa: E402
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa: E402
from lib.image_patches import extract_cz_patches, extract_hcr_patches  # noqa: E402
from lib.patch_encoder import PatchEncoder  # noqa: E402


SIX_SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]


class C2Matcher(nn.Module):
    """Image-conditioned GNN matcher.

    Fuses a 3D-CNN embedding over each cell's centroid-centred patch with
    the G1 hand-crafted descriptor; feeds the concatenation to a G1
    `GNNMatcher` with matching `in_dim`.

    Args:
        hand_dim: dimension of the hand-crafted feature vector (20 for
            `_simple_features` with k=8).
        patch_dim: dimension of the 3D-CNN patch embedding (default 64).
        hidden: matcher hidden dim.
        n_layers: number of intra-graph GNN layers.
        cross_layers: number of cross-graph attention layers.
        in_channels: number of patch input channels (1 for single-modality
            CZ or HCR-488; 2 if both channels are stacked).
    """

    def __init__(self, hand_dim: int, *, patch_dim: int = 64,
                 hidden: int = 96, n_layers: int = 4,
                 cross_layers: int = 3, in_channels: int = 1) -> None:
        super().__init__()
        self.hand_dim = hand_dim
        self.patch_dim = patch_dim
        self.patch_enc = PatchEncoder(out_dim=patch_dim, in_channels=in_channels)
        self.matcher = GNNMatcher(
            in_dim=hand_dim + patch_dim,
            hidden=hidden, n_layers=n_layers, cross_layers=cross_layers,
        )

    @property
    def dustbin(self) -> nn.Parameter:
        return self.matcher.dustbin

    def _fuse(self, f_hand: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """Concatenate hand features with CNN embedding. patches: (N, C, D, H, W)."""
        emb = self.patch_enc(patches)
        return torch.cat([f_hand, emb], dim=-1)

    def forward(
        self,
        f_a: torch.Tensor,
        p_a: torch.Tensor,
        e_a: torch.Tensor,
        f_b: torch.Tensor,
        p_b: torch.Tensor,
        e_b: torch.Tensor,
    ):
        """f_*: (N, hand_dim); p_*: (N, C, D, H, W); e_*: (2, E) long."""
        x_a = self._fuse(f_a, p_a)
        x_b = self._fuse(f_b, p_b)
        return self.matcher(x_a, e_a, x_b, e_b)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Preprocessing: load centroids, warm-start, crop HCR, extract patches.
# ---------------------------------------------------------------------------


def _preload_c2_subject(
    sid: str,
    *,
    pad_um: float = 400.0,
    k_neighbours: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
    """Load subject + pre-extract CZ and HCR-crop patches + k-NN graphs.

    The HCR crop pad is generous (400 µm by default) so that any later
    jitter on cz_init still lands within the crop. All tensors are
    kept on CPU as numpy arrays / float32 tensors.
    """
    from benchmark_data_loader import load_subject

    t0 = time.time()
    s = load_subject(sid)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ = default_warmstart_zyx(cz_um, hcr_um)

    hcr_keep_idx, hcr_crop = _crop_hcr(cz_init, hcr_um, pad_um=pad_um)

    # Coreg pairs in (cz_local_idx, hcr_local_idx in crop) space.
    cz_pos = {int(v): i for i, v in enumerate(cz_ids)}
    hcr_pos = {int(v): i for i, v in enumerate(hcr_ids)}
    hcr_remap = {int(g): i for i, g in enumerate(hcr_keep_idx)}
    pairs = []
    for _, r in s.coreg_table.iterrows():
        ci = cz_pos.get(int(r["cz_id"]))
        hi_global = hcr_pos.get(int(r["hcr_id"]))
        if ci is None or hi_global is None:
            continue
        hi_local = hcr_remap.get(int(hi_global))
        if hi_local is None:
            continue
        pairs.append((ci, hi_local))
    pairs = np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)

    cz_patches = extract_cz_patches(s, cz_um)          # (Ncz, 1, 16, 16, 16)
    hcr_patches = extract_hcr_patches(                 # (Ncrop, 1, 16, 16, 16)
        s, hcr_um[hcr_keep_idx], channel="488", level=2,
    )

    f_cz = _simple_features(cz_init, k=k_neighbours)
    f_hc = _simple_features(hcr_crop, k=k_neighbours)
    e_cz = _build_knn_graph(cz_init, k=k_neighbours)
    e_hc = _build_knn_graph(hcr_crop, k=k_neighbours)

    cache = dict(
        sid=sid,
        cz_um=cz_um,
        cz_init=cz_init,
        cz_ids=cz_ids,
        hcr_um=hcr_um,
        hcr_ids=hcr_ids,
        hcr_keep_idx=hcr_keep_idx,
        hcr_crop=hcr_crop,
        cz_patches=cz_patches,
        hcr_patches=hcr_patches,
        f_cz=f_cz,
        f_hc=f_hc,
        e_cz=e_cz,
        e_hc=e_hc,
        pairs=pairs,
    )
    if verbose:
        dt = time.time() - t0
        print(
            f"  c2_preload sid={sid} t={dt:.1f}s "
            f"cz={len(cz_um)} hcr_crop={len(hcr_crop)}/{len(hcr_um)} "
            f"pairs={len(pairs)}",
            flush=True,
        )
    return cache


def _to_tensor_batch(cache: dict, side: str) -> dict[str, torch.Tensor]:
    """Convert cached numpy arrays to torch tensors on CPU for one side."""
    if side == "cz":
        return dict(
            f=torch.as_tensor(cache["f_cz"], dtype=torch.float32),
            p=torch.as_tensor(cache["cz_patches"], dtype=torch.float32),
            e=torch.as_tensor(cache["e_cz"], dtype=torch.long),
        )
    elif side == "hc":
        return dict(
            f=torch.as_tensor(cache["f_hc"], dtype=torch.float32),
            p=torch.as_tensor(cache["hcr_patches"], dtype=torch.float32),
            e=torch.as_tensor(cache["e_hc"], dtype=torch.long),
        )
    raise ValueError(side)


# ---------------------------------------------------------------------------
# Stage 2: supervised LOSO training on real coreg_table pairs.
# ---------------------------------------------------------------------------


def _train_c2_stage2(
    model: C2Matcher,
    train_data: list[dict],
    *,
    n_iter: int = 2000,
    lr: float = 1e-4,
    rng_seed: int = 0,
    log_every: int = 50,
) -> list[float]:
    rng = np.random.default_rng(rng_seed)
    torch.manual_seed(rng_seed)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-convert each subject's tensors once (memory lives on CPU).
    tensor_cache = []
    for d in train_data:
        if len(d["pairs"]) < 5:
            continue
        if len(d["cz_init"]) < 30 or len(d["hcr_crop"]) < 30:
            continue
        tc = dict(
            a=_to_tensor_batch(d, "cz"),
            b=_to_tensor_batch(d, "hc"),
            pairs=torch.as_tensor(d["pairs"], dtype=torch.long),
            Na=int(len(d["cz_init"])),
            Nb=int(len(d["hcr_crop"])),
            sid=d["sid"],
        )
        tensor_cache.append(tc)
    if not tensor_cache:
        raise RuntimeError("No usable training subjects (each needs ≥ 5 pairs + "
                            "≥ 30 cells/side after crop).")

    losses: list[float] = []
    t_start = time.time()
    for it in range(n_iter):
        tc = tensor_cache[rng.integers(len(tensor_cache))]
        sim, _, _ = model(
            tc["a"]["f"], tc["a"]["p"], tc["a"]["e"],
            tc["b"]["f"], tc["b"]["p"], tc["b"]["e"],
        )
        loss = _pair_loss(sim, model.dustbin, tc["pairs"], tc["Na"], tc["Nb"])
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

        if (it + 1) % log_every == 0:
            dt = time.time() - t_start
            mean_loss = float(np.mean(losses[-log_every:]))
            print(
                f"  c2_stage2 it={it+1}/{n_iter} loss={mean_loss:.3f} "
                f"t={dt:.1f}s sid={tc['sid']} "
                f"Na={tc['Na']} Nb={tc['Nb']} n_pairs={len(tc['pairs'])}",
                flush=True,
            )
    return losses


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------


def _infer_c2(model: C2Matcher, held: dict) -> pd.DataFrame:
    cz_um = held["cz_um"]
    cz_ids = held["cz_ids"]
    hcr_um = held["hcr_um"]
    hcr_ids = held["hcr_ids"]
    hcr_keep_idx = held["hcr_keep_idx"]

    a = _to_tensor_batch(held, "cz")
    b = _to_tensor_batch(held, "hc")
    model.eval()
    with torch.no_grad():
        sim, _, _ = model(a["f"], a["p"], a["e"], b["f"], b["p"], b["e"])
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
            cz_id=int(cz_ids[i]),
            hcr_id=int(hcr_ids[j_abs]),
            confidence=float(conf),
            cz_x_um=float(cz_um[i, 2]),
            cz_y_um=float(cz_um[i, 1]),
            cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j_abs, 2]),
            hcr_y_um=float(hcr_um[j_abs, 1]),
            hcr_z_um=float(hcr_um[j_abs, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False)
        df = df.drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Top-level candidate entry point.
# ---------------------------------------------------------------------------


@register_candidate("C2_LOSO")
def run_c2_loso(
    s,
    *,
    n_train_iter: int = 2000,
    lr: float = 1e-4,
    k: int = 8,
    hidden: int = 96,
    n_layers: int = 4,
    cross_layers: int = 3,
    patch_dim: int = 64,
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
        print(
            f"  C2-LOSO: held-out={held} train_pool={train_subjects} "
            f"n_iter={n_train_iter}",
            flush=True,
        )

    t0 = time.time()
    train_data = [_preload_c2_subject(sid, verbose=verbose) for sid in train_subjects]
    held_data = _preload_c2_subject(held, verbose=verbose)
    if verbose:
        total_pairs = sum(len(d["pairs"]) for d in train_data)
        print(
            f"  C2-LOSO: preload t={time.time()-t0:.1f}s total_train_pairs={total_pairs}",
            flush=True,
        )

    hand_dim = train_data[0]["f_cz"].shape[1]
    torch.manual_seed(rng_seed)
    model = C2Matcher(
        hand_dim=hand_dim, patch_dim=patch_dim,
        hidden=hidden, n_layers=n_layers, cross_layers=cross_layers,
    )
    if verbose:
        print(
            f"  C2-LOSO: model hand_dim={hand_dim} patch_dim={patch_dim} "
            f"total_params={model.param_count:,}",
            flush=True,
        )

    losses = _train_c2_stage2(
        model, train_data, n_iter=n_train_iter, lr=lr, rng_seed=rng_seed,
    )

    df = _infer_c2(model, held_data)
    if verbose:
        print(f"  C2-LOSO: emitted {len(df)} pairs", flush=True)
    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=None,
        diagnostics=dict(
            held_out=held,
            train_subjects=train_subjects,
            n_train_iter=n_train_iter,
            lr=lr,
            hand_dim=hand_dim,
            patch_dim=patch_dim,
            model_params=int(model.param_count),
            train_loss_final=(float(losses[-1]) if losses else None),
            train_loss_mean50=(float(np.mean(losses[-50:])) if len(losses) >= 50 else None),
            n_hcr_kept=int(len(held_data["hcr_crop"])),
            n_emitted=int(len(df)),
        ),
    )
