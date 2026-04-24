"""S59 — F8 CZ-aware domain randomization probe on 788406.

S58 diagnosed that G1's training-to-inference gap is synthetic→real.
Training source (HCR subsample + TPS warp) doesn't look enough like a
real CZ centroid cloud. This probe adds three CZ-like perturbations to
the source side of the asymmetric F8 sampler:

  1. Gaussian position noise σ=8 µm (simulates CZ segmentation drift).
  2. Spatial dropout gradient: keep source cells with higher probability
     in the upper half of the cube (simulates CZ's partial Z coverage).
  3. Higher source drop rate 0.5 (matches ~50% CZ match rate).

Retrain G1 (500 iter, stable LR=1e-4), then infer on real 788406 pairs.
Fail-fast criterion: r@20 ≤ 0.02 → abandon CZ-aware F8 and accept
C5=1.080 plateau as final centroid-only ceiling.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._g1_gnn_matcher import (  # noqa: E402
    GNNMatcher, _build_knn_graph, _pair_loss, _simple_features, _assignment)
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa: E402
from lib.synthetic_warps import sample_asymmetric_warped_pair  # noqa: E402


def sample_cz_aware(points_um, rng, *, source_cube_um=400.0,
                    target_margin_um=400.0, source_n_target=900,
                    target_n_cap=12000, source_drop_rate=0.5,
                    source_noise_um=8.0, spatial_gradient=True):
    """Asymmetric warp with extra CZ-aware source perturbations.

    Source-side:
      - Drop rate bumped to 0.5 (CZ match-rate proxy).
      - Gaussian noise σ=source_noise_um applied to source positions.
      - Spatial gradient: dropout rate higher in lower Z half of cube.
    """
    w = sample_asymmetric_warped_pair(
        points_um, rng=rng,
        source_cube_um=source_cube_um, target_margin_um=target_margin_um,
        source_n_target=source_n_target, target_n_cap=target_n_cap,
        source_drop_rate=source_drop_rate)
    if len(w.source_um) == 0:
        return w
    # Add position noise to source side
    noise = rng.normal(0, source_noise_um, size=w.source_um.shape)
    src_perturbed = w.source_um + noise
    # Spatial gradient dropout on source: cells deeper in cube get dropped more
    if spatial_gradient and len(src_perturbed) > 10:
        z = src_perturbed[:, 0]
        z_rel = (z - z.min()) / (z.max() - z.min() + 1e-9)
        keep_p = 1.0 - 0.4 * z_rel  # 1.0 at top, 0.6 at bottom
        keep = rng.random(len(src_perturbed)) < keep_p
        src_perturbed = src_perturbed[keep]
        # Remap correspondences
        old_to_new = -np.ones(len(w.source_um), dtype=int)
        old_to_new[np.where(keep)[0]] = np.arange(int(keep.sum()))
        new_corr = []
        for s_i, t_j in w.correspondence:
            if old_to_new[s_i] >= 0:
                new_corr.append((old_to_new[s_i], t_j))
        w.correspondence = (np.array(new_corr, dtype=int) if new_corr
                            else np.empty((0, 2), int))
    # Replace
    w = type(w)(
        source_um=src_perturbed, warped_um=w.warped_um,
        correspondence=w.correspondence,
        R=w.R, scales=w.scales, translation=w.translation,
        tps_metadata=w.tps_metadata)
    return w


def train(pts_hcr_um, n_iter=500, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def _sample():
        return sample_cz_aware(pts_hcr_um, rng=rng)

    w = _sample()
    while len(w.correspondence) < 5:
        w = _sample()
    in_dim = _simple_features(w.source_um).shape[1]
    model = GNNMatcher(in_dim=in_dim, hidden=96, n_layers=4, cross_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

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
        sim, _, _ = model(
            torch.as_tensor(f_src, dtype=torch.float32),
            torch.as_tensor(e_src, dtype=torch.long),
            torch.as_tensor(f_dst, dtype=torch.float32),
            torch.as_tensor(e_dst, dtype=torch.long))
        loss = _pair_loss(sim, model.dustbin,
                          torch.as_tensor(w.correspondence, dtype=torch.long),
                          len(w.source_um), len(w.warped_um))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (it + 1) % 50 == 0:
            print(f"  it={it+1:4d} loss={np.mean(losses[-50:]):.3f} "
                  f"n_src={len(w.source_um)} n_tgt={len(w.warped_um)} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    return model, losses


def infer_and_score(model, s):
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ = default_warmstart_zyx(cz_um, hcr_um)

    fc = _simple_features(cz_init)
    fg = _simple_features(hcr_um)
    e_cz = _build_knn_graph(cz_init, k=8)
    e_hcr = _build_knn_graph(hcr_um, k=8)
    with torch.no_grad():
        sim, _, _ = model(
            torch.as_tensor(fc, dtype=torch.float32),
            torch.as_tensor(e_cz, dtype=torch.long),
            torch.as_tensor(fg, dtype=torch.float32),
            torch.as_tensor(e_hcr, dtype=torch.long))
        P = _assignment(sim, model.dustbin, n_iter=30)
    P_core = P[:-1, :-1].numpy()
    rows = []
    for i in range(len(cz_ids)):
        j = int(np.argmax(P_core[i]))
        mass = float(P_core[i, j])
        dm = float(P[i, -1].item())
        if mass < dm or mass < 1e-4:
            continue
        conf = mass / (mass + dm)
        rows.append(dict(
            cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]), confidence=conf,
            hcr_x_um=float(hcr_um[j, 2]), hcr_y_um=float(hcr_um[j, 1]),
            hcr_z_um=float(hcr_um[j, 0])))
    df = pd.DataFrame(rows)
    if not len(df):
        print("  NO PAIRS EMITTED", flush=True)
        return df, 0.0

    coreg = s.coreg_table
    gt = {int(r['cz_id']): int(r['hcr_id']) for _, r in coreg.iterrows()}
    hcr_by_id = dict(zip(hcr_ids, hcr_um))
    hits = 0; total = 0
    for _, r in df.iterrows():
        c = int(r['cz_id'])
        if c not in gt:
            continue
        total += 1
        gh = gt[c]
        if gh not in hcr_by_id:
            continue
        gt_pos = hcr_by_id[gh]
        pr_pos = np.array([r['hcr_z_um'], r['hcr_y_um'], r['hcr_x_um']])
        if np.linalg.norm(gt_pos - pr_pos) <= 20.0:
            hits += 1
    r20 = hits / max(total, 1)
    print(f"  n_pred={len(df)} n_in_gt={total} hits@20={hits} r@20={r20:.3f}",
          flush=True)
    return df, r20


def main():
    sid = "788406"
    print(f"=== S59 CZ-aware F8 probe on {sid} ===", flush=True)
    s = load_subject(sid)
    hcr_um, _ = centroids_um(s, "hcr_gfp")
    print(f"n_hcr_gfp={len(hcr_um)}", flush=True)

    model, losses = train(hcr_um, n_iter=500, seed=0)
    print(f"\nTraining complete. initial50_mean={np.mean(losses[:50]):.3f} "
          f"final50_mean={np.mean(losses[-50:]):.3f} "
          f"descent={np.mean(losses[:50])-np.mean(losses[-50:]):.3f}",
          flush=True)

    print(f"\n--- inference on real {sid} ---", flush=True)
    df, r20 = infer_and_score(model, s)

    print(f"\n=== VERDICT: r@20={r20:.3f} ===", flush=True)
    if r20 <= 0.02:
        print("Fail-fast criterion MET. Abandon CZ-aware F8.", flush=True)
    elif r20 <= 0.10:
        print("Marginal — probably not enough to move plateau.", flush=True)
    else:
        print(f"PROMISING (>0.10). Escalate to full 6-subject bench.",
              flush=True)


if __name__ == "__main__":
    main()
