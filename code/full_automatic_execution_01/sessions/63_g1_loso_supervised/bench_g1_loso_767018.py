"""S63 fallback: rerun G1-LOSO for 767018 alone if main bench was cut."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

import bench.candidates  # noqa: F401
from bench.harness import CANDIDATES  # noqa: E402
from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from bench.candidate_impls._g1_loso_matcher import (  # noqa: E402
    _preload_subject, _train_supervised_loso, _crop_hcr,
    _simple_features, _build_knn_graph, _assignment,
    SIX_SUBJECTS,
)
import torch  # noqa: E402


def score(df, s):
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    hcr_by_id = dict(zip(hcr_ids, hcr_um))
    coreg = s.coreg_table
    gt = {int(r["cz_id"]): int(r["hcr_id"]) for _, r in coreg.iterrows()}
    hits = 0
    total = 0
    errs = []
    for _, r in df.iterrows():
        c = int(r["cz_id"])
        if c not in gt:
            continue
        total += 1
        gh = gt[c]
        if gh not in hcr_by_id:
            continue
        gt_pos = hcr_by_id[gh]
        pr_pos = np.array([r["hcr_z_um"], r["hcr_y_um"], r["hcr_x_um"]])
        d = np.linalg.norm(gt_pos - pr_pos)
        errs.append(d)
        if d <= 20.0:
            hits += 1
    denom = len(gt)
    r20 = hits / max(denom, 1)
    med = float(np.median(errs)) if errs else float("inf")
    return r20, med, hits, total, denom


def _inference(model, s, k: int = 8):
    from lib.centroid_helpers import default_warmstart_zyx

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
            hcr_x_um=float(hcr_um[j_abs, 2]),
            hcr_y_um=float(hcr_um[j_abs, 1]),
            hcr_z_um=float(hcr_um[j_abs, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("confidence", ascending=False).drop_duplicates("hcr_id", keep="first")
        df = df.sort_values("cz_id").reset_index(drop=True)
    return df, len(hcr_crop)


def main():
    sid = "767018"
    c5_val = 0.315018315018315

    print("=== PRELOAD 767018 + 5 training subjects ===", flush=True)
    t0 = time.time()
    preloaded = {}
    for s_id in SIX_SUBJECTS:
        preloaded[s_id] = _preload_subject(s_id, keep_subject=(s_id == sid))
        d = preloaded[s_id]
        print(f"  {s_id}: cz={len(d['cz_init'])} hcr={len(d['hcr_um'])} "
              f"pairs={len(d['pairs'])}", flush=True)
    print(f"  preload done in {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== {sid} (held-out) ===", flush=True)
    train_data = [preloaded[tid] for tid in SIX_SUBJECTS if tid != sid]
    t0 = time.time()
    model, losses = _train_supervised_loso(
        train_data, n_iter=2000, k_neighbours=8,
        hidden=96, n_layers=4, cross_layers=3,
        lr=1e-4, jitter_um=4.0, rng_seed=0,
    )
    train_t = time.time() - t0

    s = preloaded[sid]["subject"]
    t0 = time.time()
    df, n_hcr_kept = _inference(model, s, k=8)
    infer_t = time.time() - t0

    r20, med, hits, tot, denom = score(df, s)
    delta = r20 - c5_val
    train_loss_mean50 = float(np.mean(losses[-50:])) if len(losses) >= 50 else None
    print(
        f"  G1-LOSO r@20={r20:.3f} med={med:.1f}µm hits={hits}/{tot} "
        f"emitted={len(df)} train_t={train_t:.1f}s infer_t={infer_t:.1f}s "
        f"train_loss_mean50={train_loss_mean50}",
        flush=True,
    )
    print(f"  C5 r@20={c5_val:.3f} (baseline)", flush=True)
    print(f"  Δr@20 (G1-LOSO - C5) = {delta:+.3f}", flush=True)

    row = dict(
        subject=sid,
        c5_r20=c5_val,
        g1loso_r20=r20,
        g1loso_hits=hits,
        g1loso_n=len(df),
        g1loso_med=med,
        train_loss_final=(float(losses[-1]) if losses else None),
        train_loss_mean50=train_loss_mean50,
        n_hcr_kept=n_hcr_kept,
        delta=delta,
        train_s=train_t,
        infer_s=infer_t,
    )
    pd.DataFrame([row]).to_csv(
        "sessions/63_g1_loso_supervised/bench_g1_loso_767018.csv", index=False
    )


if __name__ == "__main__":
    main()
