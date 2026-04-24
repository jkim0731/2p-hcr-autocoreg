"""S38 v3 — does the B-config matcher produce real correspondences?

v2 showed a 96×4+3 matcher hits 56% val on synthetic warps at 5000 iter.
Now test on real benchmark: train per subject (5000 iter, B config), use
_simple_features (unified with training), then evaluate on real CZ↔HCR
coreg_table. Report recall / median error with the standard harness
metrics.
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from benchmark_data_loader import load_subject, cz_px_to_um, hcr_px_to_um  # noqa
from lib.centroid_helpers import centroids_um, default_warmstart_zyx  # noqa
from lib.synthetic_warps import sample_warped_pair  # noqa
from bench.candidate_impls._g1_gnn_matcher import (  # noqa
    GNNMatcher, _build_knn_graph, _simple_features, _assignment, _pair_loss
)


def train_matcher(hcr_um, n_iter=5000, hidden=96, n_layers=4, cross_layers=3,
                  k=8, seed=1, rng=None):
    rng = rng or np.random.default_rng(seed)
    torch.manual_seed(seed)
    f0 = _simple_features(hcr_um[:50], k=k)
    model = GNNMatcher(in_dim=f0.shape[1], hidden=hidden, n_layers=n_layers,
                       cross_layers=cross_layers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for it in range(n_iter):
        w = sample_warped_pair(hcr_um, rng=rng, cube_um=400.0)
        if len(w.correspondence) < 5:
            continue
        if len(w.source_um) <= k or len(w.warped_um) <= k:
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
    return model, losses


def gt_pairs(s):
    ct = s.coreg_table
    cz = s.cz_centroids.set_index("cz_id")
    hc = s.hcr_centroids.set_index("hcr_id")
    m = ct["cz_id"].isin(cz.index) & ct["hcr_id"].isin(hc.index)
    ct = ct[m]
    cz_um = cz_px_to_um(cz.loc[ct["cz_id"]][["z_px", "y_px", "x_px"]].values, s)
    hc_um = hcr_px_to_um(hc.loc[ct["hcr_id"]][["z_px", "y_px", "x_px"]].values, s)
    return ct["cz_id"].values, ct["hcr_id"].values, cz_um, hc_um


def infer(s, model, k=8):
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_init, _ = default_warmstart_zyx(cz_um, hcr_um)

    fc = _simple_features(cz_init, k=k)
    fg = _simple_features(hcr_um, k=k)
    e_cz = _build_knn_graph(cz_init, k=k)
    e_hcr = _build_knn_graph(hcr_um, k=k)

    model.eval()
    with torch.no_grad():
        sim, _, _ = model(
            torch.as_tensor(fc, dtype=torch.float32),
            torch.as_tensor(e_cz, dtype=torch.long),
            torch.as_tensor(fg, dtype=torch.float32),
            torch.as_tensor(e_hcr, dtype=torch.long),
        )
        P = _assignment(sim, model.dustbin, n_iter=30)
    P_core = P[:-1, :-1].numpy()
    dust_col = P[:-1, -1].numpy()

    rows = []
    for i in range(len(cz_ids)):
        j = int(np.argmax(P_core[i]))
        mass = float(P_core[i, j])
        db = float(dust_col[i])
        if mass < db or mass < 1e-4:
            continue
        rows.append(dict(cz_id=int(cz_ids[i]), hcr_id=int(hcr_ids[j]),
                         confidence=mass / (mass + db),
                         cz_um=cz_um[i], hcr_um=hcr_um[j]))
    return rows, cz_um, cz_ids, hcr_um, hcr_ids


def main():
    results = []
    for subj in ["782149"]:
        print(f"\n=== {subj} ===", flush=True)
        t0 = time.time()
        s = load_subject(subj)
        hcr_um, _ = centroids_um(s, "hcr_gfp")
        print(f"  n_hcr={len(hcr_um)}; training 5000 iter 96×4+3 ...", flush=True)
        t1 = time.time()
        model, losses = train_matcher(hcr_um, n_iter=5000, hidden=96,
                                       n_layers=4, cross_layers=3)
        print(f"  train {time.time()-t1:.0f}s  final_loss={losses[-1]:.2f}",
              flush=True)

        t2 = time.time()
        rows, cz_um, cz_ids, hcr_um, hcr_ids = infer(s, model)
        print(f"  infer {time.time()-t2:.1f}s  n_pred={len(rows)}", flush=True)
        # Compare to GT
        gt_cz, gt_hcr, gt_cz_um, gt_hcr_um = gt_pairs(s)
        pred_map = {r["cz_id"]: r["hcr_id"] for r in rows}
        hits = 0
        pos_errs = []
        for k, cid in enumerate(gt_cz):
            hid_pred = pred_map.get(int(cid))
            if hid_pred is None:
                continue
            if hid_pred == gt_hcr[k]:
                hits += 1
                pos_errs.append(0.0)
            else:
                j = int(np.where(hcr_ids == hid_pred)[0][0])
                pos_errs.append(float(np.linalg.norm(hcr_um[j] - gt_hcr_um[k])))
        pos_errs = np.array(pos_errs) if pos_errs else np.zeros(0)
        n_gt_covered = int(sum(int(cid) in pred_map for cid in gt_cz))
        n_lt50 = int((pos_errs < 50).sum()) if len(pos_errs) else 0
        n_lt10 = int((pos_errs < 10).sum()) if len(pos_errs) else 0
        med = float(np.median(pos_errs)) if len(pos_errs) else float('nan')
        print(f"  n_gt={len(gt_cz)} covered={n_gt_covered} exact_id={hits} "
              f"n_lt10={n_lt10} n_lt50={n_lt50} med={med:.0f} µm", flush=True)
        results.append(dict(
            subject=subj, n_pred=len(rows), n_gt=len(gt_cz), covered=n_gt_covered,
            exact_id=hits, n_lt10=n_lt10, n_lt50=n_lt50, median_err=med,
            total_s=round(time.time()-t0, 1),
        ))

    print("\n=== SUMMARY ===")
    print(pd.DataFrame(results).to_string(index=False))
    pd.DataFrame(results).to_csv(
        "/root/capsule/code/full_automatic_execution_01/sessions/38_g1_rewrite/real_782149.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
