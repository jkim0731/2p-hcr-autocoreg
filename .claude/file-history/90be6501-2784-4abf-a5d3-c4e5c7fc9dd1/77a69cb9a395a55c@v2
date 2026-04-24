"""B3b — additive: C5 ∪ TPS-for-C5-unmatched-CZ.

B3 replaced C5 with TPS-based nearest-neighbour.  Median residual 80-111 µm
(>> 20 µm accept threshold) means the TPS fit on 23 C5-seed pairs does not
absorb the nonrigid warp; B3 trades specific C5 wins for generic NN picks.

B3b is conservative: keep every C5 pair as-is, use TPS only to propose
new pairs for CZ cells C5 did not emit a match for.  Cannot lower recall;
can only raise it if TPS proposals on C5-gap CZs beat the C5-missed rate.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult
from lib.centroid_helpers import centroids_um
from bench.candidate_impls._b3_c5_seed_tps import _fit_tps, _predict_tps, _robust_seed


@register_candidate("B3b")
def run_b3b(s, *,
            seed_k: int = 30,
            seed_min_dist_um: float = 50.0,
            tau_um: float = 25.0,
            verbose: bool = True) -> CoregResult:
    from bench.candidate_impls._c5_p1_p4_p6_ensemble import run_c5

    c5 = run_c5(s)
    c5_df = c5.pairs_df
    if c5_df is None or len(c5_df) < 8:
        return CoregResult(pairs_df=pd.DataFrame() if c5_df is None else c5_df,
                           confidence=c5.confidence,
                           transform=c5.transform,
                           diagnostics=dict(reason="c5 too small",
                                            n_c5=0 if c5_df is None else len(c5_df)))

    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_pos = {int(v): i for i, v in enumerate(cz_ids)}
    hcr_pos = {int(v): i for i, v in enumerate(hcr_ids)}

    # Which CZ cells did C5 already emit a match for?
    c5_cz = set(int(v) for v in c5_df["cz_id"].values)
    c5_hr_used = set(int(v) for v in c5_df["hcr_id"].values)

    c5_sorted = c5_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    seed_cz = []
    seed_hr = []
    for _, row in c5_sorted.iterrows():
        ci = cz_pos.get(int(row["cz_id"]))
        hi = hcr_pos.get(int(row["hcr_id"]))
        if ci is None or hi is None:
            continue
        p_cz = cz_um[ci]
        if all(np.linalg.norm(p_cz - cz_um[k]) >= seed_min_dist_um
               for k in seed_cz):
            seed_cz.append(ci)
            seed_hr.append(hi)
        if len(seed_cz) >= seed_k:
            break

    if len(seed_cz) < 6:
        return CoregResult(pairs_df=c5_df, confidence=c5.confidence,
                           transform=c5.transform,
                           diagnostics=dict(reason="not enough spatial seeds",
                                            n_seed=len(seed_cz),
                                            n_c5=len(c5_df)))

    A = cz_um[seed_cz]; B = hcr_um[seed_hr]
    A_kept, B_kept = _robust_seed(A, B, keep_frac=0.75, max_drop=8)

    rbf = _fit_tps(A_kept, B_kept)

    # Predict ONLY for CZ cells C5 did not emit a match for.
    gap_idx = [i for i, cid in enumerate(cz_ids) if int(cid) not in c5_cz]
    if not gap_idx:
        if verbose:
            print(f"  B3b: C5 covered all CZ, no gaps to fill", flush=True)
        return CoregResult(pairs_df=c5_df, confidence=c5.confidence,
                           transform=c5.transform,
                           diagnostics=dict(reason="no gap",
                                            n_c5=len(c5_df)))

    gap_cz = cz_um[gap_idx]
    gap_pred = _predict_tps(rbf, gap_cz)

    # Distances to HCR cells (restrict to not-yet-used by C5).
    free_hr = [j for j, hid in enumerate(hcr_ids) if int(hid) not in c5_hr_used]
    if not free_hr:
        return CoregResult(pairs_df=c5_df, confidence=c5.confidence,
                           transform=c5.transform,
                           diagnostics=dict(reason="no free hcr",
                                            n_c5=len(c5_df)))

    free_hr = np.array(free_hr, dtype=int)
    free_hr_um = hcr_um[free_hr]

    from scipy.spatial.distance import cdist
    D = cdist(gap_pred, free_hr_um)
    nn = np.argmin(D, axis=1)
    res = D[np.arange(len(gap_pred)), nn]

    order = np.argsort(res)
    used_hr_local = set()
    added_rows = []
    for rank in order:
        if res[rank] > tau_um:
            break
        j_free = int(nn[rank])
        if j_free in used_hr_local:
            continue
        used_hr_local.add(j_free)
        j_abs = int(free_hr[j_free])
        i_abs = int(gap_idx[rank])
        added_rows.append(dict(
            cz_id=int(cz_ids[i_abs]),
            hcr_id=int(hcr_ids[j_abs]),
            confidence=float(1.0 / (1.0 + np.exp((res[rank] - 15.0) / 8.0))),
            cz_x_um=float(cz_um[i_abs, 2]),
            cz_y_um=float(cz_um[i_abs, 1]),
            cz_z_um=float(cz_um[i_abs, 0]),
            hcr_x_um=float(hcr_um[j_abs, 2]),
            hcr_y_um=float(hcr_um[j_abs, 1]),
            hcr_z_um=float(hcr_um[j_abs, 0]),
        ))

    added_df = pd.DataFrame(added_rows)
    merged = pd.concat([c5_df, added_df], ignore_index=True) if len(added_df) else c5_df
    if len(merged):
        merged = merged.sort_values("cz_id").reset_index(drop=True)

    if verbose:
        print(f"  B3b: C5={len(c5_df)} + TPS-gap={len(added_df)} (from "
              f"{len(gap_idx)} gaps, τ={tau_um}µm) → total {len(merged)}",
              flush=True)

    return CoregResult(
        pairs_df=merged,
        confidence=float(merged["confidence"].median()) if len(merged) else 0.0,
        transform=c5.transform,
        diagnostics=dict(
            n_c5=int(len(c5_df)),
            n_seed_spatial=int(len(A)),
            n_seed_kept=int(len(A_kept)),
            n_gap=int(len(gap_idx)),
            n_added=int(len(added_df)),
            tau_um=float(tau_um),
            seed_min_dist_um=float(seed_min_dist_um),
        ),
    )
