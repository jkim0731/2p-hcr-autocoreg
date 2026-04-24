"""B3 — TPS-expansion seeded from C5's highest-confidence picks.

S23/S24's original B1/B2 failed because B1's "hull-emptiness + tight
spacing + near-surface" seed heuristic produced a constellation that
was not in the GT set on 788406, so B2's TPS bent around a wrong
anchor and rejected every downstream candidate.

B3 replaces the seed source: start from C5's emitted pairs (which
already have meaningful recall on the validation roster) and pick the
top-K most confident pairs, spatially diverse in CZ.  RANSAC-style
TPS fit iteratively drops the worst outlier until residual
distribution stabilises, then TPS-predicts every CZ cell's HCR
location.  Greedy one-to-one match by nearest HCR GFP+ within τ µm.

This trades C5's per-method putative-matching logic for a
global-consistency TPS-based re-matcher.  Expected behaviour:

- on pairs C5 already emits with high confidence: same or slightly
  better (TPS smooths local noise);
- on pairs C5 emits with lower confidence: may replace with a
  TPS-consistent nearest neighbour (which could be correct or wrong);
- may add pairs C5 skipped (CZ cells with no P1/P4/P6 vote).
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
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist


def _fit_tps(A: np.ndarray, B: np.ndarray):
    """A, B: (N, 3) zyx.  Returns per-axis Rbf list."""
    return [Rbf(A[:, 0], A[:, 1], A[:, 2], B[:, ax], function="thin_plate")
            for ax in range(3)]


def _predict_tps(rbf, pts: np.ndarray) -> np.ndarray:
    return np.stack([r(pts[:, 0], pts[:, 1], pts[:, 2]) for r in rbf], axis=-1)


def _robust_seed(A: np.ndarray, B: np.ndarray, *, keep_frac=0.8,
                 max_drop=5) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively drop the worst single seed until residuals stabilise.

    Uses leave-one-out: for each seed, fit TPS on the others, compute
    residual at the held-out point.  Drop the one with the largest LOO
    residual until we've dropped ``max_drop`` or only ``keep_frac`` remain.
    """
    A = A.copy(); B = B.copy()
    n = len(A)
    n_keep = max(4, int(np.ceil(keep_frac * n)))
    drops = 0
    while len(A) > n_keep and drops < max_drop:
        m = len(A)
        loo = np.zeros(m)
        for i in range(m):
            idx = np.arange(m) != i
            try:
                rbf = _fit_tps(A[idx], B[idx])
                pred = _predict_tps(rbf, A[i:i+1])
                loo[i] = float(np.linalg.norm(pred[0] - B[i]))
            except Exception:
                loo[i] = np.inf
        worst = int(np.argmax(loo))
        if not np.isfinite(loo[worst]) or loo[worst] < 10:  # stable
            break
        A = np.delete(A, worst, axis=0)
        B = np.delete(B, worst, axis=0)
        drops += 1
    return A, B


@register_candidate("B3")
def run_b3(s, *,
           seed_k: int = 30,
           seed_min_dist_um: float = 50.0,
           tau_um: float = 40.0,
           verbose: bool = True) -> CoregResult:
    from bench.candidate_impls._c5_p1_p4_p6_ensemble import run_c5

    c5 = run_c5(s)
    c5_df = c5.pairs_df
    if c5_df is None or len(c5_df) < 8:
        return CoregResult(pairs_df=pd.DataFrame(), confidence=0.0,
                           transform=None,
                           diagnostics=dict(reason="c5 too small",
                                            n_c5=0 if c5_df is None else len(c5_df)))

    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    cz_pos = {int(v): i for i, v in enumerate(cz_ids)}
    hcr_pos = {int(v): i for i, v in enumerate(hcr_ids)}

    # Sort C5 by confidence desc, pick top-k that are ≥ seed_min_dist_um
    # apart in CZ space.
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

    if verbose:
        print(f"  B3: C5 emitted {len(c5_df)}; seeded {len(A)} → kept {len(A_kept)}",
              flush=True)

    # Fit TPS on kept seeds; predict every CZ cell.
    rbf = _fit_tps(A_kept, B_kept)
    cz_pred = _predict_tps(rbf, cz_um)

    # For each CZ cell, find nearest HCR GFP+; keep if within τ.
    # Enforce one-to-one via greedy sort by residual.
    D = cdist(cz_pred, hcr_um)  # (n_cz, n_hcr)
    nn = np.argmin(D, axis=1)
    res = D[np.arange(len(cz_pred)), nn]
    # Sort CZ cells by residual asc; accept greedily with unique HCR constraint.
    order = np.argsort(res)
    used_hr = set()
    rows = []
    for i in order:
        if res[i] > tau_um:
            break
        j = int(nn[i])
        if j in used_hr:
            continue
        used_hr.add(j)
        rows.append(dict(
            cz_id=int(cz_ids[i]),
            hcr_id=int(hcr_ids[j]),
            confidence=float(1.0 / (1.0 + np.exp((res[i] - 20.0) / 10.0))),
            cz_x_um=float(cz_um[i, 2]),
            cz_y_um=float(cz_um[i, 1]),
            cz_z_um=float(cz_um[i, 0]),
            hcr_x_um=float(hcr_um[j, 2]),
            hcr_y_um=float(hcr_um[j, 1]),
            hcr_z_um=float(hcr_um[j, 0]),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("cz_id").reset_index(drop=True)

    if verbose:
        print(f"  B3: emitted {len(df)} pairs (tau={tau_um} µm)", flush=True)

    return CoregResult(
        pairs_df=df,
        confidence=float(df["confidence"].median()) if len(df) else 0.0,
        transform=c5.transform,
        diagnostics=dict(
            n_c5=int(len(c5_df)),
            n_seed_spatial=int(len(A)),
            n_seed_kept=int(len(A_kept)),
            n_emitted=int(len(df)),
            tau_um=float(tau_um),
            seed_min_dist_um=float(seed_min_dist_um),
        ),
    )
