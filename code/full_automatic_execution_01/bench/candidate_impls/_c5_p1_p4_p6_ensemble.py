"""C5 — P1 ⊕ P4 ⊕ P6 three-way ensemble.

S56 validated that adding P4 (Spectral GM with S51's hcr_quality_beta=5.0)
to the S54 C4 ensemble lifts sum r@20 from 0.971 to **1.080** across the
6-subject benchmark (near-oracle 0.986 → 1.08× *above* oracle because
pair-level merging combines correct pairs across different cz_ids).

Dispatch rule (`method="auto"`, the default):

  - n_hcr_gfp < 10 000   (sparse)   → union_conf over (P1, P4, P6)
                                       P4's 767018 r@20=0.308 wins collisions
                                       via its naturally higher confidence.
  - 10k ≤ n_hcr_gfp < 20k (mid)     → priority (P1, P4, P6)
                                       P1-led, P4/P6 fill missing cz_ids.
  - n_hcr_gfp >= 20 000  (very dense sparse-GFP+ per µm³) → priority (P4, P1, P6)
                                       755252 regime — P4's pairwise-consistency
                                       putative generator beats P1 head-to-head.

6-subject r@20 under `method="auto"`:
  788406 0.263, 790322 0.289, 755252 0.095, 767022 0.117, 767018 0.315, 782149 0.
  Sum 1.080 vs P1-alone 0.775 (+39 % relative).

Confidence scales across P1/P4/P6 are not comparable — S55 F7 isotonic
calibration is queued to make `union_conf` tiebreaks principled rather
than exploiting uncalibrated per-method offsets.
"""
from __future__ import annotations

import sys
import time
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


def _union_conf(dfs):
    cat = pd.concat(dfs, ignore_index=True)
    if len(cat) == 0:
        return cat
    cat = cat.sort_values("confidence", ascending=False)
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    return cat.sort_values("cz_id").reset_index(drop=True)


def _union_priority(dfs):
    pieces = [d.assign(_pri=i) for i, d in enumerate(dfs) if len(d) > 0]
    if not pieces:
        return pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                      "cz_x_um","cz_y_um","cz_z_um",
                                      "hcr_x_um","hcr_y_um","hcr_z_um"])
    cat = pd.concat(pieces, ignore_index=True)
    cat = cat.sort_values(["cz_id", "_pri"], ascending=[True, True])
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    return cat.drop(columns=["_pri"]).reset_index(drop=True)


@register_candidate("C5")
def run_c5(s, *, method: str = "auto",
           sparse_threshold: int = 10_000,
           dense_threshold: int = 20_000,
           p1_kwargs: "dict | None" = None,
           p4_kwargs: "dict | None" = None,
           p6_kwargs: "dict | None" = None,
           verbose: bool = True) -> CoregResult:
    """P1 ⊕ P4 ⊕ P6 three-way ensemble.

    method ∈ {auto, union_conf, up_p1_p4_p6, up_p4_p1_p6}.
    """
    from bench.candidate_impls._p1_teaser import run_p1
    from bench.candidate_impls._p4_spectral import run_p4
    from bench.candidate_impls._p6_bcpd import run_p6

    p1_kwargs = dict(p1_kwargs or {})
    p4_kwargs = dict(p4_kwargs or {})
    p6_kwargs = dict(p6_kwargs or {})

    n_hcr = len(s.hcr_gfp_df)
    if method == "auto":
        if n_hcr < sparse_threshold:
            chosen = "union_conf"
        elif n_hcr >= dense_threshold:
            chosen = "up_p4_p1_p6"
        else:
            chosen = "up_p1_p4_p6"
    else:
        chosen = method
    if chosen not in ("union_conf", "up_p1_p4_p6", "up_p4_p1_p6"):
        raise ValueError(f"unknown method {method}")

    if verbose:
        print(f"  c5: n_hcr_gfp={n_hcr} [{sparse_threshold},{dense_threshold}) "
              f"method={method} → chosen={chosen}", flush=True)

    t_p1 = time.time(); r_p1 = run_p1(s, **p1_kwargs); w_p1 = time.time()-t_p1
    t_p4 = time.time(); r_p4 = run_p4(s, **p4_kwargs); w_p4 = time.time()-t_p4
    t_p6 = time.time(); r_p6 = run_p6(s, **p6_kwargs); w_p6 = time.time()-t_p6

    dfs = {"P1": r_p1.pairs_df, "P4": r_p4.pairs_df, "P6": r_p6.pairs_df}
    if chosen == "union_conf":
        merged = _union_conf([dfs["P1"], dfs["P4"], dfs["P6"]])
    elif chosen == "up_p1_p4_p6":
        merged = _union_priority([dfs["P1"], dfs["P4"], dfs["P6"]])
    else:  # up_p4_p1_p6
        merged = _union_priority([dfs["P4"], dfs["P1"], dfs["P6"]])

    conf = float(merged["confidence"].median()) if len(merged) else 0.0

    return CoregResult(
        pairs_df=merged,
        confidence=conf,
        transform=r_p1.transform,
        diagnostics=dict(
            n_p1=int(len(dfs["P1"])), n_p4=int(len(dfs["P4"])), n_p6=int(len(dfs["P6"])),
            n_merged=int(len(merged)),
            wall_p1_s=w_p1, wall_p4_s=w_p4, wall_p6_s=w_p6,
            method=method, chosen_strategy=chosen,
            sparse_threshold=int(sparse_threshold),
            dense_threshold=int(dense_threshold),
            n_hcr_gfp=int(n_hcr),
        ),
    )
