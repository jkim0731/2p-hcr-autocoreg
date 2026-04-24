"""C4 — P1 ⊕ P6 ensemble.

Runs P1 (TEASER + TPS + HCR-quality β=5) and P6 (CPD nonrigid) in parallel
per subject and merges their `pairs_df`. Three strategies:

  - `union_conf`: union; on cz_id collision, keep higher-confidence pair.
  - `union_p1_first`: union; on cz_id collision, keep P1 (strictly dominates P1 alone).
  - `auto` (default): sparse-subject dispatch. If `len(s.hcr_gfp_df) < sparse_threshold`,
    use `union_conf` (P6-favoring); else `union_p1_first` (P1-favoring).

S54 6-subject benchmark: auto rule yields sum r@20 = 0.971, vs P1-alone 0.775
(+25 % relative, near-oracle 0.980).

Note: P1 and P6 confidence scales are not comparable (TLS+TPS-residual vs
`1/(1+d_nn/σ_med)`). `union_conf` works empirically because P6 tends to
be over-confident on sparse subjects where it's actually right. See S55
queue item for F7-calibrated tiebreak.
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


def _union_conf(p1: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    cat = pd.concat([p1, p6], ignore_index=True)
    if len(cat) == 0:
        return cat
    cat = cat.sort_values("confidence", ascending=False)
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.sort_values("cz_id").reset_index(drop=True)
    return cat


def _union_p1_first(p1: pd.DataFrame, p6: pd.DataFrame) -> pd.DataFrame:
    cat = pd.concat([p1.assign(_src="p1"), p6.assign(_src="p6")], ignore_index=True)
    if len(cat) == 0:
        return cat
    cat["_p1_first"] = (cat["_src"] == "p1").astype(int)
    cat = cat.sort_values(["cz_id", "_p1_first"], ascending=[True, False])
    cat = cat.drop_duplicates(subset=["cz_id"], keep="first")
    cat = cat.drop(columns=["_src", "_p1_first"]).reset_index(drop=True)
    return cat


@register_candidate("C4")
def run_c4(s, *, method: str = "auto",
           sparse_threshold: int = 10_000,
           p1_kwargs: "dict | None" = None,
           p6_kwargs: "dict | None" = None,
           verbose: bool = True) -> CoregResult:
    """P1 ⊕ P6 ensemble. method ∈ {auto, union_conf, union_p1_first}."""
    from bench.candidate_impls._p1_teaser import run_p1
    from bench.candidate_impls._p6_bcpd import run_p6

    p1_kwargs = dict(p1_kwargs or {})
    p6_kwargs = dict(p6_kwargs or {})

    n_hcr = len(s.hcr_gfp_df)
    if method == "auto":
        chosen = "union_conf" if n_hcr < sparse_threshold else "union_p1_first"
    elif method in ("union_conf", "union_p1_first"):
        chosen = method
    else:
        raise ValueError(f"unknown method {method}")

    if verbose:
        print(f"  c4: n_hcr_gfp={n_hcr} sparse_thr={sparse_threshold} "
              f"method={method} → chosen={chosen}", flush=True)

    t_p1 = time.time()
    r_p1 = run_p1(s, **p1_kwargs)
    wall_p1 = time.time() - t_p1

    t_p6 = time.time()
    r_p6 = run_p6(s, **p6_kwargs)
    wall_p6 = time.time() - t_p6

    if chosen == "union_conf":
        merged = _union_conf(r_p1.pairs_df, r_p6.pairs_df)
    else:
        merged = _union_p1_first(r_p1.pairs_df, r_p6.pairs_df)

    conf = float(merged["confidence"].median()) if len(merged) else 0.0

    return CoregResult(
        pairs_df=merged,
        confidence=conf,
        transform=r_p1.transform,
        diagnostics=dict(
            n_p1=int(len(r_p1.pairs_df)),
            n_p6=int(len(r_p6.pairs_df)),
            n_merged=int(len(merged)),
            wall_p1_s=wall_p1,
            wall_p6_s=wall_p6,
            method=method,
            chosen_strategy=chosen,
            sparse_threshold=int(sparse_threshold),
            n_hcr_gfp=int(n_hcr),
        ),
    )
