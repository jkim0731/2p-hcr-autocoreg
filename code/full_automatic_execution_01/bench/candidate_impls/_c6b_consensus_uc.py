"""C6b — consensus-first with C5's per-density fallback rule.

S60 v1 showed C6 (consensus-first + priority fallback) loses -0.080 vs C5.
Loss decomposes as sparse -0.044 (fallback divergence — C5 uses union_conf
on sparse, C6 uses priority) + mid/dense -0.036 (consensus override of
P1's certified picks with P4+P6 shared-error-mode picks).

C6b isolates "consensus-first value" by matching C5's density-dispatch
fallback exactly. If C6b still loses vs C5, consensus-first is
refuted. If C6b matches or beats C5, the S60 v1 loss was purely from
fallback-mode divergence and consensus-first has merit.

Per-cz_id decision:
  1. If any hcr_id has ≥ 2 votes among {P1, P4, P6} → pick it.
  2. Else, fallback per C5's rule:
       sparse (<10k hcr_gfp)  → union_conf (highest raw confidence)
       mid (10k–20k)           → priority(P1, P4, P6)
       dense (≥20k)            → priority(P4, P1, P6)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench.harness import register_candidate, CoregResult


def _consensus_then(dfs_ordered, *, fallback: str):
    """dfs_ordered: list of (label, df) where priority=rank in list.

    fallback ∈ {'priority', 'union_conf'}.
    """
    records = []
    for label, df in dfs_ordered:
        if len(df) == 0:
            continue
        d = df.copy()
        d["_method"] = label
        records.append(d)
    if not records:
        return pd.DataFrame(columns=["cz_id","hcr_id","confidence",
                                      "cz_x_um","cz_y_um","cz_z_um",
                                      "hcr_x_um","hcr_y_um","hcr_z_um"])
    cat = pd.concat(records, ignore_index=True)
    priority_order = {label: i for i, (label, _) in enumerate(dfs_ordered)}

    out_rows = []
    for cz_id, grp in cat.groupby("cz_id"):
        vote_counts = grp.groupby("hcr_id")["_method"].nunique()
        max_votes = vote_counts.max()
        if max_votes >= 2:
            consensus_hids = vote_counts[vote_counts == max_votes].index.tolist()
            if len(consensus_hids) > 1:
                best_hid = None; best_pri = 1e9
                for hid in consensus_hids:
                    sub = grp[grp["hcr_id"] == hid]
                    pri = min(priority_order[m] for m in sub["_method"].unique())
                    if pri < best_pri:
                        best_pri = pri; best_hid = hid
                chosen_hid = best_hid
            else:
                chosen_hid = int(consensus_hids[0])
            sub = grp[grp["hcr_id"] == chosen_hid]
            conf_mean = float(sub["confidence"].mean())
            row = sub.iloc[0].drop("_method").to_dict()
            row["hcr_id"] = int(chosen_hid)
            row["confidence"] = conf_mean
            out_rows.append(row)
        else:
            if fallback == "priority":
                grp2 = grp.assign(_pri=grp["_method"].map(priority_order))
                best = grp2.sort_values("_pri").iloc[0]
            elif fallback == "union_conf":
                best = grp.sort_values("confidence", ascending=False).iloc[0]
            else:
                raise ValueError(f"unknown fallback {fallback}")
            row = best.drop("_method").to_dict()
            if "_pri" in row:
                row.pop("_pri")
            out_rows.append(row)
    df_out = pd.DataFrame(out_rows)
    if len(df_out):
        df_out = df_out.sort_values("cz_id").reset_index(drop=True)
    return df_out


@register_candidate("C6b")
def run_c6b(s, *, sparse_threshold: int = 10_000,
            dense_threshold: int = 20_000,
            p1_kwargs: "dict | None" = None,
            p4_kwargs: "dict | None" = None,
            p6_kwargs: "dict | None" = None,
            verbose: bool = True) -> CoregResult:
    """Consensus-first with C5's density-regime fallback."""
    from bench.candidate_impls._p1_teaser import run_p1
    from bench.candidate_impls._p4_spectral import run_p4
    from bench.candidate_impls._p6_bcpd import run_p6

    p1_kwargs = dict(p1_kwargs or {})
    p4_kwargs = dict(p4_kwargs or {})
    p6_kwargs = dict(p6_kwargs or {})

    n_hcr = len(s.hcr_gfp_df)
    if n_hcr < sparse_threshold:
        priority = ("P1", "P4", "P6")
        fallback = "union_conf"
    elif n_hcr >= dense_threshold:
        priority = ("P4", "P1", "P6")
        fallback = "priority"
    else:
        priority = ("P1", "P4", "P6")
        fallback = "priority"

    if verbose:
        print(f"  c6b: n_hcr_gfp={n_hcr} priority={priority} fallback={fallback}",
              flush=True)

    t_p1 = time.time(); r_p1 = run_p1(s, **p1_kwargs); w_p1 = time.time()-t_p1
    t_p4 = time.time(); r_p4 = run_p4(s, **p4_kwargs); w_p4 = time.time()-t_p4
    t_p6 = time.time(); r_p6 = run_p6(s, **p6_kwargs); w_p6 = time.time()-t_p6

    label_map = {"P1": r_p1.pairs_df, "P4": r_p4.pairs_df, "P6": r_p6.pairs_df}
    ordered = [(label, label_map[label]) for label in priority]

    merged = _consensus_then(ordered, fallback=fallback)

    conf = float(merged["confidence"].median()) if len(merged) else 0.0

    return CoregResult(
        pairs_df=merged,
        confidence=conf,
        transform=r_p1.transform,
        diagnostics=dict(
            n_p1=int(len(label_map["P1"])), n_p4=int(len(label_map["P4"])),
            n_p6=int(len(label_map["P6"])), n_merged=int(len(merged)),
            wall_p1_s=w_p1, wall_p4_s=w_p4, wall_p6_s=w_p6,
            priority=list(priority), fallback=fallback,
            sparse_threshold=int(sparse_threshold),
            dense_threshold=int(dense_threshold),
            n_hcr_gfp=int(n_hcr),
        ),
    )
