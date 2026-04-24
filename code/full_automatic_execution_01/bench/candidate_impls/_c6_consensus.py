"""C6 — consensus-vote ensemble of P1 ⊕ P4 ⊕ P6.

S60 probe showed 2-vote consensus pairs have 1.3–2.2× the precision of
1-vote pairs across 3 tested subjects. C6 exploits this by preferring
the consensus hcr_id per cz_id when at least 2 of {P1, P4, P6} agree;
falls back to priority dispatch otherwise (same as C5).

Per-cz_id decision:
  1. Aggregate (method, hcr_id, confidence) across P1, P4, P6.
  2. If any hcr_id receives ≥ 2 votes → pick that hcr_id. Confidence =
     mean of contributing methods' confidences.
  3. Else → fall back to C5's priority rule.

Expected lift: cz_ids where C5's priority pick differs from the 2-vote
consensus get the consensus pair, which has higher precision.
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


def _consensus_per_cz(dfs_ordered):
    """dfs_ordered: list of (label, df) ordered by priority fallback.

    Returns merged df where each cz_id gets:
      - consensus hcr_id if ≥ 2 methods agree, confidence = mean(contrib)
      - else priority-ordered first method's pick
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
            grp2 = grp.assign(_pri=grp["_method"].map(priority_order))
            best = grp2.sort_values("_pri").iloc[0]
            row = best.drop(["_method", "_pri"]).to_dict()
            out_rows.append(row)
    df_out = pd.DataFrame(out_rows)
    if len(df_out):
        df_out = df_out.sort_values("cz_id").reset_index(drop=True)
    return df_out


@register_candidate("C6")
def run_c6(s, *, mid_priority=("P1", "P4", "P6"),
           dense_priority=("P4", "P1", "P6"),
           sparse_priority=("P1", "P4", "P6"),
           sparse_threshold: int = 10_000,
           dense_threshold: int = 20_000,
           p1_kwargs: "dict | None" = None,
           p4_kwargs: "dict | None" = None,
           p6_kwargs: "dict | None" = None,
           verbose: bool = True) -> CoregResult:
    """C5 with consensus-first, priority-fallback dispatch per cz_id."""
    from bench.candidate_impls._p1_teaser import run_p1
    from bench.candidate_impls._p4_spectral import run_p4
    from bench.candidate_impls._p6_bcpd import run_p6

    p1_kwargs = dict(p1_kwargs or {})
    p4_kwargs = dict(p4_kwargs or {})
    p6_kwargs = dict(p6_kwargs or {})

    n_hcr = len(s.hcr_gfp_df)
    if n_hcr < sparse_threshold:
        priority = sparse_priority
    elif n_hcr >= dense_threshold:
        priority = dense_priority
    else:
        priority = mid_priority

    if verbose:
        print(f"  c6: n_hcr_gfp={n_hcr} priority={priority}", flush=True)

    t_p1 = time.time(); r_p1 = run_p1(s, **p1_kwargs); w_p1 = time.time()-t_p1
    t_p4 = time.time(); r_p4 = run_p4(s, **p4_kwargs); w_p4 = time.time()-t_p4
    t_p6 = time.time(); r_p6 = run_p6(s, **p6_kwargs); w_p6 = time.time()-t_p6

    label_map = {"P1": r_p1.pairs_df, "P4": r_p4.pairs_df, "P6": r_p6.pairs_df}
    ordered = [(label, label_map[label]) for label in priority]

    merged = _consensus_per_cz(ordered)

    conf = float(merged["confidence"].median()) if len(merged) else 0.0

    return CoregResult(
        pairs_df=merged,
        confidence=conf,
        transform=r_p1.transform,
        diagnostics=dict(
            n_p1=int(len(label_map["P1"])), n_p4=int(len(label_map["P4"])),
            n_p6=int(len(label_map["P6"])), n_merged=int(len(merged)),
            wall_p1_s=w_p1, wall_p4_s=w_p4, wall_p6_s=w_p6,
            priority=list(priority),
            sparse_threshold=int(sparse_threshold),
            dense_threshold=int(dense_threshold),
            n_hcr_gfp=int(n_hcr),
        ),
    )
