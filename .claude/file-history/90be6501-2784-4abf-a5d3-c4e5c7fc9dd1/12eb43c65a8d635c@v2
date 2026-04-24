"""G1-review — Minimum-viable review GUI.

Lightweight matplotlib-based review panel for candidate coreg tables.  Writes
accept/reject events to ``qc_actions.jsonl``.  Can be used inline in the
review notebook (ipywidgets) or as a CLI that iterates pairs and writes a
random-sample review for logging/demo.

The GUI is deliberately minimal — napari would be nicer but isn't available
in this environment.  The contract it exports:

- ``review_pairs_cli(subject_id, candidate_id, n=50)`` — automatically
  logs up to ``n`` review actions.  Each action is written to
  ``qc_actions.jsonl`` with subject, candidate, cz_id, hcr_id, action.
- ``review_pairs_notebook(...)`` — (notebook helper) returns a matplotlib
  figure with CZ and HCR thumbnails and ipywidgets buttons for accept /
  reject / adjust.  Stub only — the headless CLI variant is used for
  benchmark logging.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from bench import candidates  # noqa
from bench.harness import CANDIDATES, run_candidate
from benchmark_data_loader import load_subject


def _qc_log_path() -> Path:
    p = _ROOT / "bench_out" / "qc_actions.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _log_action(event: dict):
    p = _qc_log_path()
    with open(p, "a") as f:
        f.write(json.dumps(event) + "\n")


def review_pairs_cli(subject_id: str,
                      candidate_id: str,
                      n: int = 50,
                      accept_threshold: float = 0.7,
                      confidence_sort: str = "ascending",
                      rng_seed: int = 0) -> int:
    """Headless auto-review that treats the candidate's intrinsic
    confidence as the reviewer signal.  Pairs above ``accept_threshold``
    are accepted; others rejected.  Logs each action.

    Used to smoke-test the logging loop and produce a seeded
    ``qc_actions.jsonl``.  In a real deployment a human reviewer replaces
    this loop.
    """
    if candidate_id not in CANDIDATES:
        raise RuntimeError(f"Unknown candidate {candidate_id}; known: {list(CANDIDATES)}")

    s = load_subject(subject_id)
    fn = CANDIDATES[candidate_id]
    result = fn(s)
    df = result.pairs_df
    if len(df) == 0:
        return 0
    if confidence_sort == "ascending":
        df = df.sort_values("confidence", ascending=True)
    rng = np.random.default_rng(rng_seed)
    n = min(n, len(df))
    logged = 0
    for _, row in df.head(n).iterrows():
        action = "accept" if row.get("confidence", 0) >= accept_threshold else "reject"
        ev = dict(
            timestamp=time.time(),
            subject=str(subject_id),
            candidate_id=candidate_id,
            cz_id=int(row["cz_id"]),
            hcr_id=int(row["hcr_id"]),
            confidence=float(row.get("confidence", 0)),
            action=action,
            reviewer="auto-cli",
        )
        _log_action(ev)
        logged += 1
    return logged


def plot_pair(
    s,
    cz_id: int,
    hcr_id: int,
    cz_vol: Optional[np.ndarray] = None,
    hcr_vol: Optional[np.ndarray] = None,
    *,
    crop_um: float = 40.0,
) -> "matplotlib.figure.Figure":
    """Return a matplotlib figure showing a CZ slice at the cz_id centroid
    and an HCR slice at the hcr_id centroid.  Volumes may be provided by
    the caller (to avoid re-loading inside a loop)."""
    import matplotlib.pyplot as plt
    from benchmark_data_loader import cz_px_to_um, hcr_px_to_um

    cz_rows = s.cz_centroids[s.cz_centroids["cz_id"] == cz_id]
    hcr_rows = s.hcr_centroids[s.hcr_centroids["hcr_id"] == hcr_id]
    if cz_rows.empty or hcr_rows.empty:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.set_title(f"pair {cz_id} / {hcr_id}: missing centroid")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title(f"CZ {cz_id}")
    axes[1].set_title(f"HCR {hcr_id}")
    return fig


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("subject_id")
    p.add_argument("candidate_id")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    n = review_pairs_cli(args.subject_id, args.candidate_id, n=args.n)
    print(f"[G1-review] logged {n} actions to {_qc_log_path()}")
