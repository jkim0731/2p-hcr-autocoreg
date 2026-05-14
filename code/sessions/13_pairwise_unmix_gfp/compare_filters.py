"""Compare the two coreg-ROI filters side by side:
  - unmix filter: ROI is "kept" iff its hcr_id appears in the cell×gene
    table (pairwise-unmixing for 5 subjects; cell-typing for 767018).
  - classifier filter: ROI is "kept" iff the v5d 4-class argmax is
    "good" or "bad_ok".
For each subject report counts, proportions, and pairwise agreement
(raw agreement + Cohen's κ).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev_code"))
_ARCHIVE_SESSIONS = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503/sessions")
for _sub in ("08_surface_vascular_match", "03c_onset_features/iterations"):
    _p = _ARCHIVE_SESSIONS / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_data_loader import load_subject  # noqa: E402

import export_bigwarp as eb  # for _find_unmixed_csv/_load_unmixed_ids

ROI_QUALITY_DIR = Path("/root/capsule/code/dev_code/cached_roi_quality")
SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]


def _proba_predicted(sid: str) -> pd.DataFrame:
    p = ROI_QUALITY_DIR / f"{sid}_stage2_4class_proba_v5d_um.parquet"
    df = pd.read_parquet(p)
    df["hcr_id"] = df["hcr_id"].astype(int)
    cols = ["p_bad", "p_bad_ok", "p_good", "p_merged"]
    names = ["bad", "bad_ok", "good", "merged"]
    df["predicted"] = np.array(names)[df[cols].to_numpy().argmax(axis=1)]
    return df[["hcr_id", "predicted"]]


def _cohen_kappa(n11: int, n10: int, n01: int, n00: int) -> float:
    N = n11 + n10 + n01 + n00
    if N == 0:
        return float("nan")
    p_obs = (n11 + n00) / N
    p1u = (n11 + n10) / N  # unmix-kept rate
    p1c = (n11 + n01) / N  # classifier-kept rate
    p_exp = p1u * p1c + (1 - p1u) * (1 - p1c)
    if p_exp >= 1.0:
        return 1.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def _csv_label(sid: str) -> str:
    p = eb._find_unmixed_csv(sid)
    if p is None:
        return "no-csv"
    return "cell-typing" if "cell-typing" in str(p) else "pairwise-unmixing"


def summarize(sid: str) -> dict:
    s = load_subject(sid)
    coreg_ids = set(int(x) for x in s.coreg_table["hcr_id"].dropna().astype(int).unique())
    cellxgene_ids = eb._load_unmixed_ids(sid)
    proba = _proba_predicted(sid).set_index("hcr_id")["predicted"]

    coreg_arr = np.array(sorted(coreg_ids), dtype=int)
    in_unmix = np.array([cid in cellxgene_ids for cid in coreg_arr])
    pred = np.array([proba.get(cid, "absent") for cid in coreg_arr])
    classifier_keep = np.isin(pred, ["good", "bad_ok"])

    n11 = int(np.sum(in_unmix & classifier_keep))
    n10 = int(np.sum(in_unmix & ~classifier_keep))
    n01 = int(np.sum(~in_unmix & classifier_keep))
    n00 = int(np.sum(~in_unmix & ~classifier_keep))
    N = n11 + n10 + n01 + n00

    miss_unmix = n01 + n00          # ~in_unmix: not in cell×gene
    rej_clf = n10 + n00              # ~classifier_keep
    return {
        "sid": sid,
        "csv_source": _csv_label(sid),
        "coreg": N,
        "miss_unmix": miss_unmix,
        "miss_unmix_pct": 100.0 * miss_unmix / N if N else float("nan"),
        "rej_classifier": rej_clf,
        "rej_classifier_pct": 100.0 * rej_clf / N if N else float("nan"),
        "n_kept_both": n11,
        "n_unmix_keep_clf_reject": n10,
        "n_unmix_miss_clf_keep": n01,
        "n_reject_both": n00,
        "agree_pct": 100.0 * (n11 + n00) / N if N else float("nan"),
        "kappa": _cohen_kappa(n11, n10, n01, n00),
    }


def main():
    rows = [summarize(sid) for sid in SUBJECTS]
    df = pd.DataFrame(rows)

    # Side-by-side comparison
    cols_main = ["sid", "csv_source", "coreg",
                 "miss_unmix", "miss_unmix_pct",
                 "rej_classifier", "rej_classifier_pct",
                 "agree_pct", "kappa"]
    cols_conf = ["sid",
                 "n_kept_both",
                 "n_unmix_keep_clf_reject",
                 "n_unmix_miss_clf_keep",
                 "n_reject_both"]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 24)
    print("\n=== Side-by-side filter comparison ===")
    print(df[cols_main].to_string(index=False,
        formatters={"miss_unmix_pct": "{:6.2f}".format,
                    "rej_classifier_pct": "{:6.2f}".format,
                    "agree_pct": "{:6.2f}".format,
                    "kappa": "{:5.3f}".format}))

    print("\n=== Confusion (per coreg ROI) ===")
    print(df[cols_conf].to_string(index=False))

    print("\n=== Legend ===")
    print("  miss_unmix              : coreg ROIs NOT in cell×gene table")
    print("  rej_classifier          : coreg ROIs with v5d argmax ∈ {bad, merged}")
    print("  n_kept_both             : kept by both (cell×gene ✓ AND classifier good/bad_ok)")
    print("  n_unmix_keep_clf_reject : in cell×gene but classifier says bad/merged")
    print("  n_unmix_miss_clf_keep   : missed from cell×gene but classifier says good/bad_ok")
    print("  n_reject_both           : rejected by both (most likely truly bad)")
    print("  agree_pct               : (n_kept_both + n_reject_both) / coreg")
    print("  kappa                   : Cohen's κ (chance-corrected agreement)")

    out = Path(__file__).parent / "outputs" / "filter_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
