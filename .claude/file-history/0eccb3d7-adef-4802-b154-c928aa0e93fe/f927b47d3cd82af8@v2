"""How do GT-matched HCR ROIs score on the S11 v5d ROI-quality metric?

For each subject:
  - all HCR vs GT-matched: histogram of binary score (good ∪ bad_ok),
    twinx (all-HCR gray, matched blue).
  - quartile/decile table of `score`.
  - 4-class probabilities (mean of p_good, p_bad_ok, p_merged, p_bad).
  - thresholded counts at score ≥ {0.3, 0.5, 0.7, 0.9}.

Output: figures/roi_quality_on_matched/{sid}.png and
        results_roi_quality_on_matched.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEV = Path("/root/capsule/code/dev_code")
SESSION = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp")
FIGDIR = SESSION / "figures" / "roi_quality_on_matched"
FIGDIR.mkdir(parents=True, exist_ok=True)
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from benchmark_data_loader import load_subject  # noqa: E402

ARCHIVE = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503")
SCORE_DIR = ARCHIVE / "cached_data" / "cached_roi_quality"
SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def analyze(sid: str) -> dict:
    s = load_subject(sid)
    matched_ids = set(int(x) for x in s.coreg_table["hcr_id"].values)
    binary = pd.read_parquet(SCORE_DIR / f"{sid}_stage2_binary_score_v5d.parquet")
    fourcl = pd.read_parquet(SCORE_DIR / f"{sid}_stage2_4class_proba_v5d.parquet")
    df = binary.merge(
        fourcl[["hcr_id", "p_good", "p_bad", "p_bad_ok", "p_merged"]],
        on="hcr_id", how="left"
    )
    df["is_matched"] = df["hcr_id"].isin(matched_ids)
    all_score = df["score"].dropna().to_numpy()
    matched_score = df.loc[df["is_matched"], "score"].dropna().to_numpy()

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    out = {
        "subject": sid,
        "n_hcr_total": int(len(df)),
        "n_matched": int(df["is_matched"].sum()),
        "all_hcr": {
            "mean": float(all_score.mean()),
            "p10": pct(all_score, 10), "p25": pct(all_score, 25),
            "p50": pct(all_score, 50), "p75": pct(all_score, 75),
            "p90": pct(all_score, 90),
            "frac_ge": {f"{t}": float((all_score >= t).mean()) for t in THRESHOLDS},
        },
        "matched": {
            "mean": float(matched_score.mean()) if len(matched_score) else float("nan"),
            "p10": pct(matched_score, 10), "p25": pct(matched_score, 25),
            "p50": pct(matched_score, 50), "p75": pct(matched_score, 75),
            "p90": pct(matched_score, 90),
            "frac_ge": {f"{t}": float((matched_score >= t).mean()) for t in THRESHOLDS},
        },
        "class_means_all": {
            c: float(df[c].mean()) for c in ["p_good", "p_bad_ok", "p_merged", "p_bad"]
        },
        "class_means_matched": {
            c: float(df.loc[df["is_matched"], c].mean())
            for c in ["p_good", "p_bad_ok", "p_merged", "p_bad"]
        },
    }

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.4))
    bins = np.linspace(0, 1, 50)
    ax.hist(all_score, bins=bins, color="#94a3b8", alpha=0.7,
            label=f"all HCR (n={len(all_score)})")
    ax2 = ax.twinx()
    ax2.hist(matched_score, bins=bins, color="#3b7dd8", alpha=0.55,
             label=f"GT-matched (n={len(matched_score)})")
    ax.set_xlabel("S11 v5d binary ROI-quality score (P(good ∪ bad_ok))")
    ax.set_ylabel("count (all HCR)")
    ax2.set_ylabel("count (matched)", color="#3b7dd8")
    ax2.tick_params(axis='y', colors="#3b7dd8")
    ax.set_title(
        f"{sid}: ROI-quality score distribution  "
        f"all_med={pct(all_score,50):.2f}  matched_med={pct(matched_score,50):.2f}  "
        f"matched ≥0.5: {100*(matched_score >= 0.5).mean():.1f}%"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    out = {}
    for sid in SUBJECTS:
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  {sid} ERROR: {type(e).__name__}: {e}")
            out[sid] = {"subject": sid, "error": str(e)}
            continue
        out[sid] = r
        a = r["all_hcr"]; m = r["matched"]
        print(
            f"{sid:6s}  matched n={r['n_matched']:4d}  "
            f"all-med={a['p50']:.3f}  matched-med={m['p50']:.3f}  "
            f"matched≥0.5={100*m['frac_ge']['0.5']:.1f}%  "
            f"matched≥0.9={100*m['frac_ge']['0.9']:.1f}%  "
            f"all≥0.5={100*a['frac_ge']['0.5']:.1f}%"
        )
    with open(SESSION / "results_roi_quality_on_matched.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
