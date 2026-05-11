"""Does the S11 v5d ROI-quality score correlate with whether a cell got
filtered out of the pairwise-unmixing dataset?

For each subject:
  1. Load full HCR centroid id list (n=total HCR).
  2. Load v5d stage2 binary score (`score` ∈ [0,1] = P(good or bad_ok)
     in S11 v5d's binary head) and 4-class probabilities.
  3. Build boolean `in_unmixed` from
     `unmixed_all_cells.csv['cell_id']`.
  4. Compute:
       - mean / quartile of `score` split by in_unmixed.
       - AUC of "dropped-by-unmix" predicted by (1 − score) — high
         AUC = unmixing preferentially drops low-quality ROIs.
       - Same diagnostic restricted to GT-matched cells
         (`coreg_table.hcr_id`).
  5. Plot KDE / histogram of `score` for kept vs dropped.

Output: figures/roi_quality_vs_unmix/{sid}.png and
        results_roi_quality_vs_unmix.json
"""
from __future__ import annotations
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEV = Path("/root/capsule/code/dev_code")
SESSION = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp")
FIGDIR = SESSION / "figures" / "roi_quality_vs_unmix"
FIGDIR.mkdir(parents=True, exist_ok=True)
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from benchmark_data_loader import load_subject  # noqa: E402

ARCHIVE = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503")
SCORE_DIR = ARCHIVE / "cached_data" / "cached_roi_quality"

SUBJECTS = ["755252", "767022", "782149", "788406"]
UNMIX_GLOB = (
    "/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/"
    "pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv"
)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def analyze(sid: str) -> dict:
    s = load_subject(sid)
    all_ids = s.hcr_centroids["hcr_id"].astype(int).to_numpy()
    matched_ids = set(int(x) for x in s.coreg_table["hcr_id"].values)

    # S11 v5d binary score and 4-class probs (covers all HCR cells)
    binary = pd.read_parquet(SCORE_DIR / f"{sid}_stage2_binary_score_v5d.parquet")
    fourcl = pd.read_parquet(SCORE_DIR / f"{sid}_stage2_4class_proba_v5d.parquet")
    score = (
        pd.DataFrame({"hcr_id": all_ids})
        .merge(binary[["hcr_id", "score"]], on="hcr_id", how="left")
        .merge(fourcl[["hcr_id", "p_good", "p_bad", "p_bad_ok", "p_merged"]],
               on="hcr_id", how="left")
    )

    # Unmixed cell ids
    matches = glob(UNMIX_GLOB.format(sid=sid))
    if not matches:
        raise FileNotFoundError(sid)
    unmixed_ids = set(
        int(x) for x in pd.read_csv(matches[0])["cell_id"].astype(int).values
    )

    score["in_unmixed"] = score["hcr_id"].isin(unmixed_ids)
    score["is_matched"] = score["hcr_id"].isin(matched_ids)

    # Per-subject summary
    kept = score.loc[score["in_unmixed"], "score"].dropna()
    dropped = score.loc[~score["in_unmixed"], "score"].dropna()
    auc_drop = auc_score(
        (~score["in_unmixed"]).astype(int).values, (1.0 - score["score"]).fillna(0.5).values
    )
    out = {
        "subject": sid,
        "n_total": int(len(score)),
        "n_unmixed": int(score["in_unmixed"].sum()),
        "n_dropped": int((~score["in_unmixed"]).sum()),
        "score_kept_mean": float(kept.mean()),
        "score_kept_median": float(kept.median()),
        "score_dropped_mean": float(dropped.mean()),
        "score_dropped_median": float(dropped.median()),
        "auc_drop_pred_by_lowscore": auc_drop,
    }

    # Same on matched-only
    sm = score[score["is_matched"]]
    kept_m = sm.loc[sm["in_unmixed"], "score"].dropna()
    dropped_m = sm.loc[~sm["in_unmixed"], "score"].dropna()
    auc_drop_m = auc_score(
        (~sm["in_unmixed"]).astype(int).values,
        (1.0 - sm["score"]).fillna(0.5).values,
    )
    out["matched"] = {
        "n_total": int(len(sm)),
        "n_unmixed": int(sm["in_unmixed"].sum()),
        "n_dropped": int((~sm["in_unmixed"]).sum()),
        "score_kept_mean": float(kept_m.mean()) if len(kept_m) else float("nan"),
        "score_kept_median": float(kept_m.median()) if len(kept_m) else float("nan"),
        "score_dropped_mean": float(dropped_m.mean()) if len(dropped_m) else float("nan"),
        "score_dropped_median": float(dropped_m.median()) if len(dropped_m) else float("nan"),
        "auc_drop_pred_by_lowscore": auc_drop_m,
    }

    # Class-prob comparison
    out["class_means_kept"] = {
        c: float(score.loc[score["in_unmixed"], c].mean())
        for c in ["p_good", "p_bad", "p_bad_ok", "p_merged"]
    }
    out["class_means_dropped"] = {
        c: float(score.loc[~score["in_unmixed"], c].mean())
        for c in ["p_good", "p_bad", "p_bad_ok", "p_merged"]
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    bins = np.linspace(0, 1, 50)
    ax = axes[0]
    ax.hist(kept, bins=bins, density=True, alpha=0.55,
            color="#2aa198", label=f"kept by unmix (n={len(kept)})")
    ax.hist(dropped, bins=bins, density=True, alpha=0.55,
            color="#cc3333", label=f"dropped by unmix (n={len(dropped)})")
    ax.set_xlabel("S11 v5d binary ROI-quality score (good ∪ bad_ok)")
    ax.set_ylabel("density")
    ax.set_title(
        f"{sid} all HCR  AUC(low-score → dropped)={auc_drop:.3f}  "
        f"kept_med={kept.median():.2f}  dropped_med={dropped.median():.2f}"
    )
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(kept_m) and len(dropped_m):
        ax.hist(kept_m, bins=bins, density=True, alpha=0.55,
                color="#2aa198",
                label=f"kept matched (n={len(kept_m)})")
        ax.hist(dropped_m, bins=bins, density=True, alpha=0.55,
                color="#cc3333",
                label=f"dropped matched (n={len(dropped_m)})")
        ax.set_title(
            f"{sid} GT-matched  AUC(low-score → dropped)={auc_drop_m:.3f}  "
            f"kept_med={kept_m.median():.2f}  dropped_med={dropped_m.median():.2f}"
        )
    else:
        ax.text(0.5, 0.5, "no dropped matched cells",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{sid} matched")
    ax.set_xlabel("S11 v5d binary ROI-quality score")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{sid}: ROI-quality score for unmixed-kept vs unmixed-dropped HCR cells",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return out


def main():
    out = {}
    for sid in SUBJECTS:
        print(f"\n=== {sid} ===")
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            out[sid] = {"subject": sid, "error": str(e)}
            continue
        out[sid] = r
        print(
            f"  all HCR    : kept={r['n_unmixed']} (med {r['score_kept_median']:.3f})  "
            f"dropped={r['n_dropped']} (med {r['score_dropped_median']:.3f})  "
            f"AUC={r['auc_drop_pred_by_lowscore']:.3f}"
        )
        m = r["matched"]
        print(
            f"  matched only: kept={m['n_unmixed']} (med {m['score_kept_median']:.3f})  "
            f"dropped={m['n_dropped']} (med {m['score_dropped_median']:.3f})  "
            f"AUC={m['auc_drop_pred_by_lowscore']:.3f}"
        )
        ck = r["class_means_kept"]; cd = r["class_means_dropped"]
        print(
            f"  class p kept   : good={ck['p_good']:.3f}  bad_ok={ck['p_bad_ok']:.3f}  "
            f"merged={ck['p_merged']:.3f}  bad={ck['p_bad']:.3f}"
        )
        print(
            f"  class p dropped: good={cd['p_good']:.3f}  bad_ok={cd['p_bad_ok']:.3f}  "
            f"merged={cd['p_merged']:.3f}  bad={cd['p_bad']:.3f}"
        )

    out_path = SESSION / "results_roi_quality_vs_unmix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
