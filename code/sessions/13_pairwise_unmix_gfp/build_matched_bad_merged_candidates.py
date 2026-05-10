"""Export a `sid,hcr_id` CSV of GT-matched HCR ROIs whose v5d 4-class
argmax is `bad` or `merged`, for use with
`v3_S11_roi_quality/04b_label_gui_app.py --candidates ...`.

Order: bad first, then merged; within each class sorted by descending
class probability (worst-looking first).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEV = Path("/root/capsule/code/dev_code")
SESSION = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp")
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from benchmark_data_loader import load_subject  # noqa: E402

ARCHIVE = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503")
SCORE_DIR = ARCHIVE / "cached_data" / "cached_roi_quality"
SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]
CLASSES = ["p_good", "p_bad_ok", "p_merged", "p_bad"]
KEEP = {"p_bad", "p_merged"}
OUT = SESSION / "outputs" / "matched_bad_merged_candidates.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in SUBJECTS:
        s = load_subject(sid)
        matched_ids = set(int(x) for x in s.coreg_table["hcr_id"].values)
        f = pd.read_parquet(SCORE_DIR / f"{sid}_stage2_4class_proba_v5d.parquet")
        f = f[f["hcr_id"].astype(int).isin(matched_ids)].copy()
        arr = f[CLASSES].to_numpy(float)
        arg = np.argmax(arr, axis=1)
        f["argmax"] = [CLASSES[i] for i in arg]
        f["argmax_p"] = arr[np.arange(len(f)), arg]
        sub = f[f["argmax"].isin(KEEP)].copy()
        # bad before merged, then by descending argmax probability
        sub["_order"] = sub["argmax"].map({"p_bad": 0, "p_merged": 1})
        sub = sub.sort_values(["_order", "argmax_p"], ascending=[True, False])
        n_bad = (sub["argmax"] == "p_bad").sum()
        n_mrg = (sub["argmax"] == "p_merged").sum()
        print(f"{sid}: matched={len(matched_ids):4d}  "
              f"bad={n_bad:3d}  merged={n_mrg:3d}  total candidates={len(sub):3d}")
        for _, r in sub.iterrows():
            rows.append({"sid": sid, "hcr_id": int(r["hcr_id"]),
                         "argmax": r["argmax"].replace("p_", ""),
                         "argmax_p": float(r["argmax_p"])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out)} rows total)")
    print(out.groupby(["sid", "argmax"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
