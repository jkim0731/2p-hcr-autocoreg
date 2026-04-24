"""Iteration 4 compute — end-to-end validation of the new public API.

Uses :func:`score_surface_quality` and :func:`auto_select_surface` added
to ``03_image_based_surface.py`` in iter 3's integration step.  Runs on
all 6 HCR subjects, builds the same candidate bank as iter 3, and
confirms the selection matches what argmax-Q-subject-to-Qcon_n≥0.9 gives
on the iter 3 scores (recomputed independently via the new API):

    755252 → mid488_tq0.85
    767018 → only594_tq0.85   (Q = 0.0144, beats all_tq0.70 at 0.0105)
    767022 → all_tq0.70
    782149 → no405_tq0.85
    788406 → no405_tq0.85
    790322 → all_tq0.70

Outputs
-------
- data/iter04_autoselect.csv — (subject, candidate) rows with Q/Qcon_n
  + selected flag; also per-subject onset/above against centroids.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

_spec = importlib.util.spec_from_file_location(
    "img_surface_v21",
    str(ROOT / "code" / "dev_code" / "03_image_based_surface.py"),
)
_img = importlib.util.module_from_spec(_spec)
sys.modules["img_surface_v21"] = _img
_spec.loader.exec_module(_img)

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import hcr_px_to_um, load_subject

OUT_DATA = ROOT / "code" / "sessions" / "03c_onset_features" / "data"

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]


def mid_channel(sid: str) -> str:
    return "514" if sid in ("755252", "767022") else "561"


def eval_against_centroids(surface, hcr_xyz):
    if surface is None:
        return (float("nan"), float("nan"))
    stats = _img.depth_profile_stats(hcr_xyz, surface)
    return (stats.get("above_frac", float("nan")),
            stats.get("onset_depth_um", float("nan")))


def build_candidates(s, sid):
    mid = mid_channel(sid)
    subsets = [
        ("all_tq0.85",    None,                     0.85),
        ("all_tq0.70",    None,                     0.70),
        ("all_tq0.95",    None,                     0.95),
        ("no405_tq0.85",  ["488", mid, "594"],      0.85),
        ("mid488_tq0.85", ["488", mid],             0.85),
        ("only488_tq0.85",["488"],                  0.85),
        ("only594_tq0.85",["594"],                  0.85),
    ]
    out = []
    for name, chs, tq in subsets:
        try:
            vol, xy, zu, _used = load_hcr_combined(s, channels=chs, level=4)
            res = _img.estimate_surface_and_l2_image_based(
                vol, zu, xy, target_quantile=tq)
        except Exception as exc:
            print(f"  {sid} {name}: FAILED ({exc})")
            res = None
        out.append((name, res))
    return out


def process_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    ref_vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    hcr_xyz = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]

    candidates = build_candidates(s, sid)
    sel_name, sel_res, scores = _img.auto_select_surface(
        candidates, ref_vol, z_um, xy_um, qcon_n_threshold=0.9)

    # attach centroid-truth columns for validation
    info = []
    name_to_res = dict(candidates)
    for _, row in scores.iterrows():
        res = name_to_res[row["name"]]
        above, onset = eval_against_centroids(res.surface if res is not None else None, hcr_xyz)
        info.append(dict(above=above, onset=onset))
    info_df = pd.DataFrame(info)
    out = pd.concat([scores.reset_index(drop=True), info_df], axis=1)
    out.insert(0, "subject", sid)

    print(f"  selected → {sel_name}")
    for _, r in out.iterrows():
        mark = "★" if r["selected"] else (" " if r["pass"] else "✗")
        onset_s = f"{r['onset']:.1f}" if np.isfinite(r['onset']) else "nan"
        print(f"  {mark} {r['name']:<18} Q={r['Q']:.4f} Qcon_n={r['Qcon_n']:+.3f}"
              f" above={r['above']:.4f} onset={onset_s}")
    return out


def main():
    all_df = [process_subject(sid) for sid in HCR]
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(OUT_DATA / "iter04_autoselect.csv", index=False)
    print(f"\nwrote {OUT_DATA/'iter04_autoselect.csv'}")

    # quick selection-summary check vs iter03
    expected = {
        "755252": "mid488_tq0.85",
        "767018": "only594_tq0.85",
        "767022": "all_tq0.70",
        "782149": "no405_tq0.85",
        "788406": "no405_tq0.85",
        "790322": "all_tq0.70",
    }
    sel = out[out.selected][["subject", "name"]].set_index("subject")["name"].to_dict()
    print("\nvalidation vs iter03:")
    ok = True
    for sid in HCR:
        got = sel.get(sid, "?")
        want = expected[sid]
        tag = "OK " if got == want else "MISMATCH"
        if got != want: ok = False
        print(f"  {sid}  want={want:<18} got={got:<18} {tag}")
    print("\nALL MATCH" if ok else "\nMISMATCH — investigate")


if __name__ == "__main__":
    main()
