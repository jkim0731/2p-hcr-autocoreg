"""Iteration 6 compute — OOT-aware scoring.

Re-runs ``estimate_pia_surface_image_autoselect`` on all 6 HCR
benchmark subjects with the updated :func:`score_surface_quality` that
masks out-of-tissue grid columns (``min_col_max_frac = 0.05``, new
default, with a shared ``global_max = np.percentile(colmax, 99.5)``
across the candidate bank).  The mask is still a correct fix even if
its effect on the 15 %-padded interior grid is small (0–9/400 OOT
columns on these 6 subjects; iter 5's "52 %/94 %/77 %" were a
reference-inflation artefact from using ``vol.max()`` as the reference,
see iter 5 correction in log.md).  The main practical win is crash
resistance on volumes with saturated voxels.

For each subject records:

- selected candidate (new scoring, shared ``global_max`` across bank)
- per-candidate Q / Qabove / Qcon / Qcon_n / n / n_oot
- centroid-truth onset + above-fraction for every candidate

Outputs
-------
- data/iter06_autoselect.csv           — per (subject, candidate)
- data/iter06_selection_summary.csv    — one row per subject,
  iter-4 vs iter-6 selection + OOT stats + centroid validation
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

OUT = ROOT / "code" / "sessions" / "03c_onset_features" / "data"
HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]

ITER04 = {
    "755252": "mid488_tq0.85",
    "767018": "only594_tq0.85",
    "767022": "all_tq0.70",
    "782149": "no405_tq0.85",
    "788406": "no405_tq0.85",
    "790322": "all_tq0.70",
}


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
        ("all_tq0.85",     None,                0.85),
        ("all_tq0.70",     None,                0.70),
        ("all_tq0.95",     None,                0.95),
        ("no405_tq0.85",   ["488", mid, "594"], 0.85),
        ("mid488_tq0.85",  ["488", mid],        0.85),
        ("only488_tq0.85", ["488"],             0.85),
        ("only594_tq0.85", ["594"],             0.85),
    ]
    out = []
    for name, chs, tq in subsets:
        try:
            vol, xy, zu, _ = load_hcr_combined(s, channels=chs, level=4)
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
    # New: auto_select_surface now passes shared global_max, and the
    # default min_col_max_frac = 0.05 in score_surface_quality masks
    # out-of-tissue columns.
    sel_name, sel_res, scores = _img.auto_select_surface(
        candidates, ref_vol, z_um, xy_um, qcon_n_threshold=0.9)

    # centroid-truth columns for validation
    info_rows = []
    name_to_res = dict(candidates)
    for _, row in scores.iterrows():
        res = name_to_res[row["name"]]
        above, onset = eval_against_centroids(
            res.surface if res is not None else None, hcr_xyz)
        info_rows.append(dict(above=above, onset=onset))
    info_df = pd.DataFrame(info_rows)
    out = pd.concat([scores.reset_index(drop=True), info_df], axis=1)
    out.insert(0, "subject", sid)

    # per-subject summary
    sel_row = out[out.selected].iloc[0]
    summary = dict(
        subject=sid,
        iter4_selection=ITER04[sid],
        iter6_selection=sel_name,
        agree=bool(sel_name == ITER04[sid]),
        above_frac=float(sel_row["above"]),
        onset_um=float(sel_row["onset"]),
        Q_selected=float(sel_row["Q"]),
        Qcon_n_selected=float(sel_row["Qcon_n"]),
        n_in_tissue=int(sel_row["n"]),
        n_oot=int(sel_row["n_oot"]),
    )
    print(f"  selected → {sel_name}  (iter4 → {ITER04[sid]}, "
          f"agree={summary['agree']})")
    print(f"  above={summary['above_frac']*100:.2f}%  "
          f"onset={summary['onset_um']:.1f}µm  "
          f"n={summary['n_in_tissue']}, n_oot={summary['n_oot']}")
    for _, r in out.sort_values("Q", ascending=False).iterrows():
        mark = "★" if r["selected"] else (" " if r["pass"] else "✗")
        onset_s = f"{r['onset']:5.1f}" if np.isfinite(r['onset']) else "  nan"
        print(f"  {mark} {r['name']:<18} Q={r['Q']:.4f}  Qcon_n={r['Qcon_n']:+.3f}"
              f"  n={int(r['n']):3d}/oot={int(r['n_oot']):3d}"
              f"  above={r['above']:.4f} onset={onset_s}")
    return out, summary


def main():
    all_df = []
    summary_rows = []
    for sid in HCR:
        out, rsum = process_subject(sid)
        all_df.append(out)
        summary_rows.append(rsum)

    df_scores = pd.concat(all_df, ignore_index=True)
    df_sum = pd.DataFrame(summary_rows)

    df_scores.to_csv(OUT / "iter06_autoselect.csv", index=False)
    df_sum.to_csv(OUT / "iter06_selection_summary.csv", index=False)

    print("\n=== per-subject summary (iter 4 vs iter 6) ===")
    print(df_sum.to_string(
        index=False,
        float_format=lambda x: f"{x:8.4f}",
    ))
    n_agree = int(df_sum["agree"].sum())
    print(f"\niter4 / iter6 agree on {n_agree}/6 subjects")
    print(f"wrote {OUT/'iter06_autoselect.csv'}")
    print(f"wrote {OUT/'iter06_selection_summary.csv'}")


if __name__ == "__main__":
    main()
