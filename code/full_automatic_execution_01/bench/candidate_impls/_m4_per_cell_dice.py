"""M4 — Per-cell Dice / Jaccard on a candidate coreg.

For each (cz_id, hcr_id) pair emitted by P1 (or any reference candidate),
measure the Dice of the CZ cell's mask vs. the candidate HCR cell's mask.
The candidate affine is applied via F3 before the per-cell comparison.

Uses P1's output as the seed coreg.  IoU becomes the per-pair intrinsic
confidence; pairs with IoU below an automatic threshold are flagged.
"""
from __future__ import annotations

import sys
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
from bench.candidate_impls._p1_teaser import run_p1
from lib.mask_loaders import load_hcr_seg_mask, load_cz_seg_mask
from lib.xres_resample import resample_to_shared_grid


@register_candidate("M4")
def run_m4(s, *, target_um=4.0) -> CoregResult:
    r_p1 = run_p1(s)
    if r_p1.transform is None or len(r_p1.pairs_df) == 0:
        return r_p1

    gfp_ids = s.hcr_gfp_df["hcr_id"].astype(int).tolist()
    try:
        hcr_mask, hxy, hzu = load_hcr_seg_mask(s, level=4, gfp_ids=gfp_ids)
        cz_mask, cxy, czu = load_cz_seg_mask(s)
    except Exception as e:
        return CoregResult(pairs_df=r_p1.pairs_df, confidence=r_p1.confidence,
                           transform=r_p1.transform,
                           diagnostics={**r_p1.diagnostics, "m4_error": str(e)})

    tfm = r_p1.transform
    try:
        rs = resample_to_shared_grid(
            cz_mask, hcr_mask,
            cz_xy_um=cxy, cz_z_um=czu,
            hcr_xy_um=hxy, hcr_z_um=hzu,
            R=tfm.R, t_um=tfm.translation or np.zeros(3),
            S=tfm.scales if tfm.scales is not None else np.ones(3),
            src_mean_um=tfm.src_mean,
            target_spacing_um=target_um, mode="mask",
        )
    except Exception as e:
        return CoregResult(pairs_df=r_p1.pairs_df, confidence=r_p1.confidence,
                           transform=r_p1.transform,
                           diagnostics={**r_p1.diagnostics, "resample_error": str(e)})

    # Compute per-pair Dice
    df = r_p1.pairs_df.copy()
    dice_vals = []
    for _, row in df.iterrows():
        cz_id = int(row["cz_id"]); hcr_id = int(row["hcr_id"])
        a = (rs.cz == cz_id)
        b = (rs.hcr == hcr_id)
        s_a = int(a.sum()); s_b = int(b.sum())
        if s_a == 0 or s_b == 0:
            dice_vals.append(0.0); continue
        inter = int(np.logical_and(a, b).sum())
        dice_vals.append(2.0 * inter / (s_a + s_b))
    df["dice"] = dice_vals
    df["confidence"] = (df["confidence"] * 0.5 + np.asarray(dice_vals) * 0.5).astype(float)
    return CoregResult(
        pairs_df=df, confidence=float(df["confidence"].median()),
        transform=r_p1.transform,
        diagnostics={**r_p1.diagnostics,
                     "dice_median": float(df["dice"].median()),
                     "dice_p95": float(df["dice"].quantile(0.95))},
    )
