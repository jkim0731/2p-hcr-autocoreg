"""Cheat-test: fit an anisotropic affine on the landmark pairs, apply it to
CZ centroids, and feed those as cz_init to P1. This tells us whether P1
CAN succeed if it receives a truly-correct coarse alignment, which isolates
the failure to the coarse stage (M1).
"""
from __future__ import annotations

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject, landmark_pairs_um
from benchmark_analysis import fit_anisotropic_similarity
from bench.candidate_impls._p1_teaser import run_p1
from lib.centroid_helpers import centroids_um


def cheat(subject_id="788406"):
    s = load_subject(subject_id)
    cz_um, cz_ids = centroids_um(s, "cz")
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    print(f"landmarks: {len(cz_lm)}")

    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    print(f"Landmark fit: R =\n{fit.R}")
    print(f"  scales={fit.scales}, translation={fit.translation}, rms={fit.rms_um:.2f}µm")

    # Apply the landmark-fit affine to ALL CZ centroids:
    # canonical convention per apply_aniso_fit: pred = (cz * scales) @ R.T + translation
    cz_init = (cz_um * fit.scales) @ fit.R.T + fit.translation

    # Call P1 with cheat warm-start
    r = run_p1(s, cz_init=cz_init, K=10, c_bar=15.0)
    print(f"\nP1 with cheat warm-start:")
    print(f"  n_pred={len(r.pairs_df)}, confidence={r.confidence:.3f}")
    print(f"  diagnostics={r.diagnostics}")

    if not r.pairs_df.empty:
        # Score vs ground truth
        gt = s.coreg_table
        pred = r.pairs_df.set_index("cz_id")
        gt = gt.set_index("cz_id")
        joined = gt.join(pred[["hcr_id"]].rename(columns={"hcr_id": "pred_hcr_id"}), how="left")
        id_match = (joined["pred_hcr_id"] == joined["hcr_id"]).fillna(False)
        print(f"  recall (ID match) = {id_match.sum() / len(joined):.3f}")
        print(f"  n_correct = {id_match.sum()}/{len(joined)}")

    # Try sweeping K to see how K affects recall
    for K in (5, 10, 20, 50, 100):
        r = run_p1(s, cz_init=cz_init, K=K, c_bar=15.0)
        if not r.pairs_df.empty:
            pred = r.pairs_df.set_index("cz_id")
            joined = s.coreg_table.set_index("cz_id").join(
                pred[["hcr_id"]].rename(columns={"hcr_id": "pred_hcr_id"}), how="left")
            m = (joined["pred_hcr_id"] == joined["hcr_id"]).fillna(False)
            print(f"  K={K}: recall={m.sum() / len(joined):.3f} n_pred={len(r.pairs_df)} n_inl={r.diagnostics.get('n_inliers')}")
        else:
            print(f"  K={K}: empty")

    out = _ROOT / "sessions/27_putative_rank_diag"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "landmark_fit": {
            "scales": fit.scales.tolist(),
            "translation": fit.translation.tolist(),
            "R": fit.R.tolist(),
            "rms_um": float(fit.rms_um),
        }
    }, open(out / "cheat_landmark_fit.json", "w"), indent=2)


if __name__ == "__main__":
    cheat()
