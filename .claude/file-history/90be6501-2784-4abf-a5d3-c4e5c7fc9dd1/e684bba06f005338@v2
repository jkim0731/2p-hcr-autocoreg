"""Reference candidate — ID mapping returning every (cz_id, hcr_id) from GT.

Used only to prove the F9 harness scoring works end-to-end. Uses the subject's
coreg_table.csv (GT) as its "prediction", so recall should be 1.0 by
construction.  This is *not* a real method — it short-circuits GT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bench.harness import (
    CoregResult,
    TransformDescriptor,
    register_candidate,
)


@register_candidate("REF_GT")
def run_ref_gt(s, **kwargs):
    # Build cz and hcr centroid tables in um
    from benchmark_data_loader import cz_px_to_um, hcr_px_to_um

    cz = s.cz_centroids.set_index("cz_id")
    hcr = s.hcr_centroids.set_index("hcr_id")

    gt = s.coreg_table.copy()
    if gt.empty:
        return CoregResult(pairs_df=pd.DataFrame(), confidence=0.0)

    # Attach centroids
    cz_px = cz.loc[gt["cz_id"], ["z_px", "y_px", "x_px"]].values.astype(float)
    hcr_px = hcr.loc[gt["hcr_id"], ["z_px", "y_px", "x_px"]].values.astype(float)
    cz_um = cz_px_to_um(cz_px, s)
    hcr_um = hcr_px_to_um(hcr_px, s)

    out = pd.DataFrame({
        "cz_id": gt["cz_id"].values,
        "hcr_id": gt["hcr_id"].values,
        "confidence": 1.0,
        "cz_x_um": cz_um[:, 2], "cz_y_um": cz_um[:, 1], "cz_z_um": cz_um[:, 0],
        "hcr_x_um": hcr_um[:, 2], "hcr_y_um": hcr_um[:, 1], "hcr_z_um": hcr_um[:, 0],
    })

    transform = TransformDescriptor(
        R=np.eye(3), scales=np.ones(3),
        translation=np.zeros(3), src_mean=np.zeros(3),
        rotation_deg_z=0.0, kind="identity",
    )
    return CoregResult(pairs_df=out, confidence=1.0, transform=transform,
                       diagnostics={"note": "GT short-circuit for harness test"})
