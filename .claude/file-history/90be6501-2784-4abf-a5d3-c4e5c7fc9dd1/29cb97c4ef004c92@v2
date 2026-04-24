"""Debug: is the GT hcr_id actually reachable via centroids_um('hcr_gfp')?

If coreg_table has 787 hcr_id values but only N% of them are in the GFP+ subset
used by P1, then P1 literally cannot predict them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from lib.centroid_helpers import centroids_um


def debug(sid="788406"):
    s = load_subject(sid)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_gfp_ids = centroids_um(s, "hcr_gfp")
    gt = s.coreg_table
    hcr_all_ids = s.hcr_centroids["hcr_id"].values

    print(f"CZ: {len(cz_ids)} cells")
    print(f"HCR (all): {len(hcr_all_ids)} cells")
    print(f"HCR (GFP+): {len(hcr_gfp_ids)} cells")
    print(f"GT pairs: {len(gt)}")
    print(f"GT cz_ids in centroids_um('cz'): "
          f"{np.isin(gt['cz_id'].values, cz_ids).sum()}/{len(gt)}")
    print(f"GT hcr_ids in hcr_all: "
          f"{np.isin(gt['hcr_id'].values, hcr_all_ids).sum()}/{len(gt)}")
    print(f"GT hcr_ids in hcr_gfp_ids (P1 candidate pool): "
          f"{np.isin(gt['hcr_id'].values, hcr_gfp_ids).sum()}/{len(gt)}")

    missing = ~np.isin(gt['hcr_id'].values, hcr_gfp_ids)
    print(f"GT hcr_ids NOT in GFP+ subset: {missing.sum()}")

    # Print hcr_gfp_df columns and thresholds
    print(f"\nhcr_gfp_df columns: {list(s.hcr_gfp_df.columns)}")
    print(f"hcr_gfp_df head:\n{s.hcr_gfp_df.head()}")


if __name__ == "__main__":
    debug()
