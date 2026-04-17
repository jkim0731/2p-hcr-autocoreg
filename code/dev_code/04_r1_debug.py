"""Quick debug for R1 on subject 790322."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject
from benchmark_data_loader import load_subject, landmark_pairs_um


def main(sid: str):
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    hcr_xyz = info["hcr_xyz"]

    print(f"subject {sid}")
    print(f"  CZ: {len(cz_xyz)} cells, extent "
          f"x={cz_xyz[:,0].ptp():.0f} y={cz_xyz[:,1].ptp():.0f} "
          f"z={cz_xyz[:,2].ptp():.0f} center={cz_xyz.mean(0)}")
    print(f"  HCR total: {len(hcr_xyz)} extent "
          f"x={hcr_xyz[:,0].ptp():.0f} y={hcr_xyz[:,1].ptp():.0f} "
          f"z={hcr_xyz[:,2].ptp():.0f}")
    print(f"  HCR GFP+: {len(gfp_xyz)} ({len(gfp_xyz)/max(1,len(hcr_xyz)):.1%}) center={gfp_xyz.mean(0)}")
    if s.hcr_gfp_df.shape[0]:
        print(f"  GFP+ spot count stats: {s.hcr_gfp_df['counts'].describe().to_dict()}"
              if 'counts' in s.hcr_gfp_df.columns else
              f"  GFP+ feature='{s.gfp_feature_name}' (no spot counts)")

    cz_um_lm, hcr_um_lm = landmark_pairs_um(s, active_only=True)
    print(f"  landmarks active: {len(cz_um_lm)}")
    print(f"  landmark CZ center: {cz_um_lm.mean(0)}")
    print(f"  landmark HCR center: {hcr_um_lm.mean(0)}")


if __name__ == "__main__":
    for sid in sys.argv[1:] or ["790322"]:
        main(sid)
        print()
