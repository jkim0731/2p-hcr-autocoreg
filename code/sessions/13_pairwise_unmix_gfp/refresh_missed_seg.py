"""Re-derive hcr_seg_missed_in_cz.tif (and metadata) from the existing
hcr_seg_coreg_in_cz.tif + a freshly-resolved cell×gene CSV. Use when the
cell×gene source has changed but the upstream HCR seg warp is still valid.

  missed = {coreg ROI hcr_id} \\ {unmixed.cell_id}

This avoids re-running the slow level-0 zarr warp (~10 min) when only the
filter set has been swapped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev_code"))
_ARCHIVE_SESSIONS = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503/sessions")
for _sub in ("08_surface_vascular_match", "03c_onset_features/iterations"):
    _p = _ARCHIVE_SESSIONS / _sub
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import export_bigwarp as eb  # noqa: E402

OUT_ROOT = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp/outputs/bigwarp_export")


def refresh(sid: str) -> dict:
    d = OUT_ROOT / sid
    coreg_path = d / "hcr_seg_coreg_in_cz.tif"
    missed_path = d / "hcr_seg_missed_in_cz.tif"
    meta_path = d / "metadata.json"
    for p in (coreg_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = json.loads(meta_path.read_text())
    csv = eb._find_unmixed_csv(sid)
    unmixed_ids = eb._load_unmixed_ids(sid)

    s = eb.load_subject(sid)
    coreg_hcr_ids = set(int(x) for x in s.coreg_table["hcr_id"].dropna().astype(int).unique())
    missed_hcr_ids = coreg_hcr_ids - unmixed_ids

    seg_coreg = tifffile.imread(str(coreg_path))
    if seg_coreg.size == 0:
        seg_missed = np.zeros_like(seg_coreg)
    else:
        max_id = int(seg_coreg.max())
        keep = np.zeros(max_id + 1, dtype=bool)
        for i in missed_hcr_ids:
            if 0 < i <= max_id:
                keep[i] = True
        seg_missed = np.where(keep[seg_coreg], seg_coreg, 0).astype(seg_coreg.dtype)

    coreg_in_overlap = int(len(np.unique(seg_coreg[seg_coreg > 0])))
    missed_in_overlap = int(len(np.unique(seg_missed[seg_missed > 0])))

    eb._save_ome_tiff(
        missed_path, seg_missed,
        xy_um=meta["output_voxel_um"]["x"],
        z_um=meta["output_voxel_um"]["z"],
        compress=True,
    )

    # update metadata
    meta.setdefault("cellxgene_csv_path_history", []).append(meta.get("cellxgene_csv_path", "?"))
    meta["cellxgene_csv_path"] = str(csv) if csv else None
    meta["id_sets"]["unmixed_cell_id_count"] = len(unmixed_ids)
    meta["id_sets"]["missed_hcr_count_total"] = len(missed_hcr_ids)
    meta["id_sets"]["hcr_seg_coreg_in_overlap"] = coreg_in_overlap
    meta["id_sets"]["hcr_seg_missed_in_overlap"] = missed_in_overlap
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"  {sid}: csv={csv}")
    print(f"    coreg={len(coreg_hcr_ids)}  unmixed_ids={len(unmixed_ids)}  "
          f"missed_total={len(missed_hcr_ids)}  missed_in_overlap={missed_in_overlap}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()
    for sid in args.subjects:
        refresh(sid)


if __name__ == "__main__":
    main()
