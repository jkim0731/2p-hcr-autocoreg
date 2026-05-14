"""Sanity-check the BigWarp export for one subject.

Reports shape, voxel size, nonzero fraction (for raw channels), unique-ID
count (for seg), and prints a few metadata fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

OUT_ROOT = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp/outputs/bigwarp_export")


def _summary(name: str, vol: np.ndarray) -> str:
    n_nz = int((vol > 0).sum())
    frac_nz = n_nz / vol.size if vol.size else 0.0
    pct = np.percentile(vol[vol > 0].ravel(), [50, 95, 99]).tolist() if n_nz else [0, 0, 0]
    return (f"{name:>18s}  shape={vol.shape}  dtype={vol.dtype}  "
            f"nz={frac_nz:.3f}  p50/95/99={pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}  "
            f"min={vol.min()}  max={vol.max()}")


def check_subject(sid: str) -> None:
    d = OUT_ROOT / sid
    print(f"\n=== {sid} ===  ({d})")
    files = [
        "cz_stack.tif",
        "cz_seg_all.tif",
        "cz_seg_coreg.tif",
        "hcr_405_in_cz.tif",
        "hcr_488_in_cz.tif",
        "hcr_seg_coreg_in_cz.tif",
        "hcr_seg_missed_in_cz.tif",
    ]
    for f in files:
        p = d / f
        if not p.exists():
            print(f"  MISSING: {f}")
            continue
        vol = tifffile.imread(str(p))
        print("  " + _summary(f, vol))

    meta_path = d / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        idsets = meta.get("id_sets", {})
        print(f"  id_sets: {idsets}")
        print(f"  sz_lp={meta.get('sz_lp'):.3f}  sz_best={meta.get('sz_best'):.3f}  "
              f"cz_mean_depth={meta.get('cz_mean_depth_um'):.1f}µm")
        print(f"  voxel_um: {meta.get('output_voxel_um')}")
        print(f"  elapsed: {meta.get('elapsed_sec'):.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="+")
    args = ap.parse_args()
    for sid in args.subjects:
        check_subject(sid)


if __name__ == "__main__":
    main()
