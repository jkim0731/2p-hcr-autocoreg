"""Probe CZ zstack + HCR 488 volume loading for one subject, print shapes/voxel sizes.

Sanity check before building the image-NCC estimator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_analysis import analyze_subject, load_hcr_volume
from benchmark_data_loader import load_subject


def probe(sid: str, hcr_level: int = 4) -> None:
    print(f"── {sid} ──")
    s = load_subject(sid)
    print(f"  cz_xy_um = {s.cz_xy_um:.4f}  cz_z_um = {s.cz_z_um:.4f}")
    print(f"  hcr_xy_um (native) = {s.hcr_xy_um:.4f}  hcr_z_um = {s.hcr_z_um:.4f}")

    cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not cz_tifs:
        cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
    if not cz_tifs:
        print("  ! no CZ zstack found")
        return
    cz_img = tifffile.imread(str(cz_tifs[0]))
    while cz_img.ndim > 3 and cz_img.shape[0] == 1:
        cz_img = cz_img[0]
    print(f"  CZ zstack path: {cz_tifs[0].name}")
    print(f"    shape = {cz_img.shape}  dtype = {cz_img.dtype}")
    print(f"    physical extent: "
          f"z = {cz_img.shape[0] * s.cz_z_um:.0f} µm, "
          f"y = {cz_img.shape[1] * s.cz_xy_um:.0f} µm, "
          f"x = {cz_img.shape[2] * s.cz_xy_um:.0f} µm")
    print(f"    intensity range: [{cz_img.min()}, {cz_img.max()}]  "
          f"mean = {cz_img.mean():.1f}")

    vol, xy_um, z_um = load_hcr_volume(s, channel="488", level=hcr_level)
    print(f"  HCR 488 level={hcr_level}")
    print(f"    shape = {vol.shape}  dtype = {vol.dtype}")
    print(f"    xy_um = {xy_um:.4f}  z_um = {z_um:.4f}")
    print(f"    physical extent: "
          f"z = {vol.shape[0] * z_um:.0f} µm, "
          f"y = {vol.shape[1] * xy_um:.0f} µm, "
          f"x = {vol.shape[2] * xy_um:.0f} µm")
    print(f"    intensity range: [{vol.min()}, {vol.max()}]  "
          f"mean = {vol.mean():.1f}")

    info = analyze_subject(s)
    pia = info["hcr_surface"]
    print(f"  HCR pia plane: z = {pia['a']:+.4f}·x + {pia['b']:+.4f}·y + {pia['c']:+.1f}")


if __name__ == "__main__":
    probe("790322")
