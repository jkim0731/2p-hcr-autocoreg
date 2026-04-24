"""Test anisotropic ICP scale estimate with a minimal R1 fit."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass

_ROOT = Path("/root/capsule/code/full_automatic_execution_01")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent / "dev_code") not in sys.path:
    sys.path.insert(0, str(_ROOT.parent / "dev_code"))

from benchmark_data_loader import load_subject
from lib.centroid_helpers import centroids_um
from anisotropic_icp import estimate_scales_icp_multi_start


@dataclass
class MinimalFit:
    R: np.ndarray
    src_mean: np.ndarray
    translation: np.ndarray
    scales: np.ndarray


def main(sid="788406"):
    s = load_subject(sid)
    cz_zyx, _ = centroids_um(s, "cz")     # (z, y, x)
    hcr_zyx, _ = centroids_um(s, "hcr_gfp")
    # anisotropic_icp expects (x, y, z) per its docstring. Permute.
    cz_xyz = cz_zyx[:, [2, 1, 0]]
    hcr_xyz = hcr_zyx[:, [2, 1, 0]]

    # Minimal R1 fit: 180° XY rotation about CZ centroid, translate to HCR
    # centroid, scales=1.
    R_xyz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)  # 180° about Z in xyz
    src_mean = cz_xyz.mean(0)
    dst_mean = hcr_xyz.mean(0)

    fit = MinimalFit(R=R_xyz, src_mean=src_mean, translation=dst_mean, scales=np.ones(3))
    res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit)
    print(f"sxy={res.sxy}, sz={res.sz}, matched={res.n_matched}, iters={res.iterations}")
    print(f"converged={res.converged}, reason={res.reason_unknown}")
    if res.fit is not None:
        print(f"fit scales = {res.fit.scales}")


if __name__ == "__main__":
    main()
