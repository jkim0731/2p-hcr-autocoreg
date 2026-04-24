"""Minimal test of mi_bspline composition: random synthetic volumes,
known affine, check that SetMovingInitialTransform works without hanging.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from lib.sitk_wrapper import mi_bspline, mi_affine  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    # Tiny volumes so this runs in seconds.
    cz = rng.random((30, 60, 60)).astype(np.float32)
    hcr = rng.random((80, 120, 120)).astype(np.float32)

    # Identity affine pre-composed.
    A = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    c = np.array([40.0, 60.0, 60.0])

    print("calling mi_bspline with sampling=0.2, iters=5", flush=True)
    t0 = time.time()
    r = mi_bspline(cz, hcr, cz_xy_um=1.0, cz_z_um=1.0,
                   hcr_xy_um=1.0, hcr_z_um=1.0,
                   initial_affine=A, initial_translation=t, initial_center=c,
                   grid_spacing_um=(30.0, 30.0, 60.0),
                   n_iterations=5, sampling_fraction=0.1)
    wall = time.time() - t0
    print(f"  wall={wall:.2f}s metric={r.metric:.4f} iters={r.iterations} "
          f"converged={r.converged}", flush=True)
    print(f"  composite type: {type(r._sitk_composite).__name__}", flush=True)

    # Test forward-apply on a few points.
    pts = np.array([[0.0, 0.0, 0.0], [40.0, 60.0, 60.0], [60.0, 100.0, 100.0]])
    out = r.apply_forward_hcr_to_cz(pts)
    print(f"  forward-apply test: in={pts.shape} out={out.shape}", flush=True)
    for i, (p_in, p_out) in enumerate(zip(pts, out)):
        print(f"    {p_in} -> {p_out}", flush=True)


if __name__ == "__main__":
    main()
