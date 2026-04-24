"""Direct mi_bspline call on 788406 downsampled volumes with per-iter print.
Diagnoses whether the per-iteration timing is within budget."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from bench.candidate_impls._i1_axial_ncc import (  # noqa: E402
    _load_cz_fullstack, _load_hcr_fullstack)
from bench.candidate_impls._i2_sitk_affine import run_i2  # noqa: E402
import lib.sitk_wrapper as sw  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sw.MI_BSPLINE_VERBOSE = True


def main():
    s = load_subject("788406")
    t0 = time.time()
    r_i2 = run_i2(s)
    print(f"I2 wall={time.time()-t0:.1f}s", flush=True)

    A = np.asarray(r_i2.diagnostics["sitk_matrix_zyx"])
    t_v = np.asarray(r_i2.diagnostics["sitk_translation_zyx"])
    c_v = np.asarray(r_i2.diagnostics["sitk_center_zyx"])

    target_um = 8.0
    cz_stack, cz_xy_um, cz_z_um = _load_cz_fullstack(s)
    hcr_vol, hcr_xy_um, hcr_z_um = _load_hcr_fullstack(s)

    cz_zoom = (cz_z_um / target_um, cz_xy_um / target_um, cz_xy_um / target_um)
    hcr_zoom = (hcr_z_um / target_um, hcr_xy_um / target_um, hcr_xy_um / target_um)
    cz_ds = zoom(cz_stack, cz_zoom, order=1).astype(np.float32)
    hcr_ds = zoom(hcr_vol, hcr_zoom, order=1).astype(np.float32)
    print(f"cz ds={cz_ds.shape} hcr ds={hcr_ds.shape}", flush=True)

    for grid_um, iters, samp_frac in [
        ((120.0, 120.0, 240.0), 5, 0.02),   # very coarse, fast
        ((60.0, 60.0, 120.0), 5, 0.02),      # medium
    ]:
        print(f"\n-- grid={grid_um} iters={iters} samp={samp_frac} --",
              flush=True)
        t0 = time.time()
        try:
            r = sw.mi_bspline(cz_ds, hcr_ds,
                              cz_xy_um=target_um, cz_z_um=target_um,
                              hcr_xy_um=target_um, hcr_z_um=target_um,
                              initial_affine=A, initial_translation=t_v,
                              initial_center=c_v,
                              grid_spacing_um=grid_um,
                              n_iterations=iters,
                              sampling_fraction=samp_frac)
            print(f"  wall={time.time()-t0:.1f}s metric={r.metric:.5f} "
                  f"iters={r.iterations} converged={r.converged}", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
