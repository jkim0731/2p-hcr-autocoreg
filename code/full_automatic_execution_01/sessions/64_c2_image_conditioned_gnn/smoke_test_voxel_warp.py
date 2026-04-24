"""S64 subgoal 2 smoke test — F8 voxel warp pipeline on 788406."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from benchmark_analysis import load_hcr_volume  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from lib.synthetic_warps import sample_voxel_warp_sample  # noqa: E402


def main():
    sid = "788406"
    print(f"=== Loading {sid} ===", flush=True)
    s = load_subject(sid)
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    print(f"  HCR GFP+ centroids: {len(hcr_um)}", flush=True)

    print("=== Loading HCR 488 level 2 volume (one-time) ===", flush=True)
    t0 = time.time()
    vol, xy_um, z_um = load_hcr_volume(s, channel="488", level=2)
    print(f"  shape={vol.shape} dtype={vol.dtype} "
          f"xy_um={xy_um:.3f} z_um={z_um:.3f} load_t={time.time()-t0:.1f}s",
          flush=True)

    rng = np.random.default_rng(42)
    print("\n=== Generating 3 voxel warp samples ===", flush=True)
    for k in range(3):
        t0 = time.time()
        sample = sample_voxel_warp_sample(
            vol, (z_um, xy_um, xy_um), hcr_um, rng=rng,
            source_cube_um=400.0, target_margin_um=400.0,
            source_n_target=900, target_n_cap=2000,
            patch_size=16, patch_spacing_um=4.0,
        )
        t_gen = time.time() - t0
        ns = len(sample.source_um)
        nw = len(sample.warped_um)
        nc = len(sample.correspondence)
        print(
            f"  sample {k}: Ns={ns} Nw={nw} corr={nc} "
            f"scales={sample.scales} yaw={sample.tps_metadata['yaw']:.1f}° "
            f"pitch={sample.tps_metadata['pitch']:.1f}° t={t_gen:.2f}s",
            flush=True,
        )

        if ns == 0 or nw == 0:
            print("  EMPTY — centroid density too low for this cube centre", flush=True)
            continue

        sp = sample.source_patches
        wp = sample.warped_patches
        print(f"    source_patches shape={sp.shape}", flush=True)
        print(f"    warped_patches shape={wp.shape}", flush=True)
        assert sp.shape == (ns, 1, 16, 16, 16), f"bad src shape {sp.shape}"
        assert wp.shape == (nw, 1, 16, 16, 16), f"bad wrp shape {wp.shape}"

        assert not np.isnan(sp).any(), "NaN in source patches"
        assert not np.isnan(wp).any(), "NaN in warped patches"
        assert not np.isinf(sp).any(), "Inf in source patches"
        assert not np.isinf(wp).any(), "Inf in warped patches"

        # Check that correspondence pairs point to centroids at warp-consistent
        # positions. For each (k_src, k_tgt) pair, source_um[k_src] should map
        # to warped_um[k_tgt] under rigid+scale (plus local TPS).
        if nc > 0:
            src_c = sample.source_um[sample.correspondence[:, 0]]
            wrp_c = sample.warped_um[sample.correspondence[:, 1]]
            RS = sample.R @ np.diag(sample.scales)
            predicted = (RS @ src_c.T).T
            residual = wrp_c - predicted
            res_rms = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
            print(
                f"    correspondence rigid-TPS residual RMS = {res_rms:.1f} µm "
                f"(expect TPS jitter ≈ 25 µm)",
                flush=True,
            )

        # Check that warped patches have different content from source patches
        # at correspondence indices (otherwise CNN learns identity).
        if nc > 0:
            sp_c = sp[sample.correspondence[:, 0]]
            wp_c = wp[sample.correspondence[:, 1]]
            diffs = np.abs(sp_c - wp_c).mean(axis=(1, 2, 3, 4))
            print(
                f"    mean |src - warped| per corresponding pair: "
                f"min={diffs.min():.3f} med={np.median(diffs):.3f} max={diffs.max():.3f}",
                flush=True,
            )

    print("\n=== PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
