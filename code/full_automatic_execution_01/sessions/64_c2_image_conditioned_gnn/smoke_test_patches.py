"""S64 subgoal 1 smoke test — verify extract_{cz,hcr}_patches on 788406."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")
sys.path.insert(0, "/root/capsule/code/dev_code")

from benchmark_data_loader import load_subject  # noqa: E402
from lib.centroid_helpers import centroids_um  # noqa: E402
from lib.image_patches import (  # noqa: E402
    extract_cz_patches,
    extract_hcr_patches,
    PatchExtractConfig,
)


def main():
    sid = "788406"
    print(f"=== Loading {sid} ===", flush=True)
    s = load_subject(sid)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_gfp")
    print(f"  CZ centroids: {len(cz_um)}", flush=True)
    print(f"  HCR GFP+ centroids: {len(hcr_um)}", flush=True)

    cfg = PatchExtractConfig(patch_size=16, sample_spacing_um=4.0)

    # CZ
    print(f"\n=== Extract CZ patches for all {len(cz_um)} CZ centroids ===", flush=True)
    t0 = time.time()
    cz_p = extract_cz_patches(s, cz_um, cfg)
    t_cz = time.time() - t0
    print(f"  shape={cz_p.shape} dtype={cz_p.dtype}", flush=True)
    print(f"  runtime: {t_cz:.2f}s", flush=True)

    assert not np.isnan(cz_p).any(), "NaN found in CZ patches"
    assert not np.isinf(cz_p).any(), "Inf found in CZ patches"
    cz_stds = cz_p.reshape(cz_p.shape[0], -1).std(axis=1)
    n_finite_std = int((cz_stds > 0).sum())
    print(f"  patches with std>0: {n_finite_std}/{len(cz_p)} ({100.0*n_finite_std/len(cz_p):.1f}%)", flush=True)
    print(f"  patch value range: [{cz_p.min():.2f}, {cz_p.max():.2f}]", flush=True)

    # HCR — take first 1000 GFP+ to bound runtime
    n_test_hcr = min(1000, len(hcr_um))
    print(f"\n=== Extract HCR patches for {n_test_hcr} GFP+ centroids (level=2) ===", flush=True)
    t0 = time.time()
    hcr_p = extract_hcr_patches(s, hcr_um[:n_test_hcr], channel="488", level=2, config=cfg)
    t_hcr = time.time() - t0
    print(f"  shape={hcr_p.shape} dtype={hcr_p.dtype}", flush=True)
    print(f"  runtime: {t_hcr:.2f}s", flush=True)

    assert not np.isnan(hcr_p).any(), "NaN found in HCR patches"
    assert not np.isinf(hcr_p).any(), "Inf found in HCR patches"
    hcr_stds = hcr_p.reshape(hcr_p.shape[0], -1).std(axis=1)
    n_finite_std = int((hcr_stds > 0).sum())
    print(f"  patches with std>0: {n_finite_std}/{len(hcr_p)} ({100.0*n_finite_std/len(hcr_p):.1f}%)", flush=True)
    print(f"  patch value range: [{hcr_p.min():.2f}, {hcr_p.max():.2f}]", flush=True)

    ok_cz = (cz_stds > 0).sum() >= 0.99 * len(cz_p)
    ok_hcr = (hcr_stds > 0).sum() >= 0.99 * len(hcr_p)
    ok_runtime = t_cz <= 30.0
    print(f"\n=== Success criteria ===", flush=True)
    print(f"  CZ ≥99% finite-std: {'OK' if ok_cz else 'FAIL'}", flush=True)
    print(f"  HCR ≥99% finite-std: {'OK' if ok_hcr else 'FAIL'}", flush=True)
    print(f"  CZ runtime ≤ 30s for {len(cz_p)} cells: {'OK' if ok_runtime else 'FAIL'} ({t_cz:.2f}s)", flush=True)
    if ok_cz and ok_hcr and ok_runtime:
        print("\nALL GREEN", flush=True)
        return 0
    else:
        print("\nFAILED", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
