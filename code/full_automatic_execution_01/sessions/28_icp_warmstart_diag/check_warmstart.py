"""Compare ICP-derived warm-start vs landmark-perfect warp, per GT pair."""
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
from benchmark_analysis import fit_anisotropic_similarity
from lib.centroid_helpers import centroids_um, default_warmstart_zyx


def main(sid="788406"):
    s = load_subject(sid)
    cz_zyx, cz_ids = centroids_um(s, "cz")
    hcr_zyx, hcr_ids = centroids_um(s, "hcr_gfp")
    print(f"CZ {cz_zyx.shape}, HCR {hcr_zyx.shape}")

    cz_init_icp, ws = default_warmstart_zyx(cz_zyx, hcr_zyx, use_icp=True)
    print(f"WS source: {ws.get('source')}")
    print(f"scales_zyx: {ws.get('scales_zyx')}")

    # GT landmarks
    from benchmark_data_loader import landmark_pairs_um
    cz_lm_xyz, hcr_lm_xyz = landmark_pairs_um(s)
    cz_lm_zyx = cz_lm_xyz[:, [2, 1, 0]]
    hcr_lm_zyx = hcr_lm_xyz[:, [2, 1, 0]]
    print(f"landmarks: {cz_lm_zyx.shape}")

    # Fit per-axis anisotropic similarity on landmarks (zyx convention)
    fit_lm = fit_anisotropic_similarity(cz_lm_zyx, hcr_lm_zyx)
    print(f"LM fit: scales={fit_lm.scales}, rms={fit_lm.rms_um:.2f} um")

    # Apply LM fit to every CZ cell; the forward formula used by fit module
    # is pred = ((src-src_mean) @ R) * scales + dst_mean
    from lib.centroid_helpers import apply_aniso_fit
    cz_lm_pred = apply_aniso_fit(cz_zyx, fit_lm)

    # Also look up coreg_table pairings between CZ cells and HCR cells
    ct = s.coreg_table.copy()
    id_to_cz_idx = {int(i): k for k, i in enumerate(cz_ids)}
    id_to_hcr_idx = {int(i): k for k, i in enumerate(hcr_ids)}
    rows = []
    for _, r in ct.iterrows():
        if int(r.cz_id) in id_to_cz_idx and int(r.hcr_id) in id_to_hcr_idx:
            rows.append((id_to_cz_idx[int(r.cz_id)],
                         id_to_hcr_idx[int(r.hcr_id)]))
    idx_cz = np.array([r[0] for r in rows])
    idx_hcr = np.array([r[1] for r in rows])
    print(f"GT pairs (both sides in GFP+): {len(rows)}")

    # Per-GT distance under ICP warp
    d_icp = np.linalg.norm(cz_init_icp[idx_cz] - hcr_zyx[idx_hcr], axis=1)
    # Per-GT distance under LM warp
    d_lm = np.linalg.norm(cz_lm_pred[idx_cz] - hcr_zyx[idx_hcr], axis=1)
    print()
    print("Per-GT distance (um) under ICP warm-start:")
    print(f"  median={np.median(d_icp):.1f}, mean={d_icp.mean():.1f}, "
          f"p90={np.quantile(d_icp,0.9):.1f}")
    print(f"  count <10um: {(d_icp<10).sum()}, <20um: {(d_icp<20).sum()}, "
          f"<50um: {(d_icp<50).sum()} / {len(d_icp)}")
    print()
    print("Per-GT distance (um) under landmark-fit warp:")
    print(f"  median={np.median(d_lm):.1f}, mean={d_lm.mean():.1f}, "
          f"p90={np.quantile(d_lm,0.9):.1f}")
    print(f"  count <10um: {(d_lm<10).sum()}, <20um: {(d_lm<20).sum()}, "
          f"<50um: {(d_lm<50).sum()} / {len(d_lm)}")


if __name__ == "__main__":
    main()
