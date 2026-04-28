"""v2-S02 iter 6 — t_z fully derived from surface anchor at each sz.

For every candidate sz in the sweep:
    t_z(sz) = z_HCR_pia(at registered xy) + sz · cz_mean_depth_below_pia
i.e. the warped CZ pia always lands on the HCR pia at the registered
xy point.  Implemented via ``couple_tz=True, tz_search_half_um=0`` —
no free t_z search, no per-sz refinement.

For one diagnostic subject (788406), saves x-MIP side views of the
warped CZ slab and the HCR-488 target slab at sz ∈ {1.5, 2.0, 2.82,
3.5, 4.0} so the surface anchor can be visually verified.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_02/lib")
sys.path.insert(0, "/root/capsule/code/dev_code")
sys.path.insert(0, "/root/capsule/code/sessions/03c_onset_features/iterations")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_data_loader import (
    load_subject, BENCHMARK_SUBJECTS, landmark_pairs_um,
)
from benchmark_analysis import fit_anisotropic_similarity, load_hcr_volume
from iter08_cz_prior import load_cz_volume
from sz_estimator import (
    estimate_sz_image_ncc,
    HCR_LEVEL,
    _warp_cz_into_hcr_crop,
)
from locked_prior_warm import compute_locked_prior_warm_start
from surface_registration_v2 import get_surface_registration

OUT = Path("/root/capsule/code/full_automatic_execution_02/sessions/v2_S02_sz_image_sweep")
SIDE = OUT / "side_views_iter6"
SIDE.mkdir(exist_ok=True)

DIAG_SID = "788406"
DIAG_SZ_VALUES = [1.5, 2.0, 2.82, 3.5, 4.0]


def save_side_views(s, lp, sz_values):
    """Save x-MIP side views (z × y) of warped CZ and HCR slab at each sz."""
    reg = get_surface_registration(s)
    cz_vol = load_cz_volume(s).astype(np.float32, copy=False)
    hcr_vol, hcr_xy_um, hcr_z_um = load_hcr_volume(
        s, channel="488", level=HCR_LEVEL,
    )
    hcr_vol = np.asarray(hcr_vol, dtype=np.float32)
    cz_xy_um = float(s.cz_xy_um)
    cz_z_um = float(s.cz_z_um)
    crop_bbox_px = tuple(reg["crop_bbox"])
    y0, y1, x0, x1 = crop_bbox_px

    # cz_mean_depth_um: same calculation that estimate_sz_image_ncc uses
    from benchmark_analysis import depth_from_surface
    from surfaces_iter08 import get_cz_surface_iter08
    cz_surface = get_cz_surface_iter08(s)
    cz_mean_xyz_um = lp.src_mean[[2, 1, 0]]
    cz_mean_depth_um = float(
        depth_from_surface(cz_mean_xyz_um[None, :], cz_surface)[0]
    )
    sz_lp = float(lp.scales[0])
    cz_z_extent_um = cz_vol.shape[0] * cz_z_um

    sz_max = max(sz_values)
    half_warped_z = sz_max * cz_z_extent_um / 2
    z_lo_um = max(
        0.0,
        lp.translation[0] + (min(sz_values) - sz_lp) * cz_mean_depth_um
        - half_warped_z - 200.0,
    )
    z_hi_um = min(
        hcr_vol.shape[0] * hcr_z_um,
        lp.translation[0] + (sz_max - sz_lp) * cz_mean_depth_um
        + half_warped_z + 200.0,
    )
    z0_idx = int(np.floor(z_lo_um / hcr_z_um))
    z1_idx = int(np.ceil(z_hi_um / hcr_z_um))
    hcr_target = hcr_vol[z0_idx:z1_idx, y0:y1, x0:x1].astype(np.float32)

    # MIP-along-x for orientation: shows z × y plane (side view).
    hcr_mip = hcr_target.max(axis=2)
    Z = hcr_target.shape[0]

    fig, axes = plt.subplots(
        3, len(sz_values), figsize=(3.0 * len(sz_values), 9.0),
        constrained_layout=True,
    )
    fig.suptitle(
        f"{s.subject_id}  side-view (x-MIP, z×y)  "
        f"sz_lp={sz_lp:.2f}  cz_mean_depth={cz_mean_depth_um:.0f} µm  "
        f"slab z=[{z_lo_um:.0f},{z_hi_um:.0f}] µm  hcr_z={hcr_z_um:.2f}",
        fontsize=10,
    )

    extent = [0, hcr_target.shape[1] * hcr_xy_um, z_hi_um, z_lo_um]
    hcr_hi = float(np.percentile(hcr_mip, 99.5))
    for j, sz in enumerate(sz_values):
        tz_offset = (sz - sz_lp) * cz_mean_depth_um
        warped = _warp_cz_into_hcr_crop(
            cz_vol, lp, float(sz), tz_offset_um=tz_offset,
            cz_xy_um=cz_xy_um, cz_z_um=cz_z_um,
            hcr_xy_um=hcr_xy_um, hcr_z_um=hcr_z_um,
            crop_bbox_px=crop_bbox_px,
            z_lo_um=z_lo_um, z_hi_um=z_hi_um,
        )
        cz_mip = warped.max(axis=2)
        cz_hi = max(1.0, float(np.percentile(cz_mip[cz_mip > 0], 99))
                    if (cz_mip > 0).any() else 1.0)

        ax = axes[0, j]
        ax.imshow(cz_mip, cmap="magma", vmin=0, vmax=cz_hi, extent=extent,
                  aspect="auto")
        ax.set_title(
            f"warped CZ  sz={sz:.2f}  t_z={lp.translation[0] + tz_offset:.0f}µm",
            fontsize=8,
        )
        ax.set_ylabel("HCR z (µm)")
        if j == 0:
            ax.set_xlabel("HCR y (µm)")

        ax = axes[1, j]
        ax.imshow(hcr_mip, cmap="gray", vmin=0, vmax=hcr_hi, extent=extent,
                  aspect="auto")
        ax.set_title("HCR 488 (slab)", fontsize=8)

        ax = axes[2, j]
        # Overlay: HCR grayscale + warped-CZ red (alpha by intensity)
        rgb = np.zeros((*hcr_mip.shape, 4), dtype=float)
        h = np.clip(hcr_mip / max(hcr_hi, 1e-6), 0, 1)
        rgb[..., 0] = h
        rgb[..., 1] = h
        rgb[..., 2] = h
        rgb[..., 3] = 1.0
        c = np.clip(cz_mip / max(cz_hi, 1e-6), 0, 1)
        rgb[..., 0] = np.maximum(rgb[..., 0], c)
        rgb[..., 1] = rgb[..., 1] * (1 - c * 0.7)
        rgb[..., 2] = rgb[..., 2] * (1 - c * 0.7)
        ax.imshow(rgb, extent=extent, aspect="auto")
        ax.set_title("overlay", fontsize=8)
        # Annotate the surface-anchor line: the HCR pia z at the cz centroid xy.
        z_pia = lp.translation[0] + sz * cz_mean_depth_um \
                - sz * cz_mean_depth_um  # = lp.translation[0] is t_z(sz_lp)
        # Actually plot t_z(sz) line — that's where the warped CZ centroid
        # lands.  And where the pia should land = t_z(sz) - sz*cz_mean_depth.
        tz_sz = lp.translation[0] + tz_offset
        z_pia_warped = tz_sz - sz * cz_mean_depth_um
        ax.axhline(tz_sz, color="cyan", linestyle="--", linewidth=0.8,
                   label=f"t_z(sz)={tz_sz:.0f}")
        ax.axhline(z_pia_warped, color="lime", linestyle="--", linewidth=0.8,
                   label=f"warped CZ pia={z_pia_warped:.0f}")
        ax.legend(loc="upper right", fontsize=6)

    out_path = SIDE / f"side_view_{s.subject_id}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  saved side view → {out_path}", flush=True)


def main():
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        cz_xyz, hcr_xyz = landmark_pairs_um(s)
        fit = fit_anisotropic_similarity(cz_xyz[:, [2, 1, 0]], hcr_xyz[:, [2, 1, 0]])
        sz_gt = float(fit.scales[0])
        sxy_gt = float(fit.scales[1])

        if sid == DIAG_SID:
            print(f"  saving side views for {DIAG_SID} ...", flush=True)
            lp = compute_locked_prior_warm_start(s)
            save_side_views(s, lp, DIAG_SZ_VALUES)

        t0 = time.time()
        res = estimate_sz_image_ncc(
            s,
            sz_grid=np.arange(1.5, 4.01, 0.10),
            scoring="smoothed_voxel",
            couple_tz=True,         # surface-anchored tz at each sz
            tz_search_half_um=0.0,  # no free search around the anchor
            verbose=False,
        )
        elapsed = time.time() - t0

        err = res.sz_peak - sz_gt if res.sz_peak is not None else float("nan")
        print(
            f"  sz_lp={res.sz_lp:.3f}  sz_peak={res.sz_peak}  sz_gt={sz_gt:.3f}  "
            f"err={err:+.3f}  ratio={res.peak_ratio:.3f}  HW={res.half_width:.3f}  "
            f"passed={res.passed}  ({elapsed:.0f}s)",
            flush=True,
        )

        rows.append({
            "subject_id": sid,
            "sz_gt": sz_gt,
            "sxy_gt": sxy_gt,
            "sz_lp": res.sz_lp,
            "sz_peak": res.sz_peak,
            "ncc_peak": res.ncc_peak,
            "ncc_median": res.ncc_median,
            "peak_ratio": res.peak_ratio,
            "half_width": res.half_width,
            "passed": res.passed,
            "fail_reason": res.fail_reason,
            "tz_offset_um": res.tz_offset_um,
            "sz_err_vs_gt": err,
            "runtime_s": round(elapsed, 1),
        })

        sweep = pd.DataFrame({
            "sz": res.sz_grid,
            "ncc": res.ncc_grid,
            "tz_offset_um": [d.get("tz_offset_um", np.nan)
                              for d in res.diagnostics["sweep_rows"]],
        })
        sweep.to_csv(OUT / f"sweep_iter6_{sid}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_iter6.csv", index=False)
    print("\n=== iter 6 (anchored t_z) summary ===")
    print(df[["subject_id", "sz_gt", "sz_peak", "sz_err_vs_gt", "peak_ratio",
              "half_width", "passed", "runtime_s"]].to_string(index=False))
    print(f"\npassed: {df['passed'].sum()}/{len(df)}")
    print(f"mean abs err: {df['sz_err_vs_gt'].abs().mean():.3f}")
    print(f"within ±0.30 GT: {(df['sz_err_vs_gt'].abs() <= 0.30).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
