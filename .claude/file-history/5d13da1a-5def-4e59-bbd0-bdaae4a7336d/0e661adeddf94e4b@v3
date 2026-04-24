"""Visualize tilt-aligned rotation and surface matching for each subject.

Produces two figures under sessions/05_R1_revised/figures/:

- tilt_alignment_side_views.png  — XZ and YZ projections of pia surfaces
                                   pre- and post-tilt correction, per subject.
- surface_overlay_3d.png         — 3D surface patches (HCR vs tilt-aligned
                                   CZ) after centroid match, per subject.

Per subject we compute, from the HCR frame:

* HCR pia sampled on a grid over its XY envelope.
* CZ pia sampled on a grid over CZ's XY envelope, then two transforms:
    - R_180 only (no tilt correction) + centroid match to HCR mean;
    - R_full = R_tilt · R_180 (full revised-R1 rotation) + centroid match.

No translation from the actual R1 output is used — we show only the
*rotational + centroid-match* effect, which is what step 2 of §R1 does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from benchmark_analysis import analyze_subject
from benchmark_data_loader import load_subject
from r1_revised import (
    _plane_normal_from_surface,
    _rotation_about_z_row,
    _rotation_between_row,
    _surface_z_at,
)

OUT_DIR = _THIS_DIR.parent / "sessions" / "05_R1_revised" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["788406", "790322", "767018", "782149", "755252", "767022"]


def sample_surface(surface: dict, xy_points: np.ndarray, n: int = 40):
    xy = np.asarray(xy_points)
    xlo, xhi = xy[:, 0].min(), xy[:, 0].max()
    ylo, yhi = xy[:, 1].min(), xy[:, 1].max()
    xs = np.linspace(xlo, xhi, n)
    ys = np.linspace(ylo, yhi, n)
    X, Y = np.meshgrid(xs, ys)
    Z = _surface_z_at(surface, X.ravel(), Y.ravel()).reshape(X.shape)
    return X, Y, Z


def compute_subject(sid: str):
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_xyz = info["cz_xyz"]
    gfp_xyz = info["gfp_xyz"]
    cz_surface = info["cz_surface"]
    hcr_surface = info["hcr_surface"]

    n_cz, _ = _plane_normal_from_surface(cz_surface, cz_xyz[:, :2])
    n_hcr, _ = _plane_normal_from_surface(hcr_surface, gfp_xyz[:, :2])

    R_180 = _rotation_about_z_row(-180.0)
    n_cz_rot = n_cz @ R_180
    R_tilt = _rotation_between_row(n_cz_rot, n_hcr)
    R_full = R_180 @ R_tilt

    X_cz, Y_cz, Z_cz = sample_surface(cz_surface, cz_xyz[:, :2], n=40)
    cz_pts = np.column_stack([X_cz.ravel(), Y_cz.ravel(), Z_cz.ravel()])
    cz_mean = cz_xyz.mean(axis=0)
    hcr_mean = gfp_xyz.mean(axis=0)

    # Translation chosen to align the pia PLANES at the CZ-XY centroid, so
    # R_tilt's effect is isolated from cell-centroid Z offsets (which exist
    # because CZ cells span ~400 µm while HCR GFP+ cells span ~1–1.5 mm at
    # sz = 1).  Everything below is in HCR coordinates.
    cz_xy_centroid = cz_mean[:2]
    # Evaluate each pia at the CZ XY centroid (pre-R in its own frame)
    z_cz_pia_at_centroid = _surface_z_at(
        cz_surface, np.array([cz_xy_centroid[0]]),
        np.array([cz_xy_centroid[1]]))[0]
    # Where does R_180·(cz_xy_centroid - cz_mean_xy) land in HCR XY?
    cz_centroid_3d = np.array([cz_xy_centroid[0], cz_xy_centroid[1],
                                z_cz_pia_at_centroid])
    cz_centroid_pre_hcr = (cz_centroid_3d - cz_mean) @ R_180
    cz_centroid_post_hcr = (cz_centroid_3d - cz_mean) @ R_full

    def _pia_aligned_translation(cz_centroid_hcr: np.ndarray) -> np.ndarray:
        """t such that transformed CZ pia matches HCR pia at HCR XY of the CZ centroid."""
        xy_in_hcr = cz_centroid_hcr[:2] + hcr_mean[:2]
        z_hcr_pia = _surface_z_at(hcr_surface,
                                  np.array([xy_in_hcr[0]]),
                                  np.array([xy_in_hcr[1]]))[0]
        t = np.array([hcr_mean[0], hcr_mean[1],
                      z_hcr_pia - cz_centroid_hcr[2]])
        return t

    t_pre = _pia_aligned_translation(cz_centroid_pre_hcr)
    t_post = _pia_aligned_translation(cz_centroid_post_hcr)

    cz_pre = (cz_pts - cz_mean) @ R_180 + t_pre
    cz_post = (cz_pts - cz_mean) @ R_full + t_post

    # HCR surface grid over its own XY envelope
    X_h, Y_h, Z_h = sample_surface(hcr_surface, gfp_xyz[:, :2], n=40)
    hcr_pts = np.column_stack([X_h.ravel(), Y_h.ravel(), Z_h.ravel()])

    # tilt angle before and after correction
    cos_pre = float(np.clip(np.dot(n_cz_rot, n_hcr), -1, 1))
    tilt_deg_pre = float(np.degrees(np.arccos(cos_pre)))
    tilt_deg_post = 0.0  # by construction

    return dict(
        sid=sid,
        cz_pre=cz_pre.reshape(X_cz.shape + (3,)),
        cz_post=cz_post.reshape(X_cz.shape + (3,)),
        hcr_grid=hcr_pts.reshape(X_h.shape + (3,)),
        tilt_deg_pre=tilt_deg_pre,
        tilt_deg_post=tilt_deg_post,
        cz_tilt=np.degrees(np.arccos(float(np.clip(n_cz[2], -1, 1)))),
        hcr_tilt=np.degrees(np.arccos(float(np.clip(n_hcr[2], -1, 1)))),
    )


def fig_side_views(rows):
    fig, axs = plt.subplots(6, 2, figsize=(11, 14), sharex=False)
    for ri, r in enumerate(rows):
        for ci, axis_pair in enumerate([(0, 2), (1, 2)]):  # XZ, YZ
            ax = axs[ri, ci]
            a, z = axis_pair
            hcr = r["hcr_grid"]
            pre = r["cz_pre"]
            post = r["cz_post"]
            # take a near-median slice on the orthogonal axis to show a clean curve
            other = 1 - a if a < 2 else 0
            # HCR: scatter all grid points (small, faint)
            ax.scatter(hcr[..., a].ravel(), hcr[..., z].ravel(),
                       s=2, color="#ff7f0e", alpha=0.25, label="HCR pia")
            ax.scatter(pre[..., a].ravel(), pre[..., z].ravel(),
                       s=2, color="#aaaaaa", alpha=0.35,
                       label="CZ: R_180 only")
            ax.scatter(post[..., a].ravel(), post[..., z].ravel(),
                       s=2, color="#1f77b4", alpha=0.5,
                       label="CZ: R_180 · R_tilt")
            # centre-row line for visual clarity
            mid = hcr.shape[other] // 2
            if a == 0:
                ax.plot(hcr[mid, :, 0], hcr[mid, :, 2], color="#ff7f0e", lw=1.2)
                ax.plot(pre[mid, :, 0], pre[mid, :, 2], color="#666", lw=1.0, ls="--")
                ax.plot(post[mid, :, 0], post[mid, :, 2], color="#1f77b4", lw=1.2)
            else:
                ax.plot(hcr[:, mid, 1], hcr[:, mid, 2], color="#ff7f0e", lw=1.2)
                ax.plot(pre[:, mid, 1], pre[:, mid, 2], color="#666", lw=1.0, ls="--")
                ax.plot(post[:, mid, 1], post[:, mid, 2], color="#1f77b4", lw=1.2)
            ax.invert_yaxis()  # pia at top
            ax.set_xlabel("x (µm)" if a == 0 else "y (µm)")
            ax.set_ylabel("z (µm)")
            ax.grid(alpha=0.3)
            if ci == 0 and ri == 0:
                ax.legend(fontsize=7, loc="lower right")
            view = "XZ" if a == 0 else "YZ"
            if ci == 0:
                ax.set_title(
                    f"{r['sid']}  {view}   "
                    f"(Δtilt {r['tilt_deg_pre']:.1f}° → 0°;  "
                    f"cz {r['cz_tilt']:.1f}°, hcr {r['hcr_tilt']:.1f}°)",
                    fontsize=9)
            else:
                ax.set_title(f"{r['sid']}  {view}", fontsize=9)
    fig.suptitle(
        "Tilt-aligned rotation: HCR pia (orange) vs CZ pia after R_180 only (grey) "
        "vs CZ pia after full R = R_180·R_tilt (blue)\n"
        "Translation chosen to pia-align at the CZ XY centroid (isolates R_tilt's "
        "effect). Everything in HCR coordinates.",
        y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tilt_alignment_side_views.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_3d_overlay(rows):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(15, 9))
    for i, r in enumerate(rows):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        h = r["hcr_grid"]; post = r["cz_post"]; pre = r["cz_pre"]
        ax.plot_surface(h[..., 0], h[..., 1], h[..., 2], color="#ff7f0e",
                        alpha=0.25, edgecolor="none")
        ax.plot_surface(pre[..., 0], pre[..., 1], pre[..., 2], color="#aaaaaa",
                        alpha=0.35, edgecolor="none")
        ax.plot_surface(post[..., 0], post[..., 1], post[..., 2], color="#1f77b4",
                        alpha=0.6, edgecolor="none")
        ax.invert_zaxis()
        ax.set_xlabel("x (µm)", fontsize=8)
        ax.set_ylabel("y (µm)", fontsize=8)
        ax.set_zlabel("z (µm)", fontsize=8)
        ax.set_title(
            f"{r['sid']}  Δtilt {r['tilt_deg_pre']:.1f}°\n"
            f"cz {r['cz_tilt']:.1f}°, hcr {r['hcr_tilt']:.1f}°",
            fontsize=9)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=18, azim=-60)
    fig.suptitle(
        "Surface matching after tilt-aligned rotation (pia-anchored translation)\n"
        "orange = HCR pia (target),  grey = CZ pia under R_180 only,  "
        "blue = CZ pia under full R = R_180·R_tilt",
        y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "surface_overlay_3d.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = []
    for sid in SUBJECTS:
        print(f"[{sid}]", flush=True)
        rows.append(compute_subject(sid))
    fig_side_views(rows)
    fig_3d_overlay(rows)
    print(f"Wrote figures to {OUT_DIR}")
    for f in ["tilt_alignment_side_views.png", "surface_overlay_3d.png"]:
        print(f"  {f}")
