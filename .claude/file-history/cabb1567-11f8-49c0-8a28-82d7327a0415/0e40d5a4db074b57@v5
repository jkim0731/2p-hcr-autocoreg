"""Generate walkthrough figures for the image-based curved-surface
estimator.  Two rows (HCR / CZ) × two columns (XZ / YZ MIP) with the
fitted quadratic surface overlaid; plus the global intensity profile
at the bottom showing the anchor threshold.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path("/root/capsule")
DEV = ROOT / "code" / "dev_code"
sys.path.insert(0, str(DEV))

_spec = importlib.util.spec_from_file_location(
    "img_surface_curved", str(DEV / "03_image_based_surface.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["img_surface_curved"] = _mod
_spec.loader.exec_module(_mod)
estimate_surface_and_l2_image_based = _mod.estimate_surface_and_l2_image_based
_surface_z = _mod._surface_z

from benchmark_analysis import load_hcr_combined
from benchmark_data_loader import (
    BENCHMARK_SUBJECTS,
    load_subject,
)

OUT_TMP = Path("/tmp/03_image_based_surface")
OUT_SESSION = ROOT / "code" / "sessions" / "03_image_based_surface_estimation"
(OUT_TMP / "figures").mkdir(parents=True, exist_ok=True)
(OUT_SESSION / "figures").mkdir(parents=True, exist_ok=True)


def _load_cz_image(s):
    cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not cz_tifs:
        cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
    if not cz_tifs:
        return None
    img = tifffile.imread(str(cz_tifs[0]))
    while img.ndim > 3 and img.shape[0] == 1:
        img = img[0]
    return img.astype(np.float32, copy=False)


def _draw_surface_xz(ax, surface, x_min, x_max, y0, n=400):
    xs = np.linspace(x_min, x_max, n)
    ys = np.full_like(xs, y0)
    if surface is not None:
        z_surf = _surface_z(surface, xs, ys)
        ax.plot(xs, z_surf, color="red", lw=1.6, label="surface (quad)")


def _draw_surface_yz(ax, surface, y_min, y_max, x0, n=400):
    ys = np.linspace(y_min, y_max, n)
    xs = np.full_like(ys, x0)
    if surface is not None:
        z_surf = _surface_z(surface, xs, ys)
        ax.plot(ys, z_surf, color="red", lw=1.6, label="surface (quad)")


def render_subject(subject_id: str, save_path: Path):
    s = load_subject(subject_id)

    # HCR
    vol_hcr, hcr_xy_um, hcr_z_um, _ch = load_hcr_combined(s, level=4)
    r_hcr = estimate_surface_and_l2_image_based(
        vol_hcr, hcr_z_um, hcr_xy_um,
        n_top_planes=2,
        k_surface=6.0,
        relative_margin_min=0.02,
        xy_tile_um=80.0,
        surface_search_frac=0.30,
        min_thick_surface_um=12.0,
        smooth_sigma_um=6.0,
        target_quantile=0.85,
    )

    # CZ
    cz_img = _load_cz_image(s)
    r_cz = estimate_surface_and_l2_image_based(
        cz_img, s.cz_z_um, s.cz_xy_um,
        n_top_planes=2,
        k_surface=6.0,
        relative_margin_min=0.02,
        xy_tile_um=60.0,
        surface_search_frac=0.30,
        min_thick_surface_um=10.0,
        smooth_sigma_um=4.0,
        target_quantile=0.70,
    )

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_h_xz = fig.add_subplot(gs[0, 0])
    ax_h_yz = fig.add_subplot(gs[0, 1])
    ax_c_xz = fig.add_subplot(gs[1, 0])
    ax_c_yz = fig.add_subplot(gs[1, 1])
    ax_h_prof = fig.add_subplot(gs[2, 0])
    ax_c_prof = fig.add_subplot(gs[2, 1])

    Z_h, Y_h, X_h = vol_hcr.shape
    half = min(20, Y_h // 4)
    y_mid = Y_h // 2; x_mid = X_h // 2
    mip_h_xz = vol_hcr[:, y_mid - half:y_mid + half, :].max(axis=1)
    mip_h_yz = vol_hcr[:, :, x_mid - half:x_mid + half].max(axis=2)
    X_h_um = X_h * hcr_xy_um
    Y_h_um = Y_h * hcr_xy_um

    ax_h_xz.imshow(
        mip_h_xz, extent=(0, X_h_um, Z_h * hcr_z_um, 0),
        aspect="auto", cmap="gray",
        vmin=np.percentile(mip_h_xz, 1), vmax=np.percentile(mip_h_xz, 99.5),
    )
    _draw_surface_xz(ax_h_xz, r_hcr.surface,
                     0, X_h_um, y_mid * hcr_xy_um)
    ax_h_xz.set_title(f"HCR {subject_id} — XZ MIP @ y={y_mid * hcr_xy_um:.0f} µm")
    ax_h_xz.set_xlabel("x (µm)"); ax_h_xz.set_ylabel("z (µm)")
    ax_h_xz.legend(loc="lower right", fontsize=8)

    ax_h_yz.imshow(
        mip_h_yz, extent=(0, Y_h_um, Z_h * hcr_z_um, 0),
        aspect="auto", cmap="gray",
        vmin=np.percentile(mip_h_yz, 1), vmax=np.percentile(mip_h_yz, 99.5),
    )
    _draw_surface_yz(ax_h_yz, r_hcr.surface,
                     0, Y_h_um, x_mid * hcr_xy_um)
    ax_h_yz.set_title(f"HCR YZ MIP @ x={x_mid * hcr_xy_um:.0f} µm")
    ax_h_yz.set_xlabel("y (µm)"); ax_h_yz.set_ylabel("z (µm)")
    ax_h_yz.legend(loc="lower right", fontsize=8)

    # CZ
    Z_c, Y_c, X_c = cz_img.shape
    halfc = min(20, Y_c // 4)
    yc_mid = Y_c // 2; xc_mid = X_c // 2
    mip_c_xz = cz_img[:, yc_mid - halfc:yc_mid + halfc, :].max(axis=1)
    mip_c_yz = cz_img[:, :, xc_mid - halfc:xc_mid + halfc].max(axis=2)
    X_c_um = X_c * s.cz_xy_um
    Y_c_um = Y_c * s.cz_xy_um

    ax_c_xz.imshow(
        mip_c_xz, extent=(0, X_c_um, Z_c * s.cz_z_um, 0),
        aspect="auto", cmap="gray",
        vmin=np.percentile(mip_c_xz, 1), vmax=np.percentile(mip_c_xz, 99.5),
    )
    _draw_surface_xz(ax_c_xz, r_cz.surface,
                     0, X_c_um, yc_mid * s.cz_xy_um)
    ax_c_xz.set_title(f"CZ {subject_id} — XZ MIP @ y={yc_mid * s.cz_xy_um:.0f} µm")
    ax_c_xz.set_xlabel("x (µm)"); ax_c_xz.set_ylabel("z (µm)")
    ax_c_xz.legend(loc="lower right", fontsize=8)

    ax_c_yz.imshow(
        mip_c_yz, extent=(0, Y_c_um, Z_c * s.cz_z_um, 0),
        aspect="auto", cmap="gray",
        vmin=np.percentile(mip_c_yz, 1), vmax=np.percentile(mip_c_yz, 99.5),
    )
    _draw_surface_yz(ax_c_yz, r_cz.surface,
                     0, Y_c_um, xc_mid * s.cz_xy_um)
    ax_c_yz.set_title(f"CZ YZ MIP @ x={xc_mid * s.cz_xy_um:.0f} µm")
    ax_c_yz.set_xlabel("y (µm)"); ax_c_yz.set_ylabel("z (µm)")
    ax_c_yz.legend(loc="lower right", fontsize=8)

    # Intensity profiles
    prof_h = np.median(vol_hcr.reshape(Z_h, -1), axis=1)
    zs_h = np.arange(Z_h) * hcr_z_um
    ax_h_prof.plot(zs_h, prof_h, "k-", lw=1, label="global median")
    ax_h_prof.axhline(r_hcr.surf_thr, color="red", ls="--", lw=0.8,
                      label=f"thr={r_hcr.surf_thr:.3f}"
                            f" (bg={r_hcr.baseline_mean:.3f}"
                            f" σ={r_hcr.baseline_sigma:.3f})")
    if r_hcr.surface is not None:
        z_s_mid = _surface_z(r_hcr.surface, X_h_um / 2, Y_h_um / 2)
        ax_h_prof.axvline(float(z_s_mid), color="red", lw=1,
                          label=f"surf @ centre={float(z_s_mid):.0f} µm")
    ax_h_prof.set_xlabel("z (µm)"); ax_h_prof.set_ylabel("median intensity")
    ax_h_prof.set_title("HCR global median profile (combined)")
    ax_h_prof.legend(fontsize=7)

    prof_c = np.median(cz_img.reshape(Z_c, -1), axis=1)
    zs_c = np.arange(Z_c) * s.cz_z_um
    ax_c_prof.plot(zs_c, prof_c, "k-", lw=1, label="all-voxel median")
    ax_c_prof.axhline(r_cz.surf_thr, color="red", ls="--", lw=0.8,
                      label=f"thr={r_cz.surf_thr:.1f}"
                            f" (bg={r_cz.baseline_mean:.1f}"
                            f" σ={r_cz.baseline_sigma:.1f})")
    if r_cz.surface is not None:
        z_s_mid = _surface_z(r_cz.surface, X_c_um / 2, Y_c_um / 2)
        ax_c_prof.axvline(float(z_s_mid), color="red", lw=1,
                          label=f"surf @ centre={float(z_s_mid):.0f} µm")
    ax_c_prof.set_xlabel("z (µm)"); ax_c_prof.set_ylabel("median intensity")
    ax_c_prof.set_title("CZ global median profile")
    ax_c_prof.legend(fontsize=7)

    fig.suptitle(
        f"{subject_id}: image-based curved surface "
        f"(HCR tiles {r_hcr.n_surface_tiles}/{r_hcr.n_tiles_total}; "
        f"CZ tiles {r_cz.n_surface_tiles}/{r_cz.n_tiles_total})",
        fontsize=11,
    )
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"saved {save_path}")


def main(subjects=None):
    subjects = subjects or BENCHMARK_SUBJECTS
    for sid in subjects:
        tgt_tmp = OUT_TMP / "figures" / f"walkthrough_{sid}.png"
        render_subject(sid, tgt_tmp)
        try:
            import shutil
            tgt_sess = OUT_SESSION / "figures" / f"walkthrough_{sid}.png"
            shutil.copy(tgt_tmp, tgt_sess)
        except Exception as exc:
            print("mirror failed:", exc)


if __name__ == "__main__":
    main()
