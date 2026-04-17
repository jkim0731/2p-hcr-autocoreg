"""Run the benchmark analysis on all 6 subjects and save tables + figures.

Usage: python code/dev_code/01_analyze_benchmark.py

Writes outputs to /root/capsule/code/sessions/01_analyze_benchmark_data/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject
from benchmark_analysis import (
    analyze_subject,
    depth_from_surface,
    depth_profile,
    estimate_pia_surface_from_image,
    load_cz_y_slab,
    load_hcr_y_slab,
    load_hcr_volume,
    load_hcr_combined,
    plot_pia_overlay,
)

OUT_DIR = Path("/root/capsule/code/sessions/01_analyze_benchmark_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    results = {}
    subjects = {}
    for sid in BENCHMARK_SUBJECTS:
        print(f"Loading {sid}...")
        s = load_subject(sid)
        subjects[sid] = s
        print(f"Analyzing {sid}...")
        results[sid] = analyze_subject(s)

    # ---- Summary table ----
    summary = pd.DataFrame([results[sid]["summary"] for sid in BENCHMARK_SUBJECTS])
    summary.to_csv(OUT_DIR / "summary_table.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string())

    # ---- Procrustes / expansion table ----
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        p = results[sid]["procrustes"]
        if p is None:
            rows.append({"subject": sid})
            continue
        sx, sy, sz = p.scales
        rows.append(
            {
                "subject": sid,
                "n_active_lm": len(p.residuals_um),
                "scale_x": sx,
                "scale_y": sy,
                "scale_z": sz,
                "xy_mean_scale": (sx + sy) / 2,
                "anisotropy_xy_over_z": ((sx + sy) / 2) / sz,
                "rotation_z_deg": p.rotation_angle_z_deg,
                "residual_rms_um": p.rms_um,
                "residual_mean_um": p.mean_um,
                "residual_max_um": p.max_um,
            }
        )
    expansion = pd.DataFrame(rows)
    expansion.to_csv(OUT_DIR / "expansion_table.csv", index=False)
    print("\n=== Expansion ===")
    print(expansion.to_string())

    # ---- Surface table (image-based primary + centroid comparison) ----
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        r = results[sid]
        row = {"subject": sid}
        for mod, xyz in (("cz", r["cz_xyz"]), ("hcr", r["hcr_xyz"])):
            img = r[f"{mod}_surface_image"]
            cen = r[f"{mod}_surface_centroid"]
            hyb = r.get(f"{mod}_surface_hybrid")
            used = ("hybrid" if hyb is not None and mod == "hcr"
                    else ("image" if img is not None
                          else ("centroid" if cen is not None else "none")))
            row[f"{mod}_source"] = used
            if img is not None:
                row[f"{mod}_img_c_um"] = img["c"]
                row[f"{mod}_img_tilt"] = img["tilt_deg"]
                row[f"{mod}_img_rough"] = img["residual_std_um"]
                d = depth_from_surface(xyz, img)
                row[f"{mod}_img_frac_above_pia"] = float((d < -5).sum()) / max(len(d), 1)
            if hyb is not None:
                row[f"{mod}_hyb_c_um"] = hyb["c"]
                row[f"{mod}_hyb_tilt"] = hyb["tilt_deg"]
                row[f"{mod}_hyb_rough"] = hyb["residual_std_um"]
                d = depth_from_surface(xyz, hyb)
                row[f"{mod}_hyb_frac_above_pia"] = float((d < -5).sum()) / max(len(d), 1)
            if cen is not None:
                row[f"{mod}_cen_c_um"] = cen["c"]
                row[f"{mod}_cen_tilt"] = cen["tilt_deg"]
                row[f"{mod}_cen_rough"] = cen["residual_std_um"]
                d = depth_from_surface(xyz, cen)
                row[f"{mod}_cen_frac_above_pia"] = float((d < -5).sum()) / max(len(d), 1)
        rows.append(row)
    surface = pd.DataFrame(rows)
    surface.to_csv(OUT_DIR / "surface_table.csv", index=False)
    print("\n=== Surfaces ===")
    print(surface.to_string())

    # ---- NN distance table ----
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        r = results[sid]
        row = {"subject": sid}
        for label, d in [("cz", r["cz_nn"]), ("hcr_all", r["hcr_nn"]), ("hcr_gfp", r["gfp_nn"])]:
            if d is None:
                continue
            row[f"{label}_mean_1nn_um"] = d["mean_1nn"]
            row[f"{label}_mean_5nn_um"] = d["mean_knn"]
        rows.append(row)
    nn = pd.DataFrame(rows)
    nn.to_csv(OUT_DIR / "nn_distance_table.csv", index=False)
    print("\n=== NN distances ===")
    print(nn.to_string())

    # ---- Depth profile figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, sid in zip(axes.flat, BENCHMARK_SUBJECTS):
        r = results[sid]
        p = results[sid]["procrustes"]
        z_exp = p.scales[2] if p is not None else 1.0

        if r["cz_surface"] is not None:
            c, d = depth_profile(r["cz_xyz"], r["cz_surface"], bin_um=20, depth_range=(-100, 1500))
            # Rescale CZ depth by Z expansion to compare with HCR on common axis
            ax.plot(c * z_exp, d / d.max() if d.max() else d, label=f"CZ (×{z_exp:.2f})", color="tab:blue")
        if r["hcr_surface"] is not None:
            c, d = depth_profile(r["hcr_xyz"], r["hcr_surface"], bin_um=20, depth_range=(-100, 1500))
            ax.plot(c, d / d.max() if d.max() else d, label="HCR all", color="tab:orange", alpha=0.7)
        if r["hcr_surface"] is not None and len(r["gfp_xyz"]) > 100:
            c, d = depth_profile(r["gfp_xyz"], r["hcr_surface"], bin_um=20, depth_range=(-100, 1500))
            ax.plot(c, d / d.max() if d.max() else d, label="HCR GFP+", color="tab:green", alpha=0.7)
        ax.set_title(sid)
        ax.set_xlabel("depth from pia (um)")
        ax.set_ylabel("normalized density")
        ax.set_xlim(-100, 1500)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "depth_profiles.png", dpi=120)
    plt.close(fig)
    print(f"Saved depth_profiles.png")

    # ---- Expansion factor figure ----
    if not expansion.empty and "scale_x" in expansion.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        subs = expansion["subject"].values
        xw = np.arange(len(subs))
        ax.bar(xw - 0.2, expansion["xy_mean_scale"], width=0.4, label="XY scale")
        ax.bar(xw + 0.2, expansion["scale_z"], width=0.4, label="Z scale")
        ax.set_xticks(xw)
        ax.set_xticklabels(subs, rotation=30)
        ax.set_ylabel("CZ → HCR expansion factor (physical um)")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.legend()
        ax.set_title("Anisotropic expansion from in-vivo 2p to ex-vivo HCR")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "expansion_factors.png", dpi=120)
        plt.close(fig)
        print("Saved expansion_factors.png")

    # ---- Residual magnitude figure ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    data = []
    labels = []
    for sid in BENCHMARK_SUBJECTS:
        p = results[sid]["procrustes"]
        if p is None:
            continue
        data.append(np.linalg.norm(p.residuals_um, axis=1))
        labels.append(sid)
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Landmark residual after anisotropic similarity (um)")
    ax.set_title("Nonrigid deformation magnitude (residual after affine fit)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_magnitude.png", dpi=120)
    plt.close(fig)
    print("Saved residual_magnitude.png")

    # ---- Pia overlay figures with BOTH image and centroid-based surfaces ----
    for sid in ("788406", "782149"):
        s = subjects[sid]
        r = results[sid]
        try:
            cz_mip, cz_yc, cz_z_um, cz_x_um = load_cz_y_slab(s, half_width_px=20)
            # HCR: take a y-slab from the combined-channel volume at level 2
            # (same resolution as the centroid NPY).  We just average channels for
            # the visualization — level-2 is too large to full-load here, so use
            # 405 level-2 for the background image.
            hcr_mip, hcr_yc, hcr_z_um, hcr_x_um = load_hcr_y_slab(
                s, channel="405", half_width_um=30.0, level=2
            )
        except Exception as e:
            print(f"Skipping pia overlay for {sid}: {e}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def overlay(ax, mip, zum, xum, surf_img, surf_cen, y_slab, title):
            Zn, Xn = mip.shape
            extent = (0, Xn * xum, Zn * zum, 0)
            vmax = np.percentile(mip, 99.5)
            ax.imshow(mip, extent=extent, cmap="gray", vmin=np.percentile(mip, 1),
                      vmax=vmax, aspect="auto", interpolation="nearest")
            xs = np.linspace(0, Xn * xum, 200)
            if surf_img is not None:
                ys_img = surf_img["a"] * xs + surf_img["b"] * y_slab + surf_img["c"]
                ax.plot(xs, ys_img, color="red", lw=1.8,
                        label=f"image-based (tilt {surf_img['tilt_deg']:.1f}°)")
            if surf_cen is not None:
                ys_cen = surf_cen["a"] * xs + surf_cen["b"] * y_slab + surf_cen["c"]
                ax.plot(xs, ys_cen, color="yellow", lw=1.2, ls="--",
                        label=f"centroid-based (tilt {surf_cen['tilt_deg']:.1f}°)")
            ax.set_xlabel("x (um)"); ax.set_ylabel("z (um)")
            ax.set_title(title); ax.legend(loc="lower right", fontsize=8)

        overlay(
            axes[0], cz_mip, cz_z_um, cz_x_um,
            r["cz_surface_image"], r["cz_surface_centroid"],
            cz_yc * s.cz_xy_um, f"{sid} CZStack  (y-slab MIP, ±20 px)"
        )
        overlay(
            axes[1], hcr_mip, hcr_z_um, hcr_x_um,
            r["hcr_surface_image"], r["hcr_surface_centroid"],
            hcr_yc, f"{sid} HCR 405 (y-slab MIP, ±30 um)"
        )
        fig.tight_layout()
        out = FIG_DIR / f"pia_overlay_{sid}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved {out.name}")

    # ---- HCR channel comparison for one subject (405 vs 488 vs combined) ----
    comp_sid = "788406"
    try:
        s = subjects[comp_sid]
        rows = []
        for ch in ("405", "488", "561", "594"):
            try:
                vol, xy_um, z_um = load_hcr_volume(s, channel=ch, level=4)
            except FileNotFoundError:
                rows.append({"channel": ch, "status": "missing"})
                continue
            surf = estimate_pia_surface_from_image(
                vol, z_um, xy_um, min_signal_abs=100.0, relative_margin=0.05
            )
            rows.append({"channel": ch,
                         "status": "ok" if surf else "failed",
                         **({"c_um": surf["c"],
                             "tilt_deg": surf["tilt_deg"],
                             "residual_std_um": surf["residual_std_um"],
                             "n_columns": surf["n_columns"]} if surf else {})})
        # Combined
        combined, xy_um, z_um, chans = load_hcr_combined(s, level=4)
        surf = estimate_pia_surface_from_image(
            combined, z_um, xy_um, min_signal_abs=0.1, relative_margin=0.05
        )
        if surf:
            rows.append({"channel": f"combined({'+'.join(chans)})",
                         "status": "ok",
                         "c_um": surf["c"],
                         "tilt_deg": surf["tilt_deg"],
                         "residual_std_um": surf["residual_std_um"],
                         "n_columns": surf["n_columns"]})
        pd.DataFrame(rows).to_csv(OUT_DIR / "hcr_channel_comparison.csv", index=False)
        print("Saved hcr_channel_comparison.csv")
    except Exception as e:
        print(f"Channel comparison skipped: {e}")

    # ---- Depth-profile falloff verification (image vs centroid surfaces) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, sid in zip(axes.flat, BENCHMARK_SUBJECTS):
        r = results[sid]
        # HCR all cells with image-based surface
        if r["hcr_surface_image"] is not None:
            c, d = depth_profile(r["hcr_xyz"], r["hcr_surface_image"], bin_um=20,
                                 depth_range=(-200, 1500))
            if d.max():
                ax.plot(c, d / d.max(), color="tab:red", lw=1.5,
                        label="HCR all / image pia")
        if r["hcr_surface_centroid"] is not None:
            c, d = depth_profile(r["hcr_xyz"], r["hcr_surface_centroid"], bin_um=20,
                                 depth_range=(-200, 1500))
            if d.max():
                ax.plot(c, d / d.max(), color="tab:orange", ls="--", lw=1.0,
                        label="HCR all / centroid pia")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(sid)
        ax.set_xlabel("depth from pia (um)")
        ax.set_ylabel("normalized density")
        ax.set_xlim(-200, 1500)
        ax.legend(fontsize=7)
    fig.suptitle("HCR cell density vs depth — verifying fall-off to 0 at pia")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "depth_profile_pia_verification.png", dpi=120)
    plt.close(fig)
    print("Saved depth_profile_pia_verification.png")

    # ---- Save a JSON dump of the quantitative results ----
    def _serial(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, tuple):
            return list(x)
        return x

    summary_all = {
        "summary": summary.to_dict(orient="records"),
        "expansion": expansion.to_dict(orient="records"),
        "surface": surface.to_dict(orient="records"),
        "nn_distance": nn.to_dict(orient="records"),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(summary_all, f, indent=2, default=_serial)

    print("\nAll outputs saved under", OUT_DIR)


if __name__ == "__main__":
    main()
