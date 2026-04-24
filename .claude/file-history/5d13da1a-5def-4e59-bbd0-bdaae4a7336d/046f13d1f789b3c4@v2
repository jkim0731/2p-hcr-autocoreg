"""Generate figures for sessions/05_R1_revised/.

Figures produced
----------------
1. error_comparison.png   — origin / rotation error vs v2.2 first-pass R1.
2. sz_score_curves.png    — 1D partial-overlap NCC vs sz per subject.
3. sxy_score_curves.png   — 2D density-map NCC vs sxy per subject.
4. depth_profiles.png     — CZ and HCR depth-from-surface histograms.
5. tilt_and_regime.png    — CZ/HCR surface tilt + coverage regime per subject.
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
from r1_revised import coarse_align_revised, depth_from_surface

OUT_DIR = _THIS_DIR.parent / "sessions" / "05_R1_revised" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["788406", "790322", "767018", "782149", "755252", "767022"]

# first-pass R1 v2.2 peakgauss results, for comparison
V22_ORIGIN = {"788406": 58.1, "790322": 74.3, "767018": 127.0,
              "782149": 334.8, "755252": 123.6, "767022": 181.0}
V22_ROT = {"788406": 2.46, "790322": 0.48, "767018": 10.26,
           "782149": 2.34, "755252": 6.99, "767022": 5.29}


def collect():
    rows = []
    for sid in SUBJECTS:
        print(f"[{sid}]")
        s = load_subject(sid)
        info = analyze_subject(s)
        r = coarse_align_revised(
            info["cz_xyz"], info["gfp_xyz"],
            info["cz_surface"], info["hcr_surface"],
        )
        cz_depth = depth_from_surface(info["cz_xyz"], info["cz_surface"])
        hcr_depth = depth_from_surface(info["gfp_xyz"], info["hcr_surface"])
        rows.append(dict(
            sid=sid,
            cz_depth=cz_depth,
            hcr_depth=hcr_depth,
            sz_grid=np.asarray(r.diagnostics["sz_grid"]),
            sz_curve=np.asarray(r.diagnostics["sz_score_curve"]),
            sxy_grid=np.asarray(r.diagnostics["sxy_grid"]),
            sxy_curve=np.asarray(r.diagnostics["sxy_score_curve"]),
            sz_best=r.diagnostics["sz_best"],
            sxy_best=r.diagnostics["sxy_best"],
            sz_conf=r.diagnostics["sz_confidence"],
            sxy_conf=r.diagnostics["sxy_confidence"],
            cz_tilt=r.diagnostics["cz_tilt_deg"],
            hcr_tilt=r.diagnostics["hcr_tilt_deg"],
            coverage=r.coverage_regime,
            origin_err=None,  # filled from r1_results.json
            rot_err=None,
        ))
    return rows


def load_json_errors(rows):
    import json
    p = _THIS_DIR.parent / "sessions" / "05_R1_revised" / "r1_results.json"
    data = {d["subject"]: d for d in json.loads(p.read_text())}
    for row in rows:
        d = data.get(row["sid"], {})
        row["origin_err"] = d.get("origin_err_um")
        row["rot_err"] = d.get("rotation_err_deg")
    return rows


def fig_error_comparison(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(rows))
    w = 0.38
    origin_rev = [r["origin_err"] for r in rows]
    origin_v22 = [V22_ORIGIN[r["sid"]] for r in rows]
    rot_rev = [r["rot_err"] for r in rows]
    rot_v22 = [V22_ROT[r["sid"]] for r in rows]
    ax1.bar(x - w/2, origin_v22, w, label="v2.2 first pass", color="#888")
    ax1.bar(x + w/2, origin_rev, w, label="revised (minimal)", color="#1f77b4")
    ax1.axhline(100, color="red", ls="--", lw=1, label="target ≤ 100 µm")
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["sid"] for r in rows], rotation=20)
    ax1.set_ylabel("origin error (µm)")
    ax1.set_title("Origin error: revised vs v2.2 first pass")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.bar(x - w/2, rot_v22, w, label="v2.2 first pass", color="#888")
    ax2.bar(x + w/2, rot_rev, w, label="revised (minimal)", color="#1f77b4")
    ax2.axhline(5, color="red", ls="--", lw=1, label="target ±5°")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["sid"] for r in rows], rotation=20)
    ax2.set_ylabel("rotation error (°)")
    ax2.set_title("Rotation error: revised vs v2.2 first pass")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_comparison.png", dpi=130)
    plt.close(fig)


def fig_sz_curves(rows):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for r, ax in zip(rows, axs.flat):
        mask = np.isfinite(r["sz_curve"])
        ax.plot(r["sz_grid"][mask], r["sz_curve"][mask], color="#1f77b4")
        ax.axvline(r["sz_best"], color="red", ls="--", lw=1,
                   label=f"sz*={r['sz_best']:.2f}")
        ax.set_title(f"{r['sid']}  (conf {r['sz_conf']:.2f}; "
                     f"regime {r['coverage']})", fontsize=10)
        ax.set_xlabel("sz")
        ax.set_ylabel("max-over-tz NCC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Z-scale search: 1D partial-overlap NCC\n"
                 "(confidence threshold = 6.0 → none emitted)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sz_score_curves.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_sxy_curves(rows):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=False)
    for r, ax in zip(rows, axs.flat):
        mask = np.isfinite(r["sxy_curve"])
        ax.plot(r["sxy_grid"][mask], r["sxy_curve"][mask], color="#2ca02c")
        ax.axvline(r["sxy_best"], color="red", ls="--", lw=1,
                   label=f"sxy*={r['sxy_best']:.2f}")
        ax.set_title(f"{r['sid']}  (conf {r['sxy_conf']:.2f})", fontsize=10)
        ax.set_xlabel("sxy")
        ax.set_ylabel("max-over-(tx,ty) NCC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("XY-scale search: 2D density-map NCC\n"
                 "(all peaks land at small sxy < 1 — geometrically wrong; "
                 "confidence < 6 → none emitted)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sxy_score_curves.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def fig_depth_profiles(rows):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    bins = np.arange(0, 1600, 20)
    for r, ax in zip(rows, axs.flat):
        cz_hist, _ = np.histogram(r["cz_depth"], bins=bins, density=True)
        hcr_hist, _ = np.histogram(r["hcr_depth"], bins=bins, density=True)
        x = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(x, cz_hist, color="#1f77b4", label=f"CZ (n={len(r['cz_depth'])})")
        ax.plot(x, hcr_hist, color="#ff7f0e", label=f"HCR GFP+ (n={len(r['hcr_depth'])})")
        ax.axvline(r["cz_depth"].max(), color="#1f77b4", ls=":", lw=1)
        ax.axvline(r["hcr_depth"].max(), color="#ff7f0e", ls=":", lw=1)
        ax.set_title(f"{r['sid']}  ({r['coverage']})", fontsize=10)
        ax.set_xlabel("depth from pia (µm)")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Depth-from-surface profiles (CZ unrescaled vs HCR GFP+)",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_profiles.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_tilt_and_regime(rows):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x = np.arange(len(rows))
    w = 0.38
    cz_tilts = [r["cz_tilt"] for r in rows]
    hcr_tilts = [r["hcr_tilt"] for r in rows]
    ax.bar(x - w/2, cz_tilts, w, label="CZ pia tilt", color="#1f77b4")
    ax.bar(x + w/2, hcr_tilts, w, label="HCR pia tilt", color="#ff7f0e")
    for xi, r in zip(x, rows):
        ax.text(xi, max(r["cz_tilt"], r["hcr_tilt"]) + 0.3, r["coverage"],
                ha="center", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([r["sid"] for r in rows], rotation=20)
    ax.set_ylabel("pia-plane tilt (deg)")
    ax.set_title("Surface tilt per subject  (labels: coverage regime)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tilt_and_regime.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_json_errors(collect())
    fig_error_comparison(rows)
    fig_sz_curves(rows)
    fig_sxy_curves(rows)
    fig_depth_profiles(rows)
    fig_tilt_and_regime(rows)
    print(f"\nWrote figures to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")
