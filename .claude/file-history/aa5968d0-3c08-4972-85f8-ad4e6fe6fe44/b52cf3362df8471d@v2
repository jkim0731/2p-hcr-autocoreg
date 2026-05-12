"""GT-matched vs all-HCR distribution of unmixed GFP+ counts (and density).

Mirrors the visualisation from `04_R1_subgoal_01` (twinx histogram of
all HCR cells gray + manually matched HCR cells blue), but applied to
the new pairwise-unmixing R*-488-GFP counts.

Two restrictions per subject:
  (A) full HCR — every cell that has an unmixed row.
  (B) overlap-restricted — only cells whose centroid lies inside the
      GT-Procrustes-mapped CZ subvolume (xy AABB ∩ pia-depth range
      from `07c_gate_gt_recheck.gt_density_gate`).

GT-matched set = `coreg_table['hcr_id']` (manual landmarks, validated
GFP+ by construction).

Outputs:
  figures/matched_vs_all/{sid}_unmixed.png — 1×2 panel (full / overlap),
  figures/matched_vs_all/{sid}_baseline.png — same for baseline feature.
  results_matched_vs_all.json — summary stats per subject.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEV = Path("/root/capsule/code/dev_code")
SESSION = Path("/root/capsule/code/sessions/13_pairwise_unmix_gfp")
FIGDIR = SESSION / "figures" / "matched_vs_all"
FIGDIR.mkdir(parents=True, exist_ok=True)
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

ARCHIVE = Path("/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503")


def _load_numeric(basename, alias):
    spec = importlib.util.spec_from_file_location(alias, DEV / basename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


gfp_thr = _load_numeric("07b_gfp_intersection_threshold.py", "gfp07b")
gtgate = _load_numeric("07c_gate_gt_recheck.py", "gtgate07c")

from benchmark_data_loader import (  # noqa: E402
    load_subject, hcr_px_to_um, cz_px_to_um,
)
from benchmark_analysis import analyze_subject, fit_anisotropic_similarity  # noqa: E402
from benchmark_data_loader import landmark_pairs_um  # noqa: E402

SUBJECTS = ["755252", "767022", "782149", "788406"]
SPOT_BASELINE = {"782149", "788406"}
INT_BASELINE = {"755252", "767022"}

UNMIX_GLOB = (
    "/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/"
    "pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_unmixed(sid: str) -> pd.DataFrame:
    matches = glob(UNMIX_GLOB.format(sid=sid))
    if not matches:
        raise FileNotFoundError(f"No unmixed CSV for {sid}")
    df = pd.read_csv(matches[0])
    gfp_col = [c for c in df.columns if c.endswith("-GFP")][0]
    return df.rename(columns={"cell_id": "hcr_id", gfp_col: "count"})[
        ["hcr_id", "count"]
    ].astype({"hcr_id": int, "count": float})


def load_baseline(sid: str) -> tuple[pd.DataFrame, str, str]:
    """Return (df, value_col, log_base). May raise if unavailable."""
    if sid in SPOT_BASELINE:
        df = gfp_thr._load_spot_feature(sid)
        return df, "density", "ln"
    # Intensity — patch the loader to find the CSV in the archive root.
    p = ARCHIVE / f"cell_data_mean_{sid}_R1.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    raw = pd.read_csv(p)
    if "channel" in raw.columns:
        raw = raw[raw["channel"] == 488]
    rename_map = {"cell_id": "hcr_id", "id": "hcr_id"}
    for k, v in rename_map.items():
        if k in raw.columns:
            raw = raw.rename(columns={k: v})
    raw = raw.copy()
    raw["mean_minus_bg"] = raw["mean"].astype(float) - raw["background"].astype(float)
    return raw[["hcr_id", "mean_minus_bg"]].dropna(), "mean_minus_bg", "log10"


def overlap_mask_um(s, hcr_um: np.ndarray) -> np.ndarray:
    """Boolean mask of HCR cells inside GT-mapped CZ overlap (xy AABB ∩ pia-depth band).

    Mirrors `gt_density_gate` overlap definition: GT-anisotropic Procrustes
    map of CZ centroids, then xy AABB intersection with HCR positives, then
    intersect pia-depth ranges.
    """
    info = analyze_subject(s)
    pia = info["hcr_surface"]
    cz_lm, hcr_lm = landmark_pairs_um(s, active_only=True)
    fit = fit_anisotropic_similarity(cz_lm, hcr_lm)
    cz_xyz_px = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
    cz_um = cz_px_to_um(cz_xyz_px, s)[:, [2, 1, 0]]  # (x, y, z) µm
    cz_mapped = (cz_um - cz_lm.mean(axis=0)) @ np.asarray(fit.R) * np.asarray(fit.scales) + hcr_lm.mean(axis=0)

    lo = np.maximum(cz_mapped.min(axis=0)[:2], hcr_um.min(axis=0)[:2])
    hi = np.minimum(cz_mapped.max(axis=0)[:2], hcr_um.max(axis=0)[:2])
    if np.any(hi <= lo):
        return np.zeros(len(hcr_um), dtype=bool)
    xy_mask = np.all((hcr_um[:, :2] >= lo) & (hcr_um[:, :2] <= hi), axis=1)

    a, b, c = float(pia["a"]), float(pia["b"]), float(pia["c"])
    depth_hcr = hcr_um[:, 2] - (a * hcr_um[:, 0] + b * hcr_um[:, 1] + c)
    depth_cz = cz_mapped[:, 2] - (a * cz_mapped[:, 0] + b * cz_mapped[:, 1] + c)
    d_lo = max(depth_cz.min(), depth_hcr[xy_mask].min())
    d_hi = min(depth_cz.max(), depth_hcr[xy_mask].max())
    z_mask = (depth_hcr >= d_lo) & (depth_hcr <= d_hi)
    return xy_mask & z_mask


def fit_threshold_log(values_pos: np.ndarray, log_base: str) -> tuple[float, dict]:
    log_fn = np.log if log_base == "ln" else np.log10
    log_x = log_fn(values_pos)
    sweep = gfp_thr.fit_gmm_sweep(log_x, k_min=2, k_max=6)
    fit = sweep["best"]
    cutoff = float(np.exp(fit["intersection_log"]) if log_base == "ln" else 10.0 ** fit["intersection_log"])
    return cutoff, fit


# ---------------------------------------------------------------------------
# Plotter
# ---------------------------------------------------------------------------
def hist_panel(ax, all_vals, matched_vals, label_x, log_base, cutoff_lin,
               title):
    log_fn = np.log if log_base == "ln" else np.log10
    pos_all = all_vals[all_vals > 0]
    pos_match = matched_vals[matched_vals > 0]
    if len(pos_all) == 0:
        ax.text(0.5, 0.5, "no positives", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(title)
        return
    log_all = log_fn(pos_all)
    log_match = log_fn(pos_match) if len(pos_match) else np.array([])
    bins = np.linspace(log_all.min() - 0.1, log_all.max() + 0.1, 60)
    ax.hist(log_all, bins=bins, color="#94a3b8", alpha=0.7,
            label=f"all HCR (n={len(pos_all)}; +{(all_vals==0).sum()} zeros)")
    ax2 = ax.twinx()
    if len(log_match):
        ax2.hist(log_match, bins=bins, color="#3b7dd8", alpha=0.55,
                 label=f"GT-matched (n={len(pos_match)}; +{(matched_vals==0).sum()} zeros)")
    cutoff_log = log_fn(cutoff_lin) if cutoff_lin > 0 else None
    if cutoff_log is not None:
        ax.axvline(cutoff_log, color="#cc3333", lw=2.0,
                   label=f"BIC-GMM cutoff = {cutoff_lin:.3g}")
    ax.set_xlabel(f"{log_base}({label_x})")
    ax.set_ylabel("count (all HCR)")
    ax2.set_ylabel("count (matched)", color="#3b7dd8")
    ax2.tick_params(axis='y', colors="#3b7dd8")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    # Stats annotation
    if len(pos_match) and cutoff_lin > 0:
        n_match_pos = int((matched_vals >= cutoff_lin).sum())
        n_match_total = int(len(matched_vals))
        n_all_pos = int((all_vals >= cutoff_lin).sum())
        n_all_total = int(len(all_vals))
        annot = (
            f"matched ≥ cut: {n_match_pos}/{n_match_total} ({100*n_match_pos/max(n_match_total,1):.0f}%)\n"
            f"all ≥ cut: {n_all_pos}/{n_all_total} ({100*n_all_pos/max(n_all_total,1):.1f}%)\n"
            f"matched-median = {np.median(matched_vals):.1f}\n"
            f"all-median (>0) = {np.median(pos_all):.1f}"
        )
        ax.text(0.98, 0.98, annot, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.7, edgecolor="#cccccc"))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze(sid: str) -> dict:
    s = load_subject(sid)
    matched_ids = set(int(x) for x in s.coreg_table["hcr_id"].values)
    hcr_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].to_numpy(float), s
    )[:, [2, 1, 0]]
    in_overlap = overlap_mask_um(s, hcr_um)
    overlap_ids = set(int(x) for x in s.hcr_centroids.loc[in_overlap, "hcr_id"].values)

    out = {
        "subject": sid,
        "n_hcr_total": int(len(s.hcr_centroids)),
        "n_matched": int(len(matched_ids)),
        "n_in_overlap": int(in_overlap.sum()),
        "n_matched_in_overlap": int(len(matched_ids & overlap_ids)),
    }

    # ---------- Unmixed ----------
    df_u = load_unmixed(sid)
    cells_u = set(int(x) for x in df_u["hcr_id"].values)
    matched_in_unmix = matched_ids & cells_u
    out["n_unmixed_rows"] = int(len(df_u))
    out["n_matched_in_unmixed"] = int(len(matched_in_unmix))
    out["n_matched_missing_in_unmixed"] = int(len(matched_ids - cells_u))

    cutoff_u, fit_u = fit_threshold_log(df_u["count"][df_u["count"] > 0].to_numpy(), "ln")
    out["unmixed"] = {
        "K_star": int(fit_u["n_components"]),
        "cutoff": cutoff_u,
        "bic": float(fit_u["bic"]),
    }

    # Build per-cell views
    df_u_idx = df_u.set_index("hcr_id")["count"]
    all_full = df_u["count"].to_numpy(float)
    matched_full = df_u_idx.reindex(list(matched_ids), fill_value=np.nan).dropna().to_numpy(float)

    overlap_mask_u = df_u["hcr_id"].isin(overlap_ids).to_numpy()
    all_over = df_u.loc[overlap_mask_u, "count"].to_numpy(float)
    matched_over_ids = matched_ids & overlap_ids & cells_u
    matched_over = df_u_idx.reindex(list(matched_over_ids), fill_value=np.nan).dropna().to_numpy(float)

    out["unmixed"]["n_all_full"] = int(len(all_full))
    out["unmixed"]["n_all_overlap"] = int(len(all_over))
    out["unmixed"]["n_matched_full"] = int(len(matched_full))
    out["unmixed"]["n_matched_overlap"] = int(len(matched_over))
    out["unmixed"]["matched_above_cut_full"] = int((matched_full >= cutoff_u).sum())
    out["unmixed"]["matched_above_cut_overlap"] = int((matched_over >= cutoff_u).sum())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    hist_panel(
        axes[0], all_full, matched_full, "unmixed_count", "ln", cutoff_u,
        f"{sid} unmixed — full HCR  (n_match={len(matched_full)}/{len(matched_ids)})"
    )
    hist_panel(
        axes[1], all_over, matched_over, "unmixed_count", "ln", cutoff_u,
        f"{sid} unmixed — within CZ overlap  (n_match={len(matched_over)}/{out['n_matched_in_overlap']})"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}_unmixed.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Baseline (where available) ----------
    try:
        df_b, col_b, lb_b = load_baseline(sid)
        df_b_idx = df_b.set_index("hcr_id")[col_b]
        cutoff_b, fit_b = fit_threshold_log(df_b[col_b][df_b[col_b] > 0].to_numpy(), lb_b)
        all_full_b = df_b[col_b].to_numpy(float)
        matched_full_b = df_b_idx.reindex(list(matched_ids), fill_value=np.nan).dropna().to_numpy(float)
        ov_mask_b = df_b["hcr_id"].isin(overlap_ids).to_numpy()
        all_over_b = df_b.loc[ov_mask_b, col_b].to_numpy(float)
        matched_over_ids_b = matched_ids & overlap_ids & set(int(x) for x in df_b["hcr_id"].values)
        matched_over_b = df_b_idx.reindex(list(matched_over_ids_b), fill_value=np.nan).dropna().to_numpy(float)

        out["baseline"] = {
            "feature": col_b,
            "log_base": lb_b,
            "K_star": int(fit_b["n_components"]),
            "cutoff": cutoff_b,
            "bic": float(fit_b["bic"]),
            "n_all_full": int(len(all_full_b)),
            "n_all_overlap": int(len(all_over_b)),
            "n_matched_full": int(len(matched_full_b)),
            "n_matched_overlap": int(len(matched_over_b)),
            "matched_above_cut_full": int((matched_full_b >= cutoff_b).sum()),
            "matched_above_cut_overlap": int((matched_over_b >= cutoff_b).sum()),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        hist_panel(
            axes[0], all_full_b, matched_full_b, col_b, lb_b, cutoff_b,
            f"{sid} baseline — full HCR  (n_match={len(matched_full_b)}/{len(matched_ids)})"
        )
        hist_panel(
            axes[1], all_over_b, matched_over_b, col_b, lb_b, cutoff_b,
            f"{sid} baseline — within CZ overlap  (n_match={len(matched_over_b)}/{out['n_matched_in_overlap']})"
        )
        fig.tight_layout()
        fig.savefig(FIGDIR / f"{sid}_baseline.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        out["baseline"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def main():
    summary = {}
    for sid in SUBJECTS:
        print(f"\n=== {sid} ===")
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            summary[sid] = {"subject": sid, "error": str(e)}
            continue
        summary[sid] = r
        u = r["unmixed"]
        print(
            f"  unmixed cut={u['cutoff']:.3g}  K*={u['K_star']}  "
            f"matched ≥cut full = {u['matched_above_cut_full']}/{u['n_matched_full']} ({100*u['matched_above_cut_full']/max(u['n_matched_full'],1):.0f}%)  "
            f"matched ≥cut overlap = {u['matched_above_cut_overlap']}/{u['n_matched_overlap']} ({100*u['matched_above_cut_overlap']/max(u['n_matched_overlap'],1):.0f}%)"
        )
        if isinstance(r.get("baseline"), dict) and "error" not in r["baseline"]:
            b = r["baseline"]
            print(
                f"  baseline cut={b['cutoff']:.3g}  K*={b['K_star']}  "
                f"matched ≥cut full = {b['matched_above_cut_full']}/{b['n_matched_full']} ({100*b['matched_above_cut_full']/max(b['n_matched_full'],1):.0f}%)  "
                f"matched ≥cut overlap = {b['matched_above_cut_overlap']}/{b['n_matched_overlap']} ({100*b['matched_above_cut_overlap']/max(b['n_matched_overlap'],1):.0f}%)"
            )
        else:
            print(f"  baseline unavailable: {r.get('baseline', {}).get('error', '?')}")
    out_path = SESSION / "results_matched_vs_all.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
