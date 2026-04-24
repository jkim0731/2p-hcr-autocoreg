"""Generate summary plots for session 07 anisotropic-ICP benchmark.

Reads ``sessions/07_scale_failure_diagnosis/icp_results.json`` and writes
figures into ``sessions/07_scale_failure_diagnosis/figures/``.

Figures:
  - ``rel_err_bar.png``      per-subject rel_err_sxy / rel_err_sz with ±20 %
  - ``est_vs_gt.png``        sxy_est vs sxy_gt, sz_est vs sz_gt
  - ``multistart_basins.png``  per-subject (sxy_final, sz_final) for all 9
                              starts, coloured by score; GT and winner marked
  - ``multistart_scores.png``  per-subject ranked score bars with at_bound
                              candidates striped
  - ``residuals_vs_radius.png`` per-subject median residual vs r_um across
                              ICP iterations of the winning start
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SESSION_DIR = Path(__file__).resolve().parent.parent / "sessions" / "07_scale_failure_diagnosis"
RESULTS_PATH = SESSION_DIR / "icp_results.json"
FIG_DIR = SESSION_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH) as f:
    R = json.load(f)

SIDS = [r["subject"] for r in R]


# ----------------------------------------------------------------------
# 1. Per-subject relative error bars
# ----------------------------------------------------------------------
def plot_rel_err_bar():
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    x = np.arange(len(SIDS))
    w = 0.38
    err_xy = [r["rel_err_sxy"] * 100.0 for r in R]
    err_z = [r["rel_err_sz"] * 100.0 for r in R]
    colors_xy = ["#3b7dd8" if abs(e) <= 20 else "#cc3333" for e in err_xy]
    colors_z = ["#62a96a" if abs(e) <= 20 else "#cc3333" for e in err_z]
    ax.bar(x - w / 2, err_xy, w, label="rel_err sxy (%)", color=colors_xy)
    ax.bar(x + w / 2, err_z, w, label="rel_err sz (%)", color=colors_z)
    ax.axhline(20, color="k", lw=0.8, ls=":")
    ax.axhline(-20, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SIDS, rotation=0)
    ax.set_ylabel("relative error (%)")
    ax.set_title("Session 07 anisotropic-ICP — per-subject rel err vs GT (±20 % dotted)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rel_err_bar.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. Scatter: est vs gt
# ----------------------------------------------------------------------
def plot_est_vs_gt():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, axis_name, gt_key, est_key, lo, hi in [
        (axes[0], "sxy", "sxy_gt", "sxy_est", 1.4, 2.1),
        (axes[1], "sz",  "sz_gt",  "sz_est",  1.9, 4.0),
    ]:
        gt = np.array([r[gt_key] for r in R])
        est = np.array([r[est_key] for r in R])
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8)
        ax.plot([lo, hi], [lo * 1.2, hi * 1.2], color="grey", lw=0.6, ls=":")
        ax.plot([lo, hi], [lo * 0.8, hi * 0.8], color="grey", lw=0.6, ls=":")
        ax.scatter(gt, est, c="#3b7dd8", s=60, zorder=5)
        for sid, g, e in zip(SIDS, gt, est):
            ax.annotate(sid, (g, e), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel(f"{axis_name}_gt (landmark Procrustes)")
        ax.set_ylabel(f"{axis_name}_est (ICP)")
        ax.set_title(f"{axis_name}: estimated vs GT  (±20 % dotted)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "est_vs_gt.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Multistart basins scatter per subject
# ----------------------------------------------------------------------
def plot_multistart_basins():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, r in enumerate(R):
        ax = axes[i]
        ms = (r.get("icp_diagnostics", {}) or {}).get("multi_start", {})
        starts = [s for s in ms.get("starts", []) if isinstance(s, dict)
                  and "sxy_final" in s]
        sxy_f = np.array([s["sxy_final"] for s in starts])
        sz_f = np.array([s["sz_final"] for s in starts])
        score = np.array([s.get("score", np.nan) for s in starts])
        # Mask out boundary-penalised (very negative scores) for the colormap
        finite_mask = score > -1e5
        if finite_mask.any():
            sc = ax.scatter(sxy_f[finite_mask], sz_f[finite_mask],
                            c=score[finite_mask], cmap="viridis", s=70,
                            edgecolor="k", linewidth=0.5, zorder=5)
            plt.colorbar(sc, ax=ax, shrink=0.85, label="score")
        if (~finite_mask).any():
            ax.scatter(sxy_f[~finite_mask], sz_f[~finite_mask],
                       facecolor="none", edgecolor="#cc3333", s=90,
                       linewidth=1.2, zorder=4, label="at_bound (rejected)")
        # GT
        ax.scatter([r["sxy_gt"]], [r["sz_gt"]], marker="*", s=260,
                   c="gold", edgecolor="k", linewidth=1.0, zorder=6,
                   label="GT")
        # Winner
        ax.scatter([r["sxy_est"]], [r["sz_est"]], marker="D", s=130,
                   facecolor="none", edgecolor="#cc3333", linewidth=1.8,
                   zorder=7, label="selected")
        ax.axvspan(1.4 * 1.2, 2.0, alpha=0)  # keep axis
        ax.axvline(1.4, color="grey", lw=0.5, ls=":")
        ax.axvline(2.0, color="grey", lw=0.5, ls=":")
        ax.axhline(1.9, color="grey", lw=0.5, ls=":")
        ax.axhline(4.0, color="grey", lw=0.5, ls=":")
        ax.set_xlim(1.35, 2.05)
        ax.set_ylim(1.8, 4.1)
        ax.set_title(f"{r['subject']}  GT=({r['sxy_gt']:.2f}, {r['sz_gt']:.2f})")
        ax.set_xlabel("sxy_final")
        ax.set_ylabel("sz_final")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Multi-start basins — score colormap; GT ★, selected ◆, bound ○",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "multistart_basins.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Multistart score bars per subject
# ----------------------------------------------------------------------
def plot_multistart_scores():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, r in enumerate(R):
        ax = axes[i]
        ms = (r.get("icp_diagnostics", {}) or {}).get("multi_start", {})
        starts = [s for s in ms.get("starts", []) if isinstance(s, dict)
                  and "sxy_final" in s]
        # Rank by score (but cap boundary at a visible floor)
        starts = sorted(starts, key=lambda s: -s.get("score", -np.inf))
        labels = [f"({s['sxy_init']:.2f},{s['sz_init']:.2f})\n"
                  f"→({s['sxy_final']:.2f},{s['sz_final']:.2f})"
                  for s in starts]
        raw_scores = np.array([s["score"] for s in starts])
        at_bound = np.array([s.get("at_bound", False) for s in starts])
        # Plot bounded-penalty bars at a fixed minimum so they are visible
        # but distinct.
        floor = float(np.min(raw_scores[~at_bound])) - 10 if (~at_bound).any() else 0.0
        display = np.where(at_bound, floor, raw_scores)
        colors = np.where(at_bound, "#cc3333", "#3b7dd8")
        x = np.arange(len(starts))
        ax.bar(x, display, color=colors)
        for j, s in enumerate(starts):
            if s.get("at_bound"):
                ax.text(j, display[j], "bound",
                        ha="center", va="bottom", fontsize=7, color="#cc3333")
            ax.text(j, display[j] + 0.3, f"nt={s['n_tight']}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, fontsize=6, ha="right")
        ax.set_title(f"{r['subject']}  sel=({r['sxy_est']:.2f}, {r['sz_est']:.2f})"
                     f"  GT=({r['sxy_gt']:.2f}, {r['sz_gt']:.2f})")
        ax.set_ylabel("score (n_tight+10·sz+1/(med+1))")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-subject multistart — red bars = boundary-penalised (rejected)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "multistart_scores.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. ICP trajectory (winner): median residual and scales vs iteration
# ----------------------------------------------------------------------
def plot_icp_trajectories():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    for i, r in enumerate(R):
        ax = axes[i]
        hist = r.get("icp_history", [])
        hist = [h for h in hist if isinstance(h, dict) and "median_residual_um" in h]
        if not hist:
            ax.set_title(f"{r['subject']} — no history")
            continue
        iters = [h["it"] for h in hist]
        med = [h["median_residual_um"] for h in hist]
        radius = [h["r_um"] for h in hist]
        n = [h.get("n_matched_inlier") or h.get("n_matched_raw") for h in hist]
        ax2 = ax.twinx()
        l1, = ax.plot(iters, med, "o-", color="#3b7dd8", label="median residual (µm)")
        l2, = ax.plot(iters, radius, "s--", color="grey", label="r_search (µm)", alpha=0.7)
        l3, = ax2.plot(iters, n, "d-", color="#62a96a", label="n_matched (inlier)")
        ax.set_title(f"{r['subject']} — winning start trajectory")
        ax.set_ylabel("µm")
        ax.set_xlabel("ICP iter")
        ax2.set_ylabel("n matched", color="#62a96a")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(handles=[l1, l2, l3], loc="upper right", fontsize=8)
    fig.suptitle("ICP inner loop — winner start radius decay + residual + matched count",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "icp_trajectories.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_rel_err_bar()
    plot_est_vs_gt()
    plot_multistart_basins()
    plot_multistart_scores()
    plot_icp_trajectories()
    print(f"Wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size/1024:.1f} KB)")
