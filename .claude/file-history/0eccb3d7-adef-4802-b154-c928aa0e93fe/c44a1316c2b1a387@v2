"""Compare baseline (07c) GFP+ thresholding vs pairwise-unmixed GFP+ counts
on the 4 subjects with unmixing data: 755252, 767022, 782149, 788406.

Apply the same BIC-GMM K∈[2,6] threshold pipeline (07b) and the same
GT-Procrustes CZ-density CV diagnostic (07c) to both feature sources, so
the comparison is apples-to-apples.

Baseline feature per subject (matches main pipeline / 07c):
  755252, 767022 — `cell_data_mean_{sid}_R1.csv` channel-488 mean − background
                   (intensity, log10).
  782149, 788406 — `*spot_488_counts.csv` density (counts/volume, ln).

Unmixed feature: `R*-488-GFP` column from
  HCR_{sid}_pairwise-unmixing*/pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv
The R-round varies per subject (R1 for the recent panel, R2 for the older
panel) — auto-detected by suffix `-GFP`.

Outputs:
  results.json — per-subject baseline + unmixed dicts.
  figures/{sid}_gmm.png      — GMM histogram + cutoff (both sources).
  figures/{sid}_density.png  — depth-density CV (both sources).
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
FIGDIR = SESSION / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))


def _load_numeric_module(basename, alias):
    spec = importlib.util.spec_from_file_location(alias, DEV / basename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


gfp_thr = _load_numeric_module(
    "07b_gfp_intersection_threshold.py", "gfp07b"
)
gt_gate_mod = _load_numeric_module(
    "07c_gate_gt_recheck.py", "gtgate07c"
)

from benchmark_data_loader import load_subject  # noqa: E402

SUBJECTS = ["755252", "767022", "782149", "788406"]
SPOT_BASELINE = {"782149", "788406"}
INT_BASELINE = {"755252", "767022"}

UNMIX_CSV_GLOB = (
    "/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/"
    "pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv"
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_unmixed(sid: str) -> pd.DataFrame:
    matches = glob(UNMIX_CSV_GLOB.format(sid=sid))
    if not matches:
        raise FileNotFoundError(f"No unmixed CSV for {sid}")
    df = pd.read_csv(matches[0])
    gfp_cols = [c for c in df.columns if c.endswith("-GFP")]
    if len(gfp_cols) != 1:
        raise RuntimeError(f"{sid}: expected exactly 1 -GFP column, got {gfp_cols}")
    col = gfp_cols[0]
    df = df.rename(columns={"cell_id": "hcr_id", col: "count"})
    df["hcr_id"] = df["hcr_id"].astype(int)
    df["count"] = df["count"].astype(float)
    return df[["hcr_id", "count"]]


def baseline_feature(sid: str):
    """Return (df_feat, value_col, log_fn, exp_fn, log_label, feature_name)."""
    if sid in SPOT_BASELINE:
        df = gfp_thr._load_spot_feature(sid)
        return df, "density", np.log, np.exp, "ln", "density"
    df = gfp_thr._load_intensity_feature(sid)
    return df, "mean_minus_bg", np.log10, lambda x: 10.0 ** x, "log10", "mean_minus_bg"


def unmixed_feature(sid: str):
    """Same return signature as baseline_feature but for unmixed counts."""
    df = load_unmixed(sid)
    # Use ln(count+1) so we can include zeros; positive-only would drop ~50%.
    # For BIC fit we still feed positives only and use ln, to mirror 07b.
    return df, "count", np.log, np.exp, "ln", "unmixed_count"


# ---------------------------------------------------------------------------
# BIC sweep + cutoff
# ---------------------------------------------------------------------------
def fit_threshold(values_pos: np.ndarray, log_fn):
    log_x = log_fn(values_pos)
    sweep = gfp_thr.fit_gmm_sweep(log_x, k_min=2, k_max=6)
    fit = sweep["best"]
    return fit, sweep["sweep"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze(sid: str) -> dict:
    out: dict = {"subject": sid}

    # Baseline (may be unavailable, e.g., intensity CSV missing for 755252/767022)
    fit_b = None
    strict_ids_b = None
    try:
        df_b, col_b, logfn_b, expfn_b, lab_b, name_b = baseline_feature(sid)
        vals_b = df_b[col_b].to_numpy(float)
        pos_b = vals_b[vals_b > 0]
        fit_b, sweep_b = fit_threshold(pos_b, logfn_b)
        cutoff_b = float(expfn_b(fit_b["intersection_log"]))
        strict_b = df_b[df_b[col_b].to_numpy(float) >= cutoff_b]
        strict_ids_b = set(int(x) for x in strict_b["hcr_id"].values)
        out["baseline"] = {
            "feature": name_b,
            "log_base": lab_b,
            "n_total": int(len(df_b)),
            "n_positive": int(len(pos_b)),
            "K_star": int(fit_b["n_components"]),
            "bic": float(fit_b["bic"]),
            "intersection_log": float(fit_b["intersection_log"]),
            "cutoff_linear": cutoff_b,
            "n_strict": int(len(strict_b)),
            "frac_strict": float(len(strict_b) / max(len(df_b), 1)),
            "sweep": [
                {"K": e["n_components"], "bic": e.get("bic")} for e in sweep_b
                if "bic" in e
            ],
        }
    except Exception as e:
        out["baseline"] = {"error": f"{type(e).__name__}: {e}"}

    # Unmixed
    df_u, col_u, logfn_u, expfn_u, lab_u, name_u = unmixed_feature(sid)
    vals_u = df_u[col_u].to_numpy(float)
    pos_u = vals_u[vals_u > 0]
    fit_u, sweep_u = fit_threshold(pos_u, logfn_u)
    cutoff_u = float(expfn_u(fit_u["intersection_log"]))
    strict_u = df_u[df_u[col_u].to_numpy(float) >= cutoff_u]
    strict_ids_u = set(int(x) for x in strict_u["hcr_id"].values)
    out["unmixed"] = {
        "feature": name_u,
        "log_base": lab_u,
        "n_total": int(len(df_u)),
        "n_positive": int(len(pos_u)),
        "K_star": int(fit_u["n_components"]),
        "bic": float(fit_u["bic"]),
        "intersection_log": float(fit_u["intersection_log"]),
        "cutoff_linear": cutoff_u,
        "n_strict": int(len(strict_u)),
        "frac_strict": float(len(strict_u) / max(len(df_u), 1)),
        "sweep": [
            {"K": e["n_components"], "bic": e.get("bic")} for e in sweep_u
            if "bic" in e
        ],
    }

    # Density gate (CV) — uses 07c's gt_density_gate, which calls load_subject(sid)
    # and reuses the same HCR centroid id space (the strict_hcr_ids set).
    if strict_ids_b is not None:
        out["baseline"]["density_gate"] = gt_gate_mod.gt_density_gate(sid, strict_ids_b)
    out["unmixed"]["density_gate"] = gt_gate_mod.gt_density_gate(sid, strict_ids_u)

    # ---------- Plot histograms ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    panels = []
    if fit_b is not None:
        panels.append((axes[0], fit_b, logfn_b(pos_b), name_b, fit_b["intersection_log"], "baseline"))
    else:
        axes[0].text(0.5, 0.5, "baseline unavailable\n(intensity CSV missing)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"{sid} — baseline (n/a)")
    panels.append((axes[1], fit_u, logfn_u(pos_u), name_u, fit_u["intersection_log"], "unmixed"))
    for ax, fit, log_x, label, cutoff_log, src in panels:
        bins = np.linspace(log_x.min() - 0.2, log_x.max() + 0.2, 100)
        ax.hist(log_x, bins=bins, density=True, alpha=0.5, color="#94a3b8")
        xs = np.linspace(bins[0], bins[-1], 400)
        from scipy.stats import norm
        total = np.zeros_like(xs)
        for mu, sig, w in zip(fit["means"], fit["sigmas"], fit["weights"]):
            comp = w * norm.pdf(xs, mu, sig)
            ax.plot(xs, comp, lw=0.9, alpha=0.8)
            total = total + comp
        ax.plot(xs, total, lw=1.6, color="black")
        ax.axvline(cutoff_log, color="#cc3333", lw=2.0)
        log_label = (lab_b if (src == "baseline" and fit_b is not None) else lab_u)
        ax.set_xlabel(f"{log_label}({label})")
        cut_lin = (np.exp(cutoff_log) if log_label == "ln" else 10 ** cutoff_log)
        ax.set_ylabel("density")
        ax.set_title(
            f"{sid} — {src}  K*={fit['n_components']}  "
            f"cut={float(cut_lin):.3g}  BIC={fit['bic']:.0f}"
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{sid}: BIC-GMM threshold — baseline vs pairwise-unmixed", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}_gmm.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot density gates ----------
    g_b = out["baseline"].get("density_gate", {"status": "no_baseline"})
    g_u = out["unmixed"]["density_gate"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, gate, src in [(axes[0], g_b, "baseline"), (axes[1], g_u, "unmixed")]:
        if gate.get("status") != "ok":
            ax.text(0.5, 0.5, gate.get("status", "no_data"),
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{sid} {src}")
            continue
        c = np.asarray(gate["depth_centers_um"])
        rho_cz = np.asarray(gate["rho_cz_per_um3"]) * 1e6
        rho_g = np.asarray(gate["rho_gfp_per_um3"]) * 1e6
        ratio = np.asarray(gate["gfp_over_cz"])
        ax2 = ax.twinx()
        ax.plot(c, rho_cz, "-", color="black", lw=2.0, label="CZ (GT-mapped)")
        ax.plot(c, rho_g, "-", color="#2aa198", lw=1.8, label=f"GFP+ {src}")
        ax2.plot(c, ratio, "--", color="#268bd2", lw=1.0, label="GFP+/CZ")
        ax.set_xlabel("depth from HCR pia (µm)")
        ax.set_ylabel("density (cells / 10^6 µm³)")
        ax2.set_ylabel("GFP+/CZ ratio")
        ax.set_title(
            f"{sid} {src}: CV={gate['cv_ratio']:.3f}  "
            f"mean={gate['integrated_ratio']:.3f}  pass={gate['gate_pass']}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{sid}: GT-Procrustes depth-density gate — baseline vs unmixed", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}_density.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return out


def main():
    records = {}
    for sid in SUBJECTS:
        print(f"\n=== {sid} ===")
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            records[sid] = {"subject": sid, "error": str(e)}
            continue
        records[sid] = r
        b = r["baseline"]; u = r["unmixed"]
        if "error" in b:
            print(f"  baseline: {b['error']}")
        else:
            cv = b['density_gate'].get('cv_ratio')
            mn = b['density_gate'].get('integrated_ratio')
            cv_s = f"{cv:.3f}" if isinstance(cv, (int, float)) else "n/a"
            mn_s = f"{mn:.3f}" if isinstance(mn, (int, float)) else "n/a"
            print(
                f"  baseline ({b['feature']:>14}): K*={b['K_star']}  cut={b['cutoff_linear']:.4g}  "
                f"n_strict={b['n_strict']}/{b['n_total']} ({100*b['frac_strict']:.1f}%)  "
                f"CV={cv_s}  mean={mn_s}"
            )
        print(
            f"  unmixed  ({u['feature']:>14}): K*={u['K_star']}  cut={u['cutoff_linear']:.4g}  "
            f"n_strict={u['n_strict']}/{u['n_total']} ({100*u['frac_strict']:.1f}%)  "
            f"CV={u['density_gate'].get('cv_ratio', float('nan')):.3f}  "
            f"mean={u['density_gate'].get('integrated_ratio', float('nan')):.3f}"
        )

    out_path = SESSION / "results.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
