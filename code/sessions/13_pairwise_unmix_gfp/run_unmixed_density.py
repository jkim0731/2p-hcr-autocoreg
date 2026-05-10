"""Variant: GFP+ threshold on (unmixed_count / cell_volume) so the unmixed
feature is continuous like the baseline density.

The earlier `run_compare.py` showed unmixed *counts* break the GMM because
the distribution is dominated by a delta-spike at 0/1 spots. Joining the
unmixed per-cell counts with `cell_body_segmentation/metrics.pickle` volumes
should restore a continuous distribution, mirroring the baseline.

Subjects: 755252, 767022, 782149, 788406 (all 4 with pairwise-unmixing data).
Writes results.json + figures/{sid}_unmixed_density.png.
"""
from __future__ import annotations
import importlib.util
import json
import pickle
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


def _load_numeric(basename, alias):
    spec = importlib.util.spec_from_file_location(alias, DEV / basename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


gfp_thr = _load_numeric("07b_gfp_intersection_threshold.py", "gfp07b")
gt_gate = _load_numeric("07c_gate_gt_recheck.py", "gtgate07c")

from benchmark_data_loader import load_subject  # noqa: E402

SUBJECTS = ["755252", "767022", "782149", "788406"]

UNMIX_GLOB = (
    "/root/capsule/data/HCR_{sid}_pairwise-unmixing_*/"
    "pairwise_unmixing/all_cells_unmixed/unmixed_all_cells.csv"
)


def load_unmixed_density(sid: str) -> pd.DataFrame:
    matches = glob(UNMIX_GLOB.format(sid=sid))
    if not matches:
        raise FileNotFoundError(f"No unmixed CSV for {sid}")
    df = pd.read_csv(matches[0])
    gfp_col = [c for c in df.columns if c.endswith("-GFP")][0]
    df = df.rename(columns={"cell_id": "hcr_id", gfp_col: "counts"})
    df["hcr_id"] = df["hcr_id"].astype(int)
    df["counts"] = df["counts"].astype(float)

    s = load_subject(sid)
    metrics_path = Path(s.hcr_dir) / "cell_body_segmentation" / "metrics.pickle"
    with open(metrics_path, "rb") as f:
        m = pickle.load(f)
    md = pd.DataFrame(m).transpose()
    md.index = md.index.astype(int)
    md.index.name = "hcr_id"
    df = df.merge(md[["volume"]], left_on="hcr_id", right_index=True, how="left")
    df = df.dropna(subset=["volume"])
    df["density"] = df["counts"] / df["volume"]
    return df[["hcr_id", "counts", "volume", "density"]]


def analyze(sid: str) -> dict:
    df = load_unmixed_density(sid)
    vals = df["density"].to_numpy(float)
    pos = vals[vals > 0]
    sweep = gfp_thr.fit_gmm_sweep(np.log(pos), k_min=2, k_max=6)
    fit = sweep["best"]
    cutoff = float(np.exp(fit["intersection_log"]))
    strict = df[df["density"].to_numpy(float) >= cutoff]
    strict_ids = set(int(x) for x in strict["hcr_id"].values)

    res = {
        "subject": sid,
        "n_total": int(len(df)),
        "n_positive": int(len(pos)),
        "K_star": int(fit["n_components"]),
        "bic": float(fit["bic"]),
        "intersection_log": float(fit["intersection_log"]),
        "cutoff_density": cutoff,
        "n_strict": int(len(strict)),
        "frac_strict": float(len(strict) / max(len(df), 1)),
        "sweep": [
            {"K": e["n_components"], "bic": e.get("bic")} for e in sweep["sweep"]
            if "bic" in e
        ],
        "density_gate": gt_gate.gt_density_gate(sid, strict_ids),
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    log_x = np.log(pos)
    bins = np.linspace(log_x.min() - 0.2, log_x.max() + 0.2, 100)
    ax = axes[0]
    ax.hist(log_x, bins=bins, density=True, alpha=0.55, color="#94a3b8")
    xs = np.linspace(bins[0], bins[-1], 400)
    from scipy.stats import norm
    total = np.zeros_like(xs)
    for mu, sig, w in zip(fit["means"], fit["sigmas"], fit["weights"]):
        comp = w * norm.pdf(xs, mu, sig)
        ax.plot(xs, comp, lw=0.9, alpha=0.8)
        total = total + comp
    ax.plot(xs, total, lw=1.6, color="black")
    ax.axvline(fit["intersection_log"], color="#cc3333", lw=2.0)
    ax.set_xlabel("ln(unmixed_count / volume)")
    ax.set_ylabel("density")
    ax.set_title(
        f"{sid} unmixed-density  K*={fit['n_components']}  "
        f"cut={cutoff:.4g}  BIC={fit['bic']:.0f}"
    )
    ax.grid(True, alpha=0.3)

    g = res["density_gate"]
    ax = axes[1]
    if g.get("status") != "ok":
        ax.text(0.5, 0.5, g.get("status", "no_data"),
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{sid} density gate")
    else:
        c = np.asarray(g["depth_centers_um"])
        rho_cz = np.asarray(g["rho_cz_per_um3"]) * 1e6
        rho_g = np.asarray(g["rho_gfp_per_um3"]) * 1e6
        ratio = np.asarray(g["gfp_over_cz"])
        ax2 = ax.twinx()
        ax.plot(c, rho_cz, "-", color="black", lw=2.0, label="CZ (GT-mapped)")
        ax.plot(c, rho_g, "-", color="#2aa198", lw=1.8, label="GFP+ unmixed-dens")
        ax2.plot(c, ratio, "--", color="#268bd2", lw=1.0, label="GFP+/CZ")
        ax.set_xlabel("depth from HCR pia (µm)")
        ax.set_ylabel("density (cells / 10^6 µm³)")
        ax2.set_ylabel("GFP+/CZ")
        ax.set_title(
            f"{sid} unmixed-density: CV={g['cv_ratio']:.3f}  "
            f"mean={g['integrated_ratio']:.3f}  pass={g['gate_pass']}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{sid}_unmixed_density.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return res


def main():
    out = {}
    for sid in SUBJECTS:
        try:
            r = analyze(sid)
        except Exception as e:
            print(f"  {sid} ERROR: {type(e).__name__}: {e}")
            out[sid] = {"subject": sid, "error": str(e)}
            continue
        out[sid] = r
        cv = r["density_gate"].get("cv_ratio", float("nan"))
        mn = r["density_gate"].get("integrated_ratio", float("nan"))
        cv_s = f"{cv:.3f}" if isinstance(cv, (int, float)) else "n/a"
        mn_s = f"{mn:.3f}" if isinstance(mn, (int, float)) else "n/a"
        print(
            f"  {sid}  K*={r['K_star']}  cut={r['cutoff_density']:.4g}  "
            f"n_strict={r['n_strict']}/{r['n_total']} ({100*r['frac_strict']:.1f}%)  "
            f"CV={cv_s}  mean={mn_s}"
        )

    with open(SESSION / "results_unmixed_density.json", "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    print(f"\nWrote {SESSION / 'results_unmixed_density.json'}")


if __name__ == "__main__":
    main()
