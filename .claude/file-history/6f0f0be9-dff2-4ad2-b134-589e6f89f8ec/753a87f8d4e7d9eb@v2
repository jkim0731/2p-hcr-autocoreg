"""Build `notebooks/03_surface_estimation_iteration.ipynb` from session outputs."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path("/root/capsule/code/sessions/03_surface_estimation")
NB = Path("/root/capsule/code/notebooks/03_surface_estimation_iteration.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Session 03 — HCR pia surface estimation: iteration log\n\n"
        "**Goal.** Revise the HCR pia surface estimator so that **ROI density at\n"
        "the estimated surface is close to zero** (the depth-from-surface density\n"
        "profile should have a clear floor at depth 0).\n\n"
        "The previous default (`hybrid`) leaves 2.5–8.2 % of ROIs above pia and,\n"
        "more importantly by the new criterion, has `r0_narrow` (density at depth\n"
        "≈ 0 relative to bulk density) up to **13.5** on some subjects.\n\n"
        "This notebook records the progression: baseline, four rounds of\n"
        "candidate methods (with failures), the chosen winner, and verification\n"
        "that the criterion is met on all six subjects.\n\n"
        "All results come from `code/dev_code/03_surface_iteration.py`; the\n"
        "session log is at `code/sessions/03_surface_estimation/log.md`."
    ))

    cells.append(code(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from IPython.display import Image, display\n"
        "OUT = Path('/root/capsule/code/sessions/03_surface_estimation')\n"
        "FIG = OUT / 'figures'"
    ))

    cells.append(md(
        "## 1. Metrics\n\n"
        "For every candidate surface we compute, against the HCR ROI centroids:\n\n"
        "| metric | definition | target |\n"
        "|--------|------------|--------|\n"
        "| `r0_narrow` | mean density in depth ∈ [−3, +3] um / bulk density ([50, 200] um) | ≲ 0.5 |\n"
        "| `r0_broad` | same but over [−10, +10] um | ≲ 0.5 |\n"
        "| `frac_above_pia` | fraction of cells with depth < −5 um | informational |\n"
        "| `spike_to_bulk` | peak density in [−100, −10] / bulk density | informational |\n\n"
        "**`r0_narrow` is the primary metric** — it is the quantitative form of\n"
        "the user's criterion. `frac_above_pia` is informational because above-\n"
        "pia cells are segmentation false positives (small ROIs clustered in the\n"
        "buffer, characterized in session 01 Stage A); if the surface is placed\n"
        "correctly, they *should* be labelled above-pia."
    ))

    cells.append(md(
        "## 2. Baseline: the problem\n\n"
        "Two existing methods on all six subjects:"
    ))
    cells.append(code(
        "base = pd.read_csv(OUT / 'round2_results.csv')\n"
        "base[base['method'].isin(['baseline_image', 'baseline_hybrid'])]\\\n"
        "    [['subject','method','c_um','frac_above_pia','r0_narrow','r0_broad','spike_to_bulk']]"
    ))
    cells.append(md(
        "- Pure image-based is the better of the two by `r0_narrow` on every\n"
        "  subject except 755252 (where it is still bad: 2.97) and 782149 (7.33).\n"
        "- Hybrid regresses catastrophically on 767018 (`r0_narrow` = 13.5).\n"
        "- Two subjects stand out as problems: **755252** and **782149**.\n\n"
        "### Diagnostic: depth profiles relative to the image-based surface\n\n"
        "Plotted for all six subjects — a clear **spike-then-trough-then-bulk**\n"
        "pattern appears at negative depths (~−50 to 0 um) in most subjects.\n"
        "The spike is the cluster of out-of-tissue segmentation false positives\n"
        "in the buffer. The image fit anchors at the spike onset, so the surface\n"
        "sits inside the spike — exactly why `r0_narrow` is large."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'baseline_image_density_profiles.png')))"
    ))

    cells.append(md(
        "## 3. Round 1 — per-tile density threshold (**M1**)\n\n"
        "**Idea.** For each (x, y) tile, find the first z (scanned from the\n"
        "top) where the rolling ROI density exceeds a fraction of that tile's\n"
        "plateau density; fit a plane to those per-tile depths.\n\n"
        "**Result.** Pushed the surface too deep — into the rising edge of the\n"
        "bulk. Improved `r0_narrow` for easy subjects but landed ~150 um below\n"
        "truth on 782149.\n\n"
        "**Why.** A density threshold does not know the difference between the\n"
        "spike peak and the bulk plateau. Subjects with no real spike trigger\n"
        "the threshold well inside the cortex. Dropped.\n"
    ))
    cells.append(code(
        "pd.read_csv(OUT / 'round1_results.csv')\\\n"
        "    [['subject','method','c_um','frac_above_pia','r0_narrow']]"
    ))

    cells.append(md(
        "## 4. Round 2 — gap-finding anchored on the image surface (**M4, M5, M6**)\n\n"
        "Hypothesis: the image fit is a correct *starting* point; the error is\n"
        "a small z-shift to place the plane past the spike, at the trough before\n"
        "bulk rise. So: *keep the image tilt, shift the intercept*.\n\n"
        "- **M4 — global trough.** Smoothed depth profile over all ROIs. Find\n"
        "  the spike peak in the negative half; the local minimum within 100 um\n"
        "  after it is the surface. Fallback to image when no spike present.\n"
        "- **M5 — per-tile trough.** Same gap-finding, per (x, y) tile.\n"
        "- **M6 — spike-vs-bulk fit.** Two-component model (Gaussian spike +\n"
        "  logistic bulk); surface at the 50 % bulk point.\n\n"
        "### Results"
    ))
    cells.append(code(
        "r2 = pd.read_csv(OUT / 'round2_results.csv')\n"
        "r2[['subject','method','c_um','frac_above_pia','r0_narrow','spike_to_bulk']]"
    ))
    cells.append(md(
        "**M4 wins on every subject by `r0_narrow`** — and by a huge margin on\n"
        "755252 (2.97 → 0.19) and 782149 (7.33 → 0.38).\n\n"
        "**Why M5 under-performs.** A 100–150 um tile contains only hundreds\n"
        "of ROIs, so the per-tile trough position is Poisson-noisy; the noise\n"
        "dominates any real tilt variation.\n\n"
        "**Why M6 under-performs.** The cortical density rise has internal\n"
        "layered structure, so a logistic's 50 % point sits deeper than the\n"
        "true trough — surface too deep, `r0` still borderline but\n"
        "`frac_above_pia` high."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'round2_depth_profiles.png')))"
    ))

    cells.append(md(
        "## 5. Round 3 — decay-from-peak (**M7, M8**)\n\n"
        "**Idea.** Instead of local-minimum-after-peak, set the surface where\n"
        "the density has decayed to `k × (peak − bulk)` below the spike\n"
        "(strict k=0.10, default 0.20, loose 0.40). M8 = per-tile.\n\n"
        "**Result.** Fragile in both directions:\n"
        "- Where the spike is weak (755252 spike/bulk 1.26; 788406 0.93;\n"
        "  790322 1.06) the decay target is below current density — fall\n"
        "  through to baseline. A `spike ≥ 1.5 × bulk` guard fires on 755252,\n"
        "  the very subject we need to fix.\n"
        "- Where the spike is strong (767018 peak at depth −92), the decay\n"
        "  point lands at negative depth and gets clipped.\n\n"
        "**Why.** Anchoring on peak height is fragile when peaks are small or\n"
        "noisy; M4's local-minimum formulation is equivalent to 'density stops\n"
        "decreasing' and is robust to peak height. Dropped."
    ))
    cells.append(code(
        "pd.read_csv(OUT / 'round3_results.csv')\\\n"
        "    [['subject','method','c_um','frac_above_pia','r0_narrow']]"
    ))

    cells.append(md(
        "## 6. Round 4 — bulk-floor (**M9, M10**)\n\n"
        "**Idea.** Instead of local-min, find the first depth ≥ 0 where the\n"
        "smoothed density drops below `threshold × bulk` (0.50 default) and\n"
        "stays below for 15 um. An absolute threshold is independent of peak\n"
        "height.\n\n"
        "**Result.** Never beats M4. First-crossing through `0.5 × bulk`\n"
        "happens on the *descending flank* of the spike, still above the\n"
        "trough — so the shift is under-sized (e.g., 782149: M4 `r0=0.38`\n"
        "vs M9 `r0=1.09`). M10 per-tile falls back to baseline because its\n"
        "spike-detection gate fails on at least one tile per subject.\n\n"
        "**Why.** M4 = crossing + one step deeper to the true trough, so M4\n"
        "dominates M9 on this profile shape. Dropped."
    ))
    cells.append(code(
        "pd.read_csv(OUT / 'round4_results.csv')\\\n"
        "    [['subject','method','c_um','frac_above_pia','r0_narrow']]"
    ))
    cells.append(code(
        "display(Image(str(FIG / 'round4_depth_profiles.png')))"
    ))

    cells.append(md(
        "## 7. Winner: **M4** (image-based surface + global trough shift)\n\n"
        "### Criterion satisfied on all six subjects"
    ))
    cells.append(code(
        "r2 = pd.read_csv(OUT / 'round2_results.csv')\n"
        "win = r2[r2['method'].isin(['baseline_image', 'baseline_hybrid', 'M4_global_trough'])]\n"
        "cols = ['subject','method','c_um','tilt_deg','frac_above_pia','r0_narrow','r0_broad','spike_to_bulk','gap_depth_um']\n"
        "win[cols].sort_values(['subject','method'])"
    ))
    cells.append(md(
        "`r0_narrow < 0.5` for every subject under M4. The hybrid default\n"
        "had `r0_narrow` up to 13.5.\n\n"
        "### Side-effect on `frac_above_pia`\n\n"
        "M4 increases `frac_above_pia` (e.g., 782149: 8 % → 22 %). This is the\n"
        "*correct* reclassification of the out-of-tissue spike: session 01\n"
        "Stage A showed those ROIs are 10–30× smaller than in-tissue ROIs and\n"
        "cluster in the buffer. Previously they were hidden below the plane.\n"
        "Using `frac_above_pia` as the quality metric implicitly rewards\n"
        "hiding them; the user's `r0_narrow` criterion correctly penalizes it."
    ))

    cells.append(md(
        "### Recommended default parameters\n"
        "- Combined-channel HCR volume at pyramid level 4 (~4 um voxels)\n"
        "- 5 % relative margin on the image first-crossing\n"
        "- ±150 um search window, 20 um smoothing, 5 um bins\n"
        "- 100 um lookahead after spike peak for local-minimum search\n"
        "- Fallback to image-only fit if no spike-then-trough detected\n\n"
        "**Promoted** to the new default in `benchmark_analysis.py`:\n"
        "`analyze_subject(..., hcr_surface_method='image_trough')`."
    ))

    cells.append(md(
        "## 8. References\n"
        "- Script: `code/dev_code/03_surface_iteration.py`\n"
        "- Log: `code/sessions/03_surface_estimation/log.md`\n"
        "- Production function: `benchmark_analysis.estimate_pia_surface_image_trough`"
    ))

    nb["cells"] = cells
    NB.parent.mkdir(parents=True, exist_ok=True)
    with open(NB, "w") as f:
        nbf.write(nb, f)
    print("Wrote", NB)


if __name__ == "__main__":
    build()
