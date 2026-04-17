"""Build the benchmark analysis notebook from the pre-computed results."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


OUT_DIR = Path("/root/capsule/code/sessions/01_analyze_benchmark_data")
NB_PATH = Path("/root/capsule/code/notebooks/01_benchmark_data_analysis.ipynb")


def _md(cell):
    return nbf.v4.new_markdown_cell(cell)


def _code(cell):
    return nbf.v4.new_code_cell(cell)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(_md(
        "# Benchmark data analysis (coregistration features)\n\n"
        "Goal: characterize the 6 benchmark subjects so that an automatic coregistration\n"
        "pipeline can be designed with correct assumptions about resolution, anisotropy,\n"
        "expansion, surface geometry, and cell distributions.\n\n"
        "**Per dev protocol, benchmark landmarks/coreg are used ONLY for characterization**,\n"
        "not for algorithm development.\n\n"
        "This notebook is a summary of the outputs computed by\n"
        "`code/dev_code/01_analyze_benchmark.py`. All tables and figures are loaded from\n"
        "`code/sessions/01_analyze_benchmark_data/`."
    ))

    cells.append(_code(
        "import sys, json\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '/root/capsule/code/dev_code')\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "OUT = Path('/root/capsule/code/sessions/01_analyze_benchmark_data')\n"
        "FIG = OUT / 'figures'"
    ))

    cells.append(_md(
        "## 1. Per-subject summary\n"
        "Cell counts, volume extents (in microns), GFP+ fraction, coregistration match\n"
        "rate, and landmark counts."
    ))
    cells.append(_code(
        "summary = pd.read_csv(OUT / 'summary_table.csv')\n"
        "summary"
    ))

    cells.append(_md(
        "## 2. Anisotropic expansion from landmarks\n\n"
        "Each subject's active landmarks are converted to physical microns using the\n"
        "per-modality resolutions (CZ: 0.78 um/px XY, 1 um/px Z; HCR level-2: \n"
        "~0.988 um/px XY, 1 um/px Z — from `step_2_automatic_mapping_for_qc.ipynb`'s \n"
        "`*4e6` XY / `*1e6` Z convention).\n\n"
        "We fit a rigid-rotation-plus-per-axis-scale mapping CZ(um) -> HCR(um).\n"
        "The resulting scale factors measure the ex-vivo HCR expansion relative to\n"
        "in-vivo 2p. **XY and Z expand differently** (anisotropic)."
    ))
    cells.append(_code(
        "expansion = pd.read_csv(OUT / 'expansion_table.csv')\n"
        "expansion"
    ))
    cells.append(_code(
        "display(Image(str(FIG / 'expansion_factors.png')))"
    ))

    cells.append(_md(
        "### Residual after anisotropic similarity fit\n\n"
        "The per-landmark residual norm after the rotation + per-axis scale fit. This\n"
        "quantifies how much nonrigid deformation the algorithm must correct after\n"
        "affine alignment."
    ))
    cells.append(_code(
        "display(Image(str(FIG / 'residual_magnitude.png')))"
    ))

    cells.append(_md(
        "## 3. Pia surface estimation and depth-from-surface\n\n"
        "We fit the pia plane `z = a*x + b*y + c` (um) by three methods:\n\n"
        "- **hybrid (default for HCR)**: (1) density filter (≥6 neighbors within\n"
        "  30 um) removes isolated out-of-tissue ROIs; (2) fit a coarse prior\n"
        "  `z_prior(x,y)` from minimum-z ROI per tile; (3) run image-based\n"
        "  first-crossing but restrict each column to z ∈ [z_prior − 100 um,\n"
        "  z_prior + 100 um]; (4) fit the final plane to the constrained map.\n"
        "  The ROI prior prevents the image search from wandering into noise\n"
        "  far above the tissue.\n"
        "- **image-based (default for CZ)**: for each (y, x) column, find the\n"
        "  first z where intensity exceeds `col_bg + 0.05*(col_top - col_bg)`. The\n"
        "  **5% margin** is essential — higher margins skip the sparse outer layer.\n"
        "  For HCR, use a **combined-channel volume** (405+488+561/514+594\n"
        "  normalized and summed) at zarr level 4 (~4 um voxels).\n"
        "- **centroid-based (diagnostic only)**: tile-top quantile of ROI\n"
        "  centroids. Underestimates pia by 50–200 um — the topmost segmented\n"
        "  ROIs sit below the real tissue boundary.\n\n"
        "We report a **fraction-above-pia** metric: share of cells whose centroid\n"
        "falls more than 5 um above the estimated pia (<~10% is good). Residual\n"
        "positives are out-of-tissue segmentation false positives."
    ))
    cells.append(_code(
        "surface = pd.read_csv(OUT / 'surface_table.csv')\n"
        "surface"
    ))

    cells.append(_md(
        "### Example pia overlays on the raw image data\n\n"
        "Each figure shows a Y-slab maximum-intensity projection in the (x, z)\n"
        "plane. Red solid = image-based fit, yellow dashed = centroid-based (fallback).\n"
        "Two representative subjects:\n\n"
        "- `788406`: near-flat HCR pia. The image-based fit sits at the tissue\n"
        "  boundary; the centroid-based fit is clearly below.\n"
        "- `782149`: strongly tilted HCR pia (~11° measured). Both fits track\n"
        "  the slope; image-based is closer to the true tissue edge."
    ))
    cells.append(_code(
        "for sid in ('788406', '782149'):\n"
        "    p = FIG / f'pia_overlay_{sid}.png'\n"
        "    if p.exists():\n"
        "        display(Image(str(p)))"
    ))

    cells.append(_md(
        "### Pia-estimate verification: HCR density vs depth\n\n"
        "For each subject we plot HCR cell density vs depth using both the\n"
        "image-based pia (red) and the centroid-based pia (orange dashed).\n"
        "The image-based curve rises sharply from 0 at depth ~0, confirming the fit\n"
        "lies at the true tissue boundary. The centroid-based curve is shifted\n"
        "50–200 um deeper. Small residual peaks at negative depths correspond to\n"
        "out-of-tissue HCR segmentation false positives."
    ))
    cells.append(_code(
        "display(Image(str(FIG / 'depth_profile_pia_verification.png')))"
    ))

    cells.append(_md(
        "### HCR channel choice for surface fitting\n\n"
        "- 405 (Rn28S) alone: dense, works, but the ‘first rise’ detection can\n"
        "  overshoot the real pia if the very outer layer is sparse.\n"
        "- 488 (GCaMP+): too sparse — only ~100–200 valid columns.\n"
        "- 561, 594: autofluorescence with noticeable tissue boundary.\n"
        "- **combined (all channels)**: normalise each channel and sum. The tissue\n"
        "  boundary is crisper because autofluorescence aligns across channels at\n"
        "  the tissue-buffer transition. This is the default for the analysis."
    ))
    cells.append(_code(
        "pd.read_csv(OUT / 'hcr_channel_comparison.csv')"
    ))

    cells.append(_md(
        "### Depth profiles from estimated pia\n\n"
        "Cells per 20-um depth bin for CZ (rescaled by Z expansion factor), HCR all,\n"
        "and HCR GFP+. Surface is image-based (405 for HCR). The 1D depth profile is\n"
        "a candidate feature for axial localization."
    ))
    cells.append(_code(
        "display(Image(str(FIG / 'depth_profiles.png')))"
    ))

    cells.append(_md(
        "## 4. Nearest-neighbor distance statistics\n\n"
        "Mean 1-NN and 5-NN distances (in um) per modality. Characterizes cell-cell\n"
        "spacing and packing density; informs kernel/neighborhood sizes for matching."
    ))
    cells.append(_code(
        "nn = pd.read_csv(OUT / 'nn_distance_table.csv')\n"
        "nn"
    ))

    cells.append(_md(
        "## 5. Surface detection exploration (Stage A–C)\n\n"
        "We explored whether ROI segmentation data could improve pia-surface estimation.\n"
        "The premise: if segmentation were perfect, the shallowest z-position per (x, y)\n"
        "column would be the pia. In practice, HCR segmentation produces out-of-tissue\n"
        "false positives that sit above the real pia.\n\n"
        "### Stage A: ROI segmentation error characterization\n\n"
        "Using the image-based pia as reference, we classified each HCR ROI as\n"
        "`above_pia` (depth < −5 um) or `in_tissue`, then compared feature distributions.\n\n"
        "**Key finding**: above-pia ROIs are **10–30× smaller in volume** (median ~7–11k px\n"
        "vs ~130–140k px for in-tissue ROIs) and paradoxically have *higher* local density\n"
        "(more clustered among themselves). Volume is the best single discriminator,\n"
        "capturing ~80–90% of above-pia ROIs at a threshold equal to the in-tissue Q25."
    ))
    cells.append(_code(
        "stageA = pd.read_csv(OUT / 'stageA_roi_feature_separability.csv')\n"
        "cols = ['subject','n_hcr_roi','n_above_pia','frac_above_pia',\n"
        "        'volume_px_above_med','volume_px_intis_med','volume_px_frac_above_captured']\n"
        "[c for c in cols if c in stageA.columns]  # verify columns present\n"
    ))
    cells.append(_code(
        "stageA[[c for c in ['subject','n_hcr_roi','n_above_pia','frac_above_pia',\n"
        "                     'volume_px_above_med','volume_px_intis_med',\n"
        "                     'volume_px_frac_above_captured'] if c in stageA.columns]]"
    ))
    cells.append(_code(
        "p = FIG / 'stageA_feature_histograms.png'\n"
        "if p.exists():\n"
        "    display(Image(str(p)))"
    ))

    cells.append(_md(
        "### Stage B: Simple filter → shallowest-z plane\n\n"
        "We applied a volume filter (keep ROIs within 30% of the in-tissue median volume)\n"
        "plus a density filter (≥6 neighbors within 30 um), then fit a plane to the\n"
        "minimum-z ROIs per (x, y) tile with quantiles q ∈ {0, 1, 2, 5}%.\n\n"
        "**Result**: Stage B improves 767018 and 782149 (the worst subjects), but\n"
        "hurts 755252 and 790322. No single filter+quantile combination beats the\n"
        "image-based baseline across all subjects."
    ))
    cells.append(_code(
        "stageB = pd.read_csv(OUT / 'stageB_best_per_subject.csv')\n"
        "stageB"
    ))

    cells.append(_md(
        "### Stage C: Hybrid (ROI coarse prior + image refinement)\n\n"
        "The hybrid approach uses ROI segmentation to constrain the image-based search:\n\n"
        "1. Density filter (≥6 neighbors within 30 um) to remove isolated artifacts.\n"
        "2. Fit a plane to the minimum-z ROIs per tile → coarse `z_prior(x, y)`.\n"
        "3. In `estimate_pia_surface_from_image`, restrict each column's search to\n"
        "   z ∈ [z_prior − 100 um, z_prior + 100 um].\n"
        "4. Fit the final plane to the constrained first-crossing map.\n\n"
        "**Result**: The hybrid is the winner:\n"
        "- 767018: 10.8% → 2.5% frac_above_pia (massive improvement)\n"
        "- 782149: 8.2% → 5.7% (improvement)\n"
        "- Other subjects: minor regressions < 2 pp\n\n"
        "The hybrid is now the **default** surface method in `analyze_subject()`."
    ))
    cells.append(_code(
        "# Summary comparison: baseline (image-only) vs best Stage B vs hybrid\n"
        "methods = pd.read_csv(OUT / 'surface_exploration_methods.csv')\n"
        "methods"
    ))
    cells.append(_code(
        "# Visual overlay: image-based vs Stage B best vs hybrid for 3 subjects\n"
        "for sid in ('788406', '782149', '767018'):\n"
        "    p = FIG / f'surface_method_overlay_{sid}.png'\n"
        "    if p.exists():\n"
        "        print(f'Subject {sid}')\n"
        "        display(Image(str(p)))"
    ))

    cells.append(_md(
        "## 6. Key takeaways for algorithm design\n\n"
        "* **Anisotropic expansion** CZ→HCR: XY ~1.6–1.9×, Z ~2.1–3.6×.\n"
        "  A single isotropic scale cannot align the modalities.\n"
        "* **HCR volume size** at level-2 centroid space is ~1.2–2.3 mm per side — much\n"
        "  larger than the old doc's ~1×1×0.5 mm claim.\n"
        "* **Rotation ~180°** around z is reproducible across subjects; can be hardcoded\n"
        "  as the initial guess.\n"
        "* **Residuals after affine** are 15–43 um RMS with max ≤ 114 um. Nonrigid\n"
        "  correction must handle this scale; kernels around 50–200 um are appropriate.\n"
        "* **Pia tilt** in HCR is up to ~12° (782149) and varies across (x, y). Surface\n"
        "  fitting is essential for depth-from-surface coordinates.\n"
        "* **HCR buffer thickness** above pia varies widely (30–410 um surface offset),\n"
        "  confirming the data description's remark that buffer depth is unknown.\n"
        "* **Depth-from-surface density profile** is a natural, 1D axial localization\n"
        "  feature that absorbs the buffer variability.\n"
        "* **Mean cell spacing** is ~20–25 um 1-NN across all subjects and both\n"
        "  modalities — a useful length scale for k-NN matching.\n"
        "* **Data quirks**: 767018 has no GFP+ data and an older processing pipeline;\n"
        "  755252/767022 use intensity-based GFP proxy (not spot counts) — most HCR\n"
        "  cells would be labeled 'GFP+' without further thresholding; 782149 has a\n"
        "  thin HCR section with stronger pia tilt and only ~34% match rate.\n"
    ))

    nb["cells"] = cells

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print("Wrote", NB_PATH)


if __name__ == "__main__":
    build()
