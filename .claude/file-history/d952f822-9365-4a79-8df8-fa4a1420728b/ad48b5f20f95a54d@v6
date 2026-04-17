"""Build the session 03 v2 notebook.

Outputs (CSV + figures) live under
/root/capsule/code/sessions/03_surface_estimation_v2/ and are
version-controlled alongside the notebook.
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

NB = Path("/root/capsule/code/notebooks/03_surface_estimation_iteration_v2.ipynb")


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


WINNING_METHODS = (
    "['baseline_image_5pct','N4_image_ceiling_0p5pct_off5',"
    "'N6_quadratic_ceiling_off3','N10_hybrid_sig120_off5',"
    "'N11_multichannel_band150_m25']"
)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Session 03 v2 - HCR/CZ pia surface under corrected criterion\n\n"
        "**Goal.** Place the pia surface such that (a) ROI density at or\n"
        "above the surface is close to zero, (b) cortex ROI density rises\n"
        "immediately below the surface, and (c) the rise depth is uniform\n"
        "across the (x, y) footprint of the section. Subjects with tissue\n"
        "tilt (e.g. 782149, 11 deg) or localized curvature (e.g. 790322\n"
        "at x~2000 um) must also be handled.\n\n"
        "**Winning protocol - HCR.** `multi_channel_image_in_band` (N11).\n"
        "A robust ROI-quadratic surface defines a ±150 um search band;\n"
        "inside that band the max-intensity projection across every HCR\n"
        "channel gives a per-column top-of-signal at a relative intensity\n"
        "margin (25 %); a second robust quadratic is then fit through\n"
        "those image-derived column tops and clamped against the ROI\n"
        "tile envelope. The image evidence carries local curvature that\n"
        "pure ROI-envelope fits miss (notably 790322 at x~2000 um).\n\n"
        "**Winning protocol - CZ.** `cz_image_with_roi_ceiling`, 2 %\n"
        "relative margin, 80 um tiles, 3 um safety offset. CZ has no\n"
        "out-of-tissue junk, so a flat image-plane clamped against ROI\n"
        "tiles is enough."
    ))

    cells.append(code(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n"
        "\n"
        "OUT = Path('/root/capsule/code/sessions/03_surface_estimation_v2')\n"
        "FIG = OUT / 'figures'\n"
        "print('Reading from', OUT)"
    ))

    # ===== HCR =====
    cells.append(md(
        "## HCR protocol walkthrough (subject 790322)\n\n"
        "Subject 790322 has the localized curvature at x~2000 um that\n"
        "broke every earlier method. The six panels show how N11 uses\n"
        "multi-channel image evidence inside a band around a ROI-quadratic\n"
        "anchor to recover that curvature."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'hcr_walkthrough.png')))"
    ))
    cells.append(md(
        "### What each step does\n\n"
        "1. **ROI tile min-z envelope.** Density-filter ROIs (>= 3\n"
        "   neighbours within 30 um), bin into 120 um (x, y) tiles, take\n"
        "   the 2nd-percentile z per tile (red dots). These trace the\n"
        "   top-of-tissue boundary visible to the ROI segmentation.\n"
        "2. **ROI-quadratic anchor + search band.** Robust IRLS-Huber\n"
        "   quadratic fit through the tile envelope gives the anchor\n"
        "   surface (blue). A ±150 um band around it (blue fill) is the\n"
        "   region in which image evidence will be trusted.\n"
        "3. **Multi-channel max-intensity MIP.** Every HCR channel is\n"
        "   loaded at level 4, normalized per-channel, then combined\n"
        "   pointwise by maximum. Tissue autofluorescence varies by\n"
        "   channel, so max-combine gives the most informative boundary\n"
        "   evidence available per (x, y, z) voxel.\n"
        "4. **Band-constrained per-column top-of-signal search.** For\n"
        "   each (x, y) column, restrict to the ±150 um band, estimate\n"
        "   a column background (P10 in band) and max (P99 in band),\n"
        "   threshold at `bg + 0.25 * (max - bg)`, and take the\n"
        "   shallowest above-threshold z. A subject-wide dynamic-range\n"
        "   floor rejects empty columns; a 10 um persistence window\n"
        "   rejects single-voxel noise.\n"
        "5. **Robust quadratic fit through image top-z.** A second\n"
        "   IRLS-Huber quadratic is fit through the column tops from\n"
        "   step 4 (red line). This surface follows real image\n"
        "   curvature — including 790322's localized dip — because the\n"
        "   constraints come from thousands of image columns rather than\n"
        "   a few dozen ROI tiles.\n"
        "6. **Final surface on MIP.** Safety-clamp against the ROI tile\n"
        "   envelope (so the fit can never sit above ROI tissue) and\n"
        "   subtract a small offset (3 um). Cyan dots are the ROI slab;\n"
        "   the red line sits right at the tissue boundary."
    ))

    cells.append(md(
        "## HCR estimated surface on every subject\n\n"
        "10 y-slabs (rows) x 6 subjects (columns). Red line is the final\n"
        "N11 surface evaluated at each slab's y; cyan dots are ROI\n"
        "centroids within +/- 20 um of the slab. Note the curvature the\n"
        "image evidence captures on 755252/782149 (strong bend along x)\n"
        "and 790322 (the x~2000 um dip that earlier N6/N9/N10 surfaces\n"
        "left 75-125 um deep under)."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'hcr_surface_overlays.png')))"
    ))

    cells.append(md(
        "## HCR validation metrics\n\n"
        "Metrics on all 6 subjects for the winner and the previous\n"
        "contenders.\n\n"
        "- **`above_frac`** - fraction of ROIs at depth < 0. Target close to 0.\n"
        "- **`onset_depth_um`** - where smoothed density first reaches\n"
        "  50 % of bulk on the positive side. Smaller = tighter fit.\n"
        "- **`onset_x_spread`, `onset_y_spread`** - per-quantile-bin spread\n"
        "  of onset_depth along x and y. Primary fit-uniformity metric:\n"
        "  small spread means cortex starts at the same depth everywhere."
    ))
    cells.append(code(
        "hcr = pd.read_csv(OUT / 'hcr_results.csv')\n"
        "winners = " + WINNING_METHODS + "\n"
        "cols = ['subject','method','c_um','tilt_deg',\n"
        "        'above_frac','onset_depth_um',\n"
        "        'onset_x_spread','onset_y_spread']\n"
        "hcr[hcr['method'].isin(winners)][cols]\\\n"
        "    .sort_values(['subject','method'])"
    ))
    cells.append(code(
        "display(Image(str(FIG / 'hcr_depth_profiles.png')))"
    ))

    cells.append(md(
        "### Why not the earlier contenders?\n\n"
        "- **Baseline image (5 % margin).** above_frac 3 - 11 %; a large\n"
        "  spike of ROIs sits above the plane. Fails the criterion.\n"
        "- **N4 image-ceiling (flat plane + global lift).** Drives\n"
        "  above_frac to ~0 but one outlier tile pushes the whole plane\n"
        "  60 - 150 um above the tissue on 755252 / 788406 / 790322,\n"
        "  giving onset_depth 32 - 82 um. A single tilted plane cannot\n"
        "  absorb real surface curvature.\n"
        "- **N6 quadratic ceiling.** Adds curvature, but a 6-parameter\n"
        "  quadratic fit through ROI tile min-z still can't follow\n"
        "  790322's localized bend (onset_depth 122 um, x_spread 85 um)\n"
        "  because the ROI envelope is too sparse there.\n"
        "- **N10 hybrid (quad + Gaussian residual).** Introduced local\n"
        "  flexibility but noisy — Gaussian smoothing of sparse tile\n"
        "  residuals produced visually worse fits on several subjects.\n"
        "  Abandoned.\n"
        "- **N11 multi-channel image in band (winner).** The ROI-quadratic\n"
        "  anchor keeps the search localized; image evidence from *all*\n"
        "  HCR channels (tissue autofluorescence varies by channel)\n"
        "  produces thousands of per-column boundary hits; a robust\n"
        "  quadratic through those is both stable and image-faithful.\n"
        "  Result on 790322: onset_depth 77 um (vs 122 for N6), x_spread\n"
        "  50 um (vs 85), above_frac < 0.01 %. Across all 6 subjects,\n"
        "  N11 gives lower onset and tighter spread than N6."
    ))

    # ===== CZ =====
    cells.append(md(
        "## CZ protocol walkthrough (subject 767022)\n\n"
        "CZ is cleaner than HCR - no out-of-tissue ROI junk - so a flat\n"
        "tilted plane plus a per-tile ROI clamp is sufficient."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'cz_walkthrough.png')))"
    ))
    cells.append(md(
        "### What each step does\n\n"
        "1. **CZ z-stack MIP.** Raw max-intensity projection across a y\n"
        "   band. Bright region = cortex, dim region above = buffer.\n"
        "2. **Image plane fit.** Per-column top-of-signal at the\n"
        "   `relative_margin = 2 %` intensity threshold; `_robust_plane_fit`\n"
        "   then fits a tilted plane through those column peaks.\n"
        "3. **Per-tile ROI min-z (80 um tiles).** Smaller tiles than HCR\n"
        "   because CZ features are more compact. Red dots trace the\n"
        "   shallowest ROIs.\n"
        "4. **Ceiling clamp + safety.** Lift the image plane so no tile's\n"
        "   min-z lies above it; subtract 3 um safety. Red line is the\n"
        "   final pia surface."
    ))

    cells.append(md(
        "## CZ estimated surface on every subject\n\n"
        "Same 10 x 6 grid for CZ. The plane is tilt-only (max tilt ~3 deg)\n"
        "and sits cleanly above every ROI column in all subjects."
    ))
    cells.append(code(
        "display(Image(str(FIG / 'cz_surface_overlays.png')))"
    ))
    cells.append(code(
        "cz = pd.read_csv(OUT / 'cz_results.csv')\n"
        "cz_cols = ['subject','method','c_um','tilt_deg',\n"
        "           'above_frac','onset_depth_um',\n"
        "           'onset_x_spread','onset_y_spread']\n"
        "cz[[c for c in cz_cols if c in cz.columns]]"
    ))
    cells.append(code(
        "display(Image(str(FIG / 'cz_depth_profiles.png')))"
    ))

    # ===== Promoted protocol =====
    cells.append(md(
        "## Promoted protocol summary\n\n"
        "**HCR** - `multi_channel_image_in_band` (N11):\n"
        "1. Density-filter ROI centroids (>= 3 neighbours within 30 um),\n"
        "   bin into 120 um tiles, take 2 %-quantile z per tile.\n"
        "2. Robust IRLS-Huber quadratic through the tile envelope ->\n"
        "   anchor surface.\n"
        "3. Load every HCR channel at level 4, combine pointwise by max.\n"
        "4. Inside a ±150 um band around the anchor, per-column relative\n"
        "   threshold (bg + 0.25 * (max - bg)) with dynamic-range floor\n"
        "   and 10 um persistence — yields per-column top-of-signal z.\n"
        "5. Second robust IRLS-Huber quadratic through the image top-z\n"
        "   values.\n"
        "6. Clamp against ROI tile envelope; subtract 3 um safety.\n\n"
        "**CZ** - `cz_image_with_roi_ceiling`:\n"
        "1. Fit tilted image plane with `relative_margin = 0.02`,\n"
        "   `min_signal_abs = 50`.\n"
        "2. Bin ROIs into 80 um tiles, per-tile min-z.\n"
        "3. Lift the plane to cover every tile; subtract 3 um safety."
    ))

    nb["cells"] = cells
    NB.parent.mkdir(parents=True, exist_ok=True)
    with open(NB, "w") as f:
        nbf.write(nb, f)
    print("Wrote", NB)


if __name__ == "__main__":
    build()
