"""Build the subgoal 04/R1/01 notebook (v2) from subgoal_01_gfp_threshold_results.json.

Produces `notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb` with markdown
narration, tables, and figures embedded via IPython.display.Image.

v2 scope: distribution-only thresholding, validated by `coreg_table` coverage.
No fixed-percentile / fixed-fraction method may be chosen as the default —
future subjects will not have a coreg table.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path("/root/capsule/code")
RESULT_JSON = ROOT / "sessions/04_R1_coarse_align/subgoal_01_gfp_threshold_results.json"
FIG_DIR = ROOT / "sessions/04_R1_coarse_align/subgoal_01_figures"
OUT = ROOT / "notebooks/04_R1_subgoal_01_GFP_positive_threshold.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> None:
    with open(RESULT_JSON) as f:
        res = json.load(f)

    cov_target = res.get("coverage_target", 0.95)
    winners = res.get("winners_detail", [])
    winner_name = winners[0]["name"] if winners else "(none)"

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Session 04 / R1 / Subgoal 01 (v2) — GFP+ threshold\n\n"
        "**Goal.** Replace the hard `counts ≥ 5` GFP+ cutoff with a principled, "
        "**distribution-only** threshold. The legacy cutoff is both arbitrary and too low "
        "(54–68 % GFP+ on the four spot-data subjects), so R1's downstream estimators "
        "see far too many candidates.\n\n"
        "**Constraints (from the user).**\n"
        "1. The method must derive the threshold from the per-subject distribution alone. "
        "*No fixed percentile or fraction* — future subjects without a coreg table must "
        "inherit the same rule.\n"
        "2. Among distribution-driven methods, prefer the one producing the **highest** "
        f"threshold while still covering ≥ {cov_target:.0%} of the subject's "
        "`coreg_table.csv` HCR IDs.\n"
        "3. The `coreg_table` is used *only* to validate. No method consults it when "
        "picking the threshold.\n\n"
        "**Scope.** Four spot-data subjects `788406, 790322, 767018, 782149`. Intensity-only "
        "subjects `755252, 767022` are out-of-scope — they carry no spot counts.\n\n"
        "**Sources.** Driver `dev_code/04_r1_subgoal_01_gfp_threshold.py`, "
        "metrics `sessions/04_R1_coarse_align/subgoal_01_gfp_threshold_results.json`, "
        "figures `sessions/04_R1_coarse_align/subgoal_01_figures/`."
    ))

    cells.append(md(
        "## 1. Why the baseline fails\n\n"
        "`benchmark_data_loader._load_gfp()` applies `counts >= DEFAULT_GFP_MIN_SPOTS (=5)`, "
        "which sits at roughly the 25th percentile of spot-bearing cells. That puts "
        "**54–68 %** of *all* HCR cells into the GFP+ set — orders of magnitude above the "
        "expected sparse-label fraction and the main reason R1's 2-D GFP-density xcorr "
        "underperformed on XY translation (cf. `sessions/04_R1_coarse_align/log.md`).\n\n"
        "The manual pipeline's heuristic (`min_counts = matched_cells.counts.min() * 1.2`) "
        "is circular at R1 (no matches exist yet) and cannot be reused."
    ))

    cells.append(md(
        "## 2. Data characterisation"
    ))

    cells.append(code(
        "from pathlib import Path\n"
        "import json\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n"
        "\n"
        "SESSION = Path('/root/capsule/code/sessions/04_R1_coarse_align')\n"
        "FIG = SESSION / 'subgoal_01_figures'\n"
        "with open(SESSION / 'subgoal_01_gfp_threshold_results.json') as f:\n"
        "    res = json.load(f)\n"
        "per_subj = res['per_subject']\n"
        "SUBJECTS = ['788406','790322','767018','782149']\n"
        "pd.set_option('display.float_format', lambda v: f'{v:.3f}')\n"
    ))

    cells.append(code(
        "rows = []\n"
        "for sid in SUBJECTS:\n"
        "    d = per_subj[sid]\n"
        "    row = {\n"
        "        'subject': sid,\n"
        "        'n_hcr_total': d['n_hcr_total'],\n"
        "        'n_spot_bearing': d['n_spot_bearing_cells'],\n"
        "        'spot_bearing_frac': d['n_spot_bearing_cells']/d['n_hcr_total'],\n"
        "        'n_coreg_rows': d['n_coreg_rows'],\n"
        "    }\n"
        "    row.update({f'count_{k}': v for k, v in d['raw_counts_percentiles'].items()})\n"
        "    if d.get('coreg_counts_percentiles'):\n"
        "        row.update({f'coreg_{k}': v for k, v in d['coreg_counts_percentiles'].items()})\n"
        "    rows.append(row)\n"
        "pd.DataFrame(rows).set_index('subject')"
    ))

    cells.append(md(
        "**Observations.**\n"
        "- Spot-bearing cells are **70–80 %** of all HCR cells — there is no clean noise-"
        "floor below `counts=1` because cells with 0 spots are not written to the CSV at "
        "all.\n"
        "- The `coreg_table.csv` HCR IDs are the most trustworthy reference we have. Their "
        "count distribution has `p1` in the **tens** and `p5`/`p10` well into the hundreds, "
        "so any threshold up to ~100 is expected to preserve ≥95 % coverage.\n"
        "- This means a distribution-based method can aim quite high (≥ 80 counts) before "
        "it starts dropping canonical pairs."
    ))

    cells.append(md(
        "## 3. Candidate strategies (all distribution-only)\n\n"
        "| ID | Description |\n"
        "|---|---|\n"
        "| baseline | `counts >= 5` (current default — reference only, not a candidate). |\n"
        "| **A** | 2-component GMM on `log(counts ≥ 1)`, posterior crossover at P=0.5. |\n"
        "| **A3** | 3-component GMM on `log(counts)` with 0-count cells encoded as `log(0.5)` to admit a background mode. |\n"
        "| **B** | Otsu's between-class variance on `log(counts ≥ 1)`. |\n"
        "| **C** | Kneedle elbow of sorted-descending counts (below-chord maximum). |\n"
        "| **D** | Kittler–Illingworth min-error on `log(counts ≥ 1)`. |\n"
        "| **Triangle** | Zack's triangle method on `log(counts ≥ 1)`. |\n"
        "| **Yen** | Yen's maximum-entropy threshold on `log(counts ≥ 1)`. |\n"
        "| **ISODATA** | Iterative-mean (Ridler–Calvard) on `log(counts ≥ 1)`. |\n"
        "\n"
        "**Deliberately excluded:** fixed-fraction / fixed-percentile strategies "
        "(\"top 15 %\", \"top 20 %\", etc.). They require a hand-picked value that cannot be "
        "generalised to future subjects without a coreg table."
    ))

    cells.append(md(
        "## 4. Per-strategy metrics\n\n"
        "For each subject × strategy we record:\n"
        "- `threshold_counts`: the integer cutoff the method picked.\n"
        "- `gfp_frac`: fraction of all HCR cells that pass (informational).\n"
        f"- `coreg_coverage`: fraction of coreg-table HCR IDs that pass — the primary "
        f"quality metric. Target ≥ {cov_target:.2f}."
    ))

    cells.append(code(
        "def strategy_table(metric):\n"
        "    rows = []\n"
        "    names = list(per_subj[SUBJECTS[0]]['strategies'].keys())\n"
        "    for name in names:\n"
        "        row = {'strategy': name}\n"
        "        for sid in SUBJECTS:\n"
        "            row[sid] = per_subj[sid]['strategies'][name].get(metric)\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('strategy')\n"
        "\n"
        "display(Markdown('**threshold_counts** — per-subject integer cutoff'))\n"
        "display(strategy_table('threshold_counts'))\n"
        "display(Markdown('**gfp_frac** — |GFP+| / n_hcr_total'))\n"
        "display(strategy_table('gfp_frac'))\n"
        "display(Markdown(f'**coreg_coverage** — target ≥ {res[\"coverage_target\"]:.2f}'))\n"
        "display(strategy_table('coreg_coverage'))"
    ))

    cells.append(md(
        "## 5. Log-count histograms\n\n"
        "Grey bars = all spot-bearing cells (log-count axis). Blue bars = the subset that "
        "appears in `coreg_table.csv`. Vertical lines mark each strategy's chosen threshold."
    ))
    cells.append(code("display(Image(str(FIG / 'log_count_histograms.png')))"))

    cells.append(md(
        "## 6. Winner selection\n\n"
        f"Rule: of the distribution-driven strategies whose coreg coverage is ≥ "
        f"{cov_target:.2f} on **every** subject, pick the one with the highest "
        "`threshold_min` across subjects (tie-break by `threshold_mean`). Baseline is "
        "excluded from the ranking (kept as a reference only)."
    ))

    cells.append(code(
        "winners = res['winners_detail']\n"
        "pd.DataFrame(winners).set_index('name')"
    ))

    cells.append(md(
        f"### Verdict: **{winner_name}**\n\n"
        "The Yen max-entropy threshold on `log(counts ≥ 1)` produces a per-subject "
        "cutoff in the **84–156 counts** range. On every subject it retains ≥ 96 % of the "
        "coreg-table HCR IDs while cutting the GFP+ fraction to roughly **10–16 %** of all "
        "HCR cells — a ~4–6× reduction vs the baseline.\n\n"
        "The other passing strategies (A3, B, Isodata, A, Triangle) land much lower — "
        "15–23 counts — and still admit 30–50 % of HCR cells. They pass the coverage bar "
        "only because they are permissive, and they leave the downstream R1 estimators in "
        "essentially the same regime as the baseline. Yen is the only method that "
        "translates the coreg-cell count distribution's natural separation into a high "
        "cutoff.\n\n"
        "Kneedle (C) and Kittler–Illingworth (D) are unstable on this histogram shape and "
        "are disqualified by coverage (C: min 0.944) or outright divergence (D flips "
        "between count=2 and count=1262 across subjects)."
    ))

    cells.append(md(
        "### Why GMM/Otsu/Triangle under-cut\n\n"
        "The log-count distribution does not have two clean modes. The zero-count population "
        "is *absent* from the CSV (it's implicit), so methods that assume two populations "
        "**inside the spot-bearing data** find a split between weakly- and strongly-labelled "
        "cells — not between background and signal. That split sits at ~15–25 counts and is "
        "biologically too low. Triangle collapses to `counts = 2` because the histogram is "
        "monotonically decreasing. Yen works here because it maximises entropy across the "
        "log-count range, which on a heavy-tailed distribution naturally lands in the tail."
    ))

    cells.append(md(
        "## 7. API wiring\n\n"
        "`benchmark_data_loader` now accepts:\n\n"
        "```python\n"
        "load_subject(sid, gfp_threshold_method='yen_log')      # new default\n"
        "load_subject(sid, gfp_threshold_method='counts_min',   # legacy\n"
        "                  gfp_min_spots=5)\n"
        "```\n\n"
        "The returned `SubjectData.gfp_min_spots` reflects the derived integer threshold "
        "(per subject), and `gfp_threshold_method` records the method used. The "
        "`fixed_frac` helper remains in the module for diagnostics only — it is no longer "
        "the default and is not used by the R1 benchmark."
    ))

    cells.append(md(
        "## 8. Validation & follow-ups\n\n"
        "1. **R1 re-validation.** `dev_code/04_r1_benchmark.py` was rerun with the new "
        "default; results land in `sessions/04_R1_coarse_align/r1_results.json` (see "
        "`log.md` for the before/after summary).\n"
        "2. **Intensity-only subjects (755252, 767022).** Not affected by this change — "
        "they carry no spot counts. A parallel \"Yen on log(mean intensity)\" check is a "
        "follow-up subgoal.\n"
        "3. **Cross-subject stability.** Yen's per-subject threshold range (84–156) is "
        "wider than A3's (21–31), but *proportionally* the resulting GFP+ fractions are "
        "much more consistent (10–16 % vs 31–54 %) — which is what the downstream "
        "estimators care about."
    ))

    nb.cells = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
