---
name: Session 13b GFP+ feature compare — spot_density wins
description: BIC GMM bimodality comparison across spot_density / unmix_density / mean_minus_bg on all 6 subjects; spot_density wins shape and recall; R2 mixed_cell_by_gene.csv unlocks the 2 R1-failed subjects.
type: project
originSessionId: edadc87d-529a-413d-bf67-8e2054d4f869
---
2026-05-14. Compared three GFP+ thresholding features on all 6
benchmark subjects under BIC-best GMM on `log(positive feature)` fit
to v5d-kept ROIs (good+bad_ok). Metric = shape_score (right-mode
separation / max σ).

**Result:** spot_density wins.

- shape_score column mean: spot 3.15 > mean−bg 2.65 > unmix 2.51
  (spot wins 4/6 subjects; loses only on 755252, 767022 — the two
  R1-failed subjects whose GFP signal comes from R2 re-probing).
- Recall on GT-matched ∩ classifier-kept: spot 0.887 > mean−bg 0.851
  > unmix 0.844.

**Why this matters:** for 755252/767022 we used to be stuck with
mean-intensity-only GFP+. The R2 `mixed_cell_by_gene.csv`
(`gene=='GFP'`, `spot_count`, `volume`) makes spot_density available
for those subjects too — same density-based threshold method now
works on all six benchmark subjects (35 750 GFP+ for 755252 @ ≥5
spots; 32 979 for 767022, vs the old 77 785 / 72 213 intensity-based
counts that overcounted autofluorescence).

**How to apply:** default to spot_density + BIC-best GMM for GFP+
thresholding everywhere. For 755252/767022 the spot source is the R2
mixed table (R1 GFP probe failed there). Code at
`code/sessions/13_pairwise_unmix_gfp/run_gfp_filter_compare.py`
(`load_spot_density` has the 3-tier source priority); summary at
`outputs/gfp_filter_compare/summary.csv` + 18 PNG histograms.
