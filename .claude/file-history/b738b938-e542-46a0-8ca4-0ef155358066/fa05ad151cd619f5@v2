---
name: S11 v2 features result
description: Stage-2 ROI-quality v2 feature redesign (405-only, opening r=3, core/shell, adjacency) — binary LOSO AUC 0.924 (v1: 0.895), missed 0.95 target.
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
S11 ROI-quality stage-2 v2 features (44 cols, 405-only + opening r=3 + adjacency) trained 2026-04-30 on 552 labels (good=315, bad=121, bad_ok=88, merged=28).

**Binary LOSO**: mean AUC=0.924, AP=0.967, Brier=0.112, acc@0.5=0.835.
- Range: 755252 0.826 → 790322 0.986; 788406 0.883 is second worst.
- Top 4 features by gain: `frac_kept_opening` (1675), `solidity_opened` (946), `volume_vox_opened` (216), `solidity_raw` (157). User's morphology hypothesis validated.
- Sign of `c405_shell_minus_core_p50` is opposite of user expectation (good cells: shell darker than core, median ≈ −54). Likely because expanded mask shell extends into background. Kept feature unflipped — model still gets signal.

**4-class LOSO**: mean acc=0.775, mean f1_macro=0.552. f1_good=0.92 ✓, f1_bad=0.74, f1_bad_ok=0.30 ✗ (often → bad or good), f1_merged=0.29 (only 28 labels).

**Why**: User specified channel biology (405=Rn28S, 488=GFP subpopulation), required opening r=3 to remove processes after expansion, asked for adjacency features. v1 had 51 mostly-405 features mixing 488 / 594 / GFP density / nucleus framing — confused signals.

**How to apply**: production models at `cached_roi_quality/roi_quality_stage2_{binary,4class}_v2.txt`; meta JSON at `roi_quality_stage2_meta_v2.json`. To push above 0.95, need more labels in 0.3–0.7 score band — 14k+ unlabelled uncertain on 755252/767018/767022, 3k on 782149.
