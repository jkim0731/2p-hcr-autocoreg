
## Automated branch (in development)

**Currently promoted (cached for all 6 benchmark subjects):**
- Pia / surface fits — `code/dev_code/surfaces_iter08.py`
  (`get_cz_surface_iter08`, `get_hcr_top_surface_iter07`,
  `get_hcr_bottom_surface_iter08`).
- `sxy` from per-cell ROI cross-sections — `code/dev_code/roi_area_sxy.py`.
  **PRODUCTION (promoted 2026-06-04): `estimate_sxy_min_rule` — min-rule 2×
  ¼-FOV.** `hcr_slab = min(p99(HCR GFP+∩ok∩¼-FOV depth), 2·p99(CZ depth))`,
  `cz_slab = hcr_slab/2` (CZ slab is HALF the HCR slab; axial ~2× expansion,
  capping CZ shallower raises sxy), `sxy = sqrt(median HCR max-xsection /
  median CZ max-xsection)`; the 2× is a heuristic, NOT measured sz (circular).
  GT-free. Recovers thin-HCR **782149 → 1.7336** (the old −15 % full-span case).
  Grid-search fallback at `SXY_GRID_SEARCH_OFFSETS` for new roaming subjects.
  Supersedes the prior full-span `estimate_sxy_roi_area` / slab-auto estimators.
- 2-D top-slab surface registration —
  `code/dev_code/surface_registration_v2.py::get_surface_registration(s)`.
  JSON-cached; picks the best of rigid / affine / PWR 3×3 / PWR 4×4,
  scored as Pearson NCC of the warped CZ binary against the raw 488
  MIP. **Registration MIP promoted 2026-06-04 to 80/150** (`CZ_SLAB=(0,80)`,
  `HCR_SLAB=(0,150)`, was 50/100 — a denser MIP lands the fit for thin-HCR
  subjects like 782149). See `docs/08 Automated surface registration.md`.
- Locked-prior warm-start (Stage A) —
  `code/dev_code/locked_prior_warm.py::compute_locked_prior_warm_start(s)`.
  Pins all 7 affine DOF (R, sxy, sxy, sz_prior, t_x, t_y, t_z) using
  `surface_registration_v2` (sxy + tilt + tx/ty) and `surfaces_iter08`
  (CZ pia / HCR top for tilt + tz anchor). Stage B refines sz.
- Slab-rigid `sz` estimator (Stage B) —
  `code/dev_code/sz_estimator.py::get_sz(s)`. JSON-cached (v2 schema)
  at `code/dev_code/cached_sz/<sid>.json`. Iter-7 slab-side-view FFT
  rigid NCC: binarized CZ ROI warped into the HCR crop, x-MIP slabs
  over the central 500 µm of x as 5×100 µm slabs, 2-D FFT cross-corr
  per slab, mean NCC across slabs scored on the sz grid. 6/6 subjects
  exact match to the iter-7 GT peak (grid step 0.10 µm/vox).
- 3-D overlap crop after (R, sxy, surface 2-D affine, sz) —
  `code/dev_code/overlap_crop.py::get_overlap_crop(s, margin_frac=0.10)`
  and `crop_hcr_volume(s, vol, margin_frac=0.10)`. **Binding rule for
  all v3 sessions: work only inside this crop + 10 % margin** (see
  "Working-volume contract" below).
- HCR per-ROI quality classifier (S11 v5d, stage 2) —
  `code/dev_code/roi_quality_v5d.py`. 92 voxel-unit + within-subject
  pct-rank features (no µm-suffixed features, no expansion-rate
  dependence); LightGBM binary + 4-class. LOSO @ 1648 labels:
  binary AUC 0.917, 4-class f1_macro 0.720. Public API:
  `extract_features(sid)`, `predict(features)`, `predict_subject(sid)`,
  `train(...)`. Models + meta + per-subject OOF parquets cached at
  `code/dev_code/cached_roi_quality/`. Feature-extraction modules
  (`roi_quality_v{2,3,4,5}.py`, `roi_v3_axis_features.py`,
  `roi_v6_voxel.py`) are also under `code/dev_code/`; the µm /
  dead-feature / low-gain peak-counter code paths have been physically
  removed.
- HCR ROI labelling GUI (standalone matplotlib app) —
  `code/dev_code/roi_label_gui.py`. CLI:
  `python code/dev_code/roi_label_gui.py --sid 788406 --reviewer alice`.
  Uses `roi_quality_v5d.py` for inference. Labels still append to the
  session JSONL `code/sessions/v3_S11_roi_quality/outputs/roi_qc_actions.jsonl`
  for label-history continuity.

**Not yet promoted (the next implementation push):**
- Surface-anchored seed constellation + TPS expansion that mimics
  the manual workflow (subgoal 4.5 in `10 Grand Plan v3`).
- Track B per-pair image-NCC scoring (subgoal 4.2).
- Pair-level GBT classifier (subgoal 4.3).

The detailed staged design lives in
**`docs/09 Full automatic v2 plan.md`** and is summarised in
`docs/07 Grand Plan.md` §13.  The ledger in §11 covers the v1 catalog
work (sessions S01–S64 in `full_automatic_execution_01/`).
The active v3 plan is `docs/10 Grand Plan v3 — Cell-cell matching and QC.md`.

**Metric definition correction (2026-06-04):** all recall/precision figures from
Session 15 used a **pose-dependent GT** (coreg pairs filtered to those with both
CZ and HCR inside the spatial pool). This is circular. The authoritative GT is now
`scoring_gt(inp)` in `sessions/15_geom_features/_data.py`: coreg pairs whose HCR
cell is GFP+∩ok, with the CZ side unfiltered. Corrected 6-subject tables are at
`sessions/15_geom_features/outputs/corrected_gt_rescore/`. Gate rankings are
qualitatively unchanged.

---

## Session 13b — GFP+ threshold feature compare (2026-05-14)

Compared three per-cell GFP signals on all 6 benchmark subjects using
BIC-best GMM on `log(positive feature)` fit to the v5d-kept (good +
bad_ok) ROI subset:

1. **spot_density** — `spot_count / volume` from the per-subject best
   spot source (`spot_488_counts.csv` / R1 aggregated spots / R2
   `mixed_cell_by_gene.csv` for the R1-failed subjects 755252 + 767022).
2. **unmix_density** — `spot_count / volume` from `unmixed_all_cells.csv`.
3. **mean_minus_bg** — channel-488 `mean − background` from
   `cell_data_mean_*_R1.csv` (computed on the fly from L2 zarrs for the
   four subjects without the CSV).

Metric: `shape_score = (μ_right − μ_next) / max(σ_right, σ_next)` on the
top two GMM components (right-most positive mode separation from its
left neighbour). Higher = cleaner bimodality.

| subject | mean−bg | spot | unmix |
|---------|--------:|-----:|------:|
| 755252  | 2.41    | 1.00 | 2.62  |
| 767018  | 2.72    | 4.17 | 4.58  |
| 767022  | 2.61    | 2.63 | 2.53  |
| 782149  | 3.07    | 4.03 | 1.53  |
| 788406  | 2.62    | 3.41 | 1.76  |
| 790322  | 2.49    | 3.67 | 2.06  |
| **mean**| **2.65**| **3.15** | **2.51** |

Fraction of coreg-matched-and-kept cells that fall above the BIC cutoff
(recall on the GT match set, per fit_subset='kept'):

| subject | mean−bg | spot | unmix |
|---------|--------:|-----:|------:|
| 755252  | 0.648   | 0.707| 0.479 |
| 767018  | 0.856   | 0.975| 0.980 |
| 767022  | 0.883   | 0.796| 0.770 |
| 782149  | 0.878   | 0.956| 0.961 |
| 788406  | 0.862   | 0.937| 0.883 |
| 790322  | 0.978   | 0.950| 0.988 |
| **mean**| **0.851**| **0.887**| **0.844** |

**Findings:**
- Spot-density wins shape_score on 4/6 subjects (767018, 782149,
  788406, 790322) and the column mean (3.15 vs 2.65 / 2.51) and recall
  mean (0.887 vs 0.851 / 0.844).
- The two R1-failed subjects (755252, 767022) are spot-density's only
  losses; their R2 GFP probe likely has lower SNR than the other
  subjects' R1 probe — but spot-density still beats unmix-density
  there.
- Unmix-density is competitive only on 767018; it underperforms on
  every spot subject (the unmixing trims spots in a way that flattens
  the bimodal mode separation).
- mean−bg is the most consistent fallback (2.4–3.1 across subjects)
  but rarely wins.

**Action:** stay with **spot_density + BIC-best GMM** for GFP+
thresholding across all 6 subjects (now feasible since the R2
`mixed_cell_by_gene.csv` is plugged in for 755252/767022). Code:
`code/sessions/13_pairwise_unmix_gfp/run_gfp_filter_compare.py`; output:
`outputs/gfp_filter_compare/summary.csv` (36 rows) + 18 PNG histograms.

---

## Working-volume contract (binding for all future v3 sessions, 2026-05-02)

**Rule.** Every future session — registration refinement, cell-cell
matching, classifier training, QC, GUI — operates **only inside the
3-D overlap crop** delivered by the locked frame, plus a **10 % margin
on each axis** (default `margin_frac = 0.10`). Out-of-crop voxels and
ROIs are dropped before any compute.

**Why.** The two assumptions that broke v1 (unknown global pose; no
shared image-level anchor) are gone. Every remaining method should
respect that and refuse to operate over the full HCR volume — both
because it's wasteful and because it dilutes signal with regions where
the registration prior says nothing useful can match.

**How to apply.** A session that needs the working volume calls:

```python
from overlap_crop import get_overlap_crop, crop_hcr_volume

crop = get_overlap_crop(s, margin_frac=0.10)
# crop["bbox_hcr_um"]    -> [z0, z1, y0, y1, x0, x1] in HCR µm
# crop["bbox_hcr_l2_vox"] -> level-2 voxel indices
# crop["sz_used"], crop["margin_frac"], crop["hcr_voxel_um"]

hcr_slab = crop_hcr_volume(s, hcr_vol, margin_frac=0.10)
```

ROI / centroid filtering is on the caller; image slicing should go
through `crop_hcr_volume(s, vol)` so the same bbox + margin definition
is used everywhere.

**Inputs that define the crop.** Rotation `R` (180° prior + tilt + PWR
θ residual), `sxy` (PWR-affine image-NCC, the source of truth —
`roi_area_sxy` is bootstrap-only per `feedback_sxy_source_of_truth`),
surface 2-D affine (`surface_registration_v2`), `sz`
(`sz_estimator.get_sz`). These four pieces are the **only**
registrations expected by `get_overlap_crop`; anything finer (TPS,
accepted-pair refit, etc.) refines *inside* the crop, never outside.

**Out-of-crop behaviour.** Functions in `code/dev_code/` should not
assume cropped input — they accept full volumes / centroid lists for
backwards compatibility. The cropping discipline is enforced by
**session-level callers**: each new session begins by calling
`get_overlap_crop(s)` and slicing every downstream input through it.
If a session's first cell is "load full HCR volume" without that crop,
the session is wrong.

**Margin choice.** 10 % is the default; sessions that need more (e.g.
surface-anchored seed search at the periphery) may override
`margin_frac=` on the call but must justify the choice in the
session's `log.md`.