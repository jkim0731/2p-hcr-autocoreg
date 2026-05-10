# Current protocol

> **Path note (2026-04-28):** `notebooks/`, `sessions/`, and
> `full_automatic_execution_01/` were moved to
> `/root/capsule/data/claude-data_ophys-mfish-autocoreg_260503/` (read-only). Any reference to
> `code/notebooks/...`, `code/sessions/...`, or
> `code/full_automatic_execution_01/...` below resolves to
> `data/claude-data_ophys-mfish-autocoreg_260503/<same suffix>`. New writes must live under
> `/root/capsule/code/`.

- Manual coregistration and cell by cell QC
- Using BigWarp

## Automated branch (in development)

**Currently promoted (cached for all 6 benchmark subjects):**
- Pia / surface fits — `code/dev_code/surfaces_iter08.py`
  (`get_cz_surface_iter08`, `get_hcr_top_surface_iter07`,
  `get_hcr_bottom_surface_iter08`).
- `sxy` from per-cell ROI-area ratio — `code/dev_code/roi_area_sxy.py`
  (`estimate_sxy_roi_area`); ±2 % on the two good spot subjects,
  −15 % on 782149 (open issue).
- 2-D top-slab surface registration —
  `code/dev_code/surface_registration_v2.py::get_surface_registration(s)`.
  JSON-cached; picks the best of rigid / affine / PWR 3×3 / PWR 4×4,
  scored as Pearson NCC of the warped CZ binary against the raw 488
  MIP. PWR 4×4 wins all 6 subjects. See
  `docs/08 Automated surface registration.md`.
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
- 3-D `sz` estimator (the binding centroid-ICP failure per the
  reassessment in
  `code/full_automatic_execution_01/reassess_with_surfaces/`).
- Locked-prior 3-D warm-start consuming all of the above.
- Surface-anchored seed constellation + TPS expansion that mimics
  the manual workflow.

The detailed staged design lives in
**`docs/09 Full automatic v2 plan.md`** and is summarised in
`docs/07 Grand Plan.md` §13.  The ledger in §11 covers the v1 catalog
work (sessions S01–S64 in `full_automatic_execution_01/`).

## Protocol files
/root/capsule/code/step_*_*.ipynb
1. Process input data files
2. Manual coregistration using BigWarp
  1) Find 4-6 initial constellation by eyes in 2p data near the surface (all ROIs near each other, unique pattern)
  2) Find GFP+ cell patterns in HCR data matched to the the 2p initial constellation
  3) Repeat 1->2 until confident match found
  4) Set the constellation and match as landmarks and apply thin plate spline transformation.
  5) Find matched patterns near the current landmarks, gradually covering more volume. Do not exhaust all cells. Total about 50 to 100 landmarks covering almost the entire volume. Activate each landmark whenever found, applying thin plate spline transformation.
3. Automatic mapping for qc: using current landmarks, find matched ROIs between 2p and GFP+ HCR ROIs based on distance.
4. Manually go through each candidate match, and add them to landmarks if matched by eyes. Look for context and cell morphology, and if activating the pair increases correlation within the context volume.
5. Repeat 3 and 4 until no more new matched pairs appear.
6. generate coregistration table.