
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