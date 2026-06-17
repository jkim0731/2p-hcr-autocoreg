# Work In Progress

_Current-state snapshot. Last major update: **2026-06-17**._
_Long-form handoff: `/scratch/sessions/17_interim_summary/README.md`._
_How to run / where the code lives: `docs/12 Code repositories and how to run.md`._

---

## Status at a glance (2026-06-17)

Automated CZ (2p-GCaMP z-stack) → HCR (light-sheet mFISH) coregistration now
runs **end-to-end as a 2-step process** and has been migrated out of the
`code/dev_code` scratch tree into two clean, installable repos.

| area | status |
|---|---|
| **Step 1 — image-based rough/warm registration** (locked prior) | ✅ working, GT-free |
| **Step 2 — soma-print 3-D cell-cell matching + iterative TPS** (fine reg, custom filters) | ✅ working |
| **HCR 3-D ROI quality classifier** (required by the matcher) | ✅ working (v5d, LightGBM) |
| **Presentation video + figures** | ✅ done (session 16) |
| 3-D shape-context matcher ("ROI ContextNet") | ❌ abandoned (soma-print won) |
| Cellpose-SAM resegmentation | ❌ abandoned for now (no quality gain; GPU-gated) |
| **QC app** | 🟡 ~half done (pass/fail labeller built; manual-improvement app not built) |

---

## The pipeline (2-step)

**Step 1 — rough/warm registration** (`autocoreg` package), order
**surfaces → sxy → top-slab 2-D reg → sz → overlap crop**:
- Pia/surface fits — `surfaces_iter08` (`get_cz_surface_iter08`,
  `get_hcr_top_surface_iter07`, `get_hcr_bottom_surface_iter08`).
- Lateral scale `sxy` — `roi_area_sxy.estimate_sxy_min_rule` (**PRODUCTION,
  promoted 2026-06-04**): min-rule 2× ¼-FOV, `hcr_slab = min(p99(HCR GFP+∩ok∩
  ¼-FOV depth), 2·p99(CZ depth))`, `cz_slab = hcr_slab/2`, `sxy = √(median HCR
  max-xsection / median CZ max-xsection)`. GT-free; the 2× is a heuristic, not
  measured sz (that would be circular). Recovers thin-HCR 782149 (1.7336).
- 2-D top-slab registration — `surface_registration_v2.get_surface_registration`:
  best of rigid/affine/PWR 3×3/PWR 4×4, scored as Pearson NCC of warped-CZ binary
  vs raw 488 MIP. **Registration MIP promoted 2026-06-04 to CZ 0–80 / HCR
  0–150 µm.**
- Axial scale `sz` — `sz_estimator.get_sz`: slab side-view FFT-NCC sweep (6/6
  subjects exact-match to the iter-7 GT peak).
- Lock + crop — `locked_prior_warm`, `overlap_crop` (cuts GFP+ search ~7×).

**Step 2 — soma-print fine registration** (`autocoreg` package):
- Descriptor — `soma_print.py` (Wang 2026 adapted): per-cell m-NN relative
  vectors; pair score = mean of n-best of m_cz × m_hcr vector distances
  (production m_cz=15, m_hcr=30, n=5).
- Matcher — `run_step3_v3.py` (spec `step3_v3_spec_2026-06-01.md`): per round,
  candidates within **fixed R_cand=150 µm** → **mutual-best** → **round-0-only
  local-flow** pre-filter → **anchor-vote** gate (production) → fit **TPS**,
  re-warp, iterate to convergence (e.g. 790322: 667→692→698). Gate alternatives
  compared: LR (high-precision/low-recall), NCC (image-validated → kept as a
  **QC validator**, not the primary sweep).
- Performance (anchor_vote, pose-independent GT, 2026-06-04): recall 0.87–0.97,
  precision dominated by silence-FP (likely-real, unlabeled). See the interim
  summary for the per-subject table.

**Why soma-print** (fig_09, load-bearing): under the realistic *locked* pose
soma-print barely degrades (AUC@50 0.997→0.993, recall@5≈1.0); shape-context
collapses (recall@1 0.86→0.56); centroid stays weak (AUC≈0.63).

**HCR 3-D ROI classifier** (`mfish-roi-classifier`): LightGBM binary + 4-class
(good/bad_ok/bad/merged) on 91 voxel-unit features. LOSO binary AUC 0.921,
4-class acc 0.703. Supplies the GFP+∩ok pool + junk removal the matcher needs;
cross-repo contract = `{sid}_stage2_4class_proba_v5d_um.parquet`.

---

## Code now lives in two repos (dev_code is legacy)

- **`2p2fish`** (`github.com/jkim0731/2p2fish`, package `autocoreg`) — full
  pipeline: rough reg + soma-print fine reg + matcher + Qt QC viewer.
- **`mfish-roi-classifier`** (`github.com/jkim0731/mfish-roi-classifier`) — the
  v5d ROI-quality classifier ("repo B"); ships trained models + label log.

`code/dev_code/` is **legacy** (development scratch). It had a real standalone
bug — `surfaces_iter08` / `surface_registration_v2` imported 6 helper modules via
a **stale, non-existent** path (`data/claude_data/...`). **Fixed 2026-06-17**:
vendored the 6 + 1 transitive modules (`iter07_compute`, `iter08_cz_prior`,
`iter08_hcr_bottom`, `compare_binarization`, `register_binary`,
`register_nonrigid_variants`, `compute_projections`) from `2p2fish` (flat
imports) and removed the dead `sys.path` hacks; all promoted modules now import
cleanly. Apply script: `/scratch/dev_code_standalone_fix/apply.py`. For new work,
**edit the repos, not dev_code.**

See `docs/12 Code repositories and how to run.md`.

---

## Abandoned approaches

- **3-D shape-context matcher ("3D ROI ContextNet")** — `shape_context.py`
  (Belongie-2002 log-spherical descriptor, pia-normal oriented). Tried as the
  geometric matcher; stopped because soma-print clearly won (fig_09). Retained as
  a comparison component, not production.
- **Cellpose-SAM resegmentation** — session 14 plan: finetune cpsam on HCR GT +
  405/488, judge with v5d. Abandoned for now: only 2 fully-labeled 405-only 3-D
  GT blocks, ~40 GPU-hr, no quality gain over released-cpsam-on-405. Parked.

---

## QC app (≈ half done)

- **Pass/fail labeller — BUILT.** Per-CZ-ROI / per-pair reviewer over matcher
  output (cube view: 488 background, CZ contour, matched HCR contour, neighbour
  contours; radio good/bad/unsure for matched, visible/not-visible for
  unmatched; autosave). In the repo as `autocoreg/qc/app.py`; launch via
  `autocoreg run <sid> --qc`. Session variants: `qc_qt_app.py`, `qc_pair_app.py`,
  `qc_app.py`.
- **Manual-improvement app — IN PROGRESS / not built.** Intended to let an
  operator improve a registration by hand, mimicking the automatic part of the
  previous manual BigWarp process (seed near-surface landmarks → TPS → add
  landmarks descending in depth → iterate). This is the remaining half.

---

## Presentation materials (done) — session 16

`16_intermediate_summary_materials` (example 790322; GT panels over 6 subjects).
13 figures (PNG@300 + PDF) + movies: **5 auto-pipeline movies** (`mov_auto_1..5`:
problem → rough reg → soma-print → customized matcher → result), **1
manual-protocol movie** (`mov_D_manual_registration`), **1 combined film**
(`mov_grand_registration`). Colours: CZ=magenta, HCR 405=white, HCR 488=cyan,
matches=gold. Read-only at
`/data/claude-data_ophys-mfish-autocoreg_soma-print_260616/sessions/16_intermediate_summary_materials/`.

---

## GFP+ thresholding decision (settled, 2026-05-14)

Use **spot_density + BIC-best GMM** on `log(spot_count / volume)` for GFP+
thresholding across all 6 subjects (beats unmix-density and mean−bg on shape and
recall; the R2 `mixed_cell_by_gene.csv` covers the R1-failed 755252/767022).

---

## Working-volume contract (binding for all v3 work, 2026-05-02)

**Rule.** Every stage — registration refinement, cell-cell matching, classifier,
QC, GUI — operates **only inside the 3-D overlap crop** delivered by the locked
frame, plus a **10 % margin per axis** (default `margin_frac=0.10`). Out-of-crop
voxels and ROIs are dropped before any compute.

**Why.** The two assumptions that broke v1 (unknown global pose; no shared
image-level anchor) are gone; nothing should operate over the full HCR volume —
it is wasteful and dilutes signal with regions the prior says cannot match.

**How.** A session that needs the working volume calls:

```python
from overlap_crop import get_overlap_crop, crop_hcr_volume   # autocoreg.overlap_crop in the repo
crop = get_overlap_crop(s, margin_frac=0.10)
hcr_slab = crop_hcr_volume(s, hcr_vol, margin_frac=0.10)
```

The crop is defined by R (180° + tilt + PWR θ), sxy, the surface 2-D affine, and
sz. Anything finer (TPS, accepted-pair refit) refines **inside** the crop, never
outside.

---

## Metric definition (authoritative, corrected 2026-06-04)

All recall/precision use the **pose-independent GT** (`scoring_gt`: coreg pairs
whose HCR cell is GFP+∩ok, CZ side unfiltered). The earlier pose-dependent GT
(both sides filtered to the pool) was circular and is **superseded**. Gate
rankings are qualitatively unchanged (anchor_vote still leads).

---

## Open items / next steps

1. Build the **manual-improvement QC app** (remaining half of QC).
2. Relabel **767018** ROI GT (under-labeled → inflated silence-FP).
3. ~~mfish-roi-classifier cold-start builders missing~~ **DONE 2026-06-17** —
   the refactor dropped two builders (tight-bbox and per-cell crops); both ported
   as `roi_classifier/feat_tight_bbox.py` + `feat_per_cell_crops.py` with CLI
   `build-bbox` / `build-crops` / `build-features` (chain: bbox → crops → 4 feature
   groups → predict). Tight-bbox verified against the segmentation on 790322
   (exact tight bounds + volume). Note: `dev_code/cached_hcr_cell_tight_bbox/` is
   **level-0** and incompatible with the current level-2 extractors — rebuild with
   `build-bbox`. (shape/v2 `stage1_score` parquet is optional → NaN if absent.)
   Also reconciled the extractors to the **µm** feature set the production `_v5d_um`
   model expects (the refactor had dropped µm outputs); verified exact vs the
   original features + `predict` reproduces the auto-coreg keep-set 97–98 % on 3
   subjects. And parallelized v2/v3 z-strip extraction (`MFISH_FEAT_WORKERS`).
   Full record: **`docs/13 ROI classifier um-vs-vox decision and reconciliation.md`**.
4. Revisit Cellpose-SAM only if a real segmentation-gain case appears.
5. Keep NCC as the soft/top-2 QC validator (ceiling ~86 % argmax / 94 % top-2;
   do not build a hard argmax gate — too lossy).
