# Code repositories and how to run (status 2026-06-17)

The automated coregistration pipeline now lives in **two installable repos**,
split along the cross-repo data contract. `code/dev_code/` is **legacy**
development scratch — read it for history, but do new work in the repos.

| repo | package | role |
|---|---|---|
| **`2p2fish`** (`github.com/jkim0731/2p2fish`) | `autocoreg` | full coregistration pipeline: rough/warm registration + soma-print fine registration + matcher + Qt QC viewer |
| **`mfish-roi-classifier`** (`github.com/jkim0731/mfish-roi-classifier`) | `roi_classifier` | self-contained HCR ROI-quality classifier ("repo B"; 101 µm features, 405-only, no stage-1 — see docs/13 §7); ships trained models + label log |

---

## Cross-repo contract

`mfish-roi-classifier` writes, and `autocoreg` reads:

```
{MFISH_ROI_QUALITY_DIR}/{sid}_stage2_4class_proba.parquet
columns: hcr_id, p_bad, p_bad_ok, p_good, p_merged   (+ human_label if labelled)
(renamed from {sid}_stage2_4class_proba_v5d_um.parquet, 2026-06-18 — 2p2fish reader to match)
```

`autocoreg` keeps cells where `argmax ∈ {p_good, p_bad_ok}` (the GFP+∩ok pool the
matcher runs on).

---

## autocoreg (2p2fish)

```bash
pip install -e .                 # core
pip install -e ".[qc]"           # + Qt QC viewer (PyQt5)

autocoreg run 790322             # full pipeline on one subject (GT-free)
autocoreg run 790322 --qc        # + launch the QC viewer after matching
```

Pipeline stages (all GT-free): 1) surface fitting (`initial_registration.surfaces`) →
2) sxy (`initial_registration.lateral_scale.estimate_sxy_min_rule`) → 3) surface registration
(`initial_registration.surface_registration`) → 4) sz (`initial_registration.axial_scale.get_sz`) →
5) matcher (`finetune_soma_print.matcher.run_subject`).

Config via env vars: `MFISH_DATA_ROOT` (default `/root/capsule/data`),
`MFISH_CACHE_DIR` (default `/root/capsule/code/dev_code`), `MFISH_ROI_QUALITY_DIR`
(default `$MFISH_CACHE_DIR/cached_roi_quality`).

---

## roi_classifier (mfish-roi-classifier)

```bash
pip install -e .                 # core (inference + training)
pip install -e ".[label]"        # + labelling GUI

roi-classifier build-features 790322   # cold-start: tight-bbox + unified feature pass
roi-classifier predict 790322          # inference (reads {sid}_features_all.parquet)
roi-classifier build-bbox 790322       # just the tight-bbox cache (--force to rebuild)
roi-classifier train                   # LOSO CV + production model training
roi-classifier-label                   # labelling GUI
```

Cold-start chain: `build-bbox` → `build-features` (a **single unified extraction
pass** — all feature families from each cell's mask/405-crop read once, written to
one `{sid}_features_all.parquet`) → `predict`. Parallelism is via
`MFISH_FEAT_WORKERS` (z-strip cell-chunking inside the pass); see
`docs/13 …` for the unify + timing details.

Python API:

```python
from roi_classifier.model import predict_subject
df = predict_subject("790322")
# → hcr_id, binary_score, proba_bad, proba_bad_ok, proba_good, proba_merged, predicted_class
from roi_classifier.features import extract_features   # feature matrix only
```

Config via env vars: `MFISH_DATA_ROOT`, `MFISH_CACHE_DIR`
(default `~/.cache/mfish-roi-classifier`), `MFISH_ROI_QUALITY_DIR`,
`MFISH_TIGHT_BBOX_DIR`, `MFISH_PER_CELL_CROPS_DIR`, `MFISH_MODELS_DIR`
(defaults to the committed `models/`, so a fresh clone is immediately runnable
for inference).

> **Resolved 2026-06-17 — the two dropped cold-start builders are ported.** The
> refactor that split feature extraction into `feat_*` modules kept only the
> *readers* of two caches and dropped their *builders*. Both are now bundled:
>
> - **tight-bbox** → `roi_classifier/feat_tight_bbox.py` (`build_tight_bbox` /
>   `_sid`), CLI `build-bbox`. Memory-safe z-strip `find_objects` over
>   `segmentation_mask_orig_res.zarr` → level-2-frame parquet. Verified on 790322:
>   per-cell `volume_vox` + tight bounds match the segmentation exactly (1-voxel
>   halo check), 101,366 cells.
> - **per-cell crops** → `roi_classifier/feat_per_cell_crops.py`
>   (`build_per_cell_crops` / `_sid`), CLI `build-crops`. Ported from the S11
>   `02_dump_per_cell_crops.py`; packbits-encoded padded mask crops (needed by the
>   surface/v4 + protrusion/v5 extractors).
>
>   `02_dump_per_cell_crops.py` — now **optional**, used only by the legacy
>   per-group extractors; the unified pass (below) doesn't need it.
>
> **µm reconciliation + unified pass (2026-06-17).** `predict` initially failed
> because the refactored extractors emitted a reduced/vox-named feature set
> matching neither shipped model. Fixed by restoring the **µm** feature set the
> production `_v5d_um` model expects, then **unifying** the 4 extractors into one
> pass (`build-features` = `bbox → unified pass → {sid}_features_all.parquet`; per
> cell the mask/opening/405-crop are computed once and shared across families).
> Verified **exact** vs the original features (782149 coldcache; worst err 3.9e-4
> on one pre-existing intensity percentile) and `predict` reproduces the auto-coreg
> keep-set 97–98 %. (µm chosen over vox: A/B-tied accuracy, µm is free vs vox's
> `v6_vox`; rank dropped for sample-prep sensitivity.) Full record: **`docs/13`**.
>
> **Parallel extraction.** The unified pass splits each z-strip's cells into
> spatially-contiguous chunks across `MFISH_FEAT_WORKERS` (default cpu−2), exact.
> Run axis: **serialize mice, parallelize within** —
> `MFISH_FEAT_WORKERS=14 build-features <sid>`. Modest speedup only
> (~16% vs the old separate pipeline) — extraction is memory-bandwidth-bound.
> `MFISH_STRIP_Z` is overridable but leave at 128 (lower skips boundary cells).
>
> ⚠️ **Do NOT reuse `dev_code/cached_hcr_cell_tight_bbox/`** with the current
> extractors: those parquets are in the **level-0** XY frame (4× larger Y/X, 16×
> volume), whereas the current `feat_*` extractors index the **level-2** `orig_res`
> seg directly (no /4). Build fresh with `build-bbox`. (Z matches — not
> downsampled — which is why only X/Y/volume differ.)

---

## How the two clones compare to the earlier `temp_repos` snapshots

Checked 2026-06-17 (the user cloned both fresh into `/`):

- **`2p2fish` (autocoreg) = identical code, cleaned.** Every flat
  `src/autocoreg/*.py` is byte-for-byte identical to
  `/scratch/temp_repos/mfish-autocoreg` (full `diff -rq` + md5). Only diffs:
  `2p2fish` adds `LICENSE`; the snapshot's 7 **empty** scaffolding dirs (figures,
  io, matching, registration, scale, surfaces, warp) were dropped. No functional
  change.
- **`mfish-roi-classifier` = improved (refactored).** Extra commit `merge to
  final version of features`: removed the versioned sprawl (`roi_quality.py`,
  `roi_quality_v2..v5/v5d/v7.py`, `roi_v6_voxel.py`, `roi_v7_features.py`) in
  favour of a clean API — `features.py` (`extract_features`) + `model.py`
  (`predict`/`predict_subject`/`train`) + per-group `feat_shape`/`feat_axis`/
  `feat_surface`/`feat_protrusion`; and it now ships trained `models/` + `labels/`.

---

## `code/dev_code` legacy + standalone fix (2026-06-17)

`dev_code` is frozen development scratch. It had a real import bug:
`surfaces_iter08` / `surface_registration_v2` pulled 6 helper modules via a
**stale path** (`data/claude_data/...`, which does not exist) — so a clean
checkout could not import them. Fixed by vendoring the 6 + 1 transitive modules
(`iter07_compute`, `iter08_cz_prior`, `iter08_hcr_bottom`, `compare_binarization`,
`register_binary`, `register_nonrigid_variants`, `compute_projections`) from
`2p2fish` with flat imports and removing the dead `sys.path` hacks. All promoted
modules (`surfaces_iter08`, `roi_area_sxy`, `surface_registration_v2`,
`locked_prior_warm`, `sz_estimator`, `overlap_crop`) now import cleanly.

Apply script (idempotent): `/scratch/dev_code_standalone_fix/apply.py`.
For new work, edit the repos — not `dev_code`.

---

## 3-capsule HCR ROI-quality pipeline (2026-06-20)

### Capsule 1 — `capsule-3D-HCR-ROI-classifier` (inference) [`ce67ff73-8963-4eed-ade8-4d3d5248a3f5`]
- **No coreg dir.** Only the subject's `HCR_{sid}_*_processed_*` asset (405-only, HCR-only;
  `benchmark_data_loader.load_subject` is coreg-optional).
- **Model = the attached CodeOcean model data asset** under `/data`, auto-detected by FORMAT at
  ANY depth (rglob to the dir holding `roi_quality_4class.txt` + `roi_quality_meta.json`), so the
  MLflow `{asset}/{name}/model/artifacts/` layout works; no vendored model at run time. The huge
  `HCR_*` asset is skipped during the scan. `--models_dir` overrides.
- Runs **in-process** (no CLI subprocess). Contract `{sid}_roi_quality_proba.parquet` =
  `hcr_id, p_bad, p_bad_ok, p_good, p_merged` (no `human_label`; no-voxel ROIs dropped).
- `--max_cells N` = quick smoke (N lowest-z ROIs, isolated temp cache; production caches untouched).

### Capsule 2 — `capsule-3D-HCR-ROI-labeling` (interactive labeling, cloud workstation)
Human-labels a smart subset of ROIs to grow the training set. **Not a Reproducible Run** — three
notebooks driven on the workstation desktop (the `roi-classifier-label` GUI needs a display):
1. **`code/attach_data.ipynb`** — per subject, finds + attaches Capsule 1's classifier output
   (`3D-HCR-ROI-4-class-label`: `{sid}_features_all.parquet` + `{sid}_roi_quality_proba.parquet`),
   the R1 HCR data it was derived from (via `provenance`), and **prior `HCR-ROI-human-labeling`
   assets** (tag search). Then builds the candidate list (`make_candidates.py` →
   `/scratch/label_candidates.csv`): mostly keep/reject-**borderline** ROIs (smallest
   `|P(good)+P(bad_ok) − 0.5|`) + a few **confident** per class, **excluding already-labeled** ROIs.
2. **`code/run_labeling.ipynb`** (= `bash code/label.sh "<sids>"`) — rebuilds the tight-bbox cache
   and opens the GUI over the candidates. Labels saved to **`/scratch/labels`** (persistent on a
   workstation) — NOT `/results` (ephemeral there). The GUI **reads** Capsule 1's proba contract
   (`score = P(good)+P(bad_ok)`); no model needed at label time.
3. **+100 more:** re-run `attach_data.ipynb` cell 3 (excludes what was just labeled → next batch)
   → re-run `run_labeling.ipynb`. Prior+current labels are read & merged newest-wins from
   `/scratch/all_labels`.
4. **`code/create_label_asset.ipynb`** — publishes `/scratch/labels` as a new
   `HCR-ROI-human-labeling` asset (`CloudWorkstationSource`), which Capsule 3 then consumes.

### Capsule 3 — `capsule-3D-HCR-ROI-classifier-training` (training, native MLflow)
- Enable **Capsule Settings → MLflow tab → "Track this Capsule"**; `MLFLOW_TRACKING_URI` is
  auto-injected (never set it; no `set_experiment`). See `code/temp/MLflow_in_CodeOcean.md`.
- Input = label data assets only (auto-discovered under `/data`, merged newest-wins); **subjects
  derived from the labels** (no subject arg; trains all labeled subjects).
- No base model: feature schema is derived from the labels' embedded features (strict same-set).
- Logs a registrable **pyfunc** model (`mlflow.pyfunc.log_model`, bundles the 4-class booster +
  meta) → register from the MLflow UI → CodeOcean Models dashboard → attach to Capsule 1 as `/data` model.

### Monitor — `Jinho_pipeline-monitor_HCR-ROI-classifier`  (origin AllenNeuralDynamics)
- Input `subject_id` only (comma-sep for many). Mirrors `Jinho_pipeline-monitor_ROICaT`.
- Finds each subject's **Round-1 HCR processed** asset (tag `HCR` + name `HCR_{sid}`, `_processed_`,
  earliest date, tie→latest-created; see `step_1_process_files.ipynb`) and triggers Capsule 1
  (`HCR_CLASSIFIER_CAPSULE_ID = ce67ff73-…`) via the all-users monitor
  (`PIPELINE_MONITOR_CAPSULE_ID = 567b5b98-8d41-413b-9375-9ca610ca2fd3`), capturing the result
  (process suffix `3D-HCR-ROI-4-class-label`). Mounts the HCR asset under its own name so Capsule 1's
  `HCR_{sid}_*_processed_*` glob resolves it.
- A reproducible run can't attach a data asset to itself — that's why the HCR asset is attached from
  the monitor's `RunParams`. (Assumes Capsule 1's pre-attached model persists on an API-triggered run.)

## autocoreg package refactor (2026-06-29, branch `refactor`)

The flat `autocoreg` package was reorganised into a modular two-stage protocol; the
matcher gate/stage names were made descriptive and the output variant renamed.
Behaviour is unchanged (790322 matcher output byte-identical: 737 pairs, soma |Δ|<1e-14).

Layout:
- `initial_registration/` — rough/warm reg: `surfaces`, `lateral_scale` (sxy),
  `surface_registration`, `axial_scale` (sz), `locked_prior`, `overlap_crop`,
  `coarse_align`, + surface/2-D-registration helpers.
- `finetune_soma_print/` — soma-print fine matching: `descriptor`, `pool_prep`, `tps`,
  `scoring`, `matcher`, `local_ncc`, `loo_image_ncc`.
- `io/` — shared loaders: `subjects`, `inputs`, `centroids`, `hcr_image`, `cz_volume`,
  `gfp_threshold`.
- `qc/` — PyQt5 viewer + artifact builder + launcher.
- `archive/` — superseded protocols kept for comparison (never imported by production):
  `shape_context`, `oracle_benchmark`, `locked_benchmark`, `refined_benchmark`,
  `iterative_matcher`.

Renames:
- module `run_step3_v3` → `finetune_soma_print.matcher`; `soma_print` →
  `finetune_soma_print.descriptor`; `surfaces_iter08` → `initial_registration.surfaces`;
  `roi_area_sxy` → `…lateral_scale`; `sz_estimator` → `…axial_scale`;
  `surface_registration_v2` → `…surface_registration`; `locked_prior_warm` → `…locked_prior`;
  `benchmark_analysis` → `io.hcr_image`; `benchmark_data_loader` → `io.subjects`;
  `data` → `io.inputs`.
- matcher gate `lr` → `likelihood_ratio`; Stage-2 `wang` / `--wang_addendum` →
  `anchor_restricted` / `--no-anchor-restricted`.
- variant `step3_v3_anchor_vote_wang_end` → **`anchor_vote_anchor_restricted`**; output file
  `matches_wang_round*.csv` → `matches_anchor_restricted_round*.csv` (QC code still reads the
  legacy names). Existing `/scratch` data + variant-keyed QC labels/manual-matches were migrated.
