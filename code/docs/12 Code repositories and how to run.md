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

## CZ↔HCR coregistration capsules (2026-06-30)

Two CodeOcean capsules that wrap the `autocoreg` package end-to-end, modeled on the
3-capsule HCR-ROI pipeline above. Staged + git-committed in `/scratch/sessions/21_coreg_capsules/`
and placed at `/`; remotes set, **not pushed yet** (`.git` root-owned → commit/push via the root
script `/scratch/sessions/21_coreg_capsules/commit_qc_changes.sh`, which commits THREE repos:
`/2p2fish` + both capsules — then push + rebuild capsule envs, postInstall cache-bust `260629-01`).

### Capsule A — `capsule-2p-3DmFISH-autocoreg` (reproducible run)  [github AllenNeuralDynamics]
- From the `capsule-3D-HCR-ROI-classifier` template. `code/run_capsule.py` runs the full GT-free
  pipeline in-process (surfaces → sxy → surface-registration → sz → matcher `anchor_vote` +
  `anchor_restricted` → `build_qc_artifacts` → `score_final_pairs`), all via the installed
  `autocoreg` package.
- App params: `subject_id` (req), `gate` (def `anchor_vote`), `build_qc` (def 1), `anchor_restricted` (def 1).
- **Requires** the Capsule-1 `HCR_{sid}_*_HCR-ROI-label_*` asset attached (matcher gates the HCR
  pool on `{sid}_roi_quality_proba.parquet`; auto-resolved, fails if absent). Also attach the
  subject's CZ/HCR assets (coreg dir, HCR processed, czstack registration + segmentation).
- **Output asset is per-subject + FLAT** (no `<variant>/<sid>` nesting — name + data_description.json
  carry sid; processing.json carries gate/variant). Layout under `/results`:
  `matches/` (matcher CSVs), `for_qc_app/` (GUI cell-matching QC inputs: warped-CZ + seg volumes +
  `final_pairs.csv` + `positions.csv`), `qc/` (per-step QC of the pipeline itself), `registration/`
  (the pose JSONs: surfaces + sz + surface_registration — persisted, reused downstream),
  `coreg_manifest.json`. Internal `<variant>/<sid>` nesting + bulky per-cell caches stay transient
  in `/root/capsule/scratch/autocoreg_work`.
- **`qc/` per-step images** (`code/pipeline_qc.py::build_pipeline_qc`, runs even at build_qc=0 for
  the non-GIF figs): 01 surface fit per modality (HCR over the COMBINED 405+488+561+594 volume it
  was fit to, not 488), 02 ROUGH locked-prior alignment xz/yz cross-sections (pre-matcher), 03 sxy
  area-hist + matched-depth bands, 04 surface-reg 4-method NCC + mode-selection, 05 sz sweep
  (sz_lp depth-ratio prior vs sz_best NCC-peak), 06 matched/unmatched centroid xy/xz/yz (z-flipped,
  green-dot/magenta-×), and **FINAL-registration sweep GIFs** `07_sweep_image_{topdown,sideways}`
  (warped CZ 488 from `cz_warped_in_hcr_um.tif` = matcher TPS, over HCR 488) + `08_sweep_rois_*`
  (slab-MIP CZ vs HCR ROIs, overlap=white) — image+ROI on the same `bbox_cz_warped` grid so they
  line up and match the QC app. Each GIF has a `<name>_frames/` PNG-pages folder.

### Capsule B — `interactive-capsule-2p-3DFISH-autocoreg-QC` (interactive QC, cloud workstation)
- From the `capsule-3D-HCR-ROI-labeling` template. **Not a Reproducible Run.** `code/`:
  1. `attach_data.ipynb` — find+attach Capsule-A result + its provenance inputs (raw CZ/HCR +
     ROI-quality); the asset is flat, so it reconstructs the `<variant>/<sid>` layout the GUI
     expects under `/scratch/autocoreg_outputs/{qc,matches}` (reading variant+sid from the
     manifest) and **seeds the package cache from `registration/`** so the GUI reuses the exact pose.
  2. `run_qc.ipynb` / `qc.sh` — the `autocoreg_qc` PyQt5 viewer (review + manual add/fix; least-
     confident first).
  3. `make_coreg_table.py` + notebook — builds `{sid}_{czstack-acq-datetime}_coreg-table.csv`.
     Positions precedence: GUI "Export all positions CSV" → Capsule-A-shipped `positions.csv`
     (+layer manual matches from the qt-labels' hcr_id via centroids_um) → headless offscreen
     export → ids+soma. **Qt-free / GUI-free** when the shipped positions exist (no 3 GB recompute).
     QC semantics: keep good/unreviewed/manual, drop bad/unsure + unmatched.
  4. `create_coreg_asset.ipynb` — publish `/scratch/coreg_tables` as a `czstack-hcr-coreg-table`
     data asset (CloudWorkstationSource; session-attached assets = provenance).
- README has a real `autocoreg_qc` screenshot (`docs/qc_app_overview.png`, captured by
  `docs/grab_qc_app_screenshot.py`).

### Shared change in `2p2fish` — `autocoreg/qc/positions.py` (single source of truth)
New Qt-free module (`POS_COLS, position_row, cz_world_from_seg, compute_pair_positions,
load_derived_from_artifacts, positions_from_artifacts, write_positions_csv`). `qc/app.py` was
refactored to route its "Export all positions CSV" + `_positions_for` + the `cz_world` cold path
through it, so the GUI export and Capsule-A's `positions.csv` are identical (790322: headless ==
GUI baseline to float32). Caveat: GUI `cz_in_hcr` is TPS-warped, the shipped baseline is the
`cz_world` seg-centroid — same correspondence, ~tiny coord diff.

### Monitor — `Jinho_pipeline-monitor_2p-3DFISH-autocoreg`  (github AllenNeuralDynamics)
Adapted from `Jinho_pipeline-monitor_HCR-ROI-classifier`. Per subject it finds + attaches the
autocoreg inputs (mirroring `code/manual workflow/step_1_process_files.ipynb`) and triggers
Capsule A via the all-users monitor (`567b5b98-…`), capturing the result with suffix
**`2p-3DmFISH-autocoreg`** (matched by Capsule B's `attach_data`):
- **R1 HCR** processed (earliest `_processed_`; tie→latest-created) — same as the HCR monitor;
- the **LATEST cortical z-stack registration** at `czstack_xy_size` µm (param, default 400, 700
  for some sessions) via `zstack_utils.get_cortical_zstack_reg_df` (filter `xy_size_um`, sort,
  iloc[-1]) + the **segmentation** derived from it (`get_derived_assets('cortical-zstack-
  segmentation')` whose `provenance.data_assets` includes the chosen reg id);
- the **`HCR-ROI-label`** classifier output (tag search, latest).
If the latest cortical z-stack at that xy_size is **not processed** (no registration +
segmentation), it stops that subject and prints that the z-stack must be processed first.
Env adds **comb + lamf-analysis** (for `zstack_utils`/`code_ocean_utils`). Params: `subject_id`
(comma-sep), `czstack_xy_size`. **Set `AUTOCOREG_CAPSULE_ID`** (constant / env var) to Capsule A's
CO id before a real run (`test=1` = search-only dry run).

> **Loader (RESOLVED 2026-06-30):** Capsule A's `autocoreg.io.subjects.load_subject` now
> **synthesizes a coreg dir** from the attached cortical-zstack registration + segmentation when
> no manual `*ctl-czstack-hcr-coreg_*` dir is present (`_synthesize_coreg_dir`, mirroring `step_1`):
> symlinks the `*_2xREG.tif` as the CZ z-stack, writes the seg binary as the `*seg-mask-outline.tif`
> (`load_cz_binary_volume` fill-holes → same solid cells), and writes the center-of-mass
> `*czstack_cell_centroids.csv`. `_find_coreg_dir` returns the manual dir if present, else the
> synthesized one (`MFISH_COREG_DIR`-overridable scratch). Verified on 790322: synthesized CZ
> centroids == the manual ones to 0.0. The GT (`coreg_table` / qced landmarks) is **matcher-optional**
> and degrades to empty — all its downstream uses (`inputs.scoring_gt`, `gfp_threshold` coverage with
> `max(.,1)` guard, `hcr_image` stats) are empty-safe, so nothing had to be removed.
>
> **GFP/sxy generalized to new subjects (RESOLVED 2026-06-30):** the GFP/sxy paths no longer
> depend on hardcoded subject lists or a pre-cached threshold.
> `gfp_threshold.detect_gfp_class(sid)` classifies **spot-vs-intensity from the attached HCR data**
> (a `*spot_488_counts.csv` in the coreg dir or `image_spot_detection/channel_488_spots/spots.csv`
> in the HCR asset → `spot`; a `cell_data_mean_{sid}_R1.csv` → `intensity`; else default `spot`),
> with the 6 hardcoded benchmark subjects still taking precedence (behaviour unchanged).
> `analyze_subject` / `strict_gfp_df` use it; `inputs.strict_gfp_ids` falls back to a **live seeded
> GMM cutoff** (`gfp_threshold.analyze_subject`) when the subject is absent from `GMM_THRESH_JSON`;
> `lateral_scale` drops its `unknown subject` gates and **guards the GT-only `sxy_gt`/`sz_gt`
> diagnostics** (→ NaN when no qced landmarks). **Verified:** disabling the hardcoded sets (i.e.
> treating 790322 as brand-new) reproduces its cutoff (`0.00169317`), strict-GFP+ ids (`9675`),
> and `sxy_median` (`1.7678`) **bit-for-bit**; the live `strict_gfp_ids` fallback yields the
> identical id set; and new subjects `800792`/`800995`/`804363` detect as `spot`. So the automated
> path now runs end-to-end for both the benchmark subjects and genuinely-new ones.
