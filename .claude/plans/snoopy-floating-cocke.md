# Plan: carve the autocoreg pipeline + ROI classifier into two standalone repos

## Context
All of this work currently lives inside the `/root/capsule` research monorepo, split across
`code/dev_code/` and `code/sessions/15_geom_features/`, and it reaches into a **read-only archive**
(`data/claude-data_ophys-mfish-autocoreg_260503/…`) by hardcoded `sys.path` for load-bearing code
(surface-fitting `iter07/iter08`, the session-08 `register_*` registration core, `centroid_helpers`).
We want two **deployable** repos:
- **Repo A — `mfish-autocoreg`**: the coregistration pipeline + the human-validation QC app (`qc_qt_app`) bundled in (default install includes QC; Qt deps kept as an optional `[qc]` extra so the headless core stays lean).
- **Repo B — `mfish-roi-classifier`**: the v5d ROI-quality classifier + its labeling GUI (`[label]` extra).

The two repos meet at exactly one **data contract** (no shared runtime code required): repo B writes
`{sid}_stage2_4class_proba_v5d_um.parquet` (cols: `hcr_id, p_bad, p_bad_ok, p_good, p_merged, human_label`);
repo A consumes it via `argmax_ok_ids` (keep where argmax ∈ {p_good, p_bad_ok}). Pipeline order: **B → A**.

The pipeline is already **GT-free / runs on any new subject** (verified): `load_subject` is pure directory-glob,
`estimate_sxy_min_rule` + `sz_estimator.get_sz` + `compute_surface_registration` + the matcher need no GT.
`coreg`/`scoring_gt`/`landmarks`/`sz_pins`/`BENCHMARK_SUBJECTS` are **validation-only** and must be split off.

## Target structure

### Repo A — `mfish-autocoreg`
```
pyproject.toml         # core: numpy scipy pandas scikit-image tifffile zarr lightgbm-free
                       # extras: [qc] = PyQt5 pyqtgraph opencv-python
src/autocoreg/
  config.py            # REPLACES _paths.py + all hardcoded /root/capsule paths:
                       #   data_root, cache_dir, roi_quality_dir  (env vars / config file)
  io/                  # subject_loader.py (← benchmark_data_loader), cz_volume.py, centroid_helpers.py(vendored),
                       #   seg/zarr/metrics access helpers (dedupe roi_area_sxy._load_hcr_metrics)
  surfaces/            # surfaces_iter08.py + VENDORED iter07_compute, iter08_cz_prior, iter08_hcr_bottom
  registration/        # surface_registration_v2.py + VENDORED register_binary, compare_binarization,
                       #   register_nonrigid_variants, compute_projections
  scale/               # roi_area_sxy.py (estimate_sxy_min_rule + SXY_GRID_SEARCH_OFFSETS fallback), sz_estimator.py
  warp/                # locked_prior_warm.py, overlap_crop.py, r1_revised.py
  matching/            # soma_print, shape_context, local_ncc, loo_image_ncc,
                       #   step1_oracle, step2_locked, step2p5_refined, step3_iterative, matcher.py(← run_step3_v3)
  data.py              # ← _data.py: subject_inputs, pools (strict_gfp ∩ argmax_ok reads the parquet contract)
  qc/app.py            # ← qc_qt_app.py  (imports autocoreg.data — single data layer, no drift)  [extra: qc]
  cli.py               # `autocoreg run <subject_dir>` → matches + warp + QC launch
  benchmark/           # VALIDATION-ONLY (core never imports this): scoring_gt, BENCHMARK_SUBJECTS,
                       #   sz_pins loader, landmark/coreg comparison, compare_3paths/scoring scripts
```
Core deps stay headless; `pip install mfish-autocoreg[qc]` adds the GUI on a reviewer workstation.

### Repo B — `mfish-roi-classifier`
```
pyproject.toml         # core: lightgbm scikit-learn pandas numpy zarr scipy
                       # extras: [label] = matplotlib (+ Tk/Qt backend)
src/roi_classifier/
  io/                  # VENDORED copy of subject_loader.py + seg/zarr access (shared foundation; see note)
  features/            # roi_quality.py (base) + v2,v3,v4,v5,v7 + helpers
                       #   (roi_v3_axis_features, roi_v4_features, roi_v5_features, roi_v5_neighbors, roi_v6_voxel, roi_v7_features)
  model.py             # ← roi_quality_v5d.py: predict_subject, train (LOSO)
  label_gui/           # ← 04b_label_gui_app.py + gui.py   [extra: label]
  cli.py               # `roi-classifier predict <sid>` → writes the contract parquet; `train`
labels/roi_qc_actions.jsonl   # training labels (append-only; see schema in code-review notes)
models/                       # trained v5d models
```

### Shared foundation (the one real coupling)
`benchmark_data_loader` (→ `subject_loader.py`: `SubjectData`, `load_subject` dir-glob, `cz_px_to_um`/`hcr_px_to_um`,
seg-zarr/metrics access) is needed by **both** repos. Decision (keeps it to 2 repos): **vendor a copy into each**
`io/` (it's small and stable). If drift becomes painful later, factor a 3rd `mfish-io` package — noted, not now.

## Critical issues to fix during carve-out (the real work)
1. **De-hardcode all paths.** Replace `_paths.py` `sys.path.insert` hacks and every `/root/capsule/...` absolute
   path (cache dirs, archive dirs, `ROI_QUALITY_DIR`, `SZ_TABLE_CSV`, `GMM_THRESH_JSON`) with `autocoreg.config`
   (env: `MFISH_DATA_ROOT`, `MFISH_CACHE_DIR`, `MFISH_ROI_QUALITY_DIR`). This is the bulk of the effort.
2. **Vendor the archive modules** (8 files) and **delete the `data/claude_data` vs `claude-data_…_260503`
   path mismatch** entirely (vendored → in-package imports; the mismatch disappears).
3. **Split core vs validation** in repo A: move `scoring_gt`, `BENCHMARK_SUBJECTS`, `load_sz_pins`/`SZ_TABLE_CSV`,
   `landmark_pairs_um`/`fit_anisotropic_similarity` GT-comparison and `prepare_subject`'s `gt_*` into `benchmark/`.
   `subject_inputs`/`prepare_subject` must run with an empty coreg table and `sz_pins=None` (→ `get_sz` computes).
4. **GFP+ threshold data**: `07b` reads `gmm_threshold_results.json`; for new subjects it recomputes the GMM.
   Bundle the JSON as an optional cache, not a hard dependency.
5. **Caches**: `cached_*` dirs are per-subject regenerable artifacts → gitignore; point at `config.cache_dir`.
   Do NOT commit benchmark subject caches/data into the repo.

## Carve-out steps (ordered)
1. Scaffold both repos (`pyproject.toml`, `src/`, extras, `.gitignore` for `cache/`, `models/`).
2. Copy core `.py` (repo A ~22 files dev_code+session15; repo B ~13 files) into the package layout above.
3. Vendor the 8 archive `.py` into the right subpackages; rewrite their imports to package-relative.
4. Write `config.py`; mechanically replace `_paths`/absolute paths; delete `_paths.py`.
5. Split repo A `benchmark/`; make `subject_inputs`/`prepare_subject` GT-optional.
6. Wire repo B `cli predict` to emit the contract parquet at `config.roi_quality_dir`; repo A reads it there.
7. Smoke-test each repo on one subject (below); then `git init` + initial commit per repo.

## Verification
- **Repo B**: `roi-classifier predict 790322` → produces `790322_stage2_4class_proba_v5d_um.parquet` with the
  exact 6-col schema; spot-check a few `argmax` classes against the current cache.
- **Repo A core (GT-free)**: point `MFISH_ROI_QUALITY_DIR` at B's output; run `autocoreg run` on **790322** and
  **782149** end-to-end with NO coreg/landmarks/sz_pin; confirm it produces matches + a warp and that 782149
  uses min-rule sxy≈1.734 + 80/150 MIP (the recovered pose).
- **Repo A validation (optional)**: in `benchmark/`, reproduce the corrected-GT recall table
  (755252 .94, 767018 .95, 767022 .87, 782149 .92, 788406 .93, 790322 .97) via `scoring_gt` — must match.
- **QC extra**: `pip install -e .[qc]`; launch `qc/app.py` on 790322; confirm it loads matches over the HCR 488.
- Confirm the headless core imports with **no** Qt/GUI deps installed (core-only env).

## Notes / open decisions
- `qc_qt_app` is the canonical QC app; the other three (`qc_pair_app`, `qc_app`, `view_in_napari`) are retired
  (not copied).
- All `roi_quality_v2..v7` feature files are LIVE (v5d merges all their parquets) → keep all; none dropped.
- Benchmark GT fixtures (6 subjects' coreg/landmarks) are validation data — keep out of the core repo; store
  as a separate fixtures bundle or pointer for the `benchmark/` harness.
