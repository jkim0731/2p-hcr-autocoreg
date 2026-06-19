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

Pipeline stages (all GT-free): 1) surface fitting (`surfaces_iter08`) →
2) sxy (`roi_area_sxy.estimate_sxy_min_rule`) → 3) surface registration
(`surface_registration_v2`) → 4) sz (`sz_estimator.get_sz`) → 5) matcher
(`run_step3_v3.run_subject`).

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
