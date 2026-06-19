# ROI classifier: µm-vs-vox feature decision + extractor reconciliation (2026-06-17)

Records (a) the historical µm-vs-vox feature decision, recovered from the session
transcripts; (b) the extractor↔model mismatch that surfaced when the refactored
`mfish-roi-classifier` ran `predict` end-to-end; (c) the reconciliation that fixed
it; (d) the within-mouse z-strip parallelization of the feature extractors;
(e) the stage-1/stage-2 architecture decision and the **self-contained C2 promotion**
(§7, 2026-06-18) — the current production model.

---

## 1. The decision: **µm features, no rank, no v6_vox** (late May 2026)

The v5d ROI-quality classifier had two candidate feature families:
- **µm features** — physical-unit shape/intensity (`volume_um3`, `surface_area_um2`,
  `bbox_*_extent_um`, `equivalent_diameter_um`, `core4um`, …), produced for free by
  the v4/v5 extractors (regionprops × the per-subject voxel pitch).
- **vox features** — calibration-free voxel-unit recomputations of the same
  measurements, produced by a *separate* extractor pass (`v6_vox`).

Decision sequence (transcript `…/0eccb3d7…jsonl`):
1. **Rank features dropped** — user: *"Do not use rank because it is affected by
   sample prep, such as thickness."* (`pct_rank_columns = []`).
2. **A/B test µm vs vox** (both rank-disabled), LOSO over the 6 benchmark subjects:

   | | µm (91 feat) | vox (88 feat) |
   |---|---:|---:|
   | binary AUC | **0.9206** | 0.9205 |
   | 4-class acc | 0.7033 | 0.7054 |
   | 4-class f1-macro | 0.6981 | 0.7003 |

   **Essentially tied** (ΔAUC −0.0001) — this cohort's voxel pitch is uniform to
   ~±1–2 %, so µm and vox carry the same information.
3. **Tiebreaker → µm.** µm features fall out of the v4/v5 extractors at **zero**
   extra cost, whereas vox needs the `v6_vox` extractor (~58 % of feature-extraction
   CPU). *Net win on extraction time at no measurable accuracy cost.*
4. **Expansion-rate note:** *neither* µm nor vox normalises per-sample expansion
   (µm = biological µm × expansion factor; the vox grid also scales with the gel).
   Only **dimensionless** features (ratios, sphericity, occupancy, aspect, …) do —
   so expansion was not a reason to prefer vox.
5. **Promotion:** µm became canonical `v5d_um` (`uses_v6_vox=False`,
   `pct_rank_columns=[]`, `n_features=91`, AUC 0.9206). Recorded in memory
   `project_S11_v5d_promoted.md`; A/B summary at
   `sessions/13_pairwise_unmix_gfp/outputs/abtest_um_vs_vox/`.

**Consistency with the auto-coreg use case:** the matcher reads
`{sid}_stage2_4class_proba_v5d_um.parquet` (`sessions/15_geom_features/_data.py`),
and session-16 reported the µm AUC (0.921). So the decision (µm) and what
auto-coreg actually used (µm) agree. The shipped `mfish-roi-classifier/models/*`
are md5-identical to the dev_code `_v5d_um` production model.

---

## 2. The mismatch (surfaced by running `predict` end-to-end)

The refactored repo's extractors had drifted off the µm contract:
- `feat_surface` (v4) computed **only sphericity** — the µm `surface_area_um2`,
  `volume_um3`, `sa_to_vol_um_inv`, and 4-µm core/shell features were suppressed
  (leftover `DROP_UM_FEATURES`).
- `feat_shape` (v2) emitted vox-named shape (`volume_vox_raw`) and dropped
  `volume_um3_raw`, `bbox_*_extent_um`, `equivalent_diameter_um_*`, `n_neighbors_30um`.

Net: the extractors produced 83 columns matching **neither** the `_um` model (91)
nor the `vox` model (92). So `predict` failed at the feature merge (19 missing).

---

## 3. The reconciliation (to the µm set)

- **`roi_v4_features.py`** — un-dropped the µm outputs: `shape_features` now returns
  `surface_area_um2`, `volume_um3`, `sa_to_vol_um_inv`, `sphericity` (raw+opened);
  `all_v4_features` adds `core_shell_intensity_features` (the 4-µm core/shell 405
  features); `feature_columns()` lists all 14. (Original functions, just re-wired.)
- **`feat_shape.py`** — added `volume_um3_raw`, `bbox_{z,y,x}_extent_um`,
  `equivalent_diameter_um_{raw,opened}` to `_shape_stats`, and `n_neighbors_30um`
  (count within 30 µm, exclude self) to the knn step.
- v3/v5 unchanged (already on the µm contract).

**Validation against the original features (coldcache, 782149 = ground truth the
`_v5d_um` model was built on):** v4 **exact** (0 error), v3 **exact** (5.7e-15),
v2 µm/neighbor columns **exact** (only a pre-existing ~3.9e-4 on one 405 percentile,
`c405_opened_p90` — not a reconciled column). All formulas were reverse-validated
against coldcache before coding (`volume_um3`, `equivalent_diameter`, `bbox_extent`,
`n_neighbors_30um` matched 0/39,291 cells).

**End-to-end (`predict` vs the `_v5d_um` OOF, the auto-coreg use case):**

| subject | argmax-agree | keep-set agree | p_good corr | keep_new / keep_oof |
|---|---|---|---|---|
| 782149 | 0.871 | 0.982 | 0.950 | 0.734 / 0.729 |
| 790322 | 0.958 | 0.974 | 0.984 | 0.855 / 0.844 |
| 767022 | 0.909 | 0.982 | 0.984 | 0.817 / 0.823 |

The **keep-set** (argmax ∈ {good, bad_ok} — exactly what the matcher consumes for
its GFP+∩ok pool) agrees **97–98 %**, keep-fractions within ~1 %. The argmax
differences (87–96 %) are the **production-vs-OOF model** difference (`predict`
uses the all-6 production model; the OOF used a leave-that-subject-out model), NOT a
feature error — Part A proves the features are reproduced exactly.

Cold-start now works: `roi-classifier build-features <sid>` (bbox → crops →
v2/v3/v4/v5) → `roi-classifier predict <sid>` produces the contract parquet.

---

## 4. Within-mouse z-strip parallelization (max cores)

The feature extractors scan the volume in z-strips. v2 (`feat_shape`) and v3
(`feat_axis`) were the bottlenecks (~11 / ~30 min serial). The z-strip loop is
embarrassingly parallel (each cell is owned by exactly one strip; strips load
disjoint sub-blocks), so it was refactored to a `ProcessPoolExecutor` over strips
(`MFISH_FEAT_WORKERS`, default cpu−2), with the nan-fill + knn kept serial in the
caller. The same worker function serves the serial and parallel paths, so values
are unchanged — verified: parallel v2 on 782149 = identical to serial = coldcache
(worst err 3.9e-4), ~3.3× faster (197 s vs ~11 min) with 7 workers.

**Run axis (per the parallelism guidance):** with few mice, **serialize mice and
parallelize within** (the z-strips use all cores) rather than running mice
concurrently (which leaves most cores idle). Use `build-features -j 1`
(serial feature groups) together with `MFISH_FEAT_WORKERS` (parallel strips) — do
NOT combine a high `-j` with a high `MFISH_FEAT_WORKERS` (nested pools
over-subscribe). Production single-mouse extraction is the main beneficiary.

---

## 5. Unified single-pass extractor + cell-chunking (2026-06-17)

The 4 extractors (`feat_shape`/`feat_axis`/`feat_surface`/`feat_protrusion`) each
re-read the volume and recomputed the per-cell opening. They were **merged into
one pass** in `feat_shape`: per cell the raw mask + opening + 405 crop are computed
once and fed to all family-math functions (`all_v3_axis_features`,
`all_v4_features`, `protrusion_features`), merged into one row (with the
`volume_um3_raw → volume_um3_raw_v4` rename that matches the old 4-parquet merge).
Output is one `{sid}_features_all.parquet`; the per-cell-crops cache is no longer
needed; `build-features` = `bbox → unified pass`. Verified **exact** vs the
coldcache originals on 782149 (worst err 3.9e-4 on the one pre-existing intensity
percentile, 0 NaN mismatches), and `predict` reproduces the auto-coreg keep-set.

**Parallelism — balanced cell-chunking.** All four extractors used identical
opening (`_CROSS_3D`, `OPENING_RADIUS=3`, pad 4), so the single shared computation
is safe. The pass parallelizes by splitting each z-strip's cells into
spatially-contiguous chunks (sorted by ymin → compact block per chunk) across
`MFISH_FEAT_WORKERS` (default cpu−2); `STRIP_Z` stays 128 so the z-context — and
thus which boundary cells are skipped — is unchanged → exact.

**Timing reality (782149).** Unified+chunked ≈ 833 s vs the old separate pipeline
≈ 987 s (~16% faster) and simpler (one parquet, no crops cache). It is **not** the
~3× one might expect: the I/O redundancy was not the bottleneck (v4/v5 already read
cheap precomputed crops), and the workload is **memory-bandwidth-bound** — 14
workers streaming GB-scale seg+405 blocks saturate RAM bandwidth, so per-cell
parallelism is sub-linear. A larger win would require optimizing the per-cell
compute itself (dominated by the v3 axis-profile math) — a separate effort.

> ⚠️ `MFISH_STRIP_Z` is overridable but **not** freely tunable: lowering it puts
> more cells at strip boundaries where large-z-extent cells overflow the smaller
> loaded block and get skipped (tested STRIP_Z=32 → 47k value mismatches). Leave
> it at 128 unless `Z_PAD` is raised accordingly.

---

## 6. Files changed (mfish-roi-classifier, uncommitted)
- `roi_v4_features.py` — restore µm v4 features (un-drop `DROP_UM_FEATURES`).
- `feat_shape.py` — v2 µm/neighbor features; **unified single-pass extractor** with
  z-strip cell-chunking; env-configurable `MFISH_STRIP_Z` (default 128).
- `features.py` — read/produce the single `{sid}_features_all.parquet`.
- `cli.py` — `build-features` = bbox → unified pass (no crops, no 4-group DAG).
- `feat_axis.py` — z-strip pool (now also superseded by the unified pass for the
  default flow; family math reused from `roi_v3_axis_features`).
- (earlier) `feat_tight_bbox.py`, `feat_per_cell_crops.py` (`build-bbox` /
  `build-crops`; crops now optional, used only by the legacy per-group extractors).

---

## 7. Stage-1 / Stage-2 architecture + self-contained C2 promotion (2026-06-18)

The classifier was a **two-model** pipeline; this session retired stage-1 and promoted a
self-contained model. Full record: `/scratch/sessions/17_interim_summary/stage1_stage2_architecture.md`.

**The two models (corrected understanding):**
- **Stage-1** — *not in this repo* (upstream/legacy). A per-cell *"is this a real, cleanly
  segmented cell?"* detector: ~48 features incl. **multi-channel** intensity (405, 488,
  probe=561 or 514 per subject, 594) + **GFP** + segmentation-provenance flags; trained on
  **pseudo-labels** (heuristics: low_solidity, nuclear_inversion, not_found_in_seg,
  volume_high/low, boundary_touching), ~291 k cells, LOSO AUC 0.99998. Emits a per-cell score
  for *all* cells → `{sid}_stage1_score.parquet`.
- **Stage-2** — *this repo*. Per-cell **human-labeled quality** classifier (4-class
  good/bad_ok/bad/merged; binary keep = good∪bad_ok). 405-only. The v5d_um set had 91 features,
  **2 of them stage-1-derived**: `mean/min_touching_score_stage1` = mean/min stage-1 score of a
  cell's *touching neighbours* — the only cross-stage coupling.

**Decision — retire stage-1, ship the self-contained "C2" model.** Drop the 2 stage-1 features
(and the whole stage-1 dependency: no second model, no extra channels) and replace them with
**12 self-contained neighbour-quality features** — mean+min over each cell's **6 nearest
neighbours** (centroid, µm) of `solidity_opened, sphericity_opened, bbox_occupancy_raw,
frac_kept_opening, volume_um3_opened, c405_shell_minus_core_p50`. Production set is now
**101 features, 405-only, no upstream dependency**; cold-start collapses to
`build-bbox → build-features → predict`.

**Evidence (LOSO, 6 subjects):**
| model | feats | binary AUC | 4-class F1m |
|---|---:|---:|---:|
| shipped v5d_um (stage-1) | 91 | 0.9206 | 0.6981 |
| 91-feat retrain + recovered cells | 91 | 0.9254 | 0.7069 |
| naïve drop (no neighbour quality) | 89 | 0.9160 | 0.6994 |
| **C2 self-contained (promoted)** | **101** | **0.9220** | 0.7021 |

- Naïve stage-1 drop costs **−0.009** binary AUC (below baseline, worse in 6/6 folds) → the
  stage-1 signal is real and *not* captured by plain adjacency features.
- The neighbour-quality proxy recovers most of it → **0.9220 > shipped 0.9206**, self-contained.
- **Other channels were tested and rejected:** adding 488/594/GFP to C2 leaves binary flat-to-noise
  and gives only a ~noise 4-class bump; 594 (SYTO59, nuclear) and 488 (GFP) did **not** improve the
  `merged` class via simple intensity summaries (the merge-detection mechanism would need a
  purpose-built intra-ROI bimodality / nucleus-count feature, not channel means). 488 also risks
  **GFP/matching-signal circularity** (it is the coreg matching channel). → **405-only retained.**

**Also this session:**
- **Adaptive per-chunk bbox** (`feat_shape`) eliminated boundary-overflow skips, recovering
  **24 previously-skipped labeled cells** (782149 ×23, 788406 ×1) into training; bit-exact on
  fitting cells.
- **`PCT_RANK_COLS` bug fix** — `model.train()` referenced an undefined name → `NameError` at
  meta-write (train was broken on a clean checkout); fixed to `PCT_RANK_COLS = []`.
- **Contract rename (this repo only; 2p2fish to match):** dropped `_v5d_um` from filenames —
  model `roi_quality_stage2_{binary,4class}.txt` + `…_meta.json`; contract output
  **`{sid}_stage2_4class_proba.parquet`**. Old stage-1 model archived at
  `models/_archive_v5d_stage1/`. Meta `version = v6_selfcontained_c2` + provenance (commit,
  n_labels, feature lineage). All changes uncommitted (working tree).
