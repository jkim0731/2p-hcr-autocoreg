# Cellpose-SAM 4-Combo Segmentation Comparison

## Context

We want to know whether (a) finetuning Cellpose-SAM on our HCR ground-truth labels and (b) feeding it both 405 and 488 channels improves HCR ROI segmentation quality versus the current released-cpsam-on-405 baseline. Quality is judged inside the coregistration overlap region of each autocoreg subject using the existing v3_S11 GBT-based ROI-quality classifier (`roi_quality_v5d.py`).

Decisions confirmed by user:
- **Segmenter**: Cellpose-SAM (`cpsam`) throughout. 4 cells = {released cpsam, cpsam finetuned on GT} × {405 only, 405+488}.
- **Train / test cohorts are disjoint**: finetune on GT cohort {583560, 608368, 629294}; evaluate on autocoreg cohort {782149, 788406, 790322, 755252, 767022, 663991} inside `get_overlap_crop(s)`.
- **Compute**: pause execution; user provisions GPU (~40 GPU-hr A100 budget) before any run.

## Hard constraints discovered in exploration

- GT raw tiffs are **single-channel** (`uint16`, axes `QYX`, 60×150×150). The 488 channel is unavailable on GT blocks — finetuning is necessarily 405-only. The 488 condition only differs at inference on the autocoreg volumes.
- Only **2 fully-labeled 3D blocks** (`583560_R1_block_3/merged_gt.tif`, 192 instances; `608368_R1_block_3/merged_gt.tif`, 105 instances). Other annotations are partial (2 seg.tiffs, 1 geojson) — defer to optional weak supervision.
- Cellpose 4.1.1 already installed; no CUDA in current capsule.
- GBT v5d classifier was trained on production-cellpose outputs → applying it to alternate segmentations is out-of-distribution. Reported P(good)/P(merged) are a *relative* comparison signal, not a calibrated quality measure.
- `cached_per_cell_crops/` does not exist in the repo; rebuild required per (subject, setting).

## Implementation steps

### Stage 0 — Compute provisioning (user)
- Move capsule to GPU instance (A100 40 GB or L4). Confirm `torch.cuda.is_available()` and `cellpose` import succeed before stage 1.

### Stage 1 — Finetune Cellpose-SAM on GT (3 GPU-hr)
- New file: `code/sessions/v3_S14_seg_compare/finetune_cpsam.py`.
- Data: stack `merged_gt.tif` + matching `raw_data/*.tiff` for `583560_R1_block_3` and `608368_R1_block_3`. Slice into 2D z-frames (≈240 frames total). Normalize per-tile to cellpose conventions.
- Split: LOSO across the 2 subjects. Two folds → two checkpoints, plus a final "train on both" checkpoint used for autocoreg inference.
- Run cellpose's 2D finetune loop starting from `cpsam` weights; `channels=[1,0]` (single channel = 405). Standard cellpose augments + light random affine; 200 epochs, batch 8, lr 1e-4.
- Artefacts: `code/sessions/v3_S14_seg_compare/models/cpsam_ft_405_{loso_583560,loso_608368,full}.pt` + train log JSON.
- Verification: report 2D Jaccard / detection F1 of each LOSO fold on the held-out subject's slices. If F1 < pretrained-cpsam baseline on the same held-out slices, flag and stop.

### Stage 2 — Inference on autocoreg overlap (40 GPU-hr)
- New file: `code/sessions/v3_S14_seg_compare/run_inference.py`.
- For each `(sid, setting)` in 6 subjects × 4 settings = 24 runs:
  1. `s = SubjectData(sid)`; `crop = get_overlap_crop(s, margin_frac=0.10)` (use `dev_code/overlap_crop.py`).
  2. Convert level-2 bbox to level-0 voxel slice (×4 in Y,X; same in Z).
  3. Load 405 (and 488 if 2-channel setting) via `benchmark_analysis.load_hcr_volume(s, channel, level=0)` and slice.
  4. `CellposeModel(pretrained_model=...).eval(img, do_3D=True, anisotropy=hcr_z_um/hcr_xy_um, channels=...)`. For 405-only: `channels=[1,0]`. For 405+488: `channels=[1,2]` with `img` stacked `(C,Z,Y,X)`.
  5. Write predicted instance labels to `outputs/seg_compare/{sid}/{setting}/segmentation_mask_orig_res.zarr` at the **level-2 contract** (mode-pool 4× in YX, keep Z), restricted to the overlap bbox. This is what the v5d feature extractor reads.

### Stage 3 — v5d feature extraction + GBT inference (2 CPU-hr, parallel)
- New file: `code/sessions/v3_S14_seg_compare/score_v5d.py`.
- Per `(sid, setting)`:
  1. Point the v5d extractors at the per-setting zarr by patching `_orig_res_path(s)` (or via an env var override). Keep the production zarr untouched.
  2. Rebuild `cached_per_cell_crops/{sid}_{setting}_per_cell_crops.parquet` from the new mask + level-2 `channel_405.zarr`.
  3. Call `roi_quality_v2.extract_roi_features_v2`, v3, v4, v5, v6 → 5 parquets.
  4. Call `roi_quality_v5d.extract_features(sid)` then `roi_quality_v5d.predict(features_df)` using `cached_roi_quality/roi_quality_stage2_{binary,4class}_v5d_um.txt`.
  5. Filter to ROIs whose level-2 centroid lies inside the overlap bbox.

### Stage 4 — Metrics + plotting (1 CPU-hr)
- New file: `code/sessions/v3_S14_seg_compare/metrics.py`.
- Outputs:
  - `code/sessions/v3_S14_seg_compare/outputs/seg_compare_4combos.parquet`: one row per `(subject, setting, roi_id)` with centroid, in_overlap flag, P(good), P(merged), 4-class label, volume_µm³.
  - `code/sessions/v3_S14_seg_compare/outputs/seg_compare_summary.csv`: one row per `(subject, setting)` with ROI count, mean P(good), mean P(merged), 4-class fractions, mean ROI volume.
- **Headline metric**: per-setting mean of `P(good) − P(merged)` across all in-overlap ROIs, then averaged over subjects. Pairwise paired-by-subject deltas vs the released-cpsam-on-405 baseline.
- Secondary: cross-setting centroid-Hungarian retention rate at 5 µm; 4-class fraction stacked bars per subject.

## Critical files

- New: `code/sessions/v3_S14_seg_compare/{finetune_cpsam,run_inference,score_v5d,metrics}.py` + `outputs/`, `models/`.
- Reused (read-only): `dev_code/overlap_crop.py`, `dev_code/sz_estimator.py`, `dev_code/benchmark_analysis.py::load_hcr_volume`, `dev_code/roi_quality_v{2,3,4,5,6}.py`, `dev_code/roi_quality_v5d.py`, `cached_roi_quality/roi_quality_stage2_{binary,4class}_v5d_um.txt`.

## Wall-clock estimate (with GPU)

| Stage | Time |
|---|---|
| Compute provisioning (human) | ~0.5 hr |
| 1. Finetune (2 LOSO + full) | ~3 GPU-hr |
| 2. Inference (24 runs) | ~40 GPU-hr |
| 3. Features + GBT | ~2 CPU-hr (parallel) |
| 4. Metrics | ~1 CPU-hr |
| **Total** | **~44 GPU-hr / ~2 calendar days** |

## Verification

End-to-end checks at each stage:
1. After finetune: held-out 2D F1 ≥ pretrained on the same slices, else stop and diagnose.
2. After inference: per (sid, setting) report ROI count + median volume; sanity-check against production cellpose output count on the same overlap (expect within 0.5×–2×).
3. After v5d: spot-check 10 random ROIs per setting visually — overlay predicted mask on raw 405; flag if obvious merges/over-segments are common.
4. Reproducibility: every output file carries a sidecar JSON with the cpsam weight hash, the cellpose version, `get_overlap_crop` parameters, and the git commit hash.

## Top risks

1. **GBT classifier OOD** — v5d was trained on production-cellpose; absolute P(good) is biased. Mitigate by treating it as a *relative* metric and adding a small manual-label sanity check (~20 ROIs/setting).
2. **Tiny train set** (2 blocks, 1 per GT subject) — LOSO ≈ n=1 test; finetuned model may underperform pretrained on the autocoreg cohort.
3. **Cross-cohort domain shift** — GT cohort was acquired separately from autocoreg; 405 staining intensity / sectioning may differ.
4. **488 unseen at train time** — the 405+488 finetuned cell is effectively pretrained behaviour on channel 2; gains are not guaranteed.
5. **Compute slip** — if A100 unavailable, falling back to L4 or level-1 inference biases comparisons; document and treat results as preliminary.
