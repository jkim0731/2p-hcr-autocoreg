---
name: S11 v4 ROI-quality features
description: Surface area + sphericity + 4-µm calibrated core/shell features; v4 binary AUC 0.930 (vs v3 0.924), 4-class f1_macro 0.660 (vs v3 0.650).
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
S11 v4 features land 2026-04-30. New families on top of v2+v3-extra:
1. Per-mask geometry: `surface_area_um2_{raw,opened}`, `volume_um3_*`, `sa_to_vol_um_inv_*`, `sphericity_*` (face-counting on anisotropic voxels — biases SA up ~1.5× on a perfect sphere but uniform across cells, fine for discrimination).
2. Calibrated 4-µm core/shell on opened mask via `ndi.distance_transform_edt(sampling=(vz,vy,vx))`: `c405_core4um_p50_opened`, `c405_shell4um_p50_opened`, `c405_shell_minus_core4um_p{50,90}`, `c405_shell_over_core4um_p50_ratio`, `core4um_voxel_frac_opened`. Replaces v2's 1-voxel erosion.

LOSO results (1011 labels, 6-subj):
- binary AUC 0.9295 (v3 0.924, v2 0.897); AP 0.942; Brier 0.115; acc@0.5 0.844
- 4-class f1_macro 0.660 (v3 0.650), acc 0.753

Key importance gains in 4-class:
- `c405_shell_over_core4um_p50_ratio` is #2 (gain 1829, ~3× the v2 1-vox shell-minus-core)
- `sphericity_opened` is #3, `sphericity_raw` is #6
- `c405_shell_minus_core4um_p50` is #7

**Why:** Deeper shell-vs-core separation comes from physically-grounded 4-µm radius (calibrated from intensity profiles of `good` ROIs in `08_core_shell_calibration.py`; pooled r_thresh = 4 µm; per-subject 3-4 µm).

**How to apply:** When working on stage-2 ROI quality, prefer the v4 model (`roi_quality_stage2_binary_v4.txt`, `roi_quality_stage2_4class_v4.txt`); training script at `code/sessions/v3_S11_roi_quality/05d_train_stage2_v4.py`; helper module `code/dev_code/roi_v4_features.py`.

Cache infra (re-usable for v5+):
- `code/dev_code/cached_per_cell_crops/{sid}_per_cell_crops.parquet` — 510k cells, 616 MB total, mask-only (bool packed). 6-worker MP dump took 124 s. Eliminates seg-zarr label decode for future feature iterations. /scratch is root-only in this environment, so cache lives in /root/capsule.
- v4 extraction with 6-worker MP: 14 min wall (vs v3 sequential 125 min).
