# 260429
- HCR ROI segmentation errors.
- Initial landmark search not great or difficult to verify.
- Cell-cell matching classifier would be great to automate the procedure (how can I deal with orientation variation?)
- Next steps:
    - Build HCR ROI classifier (compare with Matt's classifier). How long does it take to get HCR ROI segmentation? Within and nearby the overlapping volume + margin only?
    - Redo automatic registration algorithm with the initial registration constraint.
        - Centroid-based: Both registration and cell matching. Requires good segmentation and GFP+ filtering. cell matching can be improved after initial registration. How to assess registration improvement?
        - Image-based: registration only. Apply cell-cell matching later (maybe similar to manual workflow after the fine registration - but requires cell-cell matching classifier to be fully automatic)

# 260514
- HCR ROI classifier is built and quite reliable. Use good + bad_ok cells when filtering is necessary.
- GFP+ cells can be defined from BIC GMM from 488 spot density (raw "unmixed" spots - R2 for 755252 and 767022)
- CP SAM test is running but not promising. Current segmentation may be the best option.
- 3 potential options left: 
    - 0. All after initial surface registration 
    - 1. Matching using geometric features (shape contexts - Belongie et al., 2002; soma-print - Wang et al., 2026 https://www.biorxiv.org/content/10.64898/2026.04.28.719500v1.full.pdf)
    - 2. Piecewise volumetric nonrigid registration - pyramid. 
    - 3. Mimicking manual workflow (requires evaluating matches - maybe the same features in #1, or using 3D image networks)

# 260617
- RESOLVED. Automatic coregistration works end-to-end as a **2-step** process:
  (1) image-based rough/warm registration (locked prior: surfaces → sxy → top-slab
  2-D reg → sz → overlap crop), then (2) **soma-print 3-D cell-cell matching** +
  iterative TPS with customized filters (R_cand=150 → mutual-best → round-0
  local-flow → anchor-vote gate → TPS, iterate).
- Of the 3 options from 260514: **#1 soma-print won.** Shape-context ("3D ROI
  ContextNet") was tried and dropped (degrades under realistic pose; fig_09).
  #3 manual-workflow-mimic is the in-progress QC manual-improvement app. #2
  piecewise volumetric nonrigid not pursued (TPS on matched cells suffices).
- CP-SAM confirmed not worth it for now (no quality gain; GPU-gated) → abandoned.
- HCR ROI classifier is required by the matcher (GFP+∩ok pool + junk removal).
- Presentation video + figures done (session 16).
- Code migrated to two repos: **2p2fish** (autocoreg) + **mfish-roi-classifier**.
  `dev_code` is now legacy (standalone import bug fixed 2026-06-17).
- QC app ~half done: pass/fail labeller built; manual-improvement app to build.
- See `docs/11 Work In Progress.md`, `docs/12 Code repositories and how to run.md`,
  and `/scratch/sessions/17_interim_summary/README.md`.

# 260618
- **HCR ROI classifier → self-contained.** Retired the upstream stage-1 dependency
  (multi-channel + GFP); production model is now **C2**: 101 µm features, 405-only, with
  12 neighbour-quality features replacing the 2 stage-1 neighbour scores. Binary LOSO 0.922
  (> prior 0.9206). Other channels (488/594/GFP) tested, **not** adopted (binary flat-to-noise;
  GFP = matching-signal circularity). Adaptive bbox recovered 24 boundary-skipped labeled cells.
  Contract renamed `…_proba_v5d_um.parquet → …_proba.parquet` (2p2fish to match). Fixed a
  `PCT_RANK_COLS` train bug. Record: docs/13 §7 +
  `/scratch/sessions/17_interim_summary/stage1_stage2_architecture.md`.
- Next (session 18): plan to split into 3 CodeOcean capsules (extract+infer | labeling |
  training+MLflow); per-session timestamped label assets, newest-wins merge.
