# Session 16 — Intermediate Summary Materials (publication figures + presentation movies)

## Context

The automated CZ(2p-GCaMP)→HCR(light-sheet) coregistration protocol is now mature: rough
registration (surfaces → sxy → tilt/tz → top-slab 2-D → sz → overlap crop) is **done and
cached for all 6 subjects**, soma-print cell-cell matching clears the Step-1/Step-2 bars, and
Step-3 iterative expansion runs end-to-end (`local_flow` variant complete; the NCC-gated path's
final comparison is still ongoing). The ROI-quality classifier (S11 v5d) is promoted with cached
scores + ~1.9k human labels.

We need a coherent set of **publication-quality figures** and **presentation-quality movies**
that explain the protocol step by step, using subject **790322** for all image/movie examples.
The deliverables must also include (a) GT-based analysis of tilt, xy/z expansion rate, and
nonrigid deformation, and (b) a description + results + example imagery for the ROI classifier.

This is a **materials-generation** session — no protocol logic changes. We reuse cached numeric
artifacts and existing renderers; new code only assembles figures and frame-stack movies.

**Confirmed decisions:** movies as **both MP4 + GIF**; GT analysis as **6-subject panel +
790322 detail**; matching figures feature **soma-print + `local_flow`** (NCC path marked "in
progress"). Rendering is **offscreen matplotlib/PIL/imageio** (no napari — headless host;
`imageio_ffmpeg` present so MP4 works; no system `ffmpeg`, so always use the imageio writer).

---

## Deliverable layout

New session dir: `code/sessions/16_intermediate_summary_materials/`

```
16_intermediate_summary_materials/
├── README.md                  # narrative index: what each figure/movie shows, in protocol order
├── _common.py                 # shared style (fonts, dpi=300, colormaps, save_fig/save_movie helpers, 790322 loader)
├── fig_00_problem_overview.py # scale/context: tiny CZ box inside huge HCR
├── fig_01_gt_expansion.py     # GT panel: xy vs z expansion, anisotropy, tilt
├── fig_02_gt_deformation.py   # GT panel: residual RMS/max after affine + 790322 deformation field
├── fig_03_surfaces.py         # pia surface fits (CZ + HCR), tilt
├── fig_04_sxy_roi_area.py     # per-cell bbox-area-ratio distribution + example CZ/HCR cell crops
├── fig_05_topslab_reg.py      # CZ binary warped onto HCR-488 MIP, before/after, NCC
├── fig_06_sz_fft.py           # NCC-vs-sz curve + side-view slab
├── fig_07_overlap_crop.py     # localized CZ box placed in HCR (payoff)
├── fig_08_somaprint_concept.py# descriptor schematic (m-NN vectors, n-best match)
├── fig_09_step2_robustness.py # AUC degradation: soma vs shape-context vs centroid (the load-bearing result)
├── fig_10_final_matches.py    # local_flow matches colored by GT agreement (790322)
├── fig_11_roi_classifier.py   # ROI classifier: metrics + confusion + per-class example crops (X-sec + side MIP)
├── mov_A_localization.py      # CZ subvolume flies/rotates into place inside HCR
├── mov_B_roughreg.py          # top-slab overlay forming + sz sweep
├── mov_C_matching_expansion.py# Step-3 rounds growing + TPS tightening
├── outputs/
│   ├── figures/*.png|pdf
│   └── movies/*.mp4 + *.gif
└── log.md                     # hypothesis/method/result per CLAUDE.md logging
```

---

## What matters, and why each gets visual treatment

Ordered by the protocol narrative. **P1 = must-have / high-impact**, P2 = supporting, P3 = nice-to-have.

### A. The problem (P1) — `fig_00`, `mov_A`
The single most impressive idea: a ~400 µm in-vivo box localized inside a multi-mm ex-vivo volume,
under ~180° rotation + anisotropic expansion + nonrigid warp + partial correspondence.
- **fig_00**: HCR 405/combined MIP (level 4) with the CZ overlap box drawn to true scale; inset of CZ.
- **mov_A** (movie, impressive): CZ centroid cloud rotating/“flying” into its located pose inside the
  HCR cloud — the localization payoff. 3-D matplotlib scatter, rotating azimuth, ~120 frames.

### B. GT data analysis (P1, explicitly requested) — `fig_01`, `fig_02`
Numbers already exist in `docs/01 Data Description.md` (landmark anisotropic-similarity fit); we
**recompute from the GT landmark/coreg data** so the figure is self-contained and reproducible.
- **fig_01** (6-subject panel): grouped bars of XY vs Z expansion per subject + XY:Z anisotropy;
  pia-tilt (CZ vs HCR) bars. Story = expansion is anisotropic (Z ~1.5× XY) and tilt varies 1–12°.
- **fig_02**: residual-RMS / residual-max-after-affine bars (6-subject) **+ 790322 detail**: a
  quiver/displacement field of the post-affine residual (the nonrigid deformation), magnitude-colored,
  shown in an xy and a depth (xz) projection. This is the "degree of nonrigid deformation" deliverable.

### C. Rough registration, the locked-prior chain (P1/P2) — `fig_03..07`, `mov_B`
This is the bulk of the engineering. Show the 4-step pose chain (sxy → tilt/tz → top-slab θ → sz):
- **fig_03** (P2): pia plane fits over CZ and HCR MIPs (side view), annotated tilt angles.
- **fig_04** (P2): the sxy mechanism — histogram of per-cell CZ↔HCR **tight xy-bbox-area** ratios with
  the median (sxy≈1.76 for 790322), plus 2–3 matched example cell crops (CZ vs HCR) showing the size
  jump. Use the shipped `roi_area_sxy.py` bbox-area definition **as-is** (the bbox→max-ROI-cross-section
  swap is deferred to a separate session, per user). Caption states the area measure is the tight xy
  bbox so the method is documented honestly.
- **fig_05** (P1): top-slab 2-D registration — warped CZ binary overlaid on raw HCR-488 MIP,
  **before vs after**, with the PWR NCC score. Visually compelling alignment.
- **fig_06** (P2): sz estimator — FFT-NCC vs sz curve with the peak marked (sz≈3.1), plus a side-view
  slab montage; explains why z is the weak axis.
- **fig_07** (P1): the overlap crop — final localized CZ box inside HCR, the rough-reg payoff.
- **mov_B** (movie): the top-slab CZ overlay sweeping into alignment, then the sz NCC sweep animating.

### D. Cell-cell matching (P1) — `fig_08..10`, `mov_C`
- **fig_08** (P1, clarity-critical): soma-print descriptor schematic — a cell's m nearest-neighbour
  vectors and the n-best m×m matching to a candidate. Hand-drawn-style matplotlib diagram.
- **fig_09** (P1, the load-bearing result): Step-2 AUC@50 degradation table/plot — soma-print
  (~0.99, pose-robust) vs shape-context (degrades) vs centroid (weak). Why the pipeline went soma-only.
- **fig_10** (P1): final `local_flow` matches on 790322 rendered as endpoint pairs over HCR-488 MIP
  slabs, colored by GT agreement (reuse/extend `run_visualize_matches.py`); caption notes NCC path TBD.
- **mov_C** (movie, impressive): Step-3 iterative expansion — accepted matches growing round by round
  with the TPS warp tightening (drive from `outputs/step3_v2_path_a_local_flow/790322/matches_round{0..2}.csv`).

### E. ROI classifier (P1, explicitly requested) — `fig_11`
- Brief description (in README + figure caption): per-HCR-ROI quality classifier (LightGBM v5d),
  92 voxel-unit + pct-rank features, no re-segmentation; 4 classes good / bad_ok / merged / bad.
- Results panel: LOSO binary AUC (~0.92) + 4-class f1_macro (~0.72) + a **confusion matrix**, and the
  790322 score distribution. Computed from `cached_roi_quality/` OOF parquets + the 1.9k labels.
- **good+bad_ok vs rest** example imagery (explicit request): for 790322, montage of example ROI crops
  per group — **cross-sectional (axial slice through centroid)** and **side maximum-projection (xz)** —
  drawn from `segmentation_mask.zarr` (level-0; xy /4) + 405/488 channels, grouped good∪bad_ok vs
  merged∪bad. Shows visually what the classifier separates.
- **Full-volume junk-removal view (per user request)**: score **all** HCR ROIs in the full 790322
  volume with `roi_quality_v5d.predict_subject(s)`, then render the **whole-volume** centroid cloud
  (and/or 405 MIP) colored **kept (good∪bad_ok) vs removed (merged∪bad)**, with the fitted pia surface
  drawn as reference. The point: visualize **how many out-of-tissue / debris ROIs the classifier culls**
  across the entire volume (not just hand-picked crops) — e.g. removed ROIs concentrated above the pia
  / outside the tissue band. Report kept/removed counts and the fraction of removed ROIs lying above the
  pia surface as a quantitative cleanup measure. This is a separate sub-figure (`fig_11b`) from the
  per-class crop montage (`fig_11a`).

---

## Reuse map (do not rebuild)

- **Data loading**: `dev_code/benchmark_data_loader.load_subject('790322')`; `cz_px_to_um`/`hcr_px_to_um`.
- **Image volumes**: `cz_volume.load_cz_volume(s)`; `benchmark_analysis.load_hcr_volume(s, channel, level)`
  and `load_hcr_combined(...)` (zarr pyramid; level 4 ≈ 4 µm).
- **Cached pose artifacts (790322, all present)**: `cached_surfaces/790322_*`,
  `cached_surface_registration/790322.json`, `cached_sz/790322.json`; `overlap_crop.get_overlap_crop(s)`,
  `locked_prior_warm.compute_locked_prior_warm_start(s)` + `apply_to_cz_um`, `roi_area_sxy.estimate_sxy_roi_area(s)`.
- **Matching**: `soma_print.py` (descriptor for the schematic); existing match CSVs under
  `sessions/15_geom_features/outputs/step2p5_matches/790322.csv` and
  `outputs/step3_v2_path_a_local_flow/790322/matches_round*.csv`.
- **Renderers to extend** (not duplicate): `run_visualize_matches.py`, `run_visualize_paired.py`,
  `build_warped_cz_volume.py`, `build_seg_volumes.py` (feed slice-through movie frames).
- **ROI classifier**: `roi_quality_v5d.predict_subject(s)`; OOF parquets in `cached_roi_quality/`;
  labels in `sessions/v3_S11_roi_quality/outputs/roi_qc_actions.jsonl` (790322 = 314 labels).
- **GT**: `s.coreg_table`, `s.landmarks_qced` for expansion/tilt/deformation recompute.

## Movie rendering approach (headless)
Render frames with matplotlib (`Agg`) / numpy MIPs → list of RGB arrays → `imageio.mimsave` for GIF
and `imageio.get_writer(..., fps=...)` (ffmpeg plugin) for MP4. Helper `save_movie(frames, stem, fps)`
in `_common.py` writes both. Keep movies short (5–10 s, ~640–960 px) for slide embedding.

---

## Execution order
1. Scaffold session dir + `_common.py` (style, loaders, save_fig/save_movie). Smoke-test 790322 load
   + one HCR MIP render.
2. **GT analysis** (`fig_01`, `fig_02`) — recompute expansion/tilt/residual from landmarks; verify
   numbers match `docs/01 Data Description.md` within rounding (sanity gate).
3. Rough-reg statics (`fig_00`, `fig_03..07`) from cached pose artifacts.
4. Matching statics (`fig_08`, `fig_09`, `fig_10`).
5. ROI classifier (`fig_11`) — metrics/confusion from OOF + label crops.
6. Movies (`mov_A`, `mov_B`, `mov_C`) last (heaviest).
7. `README.md` narrative index + `log.md`.

## Verification
- Each `fig_*`/`mov_*` script is runnable standalone (`python code/sessions/16_.../fig_XX.py`) and
  writes to `outputs/`. Run all; confirm every PNG/PDF + MP4/GIF is produced and non-empty.
- GT figures: assert recomputed 790322 XY≈1.77×, Z≈3.04×, residual RMS≈22 µm match the doc (±1).
- Visual check: open 3–4 representative PNGs and the 3 movies (read first/last frame dims) to confirm
  they render the intended overlays (alignment visible in fig_05; rounds grow in mov_C).
- README cross-references every artifact in protocol order with a one-line "what it shows".

## Out of scope
- No changes to pipeline/matching logic. NCC-gated final-match figure deferred (placeholder note in
  README) until that comparison completes. No interactive GUI work.
