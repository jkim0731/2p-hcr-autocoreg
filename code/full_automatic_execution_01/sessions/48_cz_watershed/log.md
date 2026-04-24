# S48 (= S46-c) — Per-cell CZ labels via centroid-seeded voronoi

**Status:** validated (foundation shipped). M4 wrapper NOT re-run this session; deferred as low-ROI (see § Decision).

## Problem (from S45)

S45 established that the shipped `*_seg-mask-outline.tif` is binary uint8 {0,1}
on all 4 subjects. F2 / M4 are blocked by the absence of per-cell CZ labels.
Cellpose 3D on CPU would take 30–60 min/subject × 4 subjects = 2–4 h, blocking
the auto-mode iteration loop.

## Approach

Centroid-seeded voronoi with two caps:

1. **Anisotropic EDT + indices.** `scipy.ndimage.distance_transform_edt` with
   `sampling=(cz_z_um, cz_xy_um, cz_xy_um)` on the inverted seed mask returns
   both per-voxel distance and per-voxel *nearest seed coordinates*. Indexing a
   seed→cz_id lookup produces per-voxel labels in one vectorized pass.
2. **Radius cap at R=8 µm** (≈ cortical-cell radius). Anything outside the
   ±8 µm sphere around the nearest centroid → 0.
3. **Intensity cap at p50 + 0.3·(p90 − p50)** of the Gaussian-smoothed z-stack.
   Removes voxels that are clearly tissue background (below autofluorescence).

On 788406 this produces 923 / 932 labeled cells, median 2038 µm³, p10/p90
1362/2120 µm³, wall 44 s.

## Tried first: watershed (rejected)

`skimage.segmentation.watershed(-vol_smooth, markers, mask=vol_smooth>p5)` runs
but floods uncontrollably — per-cell median 51 k µm³, max 520 k µm³ (25× too
large). The CZ z-stack has no strong inter-cell basins after smoothing, so
watershed expands each label until it collides with a neighbour. Added cost
(117 s) with bad output → abandon.

## Stats across 4 subjects

| subject | n_cz | labeled | median µm³ | p10 µm³ | p90 µm³ | wall |
|---------|------|---------|------------|---------|---------|------|
| 788406  | 932  | 923     | 2038       | 1362    | 2120    | 44 s |
| 755252  | 835  | 812     | 2074       | 1388    | 2120    | 44 s |
| 767022  | 926  | 905     | 2042       | 1206    | 2120    | 45 s |
| 782149  | 894  | 880     | 2017       | 1074    | 2120    | 44 s |

The tight p90 ≈ 2120 µm³ on every subject = the R=8 µm sphere ceiling
(4/3·π·8³ ≈ 2145 µm³ minus intensity-mask trim). The p10 spread (1074–1388)
captures per-subject intensity-threshold tightness. This is a **footprint
approximation**, not a morphology-faithful segmentation — the radius cap
dominates. Good enough for M4's per-pair IoU / bounding-box features; not a
drop-in replacement for cellpose if we later want per-cell *volume variance*
as an F6 feature.

## Shipped

`lib/cz_labels.py`:
- `cz_voronoi_labels(s, *, R_um=8.0, ...) -> np.ndarray(Z,Y,X) int32` (cached)
- `cz_cell_bboxes(s, *, R_um=8.0) -> dict[cz_id, (z0, z1, y0, y1, x0, x1)]`

The full label volume is 450×512×512 int32 ≈ 450 MB — cached per subject in a
module-level dict. A future F2-style mask loader can call this directly.

## Decision (do NOT re-run S45 M4-augmented P1)

S45's top-K ceiling analysis is the binding constraint on M4's expected value:

- 788406 K=100 → 0.954 (ranker-bounded). Current S47 pipeline at K=5 β=5 is
  0.234 r@20; theoretical ceiling is 0.954 at K=100. M4 could close *some* of
  that gap but only for this subject.
- 755252/767022 K=500 → 0.84/0.86 (generator-broken). M4 can't help — the GT
  partner isn't in the putative set.
- 782149 K=500 → 0.000 (generator structurally broken). Unreachable by M4.

S45's in-sample LR analysis on HCR-only features gave +0.027 AUC, likely
< 0.01 CV-adjusted. Adding per-cell CZ volume (now that it's available) would
plausibly add another ~0.005–0.015. Net expected 788406 r@20 lift from an
M4-augmented ranker: **< 0.02 pp**, at the cost of wiring a new candidate
through F9.

S46-d (alt coarse for 782149) has **unbounded upside** — 782149 is currently 0
on every pipeline stage and a working coarse-alignment path could unlock it
completely. **Pivot to S46-d before M4 re-run.**

## Follow-ups (queued but not blocking)

- **F2-mask-loader** — wrap `cz_voronoi_labels` as a candidate-registry-ready
  loader that returns `(mask_zyx, cz_xy_um, cz_z_um)` in the F1/F2 contract.
- **M4 re-run** — after S46-d lands, re-run the M4-augmented P1 probe on 788406
  (the only subject that will benefit) as a precision-only upgrade; skip if the
  S47 numbers are sufficient.

## Files

- `lib/cz_labels.py` — shipped.
- `sessions/48_cz_watershed/probe_watershed.py` — rejected, kept for record.
- `sessions/48_cz_watershed/probe_voronoi.py` — per-subject probe.
- `sessions/48_cz_watershed/probe_all_subjects.py` — 4-subject run.
- `sessions/48_cz_watershed/probe_*.log` — outputs.
- `sessions/48_cz_watershed/cz_labels_vor_788406_R8.tif` — 788406 sample output
  (other subjects not written to disk; compute on demand).
