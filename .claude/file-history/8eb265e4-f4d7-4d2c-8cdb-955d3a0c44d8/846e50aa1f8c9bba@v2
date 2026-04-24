---
name: Session 07d image-level 488 NCC sz FAILED
description: Image-level depth-profile NCC (R1 + pia-normal alignment, 1D intensity profile) tested 3 preprocessing variants; best = 488-only + p10 baseline-subtract at 2/6 pass; combined-channel 0/6; per-subject oracle over variants 3/6. Shape mismatch between CZ GCaMP and HCR 488 replaces threshold-bias as the dominant failure.
type: project
originSessionId: 8eb265e4-f4d7-4d2c-8cdb-955d3a0c44d8
---
Session 07d implemented the "next candidate" flagged by 07b/07c:
image-level 488 NCC for sz, bypassing centroid threshold-bias.

**Design (all variants).** Apply R1 (rotation + translation, no
scaling) to CZ voxels so CZ and HCR share xy centroid + pia-aligned
z. Compute depth-from-HCR-pia for every voxel. Bin CZ and HCR into
1D depth profiles `I(d)` at 10-µm bins. For each candidate `sz`,
stretch CZ profile around the R1-translation anchor and evaluate
NCC on the positive-positive overlap. Report argmax sz. GT from
`fit_anisotropic_similarity(landmark_pairs_um(...))`, scoring only.

**Three variants tested end-to-end:**
1. 488-only, no baseline, sz∈[1.0, 5.0]: **1/6** (790322 only).
2. Combined multi-channel (405+488+514+561+594) + p10 baseline,
   sz∈[1.5, 4.5]: **0/6** — worse than 488-only.
3. 488-only + p10 baseline, sz∈[1.5, 4.5]: **2/6** (755252 −4.2%,
   767018 +0.5%). Best.

**Per-subject oracle over variants: 3/6** (adds 790322 at −3.3 %
from 488-only-no-baseline). Oracle requires GT knowledge so not a
valid estimator. Fails 6/6 bar under every framing.

**Failure modes.**
- **Structural truncation (782149):** HCR z-extent 1004 µm;
  GT-stretched CZ needs 450·2.93 = 1318 µm. NCC can never find
  GT sz regardless of preprocessing — the physical match isn't
  inside the observed HCR volume. −37 to −47 % in all variants.
- **Profile-shape mismatch (767022, 788406):** CZ GCaMP (shallow
  expression peak) and HCR 488 (layer-dependent retention, sharp
  tissue fall) have qualitatively different depth shapes; NCC
  aligns shapes, not physical tissue.
- **Preprocessing sign flips (790322 vs 755252/767018):** Raw 488
  has a clean plateau for 790322 that matches CZ; baseline
  subtraction creates artificial peaks and pushes sz to grid top.
  Opposite for 755252/767018 — raw 488 is flat, baseline reveals
  structure. No universal preprocessing rule works for all six.

**Core insight.** Intensity NCC bypasses the threshold-bias that
blocked 06/07/07b/07c, but surfaces a new bias — **GCaMP-vs-488
depth-shape mismatch** — that's at least as severe and has no
knob to tune. 1D intensity profile is too coarse: it collapses all
xy structure into a scalar, losing exactly the lateral features
that distinguish stretch-sz from shape-sz.

**Where.**
- `dev_code/07d_image_ncc_sz.py` — estimator (variant dispatch via
  argv; configs `488_only / 488_baseline / combined_baseline`).
- `dev_code/07d_probe_images.py` — one-subject probe.
- `sessions/07d_image_ncc_scale/log.md` — full write-up including
  all three variant tables and cross-variant oracle.
- `sessions/07d_image_ncc_scale/figures/` — 12 PNGs (6 per primary
  variant).
- `sessions/07d_image_ncc_scale/sz_ncc_summary_488_baseline.json`
  (best), `..._combined_baseline.json`, `archive_488_only/` (initial).

**Do NOT** try another 1D intensity-profile variant (different
channel, different baseline, different smoothing) — the ceiling is
set by GCaMP-vs-488 shape mismatch, not by preprocessing.

**Next candidates (each its own session, NOT implemented here):**
(a) 2D `I(x, d)` and `I(y, d)` projection NCC — optimise `(sxy, sz)`
    jointly; recovers lateral structure that a 1D profile averages
    away. 30× more NCC evaluations but tractable.
(b) Segmentation-based layer-boundary matching — segment cortical
    layers independently in each modality, match boundary depths
    in µm. Side-steps shape mismatch since both report the same
    layer boundary regardless of stain.
(c) Co-estimate sz from sxy + cell-count conservation:
    `sz = N_ratio / sxy²`. Bypasses 782149's truncation entirely
    since it never matches stretched depth profiles.
