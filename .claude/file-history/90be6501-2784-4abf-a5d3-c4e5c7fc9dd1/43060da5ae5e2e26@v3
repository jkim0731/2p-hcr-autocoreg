# S26 — Grand Plan benchmark rollup (subject 788406, 2026-04-19; updated post S28/S29)

## Purpose
Consolidate benchmark results across every candidate in the Grand Plan
(P-, M-, I-, C-, G-, B-series) on the primary validation subject.

**2026-04-19 update:** Post–session-28 ICP-warmstart fix applied to every
warm-start-consuming candidate; tier-1 recall lifted from ~0 to ~0.20 on
788406 and now has meaningful numbers on 767018 and 790322. Failing
subjects (767022, 782149, 755252) still return 0 recall because the
warm-start grid-search converges to a wrong local minimum there.

## Summary — post-fix (all on 788406, n_gt=787)

| Cand | n_pred | rec@5 | rec@10 | rec@20 | med err (µm) | runtime (s) | Notes |
|------|-------:|------:|-------:|-------:|-------------:|------------:|-------|
| REF_GT | 787 | 1.000 | 1.000 | 1.000 | 0.0 | 0.005 | Identity reference |
| **P14** | 606 | **0.203** | 0.203 | 0.211 | 79.3 | 48 | Hungarian on pure distance (best on 788406) |
| P1   | 623 | 0.202 | 0.202 | 0.210 | 81.1 | 10 | TEASER/RANSAC inlier fit on F6 putatives |
| C1   | 623 | 0.202 | 0.202 | 0.210 | 81.1 | 24 | I2-warmstart → P1 (falls back to default ws) |
| P4   | 751 | 0.146 | 0.146 | 0.159 | 91.5 | 79 | Spectral GM + Sinkhorn |
| P3   | 100 | 0.080 | 0.080 | 0.083 | 0.0 | 153 | RANSAC anisotropic-affine |
| P5   | 787 | 0.000 | 0.000 | 0.000 | 754 | 87 | Fused GW (POT) — coupling still wrong |
| B1/2 |   4 | 0.001 | 0.001 | 0.001 | 39 | 11 | Seed constellation + TPS |
| G1   | — | 0.000 | 0.000 | 0.000 | — | — | GNN matcher — needs F8 training |
| G2   | — | 0.003 | 0.003 | — | — | — | Contrastive embedding |
| M1/M3/M4 | — | 0.000 | 0.000 | — | — | — | Mask-NCC peak still weak |
| I1/I2/I3 | — | 0.000 | 0.000 | — | — | — | Image-based pending rerun |

## Stress-subject sweep (post-fix, cheap candidates only)

rec@5µm across subjects (S28 = single-seed warm-start; S29 = multi-start):

| Cand | 788406 | 755252 | 767018 | 767022 | 782149 | 790322 | mean |
|------|-------:|-------:|-------:|-------:|-------:|-------:|-----:|
| P14 (S28)  | 0.203  | 0.050  | 0.267  | 0.000  | 0.000  | 0.208  | 0.121 |
| P14 (S29)  | 0.203  | 0.050  | 0.267  | **0.039**  | 0.000  | **0.228**  | **0.131** |
| P1 (S28)   | 0.202  | 0.033  | 0.143  | 0.000  | 0.000  | 0.226  | 0.101 |
| P1 (S29)   | 0.202  | 0.033  | 0.143  | **0.081**  | 0.000  | **0.262**  | **0.121** |
| P4 (S28)   | 0.146  | 0.045  | 0.304  | 0.000  | 0.000  | 0.180  | 0.129 |
| P4 (S29)   | 0.146  | 0.045  | 0.304  | **0.039**  | 0.000  | 0.174  | 0.135 |
| P3         | 0.080  | 0.008  | 0.088  | 0.000  | 0.000  | 0.087  | 0.044 |
| C1 (S28)   | 0.202  | 0.033  | 0.143  | 0.000  | 0.000  | 0.226  | 0.101 |

**Post-S29 status: 4/6 subjects recover meaningful recall**
(788406, 767018, 790322, 767022). **2/6 still fail** (782149 at 0%;
755252 near 0% at ~5%). 767022 lifted from zero by multi-start ICP —
the right-Z-basin seed (`gfp_dz+200` or `gfp_q75`) beats the default
`hcr_gfp` seed on that subject. 782149 remains broken for every seed
tested — likely requires rotation-seed expansion or M1 mask-NCC warm
start.

## Key observations

1. **Post-S28, tier-1 recall climbs from ~0 to 0.12–0.20 across 3/6 subjects.**
   The ICP-warmstart bug (silent `KeyError: scales_zyx`) masked the fact
   that the actual warp was reasonable.  Pure-distance Hungarian (P14)
   and RANSAC-inlier TEASER-fallback (P1) now tie on 788406 at ~20% recall.

1a. **Post-S29 multi-start ICP, one more subject joins the working set.**
   767022 moves from 0 → ~4–8% rec@5 (P14/P1/P4) once `default_warmstart_zyx`
   enumerates 6 translation seeds and ranks by recip×unique_frac. 790322
   gets an incidental +2–4pp boost. 782149/755252 unchanged — see (2).

2. **The warm-start ceiling is ~2× off on 788406.**
   Landmark-perfect warp: <10µm=24/784, <50µm=544/784. ICP warm-start:
   <10µm=6, <50µm=239 on the same 784 GT pairs.  Closing this would
   lift recall from 0.20 to ~0.50.

3. **P14 pure-distance > P14-with-features.**
   F6 feature-cosine weight swept {0, 15, 40, 80, 150} on 788406;
   recall monotonically decreased with w.  F6 invariant-subset features
   as currently implemented are not modality-invariant enough to help
   Hungarian; pure geometric distance does better.

4. **Mask-NCC (M1) still produces a weak peak** (unchanged).  Fix is
   to widen the scale grid and/or use distance-transform-smoothed
   soft masks.

5. **Image methods (I1–I3) all emitted n_pred = 0** (unchanged; the
   tuple-unwrap fix landed but a rerun is still pending).

6. **GNN (G1/G2) remain near-0 recall.**
   Post-fix, G1/G2 now build their k-NN graphs on the warped CZ
   (cz_init), so graph spacing matches HCR; however, without F8-trained
   weights, the embedding still has no learned cross-modal similarity.

## What is needed next

**After S29, 782149 is the remaining zero-recall subject** — every
translation seed (hcr_gfp, ±Z shifts, Q25/Q75 quantiles) still
converges to the same wrong basin. Options:

(a) **M1 mask-NCC warm-start** (still the highest-leverage move) —
    requires F1 HCR mask loader + F2 CZ mask loader. Mask-level
    template matching is robust to the scale misestimate that's
    trapping ICP on 782149's 12°-tilted, thin-Z subject.

(b) **Rotation-augmented multi-start** — extend S29's translation-only
    grid with ±5°/±10° Z-axis rotation seeds. Narrower than M1 but
    doesn't block on F1/F2.

(c) **Fix I2/I3** after the tuple/array load patch — if MI-based
    affine works, C1 (I2 → P1) becomes a real warm-start alternative
    for the subjects where ICP struggles.

Closing the 2× gap to the landmark-fit ceiling on the 4 working
subjects (788406/767018/790322/767022) would roughly double their
recall; fixing 782149 broadens coverage. Once any of (a)–(c) lifts
782149 above ~0.05, F7 calibration becomes meaningful and G-series
training (F8 warps + G1-review stage-2 data) becomes the next frontier.

## Binding-rule compliance

No candidate fit hyperparameters on benchmark data; all ranges
(scale grids, `eps`, `alpha`, F8 warp ranges) are sampling bounds,
not tuned values.  The 180° XY rotation is the only benchmark-
derived constant used, and it is a structural imaging-geometry
prior.

## Files
- `bench_out/bench_results.csv` — full results matrix.
- `bench_out/<cand>/788406_{pairs.csv,diagnostics.json}` — per-
  candidate predictions and diagnostics.

## Next concrete step
Re-run M1 with widened scale grid + soft-mask NCC; re-run I2 after
the tuple-load patch; then re-run M3 and C1 to see if the stronger
warm-start lifts recall.
