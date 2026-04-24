# S44 — Cross-modality bakeoff (P1, M1, M3, C1, I2) on 4 subjects

**Status.** validated — **C1 is a free 2× speedup for P1** (identical
r@20 on all 4 subjects); **M1/M3 as registered are the S31-abandoned
density-NCC variants, not the seg-mask variants from Grand Plan §4.2**;
**no cross-modality lift on any subject**; 782149 remains unreachable.

## Setup

P1, M1, M3, C1, I2 via F9 harness on {788406, 755252, 767022, 782149}.
Primary question: does any mask/image candidate lift 782149 (r@20=0)
or lift 755252/767022 over the P1 centroid plateau (0.044 / 0.108)?

## Results — r@20 matrix

| Subject | P1    | M1    | M3    | C1    | I2    |
|---------|------:|------:|------:|------:|------:|
| 788406  | **0.210** | 0.000 | 0.000 | **0.210** | 0.000 |
| 755252  | **0.044** | 0.000 | 0.000 | **0.044** | 0.000 |
| 767022  | **0.108** | 0.000 | 0.000 | **0.108** | 0.000 |
| 782149  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Results — median_err_um, wall time, n_pred

| Subj | P1 med | P1 s | M1 | M3 med | M3 s | C1 med | C1 s | I2 |
|------|-------:|-----:|---:|-------:|-----:|-------:|-----:|---:|
| 788406 | 81   | 55 | — | 436  | 19 | 81  | 25 | — |
| 755252 | 103  | 53 | — | 898  | 16 | 103 | 32 | — |
| 767022 | 107  | 37 | — | 664  | 11 | 107 | 18 | — |
| 782149 | 1136 | 35 | — | 1136 |  7 | 1136| 12 | — |

M1 and I2 emit transforms only (n_pred=0 → unscorable) — expected for
coarse-affine candidates.

## Findings

1. **C1 (I2 → P1) exactly ties P1** on all 4 subjects for r@20, rec_id,
   median error, and n_pred. But wall time is **2× faster** (12–32 s
   vs 35–55 s for P1). I2's image-level MI-affine provides a warm-start
   at least as good as P1's default ICP-multi-start, without changing
   the basin. **C1 is a free drop-in speedup for P1**.

2. **M1 as registered is the S31-abandoned density-NCC**, not the
   seg-mask-NCC from Grand Plan §4.2. The file `_m1_mask_ncc.py`
   header states "centroid-density representation" and emits empty
   output when robust-z < 3 (which happens on all 6 subjects per S31).
   The seg-mask-NCC variant (using F1 HCR seg-mask loader + F2 CZ
   seg-mask loader + F4 mask-overlap scorer) has **never been tested**
   despite F1/F2/F4 helpers existing in `lib/`.

3. **M3 (M1 → P1 hybrid) actively regresses P1** on 3/4 subjects:
   788406 med 81 → 436; 755252 103 → 898; 767022 107 → 664 (4–9× worse
   than P1 alone). M3 inherits M1's abandoned density-NCC transform
   and feeds it to P1 as warm-start, landing P1 in a wrong basin that
   P1's own default warm-start would have avoided. **Do NOT ship M3
   until M1 is rewritten for seg-mask NCC.**

4. **782149 remains unreachable.** M3 converges to the same wrong basin
   as P1 (1136 µm median); C1 converges to the same wrong basin as P1
   (1136 µm); none of the image/mask candidates provide the orthogonal
   signal needed. Consistent with S32/S34/S35/S41: 782149 centroid
   methods exhausted regardless of warm-start modality.

5. **P1 remains the centroid-only production baseline at 0.362 total
   r@20.** No candidate tested lifts any subject.

## Decision

- **Promote C1 as the production-default P1 wrapper** — same recall at
  half the runtime. Low risk: if I2 fails, C1 falls through to the
  centroid-only warm-start (C1's design already handles this). Verify
  C1's fallback behaviour on a broken-I2 synthetic before shipping.

- **Kill M1/M3 as currently registered.** Both rely on the S31-abandoned
  density-NCC. Mark them as `superseded` in the Grand Plan ledger. The
  Grand Plan §4.2 M1 (seg-mask NCC via F1/F2/F4) is a different
  candidate and needs a distinct `_m1_segmask_ncc.py` implementation.

- **Do NOT invest in the seg-mask M1 pivot without first diagnosing
  why 782149 fails in the mask domain.** Prior sessions established the
  problem is not coarse localization (S34 I2-crop gets median 267 µm on
  782149; S41 all P-variants have predictions in the right XY region)
  but per-cell ranking / assignment at the correct XY scale. Seg-mask
  NCC addresses coarse localization, which is already solved — it will
  not unlock 782149 unless combined with a different ranker.

- **Pivot S45 to the actual bottleneck: ranker / assignment at the
  correct scale.** Two cheap directions:
  - **S45-a**: M4 per-cell Dice / IoU as an additional per-pair feature
    inside P1's putative scoring (requires F1/F2 seg-masks in memory;
    reuses F4 scorer). If IoU is identity-informative on 755252 where
    F6 alone is not, we get a lift.
  - **S45-b**: Spatial cluster-level P1 — run P1 independently on
    small XY sub-regions of 782149 (each a k-NN cluster), then TPS
    expansion from the highest-confidence sub-region. Tests whether
    the problem is global-TLS basin-commitment vs local-region-level
    solvability.
  Start with S45-a since F4 exists; S45-b requires more scaffolding.

## Artifacts

- `probe_mi_bakeoff.py` — 5 candidates × 4 subjects = 20 F9 runs.
- `probe_mi_bakeoff.log` — stdout per run.
- `mi_bakeoff.csv` — machine-readable matrix.
