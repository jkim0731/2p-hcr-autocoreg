# S40 — P1 multi-start with I2-crop + I2-direct seeds

**Status.** validated_negative — multi-start with I2-based seeds brings no
lift; SS ranker correctly selects default on all 4 subjects (safe, not
useful).

## Motivation

S39 showed P1 TEASER at default (K=5, c_bar=15) is stuck on 755252
(r@20=0.044), 767022 (0.108) and 782149 (0.000); only 788406 has real
recall. S34 showed I2-crop W=600 seed unlocks 755252 on raw anisotropic
ICP (+53 n_lt50). Hypothesis: wrap P1 with multi-seed selection — try
default (`default_warmstart_zyx`), I2-direct (MI-affine applied to CZ),
and I2-crop600 (CZ rotated 180° + cropped-HCR XY centroid) — and rank by
a no-peek SS score (count of P1 output pairs with residual_um < 30 µm).

## Setup

`probe_multistart.py` — per subject runs three P1 calls with different
`cz_init`:

- **default** — `cz_init=None` (P1 uses its 6-translation grid multi-start
  ICP warm-start internally).
- **i2_direct** — `cz_init = r_i2.apply_inverse(cz_um)` where `r_i2` is
  SimpleITK 3D MI-affine at 8 µm isotropic (I2 candidate).
- **i2_crop600** — I2 gives predicted CZ center; crop HCR-GFP+ to a
  ±600 µm XY window; set `cz_init = 180°-rotated CZ + cropped-HCR
  centroid`. Same construction as S34's seed.

SS score: `sum(pairs_df.residual_um < 30)` from run_p1's output.

## Results

| Subject | Seed        | SS  | r@5   | r@10  | r@20  | rec_id | median µm | wall |
|---------|-------------|----:|------:|------:|------:|-------:|----------:|-----:|
| 788406  | default     | 370 | 0.202 | 0.202 | 0.210 |  0.202 |      81   | 54 s |
| 788406  | i2_direct   |  36 | 0.000 | 0.000 | 0.000 |  0.000 |     333   |  7 s |
| 788406  | i2_crop600  | 211 | 0.003 | 0.003 | 0.003 |  0.003 |     246   |  6 s |
| 755252  | default     | 500 | 0.033 | 0.033 | 0.044 |  0.033 |     103   | 51 s |
| 755252  | i2_direct   | 116 | 0.000 | 0.000 | 0.000 |  0.000 |     259   | 10 s |
| 755252  | i2_crop600  | 459 | 0.000 | 0.000 | 0.000 |  0.000 |     170   | 10 s |
| 767022  | default     | 242 | 0.081 | 0.082 | 0.108 |  0.081 |     107   | 35 s |
| 767022  | i2_direct   |  41 | 0.000 | 0.000 | 0.000 |  0.000 |     395   |  5 s |
| 767022  | i2_crop600  | 216 | 0.001 | 0.001 | 0.001 |  0.001 |     312   |  5 s |
| 782149  | default     | 113 | 0.000 | 0.000 | 0.000 |  0.000 |    1136   | 33 s |
| 782149  | i2_direct   |  12 | 0.000 | 0.000 | 0.000 |  0.000 |     323   |  1 s |
| 782149  | i2_crop600  |  81 | 0.000 | 0.000 | 0.000 |  0.000 |     267   |  2 s |

**SS-picker matches oracle on every subject** (default wins 4/4). No
regression risk. No lift either.

## Diagnosis

Two seed-specific failures:

1. **I2-direct is simply worse than default** on all 4 subjects (SS 12–
   116 vs default 113–500). Raw MI-affine at 8 µm isotropic does not
   reach default's ICP-based warm-start + local-refit quality for point
   clouds.

2. **I2-crop600 gets close in bulk but TEASER still picks wrong pairs**.
   Most striking: 782149 default median=1136 µm (wrong basin), i2_crop600
   median=267 µm (right XY bulk region) — yet r@20=0.000 both. The I2
   crop puts prediction bulk near truth, but GNC-TLS settles on wrong-
   assignment pairs at that location because F6 cosine + distance-based
   putative generator does not separate GT partners from distractors at
   the anisotropic scale in that crop region.

   On 755252 i2_crop600: SS=459 vs default SS=500 (close; both
   "tight"), but oracle r@20=0 for the crop — the basin's close in
   residuals-of-putative-pairs sense but the putatives themselves are
   wrong-pair by ID.

**Implication**: the bottleneck on stress subjects is NOT warm-start
quality. All four baselines and all I2 seeds put predictions in the
correct large-scale region for 755252/767022 (medians 100–400 µm); P1
then re-commits to a wrong-pair basin regardless of seed. This matches
S32's finding that 782149's centroid-ICP objective has no minimum at
truth, extended to 755252/767022: the *F6 putative ranking plus TLS
outlier rejection* is the bottleneck, not coarse localization.

## Decision

- **Accept** multi-start with SS ranking as safe infrastructure (no
  regression, 4/4 oracle-matched). Do not merge into production P1 yet,
  since it adds 2 I2 runs (≈ 8 s each) for no benefit.
- **Reject** I2-direct and I2-crop600 as seed sources for P1 on all 4
  benchmark subjects.
- **Redirect** to attacking the putative-ranking / TLS step, not the
  warm-start:
  - S41 idea A: feed M3 (mask-centroid hybrid with F4 per-cell IoU) as
    a per-pair feature into P1's scoring; IoU is identity-informative
    where F6 is not.
  - S41 idea B: reconsider P4 (spectral graph matching) on 755252 —
    spectral GM uses pairwise-consistency, not point-wise TLS, so its
    failure modes may differ from P1.
  - S41 idea C: G2 contrastive embedding on real data (if the F8 /
    synthetic-to-real gap can be reduced with a domain-adaptation trick
    like pia-aligned normalization).

## Artifacts

- `probe_multistart.py` — 3 seeds × 4 subjects = 12 P1 runs.
- `probe_multistart.log` — stdout with per-seed SS + GT metrics.
- `multistart.csv` — machine-readable results.
