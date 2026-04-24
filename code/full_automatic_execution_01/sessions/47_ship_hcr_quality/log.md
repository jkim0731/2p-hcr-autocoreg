# S47 — Ship HCR image-quality bonus into P1 scorer

**Status:** validated. P1 default flipped to `hcr_quality_beta=5.0`.

## Goal

S46-b established that subtracting `beta * hcr_quality(j)` from the per-pair score
(where `hcr_quality(j)` is the within-HCR z-scored sum of channel-488 patch
[mean, std, p90, |laplacian|_mean] around each GFP+ centroid) lifts raw-argsort
GT-in-top-K on the stress subject 755252 by +14.9 pp at beta=5 (0.280 → 0.429,
K=20). S47 ships the bonus into the production P1 pipeline and measures whether
the raw-argsort lift survives the downstream GNC-TLS + TPS + one-to-one stages.

## Implementation

1. `lib/image_quality.py` — `hcr_quality(s)` returns a 1-D float array aligned to
   `centroids_um(s, "hcr_gfp")` row order. Cached per
   `(subject_id, channel, level, bbox_um)` in a module-level dict (per-centroid
   extraction is ~30–50 s per subject; the cache makes repeated sweeps cheap).
2. `bench/candidate_impls/_p1_teaser.py::_seed_putative` — added optional
   `hcr_quality`/`beta` kwargs. Score becomes
   `score = D - 25*cos_F6 - beta*hcr_quality(j)` then `argsort` top-K as before.
3. `run_p1` signature changed: `hcr_quality_beta: float = 5.0` (shipping default,
   was 0.0 during S46-b).

## Probes

### probe_p1_beta_sweep.py — K=5 × beta ∈ {0, 3, 5, 8} × 4 subjects

| subject | β=0 r@20 | β=3 r@20 | β=5 r@20 | β=8 r@20 | β=0 rec_id | β=5 rec_id |
|---------|----------|----------|----------|----------|------------|------------|
| 788406  | 0.210    | 0.207    | **0.234**| 0.203    | 0.202      | **0.225**  |
| 755252  | 0.044    | **0.056**| 0.044    | 0.047    | 0.033      | 0.038      |
| 767022  | **0.108**| 0.097    | 0.103    | 0.062    | 0.081      | **0.096**  |
| 782149  | 0.000    | 0.000    | 0.000    | 0.000    | 0.000      | 0.000      |

Cross-subject sum r@20: β=0 = 0.362 → β=5 = 0.381 (+0.019, +5.2 % relative).

### probe_k20_beta_sweep.py — does widening K unlock 755252 lift?

| subject | K=5 β=0 | K=5 β=5 | K=20 β=0 | K=20 β=5 |
|---------|---------|---------|----------|----------|
| 788406 r@20 | 0.210 | **0.234** | 0.180 | 0.193 |
| 755252 r@20 | 0.044 | 0.044     | 0.027 | 0.023 |
| 767022 r@20 | 0.108 | 0.103     | 0.127 | **0.126** |
| 782149 r@20 | 0.000 | 0.000     | 0.000 | 0.000 |

K=20 helps only on 767022 (+1.8 pp r@20 at β=0); it regresses 788406 by
3.0 pp and 755252 by 1.7 pp at β=0. TLS basin commitment on a 4× larger
putative pool makes the top-K widening a net loss. Keep K=5.

## Why the raw-argsort lift shrinks end-to-end

S46-b measured GT-in-top-K at K=20 on the raw per-CZ ranking. P1 end-to-end
r@20 is gated by three stages downstream of `_seed_putative`:

1. **K=5 truncation.** Putatives fed to GNC-TLS are the top-5 only; the 755252
   lift at K=5 (0.370 → 0.476, +10.6 pp) is smaller than at K=20 but still
   substantial. The full pipeline uses K=5.
2. **GNC-TLS basin.** The affine fit commits to the largest consensus set; a
   newly-promoted correct pair only helps if it also agrees with the rest of the
   inliers under a single anisotropic affine.
3. **One-to-one `drop_duplicates('hcr_id')`.** If a lifted correct pair and an
   incorrect pair target the same `hcr_id`, the higher-confidence (possibly
   wrong) one wins. 755252 has severe collisions due to low GFP+ contrast.

Net: on primary subject 788406 the lift converts cleanly (+2.4 pp r@20,
+2.3 pp rec_id). On 755252/767022 the conversion is weak or unstable. On
782149 the coarse alignment itself is wrong (median err ~1135 µm), so
downstream lifts can't help.

## Decision

Ship K=5, β=5.0 as the P1 default. No subject meaningfully regresses.
767022 r@20 loses 0.5 pp but gains 1.5 pp rec_id — acceptable. Primary
subject 788406 is the tiebreaker and it gains clearly.

## Files

- `lib/image_quality.py` (created)
- `bench/candidate_impls/_p1_teaser.py` (default `hcr_quality_beta=5.0`)
- `sessions/47_ship_hcr_quality/probe_p1_beta_sweep.py` + `.csv` + `.log`
- `sessions/47_ship_hcr_quality/probe_k20_beta_sweep.py` + `.csv` + `.log`

## Follow-ups queued (not part of S47)

- **S46-c** — cellpose-on-CZ-zstack to unlock M-series mask overlap.
- **S46-d** — alt coarse-alignment path for 782149 (1135 µm median err means
  the seed transform is off; HCR-quality bonus can't help when the crop is
  wrong).
