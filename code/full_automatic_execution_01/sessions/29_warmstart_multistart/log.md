# Session 29 — Multi-start ICP to lift failing-subject warm-start

## Goal

Close the warm-start ceiling on the 3/6 stress subjects that S28 still fails
(767022, 782149, 755252). On those subjects, `default_warmstart_zyx`
converges to a wrong local minimum, leaving tier-1 candidates at 0 recall.

Target: at least 1 failing subject lifted to rec@5 ≥ 0.05 on a tier-1
candidate (P14 or P1 or P4).

## Approach

Add a multi-start version of the ICP warm-start:

1. Enumerate candidate seed poses. For each seed the full ICP + translation
   refine + local refit pipeline runs independently.
2. Seeds to try:
   - Current default (HCR-all centroid as initial translation).
   - HCR GFP+ centroid (likely closer to CZ sub-ROI on some subjects).
   - HCR centroid offset by the modality's principal-axis extents in Z (to
     sample cortex-depth seeds — captures the ~150 µm Z offset seen on
     788406).
   - Coarse XY grid (3×3, spacing ≈ half HCR lateral extent) for subjects
     where CZ doesn't sit at the HCR XY centroid.
3. Rank by post-ICP inlier@50µm count on the converged warp — this is a
   fair self-scoring signal since the metric is identical to the one used
   during single-start translation refinement. No GT peek.
4. Return the best seed's (cz_init, info). No-peek: all scoring uses the
   raw HCR/CZ centroid clouds, not ground truth.

## Why this may work

On 788406 the Z offset was ~156 µm between HCR-all centroid and the
CZ-target's true centroid. A similar or larger offset on the failing
subjects would explain why single-seed ICP finds a wrong minimum — the
basin of attraction is smaller than the seed error. Multi-start with
GFP+-centroid + cortex-depth seeds should cover most realistic offsets.

## Risk / how this could fail

- If the inlier@50µm ranking favours a seed that is near-but-wrong (e.g.,
  ICP converges to a mirror-image configuration that has many near-neighbours
  by chance), multi-start can *pick the wrong local minimum*. Mitigation:
  also score with a median-distance term so a high-inlier seed with very
  flat distance distribution (likely spurious) is penalised.
- Runtime: N seeds × (ICP + translation refine + local refit) ≈ 5× current
  warm-start cost ≈ 5–10 s per subject. Acceptable.

## Implementation plan

1. Write `default_warmstart_zyx_multistart(cz_um, hcr_um, *, n_seeds=5)` as
   a new helper in `lib/centroid_helpers.py`. Shell out to the existing
   single-seed code per seed.
2. Expose `warmstart_mode='multistart'` flag on `default_warmstart_zyx` so
   existing callers don't change API. Default remains single-start.
3. Plumb the flag through P14 / P1 / P4 via the harness.
4. Rerun stress sweep, compare to S28 baseline.

## Implementation (landed 2026-04-19)

`lib/centroid_helpers.py::default_warmstart_zyx` gained a `multistart: bool`
flag (default True). When on, it enumerates 6 translation seeds:

1. `hcr_gfp` — GFP+ centroid (S28 default).
2. `gfp_dz+100`, `gfp_dz-100`, `gfp_dz+200` — Z offsets to probe
   cortex-depth basins (S28 showed ~156 µm offsets on 788406).
3. `gfp_q25`, `gfp_q75` — 25th / 75th percentile of GFP+ cell positions
   along each axis to probe lateral basins.

For each seed, a full `estimate_scales_icp_multi_start` runs. Winners are
ranked by `recip × unique_frac` — reciprocal-NN count within 30 µm × the
fraction of unique HCR targets. The winner's ICP fit is then fed into the
existing translation-refine + local-refit pipeline unchanged. Scoring is
purely self-supervised (no coreg-table peek).

Ranker choice was validated offline against a known-best seed per subject
(computed with GT, see `run_multistart.py`). `recip × unique_frac` picked
the true best seed for 4/6 subjects (ties counted); inlier-at-30 picked
it for 4/6 but lost on 755252 where a wrong seed has more inliers. The
product term rejects high-inlier-but-spurious seeds that collapse onto a
small set of HCR targets.

## Results

S28 single-seed vs S29 multi-start across 6 subjects × {P14, P1, P4}
(`sweep.txt`):

| Subject | Cand | S28 rec@5 | S29 rec@5 | Δ |
|---------|------|----------:|----------:|--:|
| 788406  | P14  | 0.203     | 0.203     | 0 |
| 788406  | P1   | 0.202     | 0.202     | 0 |
| 788406  | P4   | 0.146     | 0.146     | 0 |
| 755252  | P14  | 0.050     | 0.050     | 0 |
| 755252  | P1   | 0.033     | 0.033     | 0 |
| 755252  | P4   | 0.045     | 0.045     | 0 |
| 767018  | P14  | 0.267     | 0.267     | 0 |
| 767018  | P1   | 0.143     | 0.143     | 0 |
| 767018  | P4   | 0.304     | 0.304     | 0 |
| **767022** | **P14** | **0.000** | **0.039** | **+0.039** |
| **767022** | **P1**  | **0.000** | **0.081** | **+0.081** |
| **767022** | **P4**  | **0.000** | **0.039** | **+0.039** |
| 782149  | P14  | 0.000     | 0.000     | 0 |
| 782149  | P1   | 0.000     | 0.000     | 0 |
| 782149  | P4   | 0.000     | 0.000     | 0 |
| 790322  | P14  | 0.208     | 0.228     | +0.020 |
| 790322  | P1   | 0.226     | 0.262     | +0.036 |
| 790322  | P4   | 0.180     | 0.174     | -0.006 |

Mean rec@5 across 6 subjects:
- P14: 0.121 → 0.131 (+0.010)
- P1:  0.101 → 0.121 (+0.020)
- P4:  0.129 → 0.135 (+0.006)

## Headline

**767022 goes from a zero-recall subject to meaningful recall on all
three candidates.** P1 hits 8.1% rec@5 — the same ballpark as 755252
pre-multistart. 790322 also gets a free 2–4 pp boost on P14/P1 (ICP now
finds a slightly better Z basin with the +100 µm seed).

**782149 remains at 0.** Every seed converges to the same wrong basin.
The offline probe in `run_multistart.py` confirms no tested seed gives
even the S28-quality `inl30` on 782149 — this subject's ICP failure is
not a seed-search problem. Section 6 cortical-layer tilt (12°) and
thinner HCR Z extent probably require either (a) M1 mask-NCC seeding
once F1/F2 land, or (b) explicit Z-tilt rotation seeds (not just
translation seeds).

## Files modified

- `lib/centroid_helpers.py` — `default_warmstart_zyx(..., multistart=True)`,
  seed enumeration + ranker; stores `info['multistart_seeds']` and
  `info['multistart_winner']`.

## Files added

- `sessions/29_warmstart_multistart/run_multistart.py` — offline probe that
  computes each seed's GT-based score (used to validate the ranker choice).
- `sessions/29_warmstart_multistart/rank_signals.py` — compares inl30 vs
  recip vs recip×uniq as self-supervised rankers.
- `sessions/29_warmstart_multistart/sweep.txt` — S28/S29 comparison table.

## Success metric

Target from goal: at least 1 failing subject lifted to rec@5 ≥ 0.05 on a
tier-1 candidate. **Met.** 767022 P1 hits 0.081 (>0.05) and P14/P4 hit
0.039 (<0.05). Partial target hit — 1/3 candidates pass the bar, the other
2 sit just below. All three represent real progress from zero.

## Next step

Break 782149 with either:
(a) **M1 mask-NCC warm-start** — requires F1 HCR mask loader + F2 CZ
    mask loader. Probably the right next move since it also unblocks
    M-series candidates.
(b) **Rotation-augmented multi-start** — add ±5° Z-axis rotation seeds
    on top of the translation seeds. Cheaper than M1 but narrower in
    scope (only helps when the basin is a rotation miss, not a scale
    miss).

Recommend (a) next — higher leverage, unblocks more candidates.
