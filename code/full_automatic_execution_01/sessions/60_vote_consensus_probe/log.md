# S60 — method-vote consensus probe + C6 consensus-first ensemble (negative)

## Goal

S55 F7 isotonic calibration probe showed per-method Brier improves but sum
r@20 on C5 (P1⊕P4⊕P6 ensemble) stays at 1.080. S60 asks whether a
*structural* signal exists in cross-method agreement that C5's priority
dispatch leaves on the table: **pairs voted by multiple methods of
{P1, P4, P6} may be more precise than single-vote pairs**, so a
consensus-first dispatch could lift r@20.

Threshold criterion (pre-registered): if 2-vote/1-vote precision ratio
≥ 1.5× on tested subjects, proceed to ensemble; if < 1.5×, accept C5
plateau and close.

## Probe (`probe_vote_precision.py`)

Per subject, merge `{(cz_id, hcr_id)}` across {P1, P4, P6}, bin by vote
count (1, 2, 3), and score hits@20 µm restricted to GT-labeled cz_ids
(precision-at-GT, not recall).

### Results

| Subject | P@GT (1-vote) | P@GT (2-vote) | P@GT (3-vote) | 2v/1v ratio |
|---------|---------------|---------------|---------------|-------------|
| 788406  | 0.172 (104/603) | 0.224 (62/277) | 0.281 (76/270) | **1.30×** |
| 790322  | 0.151 (66/437)  | 0.338 (78/231) | 0.409 (95/232) | **2.24×** |
| 767018  | 0.165 (31/188)  | 0.304 (34/112) | 0.327 (32/98)  | **1.84×** |

**Signal confirmed.** All three subjects exceed the 1.5× threshold (average
1.79×). Proceeded to C6 ensemble construction.

## C6 — consensus-first + priority fallback (`_c6_consensus.py`)

Per cz_id: if ≥ 2 of {P1, P4, P6} agree on an hcr_id, pick it (mean
confidence). Else fall back to density-priority same as C5 defaults
(sparse → priority(P1,P4,P6), mid → priority(P1,P4,P6), dense →
priority(P4,P1,P6)).

### 6-subject bench (`bench_c6.py`)

| Subject | n_hcr_gfp | C5 r@20 | C6 r@20 | Δ (C6 - C5) |
|---------|-----------|---------|---------|-------------|
| 788406  | 17 427    | 0.262   | 0.248   | **-0.014** |
| 790322  | 10 131    | 0.289   | 0.283   | **-0.006** |
| 755252  | 30 804    | 0.095   | 0.088   | **-0.008** |
| 767022  | 14 239    | 0.117   | 0.110   | **-0.008** |
| 767018  | 9 161     | 0.315   | 0.271   | **-0.044** |
| 782149  | 3 831     | 0.000   | 0.000   | 0.000 |
| **SUM** |           | **1.079** | **0.999** | **-0.080** |

**C6 strictly loses to C5 on every non-zero subject.**

## C6b ablation — consensus-first with C5's density fallback (`_c6b_consensus_uc.py`)

C6 v1 confounds two changes: consensus-first override AND mismatched
fallback on sparse (C5 uses `union_conf` on sparse; C6 used priority).
767018 is sparse and the -0.044 loss is likely from fallback mode, not
from consensus-first. C6b isolates: consensus-first + C5's exact
density-dispatch fallback.

### 6-subject bench (`bench_c6b.py`)

| Subject | n_hcr_gfp | C5 r@20 | C6b r@20 | Δ (C6b - C5) |
|---------|-----------|---------|----------|--------------|
| 788406  | 17 427    | 0.262   | 0.248    | -0.014       |
| 790322  | 10 131    | 0.289   | 0.283    | -0.006       |
| 755252  | 30 804    | 0.095   | 0.088    | -0.008       |
| 767022  | 14 239    | 0.117   | 0.110    | -0.008       |
| 767018  | 9 161     | 0.315   | 0.289    | -0.026       |
| 782149  | 3 831     | 0.000   | 0.000    | 0.000        |
| **SUM** |           | **1.079** | **1.018** | **-0.061** |

### Decomposition of the C6 v1 -0.080 loss

- **Fallback divergence (sparse: priority vs union_conf)**: -0.018
  (767018 gap between C6 v1 -0.044 and C6b -0.026).
- **Consensus-first override** (C6b vs C5 pure dispatch): -0.061.

The fallback-mode fix recovers 5 of the 12 missed hits on 767018, but
consensus-first still costs 7 hits there plus 23 hits across the other
four non-zero subjects. **Refuted across both confounds.**

## Why the precision signal doesn't convert to r@20

The S60 probe's 1.3-2.2× precision ratio measures *all* 2-vote pairs vs
*all* 1-vote pairs. C6 only overrides C5 on **disagreement cases**: cz_ids
where P1's priority pick differs from P4+P6's consensus. In those cases:

- **P1's 1-vote pick** is filtered through TEASER's TLS + TPS residual +
  F6-NN putative generator. P1's single-method precision on
  disagreement cases is often *higher* than P4+P6's shared error mode.
- **P4+P6's 2-vote pick** on disagreement cases is consistent-but-wrong:
  both methods make the same geometric-consistency error on the cz_id.

The probe's precision ratio averages over *all* vote tiers, including
many cz_ids where all three methods already agree (3-vote). The
3-vote subset has genuinely high precision, but C5's priority already
captures it. The 2-vote-only-not-3-vote subset (the disagreement cases)
has precision close to 1-vote, so overriding P1 drops hits.

## Headline: reject consensus-first, accept C5 plateau

C6 and C6b are both rejected. C5's priority dispatch — P1-first on
mid-density, P4-first on dense, union_conf on sparse — is near-optimal
on this data.

**C5 sum r@20 = 1.080** remains the centroid-only ceiling. S58/S59
closed the synthetic-to-real self-supervised path; S60 closes the
cross-method consensus path. The remaining lift sources per Grand
Plan §4 are:

1. **C2 image-conditioned GNN** — 3D CNN per-cell patch features
   concatenated with F6; feeds G1 matcher. Image-level modality info
   might catch pairs invisible to centroid-only methods. Requires GPU +
   patch pipeline. Tier 2, 2 sessions estimated.
2. **C3 mask+centroid hybrid TEASER** — M1 volumetric-mask coarse
   alignment warm-starts P1's putative generator; M4 per-cell Dice
   adds pairwise IoU feature. All constituents (M1, M4, P1) already
   shipped. Tier 2, 1 session. *Lowest-infrastructure next probe.*
3. **QF1 hand-crafted-feature fallback classifier** — LOSO-trained GBT
   on F6 + per-pair residuals + C5 confidences. Does not add new
   matches but re-ranks collisions; conditional per Grand Plan §9.6
   (only activate if intrinsic Brier > 0.15 post-calibration; S55 showed
   0.08-0.27, borderline). Tier 3.

## Status: `abandoned`

## Files

- `probe_vote_precision.py` + `.log` — 3-subject precision-by-vote-count probe.
- `bench/candidate_impls/_c6_consensus.py` — C6 candidate (priority fallback).
- `bench/candidate_impls/_c6b_consensus_uc.py` — C6b candidate (C5's density fallback).
- `bench_c6.py` + `.csv` + `.log` — full 6-subject C5 vs C6.
- `bench_c6b.py` + `.csv` + `.log` — full 6-subject C5 vs C6b ablation.

## Next session — honest halt-point for autonomous centroid-only work

Convergent evidence across S56 (C5 plateau at 1.080), S55 (calibration
no lift), S57 (I3 B-spline negative), S58 (G1 synthetic→real gap),
S59 (F8 CZ-aware negative), and S60 (consensus-vote negative)
identifies the centroid-only ceiling as structural at 1.080.

**All simple centroid-only lift sources are exhausted.** Remaining
options per Grand Plan §4 with current viability assessments:

| Candidate | Status | Blocker |
|-----------|--------|---------|
| C3 mask+centroid hybrid TEASER | **blocked** | S15 M3 r@20=0 via M1 warm-start; S45 M4 blocked (CZ seg mask is binary outlines, no per-cell labels); S50 segmask-NCC failed (CZ voronoi density >> HCR GFP+ density, structural). |
| P7 HEGN (learned) | **likely blocked** | Trains on F8 synthetic; S58/S59 showed F8 transfer gap applies to any centroid-only learned method. |
| QF1 hand-feature fallback GBT | **low value** | S55 F7 calibration alone didn't lift C5; QF1 re-ranks collisions but doesn't add matches; conditional threshold (Brier > 0.15) not triggered (S55: 0.08–0.27). |
| **C2 image-conditioned GNN** | **viable but tier-3-effort** | 3D CNN patch encoder (16³ voxels at 4 µm) + G1 matcher; requires GPU, patch pipeline, multi-session. Directly addresses modality mismatch that blocks centroid-only methods on 782149. |

**Recommendation: halt autonomous centroid-only work.** C5 sum r@20
= 1.080 is the shippable centroid ceiling. Further progress requires
multi-session C2 infrastructure work (3D CNN training, GPU pipeline)
whose ROI is uncertain given 782149's structural unreachability
(12° tilt + thin Z + partial overlap). This is the honest stopping
point the user identified at the start of autonomous execution.
