---
name: S10 constrained-revival of v1 candidates
description: 2026-04-30 — P14/P14_full/P1 inside locked-frame overlap + roi_quality v5d τ=0.3; sum r@20 best=1.72 (P14_full), 1.6 (P14), 1.3 (P1); LP residual is the ceiling, not the matcher. None tie v2 ensemble's 2.95.
type: project
originSessionId: a6b80cf6-8ea2-4333-9758-4422dc184b49
---
S10 ran §4.1 of Grand Plan v3 — re-evaluated v1 P14 (Hungarian) and P1
(TEASER++/GNC-TLS) inside the LP-locked overlap, with HCR optionally
filtered by `roi_quality_stage2_binary_score_v5d ≥ τ` ({good, bad_ok}=1).

**Why:** the §4.1 thesis is that v1 candidates failed under v1's
unbounded HCR volume, and may succeed under crop + LP.

**Results (sum r@20 across 6 benchmark subjects, accept_radius=200,
expand_um=100):**

| method     | sum r@20 |
|------------|---------:|
| `P14_full` | 1.72     |
| `P14`      | 1.60     |
| `P1`       | 1.29     |
| v2 gated ensemble (reference) | 2.95 |
| v1 P14/P1 (no crop)           | 0.78 |

Crop alone lifts P14 ~2.2× over v1; the quality filter at τ = 0.3
slightly *hurts* P14 because GT-HCR survival is 88–99 % (drop = direct
recall loss, no upside for pose-only matchers).  P1's GNC-TLS does NOT
beat raw Hungarian inside the crop — TLS converges to a tighter wrong
basin on hard subjects.

**LP residual is the ceiling.** Median LP-vs-GT distance per subject:
790322 35 µm, 767022 50 µm, 788406 57 µm, 767018 61 µm, 755252 75 µm,
782149 94 µm.  Hungarian's r@20 ≈ 0.5 × `frac<50µm`.  No pose-only
matcher can rescue 755252 / 782149 (`frac<50` ≤ 0.023); they need
M3-ICP or §4.5 anchor seeding.

**How to apply:** §4.1 stop rule fires — P14/P1 do not earn promotion
to the production candidate menu.  Use the crop (overlap bbox + roi
quality) as input to the per-pair classifier (§4.2/4.3) and TPS
expansion (§4.5), not as a standalone matcher.  Quality threshold
τ = 0.3 keeps ≥ 88 % of GT-HCR on every subject and is the working
default for downstream sessions.

Files: `code/sessions/v3_S10_revival_pass/{run_constrained_revival.py,
log.md, constrained_revival_results.csv}`.  M4, P4, P15 deferred — they
share the same LP-residual ceiling.
