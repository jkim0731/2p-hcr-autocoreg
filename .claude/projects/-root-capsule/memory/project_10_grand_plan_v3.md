---
name: Grand Plan v3 — cell-cell matching + QC
description: Rough registration is DONE; v3 plan focuses on cell-cell matching, QC, optional volume refinement; saved at code/docs/10 ...
type: project
originSessionId: b738b938-e542-46a0-8ca4-0ef155358066
---
**Fact.** A v3 plan was opened on 2026-04-29 at `code/docs/10 Grand Plan v3 — Cell-cell matching and QC.md`. It supersedes the rough-registration sections of `09 Full automatic v2 plan.md` and reframes Stage D / E / F as cell-cell matching + QC subgoals (4.1–4.7).

Baseline at the time of writing: v2-S03 gated ensemble = sum r@20 = 2.95 vs v1 = 1.05; per-subject 755252=0.07 / 767018=0.86 / 767022=0.66 / 782149=0.16 / 788406=0.44 / 790322=0.76. Stage A (locked-prior warm-start) and Stage B (slab-rigid sz, 6/6 within ±0.30 GT) are shipped. Surface registration v2 (PWR 4×4) caches `crop_bbox` for 6/6.

v3 subgoals: (4.1) constrained revival of v1 candidates inside the locked overlap; (4.2) Track B per-pair image scores; (4.3) cell-cell matching classifier (GBT first → patch CNN / G1-retrain only if GBT misses AUC ≥ 0.92); (4.4) HCR ROI pass/fail classifier on **existing** segmentation (no cellpose rerun) with Stage-1 heuristic+synthetic pseudo-labels then Stage-2 human-uncertain-band refinement; (4.5) TPS expansion seeded by Stage D iter 8; (4.6) optional conservative volume refit when ≥ 30 pairs accepted; (4.7) GUI integration (central — v3 keeps the human in the loop).

**Operating mode (per user 2026-04-29).** v3 is *not* aiming for hands-off automation; the loop is "algorithm proposes → human reviews / accepts / re-anchors in the GUI". Full automation is post-v3.

**Session numbering (per user 2026-04-29).** v2 ended at `v2_S05_tps_expansion`; v3 sessions continue at **S06** and live under `code/sessions/v3_S0{N}_*/`. Lib code lands in `code/dev_code/` (no `full_automatic_execution_02/` dir under `code/`). Read-only v1/v2 references stay in `data/claude_data/`.

Recommended session order: S06 promotion → S07 pair scoring → S08 GBT classifier → S09 TPS expansion → S10 revival pass → S11 HCR ROI pass/fail (CPU) → S12 volume refit → S13 GUI → S14 patch CNN (GPU-gated, only if S08 misses AUC ≥ 0.92).

**Why:** rough registration ceiling broken by surface_registration_v2 + Stage A/B; the failing pieces in v1/v2 (cell-cell matching, QC) deserve constrained revivals because they were limited by global-pose ambiguity, not by the matchers themselves.

**How to apply:** treat this doc as the authoritative roadmap; consult §6 for session order; §5 for compute estimate (one GPU recommended; more CPU has diminishing returns); §7 for stop/pivot rules. v1 ledger in `07 Grand Plan.md` §11 is still the source of truth for past failure modes.
