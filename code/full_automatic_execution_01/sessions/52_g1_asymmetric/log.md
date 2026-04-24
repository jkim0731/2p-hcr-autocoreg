# Session 52 — G1 asymmetric F8 retrain

**Status:** abandoned (2026-04-20)
**Goal:** fix S38 G1's 0% real-data recall by switching F8 from symmetric (source ≈ target size) to asymmetric (source ~150 pts inside target ~4200 pts) to match CZ⊂HCR count asymmetry.

## Changes landed

1. **`lib/synthetic_warps.py::sample_asymmetric_warped_pair`** — new sampler. Source = small cube (~400 µm, ~150 GFP+ cells on 788406); target = source cube + margin (~1200 µm, ~4200 cells). Correspondence via `tgt_idx_of = {global_src_idx: position_in_tgt_array}`. Validated on real 788406 HCR: n_src=124-200, n_tgt=4163-4292.
2. **`bench/candidate_impls/_g1_gnn_matcher.py`** — added `asymmetric=True` default + progress logging every 100 iters + hidden/n_layers/cross_layers kwargs.
3. **`sessions/52_g1_asymmetric/probe_single.py`** — single-subject runner with `use_f6`/`asymmetric` CLI args.

## Benchmark (788406, hidden=96, n_layers=4, cross_layers=3)

| config | n_iter | wall (s) | final loss | r@5 | r@10 | r@20 | median err (µm) |
|---|---|---|---|---|---|---|---|
| asymmetric, use_f6=True  | 200 | 328 | 4.950 | 0.000 | 0.000 | 0.000 | 922 |
| asymmetric, use_f6=False | 300 | 437 | 4.987 | 0.000 | 0.000 | 0.000 | 1105 |

Baseline (S38, symmetric): 56 % synthetic-val hit, 0 % real-data recall.

## Diagnosis — why asymmetric doesn't converge

Loss formulation in `_pair_loss`: `(sum_matched(-log P[i,j]) + sum_unmatched_src(-log P[i,-1]) + sum_unmatched_tgt(-log P[-1,j])) / (n_a + n_b)`.

With n_a ≈ 150, n_b ≈ 4200, n_matched ≈ 100:
- matched contribution (~100 terms, perfect match → 0): small.
- unmatched-source dustbin (~50 terms × -log(~0.99) ≈ 0): small.
- **unmatched-target dustbin (~4000 terms × -log(1/151) ≈ 5) ≈ 20 000**, divided by 4350 ≈ **4.6**.

This dustbin-column term is essentially **fixed** by Sinkhorn's normalization — the model has almost no gradient path to push every single target's dustbin probability to 1.0 simultaneously. So loss saturates at ~4.95 regardless of how well the model actually matches.

Symptoms confirm: loss 4.987 → 4.950 → 4.987 across 100/200/300-iter checkpoints (oscillation, no downward trend).

## Root-cause triage (what would actually fix G1)

1. **Loss rewrite.** Replace the uniform `sum(-log P[-1,j])` with InfoNCE over matched embeddings (dropping the per-target-dustbin term). Or weight the loss by `1 / n_b` on target-unmatched and `1 / n_a` on source-unmatched so the total is balanced by set.
2. **Domain gap CZ↔HCR.** Even S38's symmetric version trained to 56% synthetic-val with 0% real transfer. Training on HCR↔HCR warps teaches geometric neighborhood matching on **HCR-density clouds**; real CZ has different density, different selection bias (CZ cells chosen via microCT slicing → higher-quality subset), and different feature distribution. Training distribution must include CZ-like dropout + feature noise.
3. **Feature alignment train↔inference.** Training uses `_simple_features` (16 dims: norm-k-NN distances + elevation angles). Inference with `use_f6=True` z-scores F6's 41 invariant features to HCR stats and truncates to 16 dims. These are **different feature columns** — the model never sees these at train. Tested `use_f6=False`: still 0 recall (not the dominant cause, but is a real bug).

Fixing all three would take multiple sessions. Given P1+β and P4+β already landed at r@20 ≈ 0.17–0.20 (S47, S51), the tier-2 plan (Grand Plan Section 9.4) calls for P6 BCPD / B1-B2 rather than deep G-series rework.

## Decision

**Abandon S52.** G1 requires a full re-architecture of the loss + feature-alignment + domain-adaptation pipeline — treat as tier-3 rework (Grand Plan Section 9.5) if P6/B-series don't close the gap. For now, proceed to S53 (P6 BCPD) per Section 9.4.

## Next up

- **S53 P6 BCPD** — Bayesian CPD on HCR GFP+ centroids with P1 warm-start (tier 2, likely high-value).
- If BCPD plateau too: S54 B1/B2 seed-constellation + TPS expansion.
