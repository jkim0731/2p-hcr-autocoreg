# Plan — Step 3 v2 redesign (two paths + final-round addendum)

## Context

Step 3 v1 ran but acted only in round 0 (R_cand collapsed to 15 µm afterwards; TPS residual is structurally ≈ 0 at every landmark and useless as a signal; per-pair local image NCC is weak in dense tissue). User wants v2 that actually iterates and that **uses the same mechanism in every round** so round 0 isn't a special case.

**Framing (per user, 2026-05-20):** Round 0 is a **non-rigid registration pass**. Its output is the TPS landmark set that anchors all subsequent rounds. Because TPS is sensitive to outlier landmarks, round 0 needs filtering on top of mutual-best — without it, wrong matches contaminate the TPS warp and distort everything downstream.

The two paths the user proposed:

* **Path A — centroids only.** Each round: soma-print → mutual-best → (some filter) → refit TPS → repeat. Round ≥ 1 candidate set = K=10 mutual-K-NN. No image data.
* **Path B — Path A + LOO image NCC δ.** Same as A but additionally drop pairs whose `Δ_NCC = NCC_with − NCC_without < 0` over a *large neighbourhood cube* (initial threshold 0; relax to −0.1 if too strict).

This plan also has to address three open points the user raised:

1. **Wang's LR-threshold is too noisy on our data — confirm empirically and write it down.**
2. **The same-mechanism constraint means no Wang-anchor descriptor in round 0 and no prior-round anchor-vote in round 0 either; the natural round-0 candidate is self-referential anchor-vote, but it has logical issues — critique them carefully.**
3. **Optional final addendum round with Wang's anchor-restricted descriptor as a falsifiability test** — compare against m-NN descriptor in the same round.

## 1. Empirical check: is Wang's LR threshold noisy on our pool sizes?

Diagnostic experiment (one-off, before launching the main iteration):

* For each subject, run Step 2.5's mutual-best procedure to get soma scores.
* For each CZ cell, compute its best soma score and its 2nd-best soma score across its candidate set.
* Build two histograms per subject:
  * Best-match scores → expected to be a 2-Gaussian mixture (correct + incorrect).
  * 2nd-best scores → expected to be a single Gaussian (incorrect-only).
* Fit both with `sklearn.mixture.GaussianMixture`; record AIC/BIC, log-likelihood, and component means/widths.
* Diagnostics:
  * Visual: does the best-score histogram show a clean bimodality?
  * Component overlap (Bhattacharyya distance between the "correct" and "incorrect" Gaussians of the best-score mixture).
  * Stability: bootstrap-sample 50% of the cells, refit, see how much component means shift.
* Compute Wang LR for every candidate pair; threshold at LR < 0.05; report:
  * Per-subject accept count vs Step 2.5 mutual-best count.
  * Per-pair agreement between LR<0.05 acceptance and mutual-best.

**Predicted outcome (the reason we diverge from Wang):**

* 755252, 767018 — likely the worst — score distributions are not cleanly bimodal due to high contamination by non-GT-matchable pairs; LR fit gives unreliable component parameters; LR<0.05 set differs from mutual-best by a large fraction.
* 790322, 788406 — the best — distributions show cleaner bimodality and LR<0.05 ≈ mutual-best.

If the diagnostic confirms this, the plan is justified in dropping LR<0.05 and using **mutual-best** as the universal gate.

Output: `outputs/step3_v2_diag/lr_diagnostic_<sid>.png` + `lr_diagnostic_summary.csv` with the component means, BIC, overlap, and accept-set agreement.

## 2. Critique: anchor-vote 5/5 in round 0 (self-referential)

The user pushed back that anchor-vote should be usable in round 0 too. The only round-0-capable version is **self-referential**: filter M_t_raw (the round's mutual-best set) by whether each pair's 5 supporting vector-pair indices name cells that are themselves in M_t_raw. Below are the real logical issues with this:

**Issue 1 — Circularity / no independent evidence.**
In round 2+, the "anchor set" is the output of a *previous* round's filtering — TPS-validated, score-validated, an independent quality check. Self-referential anchor-vote in round 0 uses only the current round's mutual-best to validate the current round's mutual-best. A pair survives iff its own neighbours survive, and their survival depends on the same circle. This finds **self-consistent cliques** but doesn't bring any independent quality information.

**Issue 2 — Self-amplification of correlated errors (the FAILURE MODE the user already saw).**
Earlier visualisation showed that many real-FP matches are flow-conforming — their soma-print scores agree because the descriptor is confused by similar-looking nearby cells, and so are their neighbours'. A cluster of wrong-but-mutually-supporting matches will **all pass** self-referential anchor-vote: each one's 5 best vector-pair-indices name cells that are also wrong-but-mutually-supporting. Anchor-vote in this circular form does not detect the very failure mode it was meant to catch.

**Issue 3 — Density bias.**
A correct match in a sparse region has 5 supporting vector-pairs whose implicit cells may or may not be mutual-best (depends on how many of cz_i's 15 m-NN happen to also be mutual-best). In dense, well-matched regions, almost every supporter is a mutual-best, anchor-vote = 5/5 trivially. In sparse regions, even correct matches lose support. Self-referential anchor-vote therefore **biases toward dense clusters** and systematically drops correct-but-isolated matches.

**Issue 4 — Regime-dependent strictness.**
If round-0 mutual-best already contains >70 % of CZ cells, self-referential anchor-vote is mostly self-fulfilling (most pairs find their 5 supports in M_raw). If mutual-best is sparse (e.g., 25 % of CZ cells), the filter collapses to a small inner clique and may strip out the majority of correct matches. The same filter, applied identically, behaves very differently across subjects.

**Conclusion: drop self-referential anchor-vote in round 0.**
Use mutual-best + a *non-circular* filter in round 0 (see §2b below). Anchor-vote only becomes informative once an **independently-validated** accepted set exists from a prior round.

## 2b. Round-0 filter for Path A — local-flow consistency (= LOO TPS residual)

Since round 0 is a registration pass, its filter should target the kind of outlier that breaks TPS:

* For each pair `(i, j)` in `M_0`, compute its displacement `v_ij = hcr_j − cz_lp_i` in HCR µm.
* Find the K=10–20 nearest CZ-side neighbour pairs in `M_0`.
* Compute the local-median displacement `v_local = median(v over those K)`.
* Compute residual `r_ij = ||v_ij − v_local||₂` in µm.
* Filter: keep `(i, j)` if `r_ij ≤ p90(r)` of the subject's own distribution.

This is mathematically equivalent to a **leave-one-out TPS residual**: the displacement implied by `M_0 \ {(i, j)}` at position `i` minus this pair's own displacement, evaluated locally. It directly answers the question TPS cares about ("does this pair agree with the warp implied by everyone else?") without the round-0 self-referential circularity of anchor-vote.

**Limitations (honestly):**

* Can't catch flow-conforming FPs (real-FPs whose displacement happens to agree with neighbours' — the failure mode visible on 755252, 767022, 790322 in earlier diagnostics). Those require image evidence (Path B).
* Sparse-region penalty: if a CZ cell has fewer than K neighbours in `M_0`, the local-median is from a wider region and noisier. Mitigate by requiring at least 5 of the K to be within some radius (e.g., 100 µm).
* Threshold choice `p90` is a design parameter; sweep {p80, p90, p95} only if first results look off.

Earlier diagnostics showed GT pairs sit at deviation 5–9 µm median, 11–17 µm p90; real-FPs that disagree with flow sit higher. So `p90`-based threshold typically lands at 15–20 µm, naturally separating clear outliers without GT info.

**Alternative considered: top-50th percentile by soma score.**
Critique: v1 already showed q=0.5 on combined-confidence drops correct matches (790322 lost ~200 GT). Soma score isn't aligned with TPS quality — pairs with moderate soma score can still have perfect displacement. Use only as fallback if local-flow filter strips too aggressively.

This means the "same mechanism every round" constraint can't be strictly enforced *if* we want anchor-vote — round 0 is necessarily different (no anchors yet to vote against). The natural design instead is:

* **Round 0** = mutual-best only.
* **Round 1+** = mutual-best + anchor-vote 5/5 against `M_{t-1}` (prior round's accepted set, NOT self).

This loses the surface "same mechanism" symmetry but is logically clean. Path B can additionally have LOO image NCC δ from round 0 onward, since LOO image NCC δ doesn't have the circularity problem (TPS is independent of the descriptor evidence).

## 3. The two paths (locked-in form)

### Path A — centroids only

```
Round 0 (bootstrap / nonrigid registration pass):
    cz_cur = cz_lp_um                          # locked frame
    candidates per CZ_i = HCR cells within R_cand_um (Step 2.5 formula)
    D[i, j] = soma_score(i, j; m_cz=15, m_hcr=30, n=5)
    M_0_raw = mutual_best(D)

    # Local-flow filter (§2b) — keep pairs whose displacement agrees with neighbours
    for (i, j) in M_0_raw:
        v_ij = hcr_j_pos - cz_lp_i_pos
        K nearest CZ-side neighbours of i within M_0_raw
        v_local = median of those K neighbours' v
        r_ij = ||v_ij - v_local||
    accepted_0 = {(i, j) : r_ij <= p90(r)}     # data-derived threshold
    TPS_0 = fit per-axis Rbf(thin_plate) on accepted_0

Round t ≥ 1:
    cz_cur = TPS_{t-1}(cz_lp_um)
    candidates per CZ_i = K=10 nearest HCR in HCR µm
    candidates per HCR_j = K=10 nearest CZ_cur in HCR µm
    D[i, j] = soma_score for pairs in mutual K-NN intersection
    M_t_raw = mutual_best(D)
    M_t = {(i, j) ∈ M_t_raw : anchor_vote(i, j; accepted_{t-1}) == 5/5}
    accepted_t = M_t
    TPS_t = fit on accepted_t
    if |accepted_t Δ accepted_{t-1}| / |accepted_{t-1}| < 0.02: stop
    if t >= MAX_ROUNDS = 5: stop
```

### Path B — Path A + LOO image NCC δ from round 0

```
Round 0:
    Same as Path A round 0 to get M_0_raw.
    Apply local-flow filter to get pre_accepted (registration outlier pre-filter).
    For each (i, j) in pre_accepted:
        TPS_plus  = fit Rbf on pre_accepted                   # has (i, j)
        TPS_minus = fit Rbf on pre_accepted \ {(i, j)}
        region    = 80-µm cube at hcr_j's HCR µm position
        warped_plus  = sample CZ image at region via TPS_plus
        warped_minus = sample CZ image at region via TPS_minus
        hcr_patch    = HCR-488 at region
        Δ_NCC(i, j)  = pearson(warped_plus, hcr_patch) − pearson(warped_minus, hcr_patch)
    accepted_0 = {(i, j) in pre_accepted : Δ_NCC ≥ 0}
    TPS_0 = refit on accepted_0

Round t ≥ 1:
    Same as Path A round 1+, except after anchor-vote 5/5 filter, apply Δ_NCC ≥ 0 filter.
```

The local-flow filter is applied first in round 0 to give Δ_NCC a clean TPS baseline. This keeps the registration outliers out of the TPS that Δ_NCC's "with-pair" warp depends on.

(If Path B's filter is too strict, fall back to Δ_NCC ≥ −0.1.)

## 4. Final addendum round — Wang's anchor descriptor as a falsifiability test

After Path A or Path B converges (let's call the converged set `M_*` with TPS `TPS_*`), run **one additional round** with Wang's anchor-restricted descriptor:

* For each CZ cell, take its n=10 nearest cells **that are in M_***. Those n cells are its anchors. On the HCR side, their matched HCR partners are the HCR anchors (one-to-one correspondence).
* Build each cell's Wang-anchor descriptor.
* Score every candidate `(cz_i, hcr_j)` in the K-NN candidate set using the Wang-anchor scoring (mean Euclidean over n=10 paired vectors, no n-best selection).
* Take mutual-best on the new scores.
* Filter by anchor-vote 5/5 against `M_*`.
* Call this `M_Wang_final`.

**What to compare against (the test):**

Without rerunning the iteration: in the SAME final round, also compute the **m-NN score** for the same candidates (the score Path A would have produced), take its mutual-best, apply the same anchor-vote 5/5 filter against `M_*`. Call this `M_mNN_final`.

So we get two sets:
* `M_Wang_final` — from Wang's anchor-restricted descriptor.
* `M_mNN_final` — from m-NN descriptor (= one more iteration of Path A, no descriptor change).

**Comparisons to draw (GT used only for reporting, never to tune):**

| Diff | What it means |
|---|---|
| (`M_Wang_final` ∖ `M_mNN_final`) | Pairs the Wang descriptor uniquely promotes. Are they GT-correct? If yes, Wang's descriptor change is buying us something. |
| (`M_mNN_final` ∖ `M_Wang_final`) | Pairs the Wang descriptor uniquely rejects. Are they real-FPs that the Wang descriptor catches? |
| Per-pair score correlation Wang ↔ m-NN | If they agree (high rank correlation), Wang's descriptor change is doing nothing new — m-NN is enough. If they disagree on a meaningful subset and Wang's set is more accurate, the descriptor change matters. |
| Final precision and recall | `M_Wang_final` vs `M_mNN_final` vs `M_*` (the pre-addendum converged set). Did the addendum round move us off `M_*`? In which direction? |

**Decision rule:**

* If `M_Wang_final` matches `M_mNN_final` within 2 % of `|M_*|`, the Wang descriptor change isn't necessary for our data — anchor-vote on m-NN does the same work.
* If `M_Wang_final` is strictly better (more GT-agree, fewer real-FPs), the Wang descriptor change matters and should be promoted as part of the iteration (every round, not just the addendum).
* If `M_Wang_final` is strictly worse, the m-NN descriptor is fine and Wang's restriction hurts.

This is a *single extra round* of compute (a few minutes), keeps the main iteration simple, and gives us a clean go/no-go on Wang's descriptor change without complicating Path A or B.

## 5. K-NN candidate set (round ≥ 1)

Replaces v1's R_cand_um radius. For each CZ cell i (under current TPS warp), candidate HCR partners = K=10 nearest HCR cells by HCR-µm Euclidean. For each HCR cell j, candidate CZ partners = K=10 nearest CZ-TPS-warped cells. Score only pairs in the mutual K-NN intersection (j ∈ KNN(i) AND i ∈ KNN(j)) — that's the only set where mutual-best is even possible.

K=10 is the first-trial value; sweep to {5, 20} only if first results are unstable.

## 6. Files to create

```
code/sessions/15_geom_features/
├── run_step3_v2_diag_lr.py      # Wang LR diagnostic per subject
├── run_step3_v2.py              # main driver with --path {a, b} and --final_addendum_wang
├── loo_image_ncc.py             # LOO image NCC δ helper (Path B only)
```

Reuse from existing code (already present):

| Need | Where |
|---|---|
| Subject prep, R_cand from data, Z bounds, pia anchor | `run_step2p5_refined.py::prepare_subject` |
| Soma score + neighbour indices for anchor-vote | `run_step3_iterative.py::soma_score_with_neighbour_indices` |
| Anchor-vote computation | `run_step3_iterative.py::anchor_vote` |
| TPS fit / apply | `run_step3_iterative.py::fit_tps, apply_tps` |
| Mutual-best on score matrix | `run_step3_iterative.py::mutual_best_pairs` |
| CZ image loader | `code/dev_code/cz_volume.py::load_cz_volume` |
| HCR image loader | `code/dev_code/benchmark_analysis.py::load_hcr_volume` |

## 7. GT-free discipline

* `is_gt` used ONLY in the final summary CSV — never to set thresholds, weights, K, R_cand, cube size.
* `anchor_vote` uses only the *prior round's* accepted set (no GT). Round 0 has no anchor-vote (per the critique in §2).
* `Δ_NCC` uses only image content and TPS (no GT).
* Default filter values (`< 0`, K=10, 80-µm cube) are universal, not GT-tuned.
* Wang LR<0.05 diagnostic is informational only — even if it agrees with mutual-best, we don't adopt it as a tuned gate.

## 8. Outputs

```
outputs/step3_v2_diag/
    lr_diagnostic_<sid>.png              # mixture-fit visualisation
    lr_diagnostic_summary.csv            # AIC/BIC, overlap, accept-set agreement

outputs/step3_v2_path_a/<sid>/
    matches_round{0..T}.csv              # is_gt flag in column for reporting only
    rounds_log.csv                       # |M|, |Δ|, n_added/removed, n_filtered_by_anchor_vote
    tps_warp_round{0..T}.pkl
    matches_final_wang_addendum.csv      # Wang-anchor addendum result
    matches_final_mNN_addendum.csv       # m-NN equivalent for the comparison

outputs/step3_v2_path_b/<sid>/
    (same + Δ_NCC column per matched pair)
```

## 9. Verification

* Smoke test on **790322** (cleanest) for each path before full sweep.
* LR diagnostic must run on all 6 subjects before either path is launched (cheap, ~1 min total).
* Sanity check that round 0 in Path A gives exactly the same match set as Step 2.5 mutual-best (no filter, just descriptor + R_cand).
* Confirm GT-free by grepping driver scripts for `is_gt`, `coreg_table` references — only allowed in summary writeout.

## 10. Stop / abort

* `MAX_ROUNDS = 5`.
* Per-round `|M_t Δ M_{t-1}| / |M_{t-1}| < 0.02` → stop converged.
* `|M_t| < 20` → abort subject (filter too strict for this subject), record and continue.

## 11. Open questions to lock down before launching

1. **Round-0 local-flow K and threshold.** K=10–20 neighbours; threshold = p90 of own deviation distribution. Sweep K∈{10, 20} and threshold∈{p80, p90, p95} only if first results are unstable. Default K=15, threshold=p90.
2. **Round-≥1 K-NN size.** Default K=10. Sweep {5, 10, 20} only if 10 looks unstable.
3. **Path B cube size.** Default 80 µm. Sweep {50, 80, 120} only if results are noisy.
4. **Path B Δ_NCC threshold.** Default `0`. Relax to `−0.1` if too strict.
5. **Anchor-vote strictness in rounds ≥ 1.** Default 5/5 (all 5 supporting pairs must be in prior-round accepted set). Could relax to ≥ 4/5 if 5/5 collapses the match set.
