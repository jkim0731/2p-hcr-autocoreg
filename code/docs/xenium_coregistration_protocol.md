# Protocol: co-registering a 2P cortical z-stack to Xenium slices
### (ported from the 2P↔HCR soma-print matcher — handoff to the Xenium agent)

_Written 2026-06-02. Audience: a Claude Code agent that already knows the Xenium
data and the soma-print paper (Wang et al.), but **not** our customised matching
approach. This document is that approach, translated to the **2D / thin-slice**
Xenium setting, with the **density-mismatch** handling spelled out (you do not
know this part)._

---

## 0. Goal & setup

- **Moving (in-vivo):** 2P cortical z-stack (CZ) — same modality we used before;
  dense GCaMP+ somata, FOV ≈ 700 µm, full cortical depth.
- **Fixed (ex-vivo):** **Xenium** — multiple **thin 2D slices, ~20 µm each**.
  **Register only the TOP slice**; once that single 2D registration is solved, the
  rest of the slices are already automated downstream. So this is fundamentally a
  **2D registration problem** (xy), not the 3D volume registration we did with HCR.
- **Warm-start prior:** the CZ stack is roughly **centred** on the sliced tissue,
  but the residual offset can be as large as **±200 µm** in xy. Tissue box ≈
  1.8 mm × 1.7–2.7 mm.
- **Shrinkage:** ex-vivo tissue is **~80 %** of in-vivo size. So the CZ→Xenium
  scale is **sxy ≈ 0.80** (apply this as the initial scale; see §5).

You will **not** use any image-intensity / MIP-NCC overlap registration. That is
deliberate — see §4 (it fails under density mismatch; that was our hardest-won
lesson). The whole pipeline is **geometric** (cell-centroid soma-print).

---

## 1. The core matcher (our approach, in 2D)

This is the iterative soma-print + anchor-vote matcher we validated. It runs
between **one CZ slab** (a thin depth band, treated as 2D in xy — see §3) and the
**Xenium top slice** (2D), warm-started from **one seed** (§5). Everything below is
**2D (xy)**.

### Per-round loop (identical every round except round 0 has no TPS yet)
```
warm-start: place CZ-slab cells in Xenium-xy via the seed (rigid offset + sxy≈0.8)
accepted ← ∅ ; tps ← None
for rd in 0 .. MAX_ROUNDS-1:
    if tps: cz_pos ← apply_tps_2d(tps, cz_slab_xy)        # re-centre on improving warp
    D     ← soma_score(cz_pos, xen_xy, R_cand=200 µm)      # 2D soma-print, radius candidates
    pairs ← mutual_best(D)                                 # symmetric best-best
    if rd == 0: pairs ← local_flow_filter(pairs)           # OPTIONAL round-0 outlier reject
    kept  ← anchor_vote_gate(pairs)                        # WITHIN-ROUND anchor-vote (§1c)
    accepted ← set(kept)                                   # RE-EVALUATE, do NOT lock (§1d)
    if |accepted| ≥ 3: tps ← fit_tps_2d(accepted)          # 2D thin-plate spline
    if rd>0 and rel_delta < 0.02: break                    # converged
```

### 1a. soma-print descriptor — 2D
For each cell, build its descriptor from the **displacement vectors to its
neighbours** (2D xy). Score a candidate pair `(cz_i, xen_j)` = the **mean of the
`n` smallest distances** between cz_i's neighbour-vector set and xen_j's
neighbour-vector set (this is exactly the paper's soma-print, in 2D). We used
`n = 5`. **Neighbour-set definition is critical under density mismatch — see §4.**

> **SWEEP the descriptor sizes — do not take them as fixed.** Treat both as
> parameters to tune on the Xenium data: (i) the **number of neighbour vectors per
> modality** `m_CZ`, `m_Xenium` — keep them **independent/asymmetric** (different
> densities ⇒ different optimal counts), and (ii) the **number of best-matching
> vector pairs `n`** used for the score. Our HCR values (`m_CZ=15`, `m_HCR=30`,
> `n=5`) were tuned for *that* data and condition; the soma-print paper's values are
> for *theirs*. Neither transfers — sweep `m` per modality and `n`, and pick by the
> GT-free quality score (§6). (Same lesson that drove our original parameter sweep:
> the condition is different, so re-sweep rather than inherit.)

### 1b. Candidate selection — mutual-best, every round
`mutual_best(D)`: keep `(i,j)` iff `j` is `i`'s best (lowest score) and `i` is
`j`'s best. Symmetric. This is your primary discriminator and is **density-robust**
(a surplus cell in the denser modality cannot form a mutual pair — see §4).

### 1c. Gate — WITHIN-ROUND anchor-vote (the validated winner)
For a candidate `(cz_i, xen_j)`, the soma-score is produced by its `n=5`
best-matching neighbour-vector pairs, which imply `n=5` **neighbour
correspondences** `(cz_a, xen_b)`. **anchor-vote = the fraction of those `n`
neighbour-pairs that are themselves in the CURRENT ROUND'S mutual-best set.** Keep
the pair if that fraction ≥ `ANCHOR_VOTE_FRAC` (we use **0.6 = 3/5**).

> **Do this within-round, NOT against accumulated/accepted matches.** Checking
> against the *current round's mutual-best* (not a propagated accepted set) removes
> the bootstrap circularity (fresh regions can seed themselves), avoids
> self-amplification of correlated errors, and lets the gate run at round 0. We
> tested the cross-round version — it is biased and over-conservative; don't use it.

We compared three gates (likelihood-ratio, local-flow, image-NCC). **anchor-vote
won** on recall *and* precision and is cheap — use it. (LR is high-precision but
loses ~30 % recall; image-NCC is QC-only and breaks under density mismatch.)

> **Threshold is data-dependent — TUNE it, don't port the constant blindly.**
> `ANCHOR_VOTE_FRAC` trades recall vs false positives: lower (3/5) = more recall +
> slightly more FP risk; higher (5/5) = fewer FP but lower recall. On *our* HCR data
> 3/5 gave high recall with only ~2 % genuine wrong-cell matches (the rest of the
> precision gap was unlabeled "silence" matches, mostly real). **But under Xenium's
> density mismatch, spurious mutual-best + neighbour agreement is more likely, so the
> FP risk is higher.** Therefore **start stricter for Xenium — 4/5 or 5/5 — and relax
> toward 3/5 only if recall is too low.** Treat it as a small sweep validated on the
> new data, not a fixed constant.

> **Final-round Wang anchor-descriptor — INCLUDE it as an optional polish (re-tested
> 2026-06-02).** After Stage-1 (anchor-vote) converges, run a Stage-2 that rebuilds
> each cell's soma-print from its ~10 *accepted-anchor* neighbours (Wang's iterative
> descriptor) and re-matches with the **same 3/5 within-round anchor-vote gate** +
> re-evaluation. Re-tested on our 4 valid subjects: **net-positive — +19 GT, 0 GT
> removed** (recall 0.958→0.966 under prior pose-dependent GT; corrected 2026-06-04 figures in `sessions/15_geom_features/outputs/corrected_gt_rescore/`; precision −0.7 pp, all silence-FP). ⚠️ An *earlier*
> result found it net-negative, but that was a **different regime** (5/5 cross-round
> anchor-vote, where wrong Stage-1 anchors poisoned the Wang descriptor). At 3/5
> within-round with a large, accurate Stage-1 set the descriptor gets clean reference
> pairs and only *adds* good matches the radius search missed. So run it as a final
> polish — but it's a *small* gain, so **validate on Xenium** (density mismatch may
> change this), and remember anchor-vote (gate) ≠ Wang (descriptor) — distinct
> mechanisms. Skip Stage-2 if Stage-1 yields too few anchors (< ~10).

### 1d. Accepted-set policy — re-evaluate, do NOT lock
Each round rebuilds `accepted` from the filters; a pair accepted in round t−1 can
be dropped in round t if it newly fails. We checked: this is **stable** (no
oscillation, no partner-switching) and loses ~no real matches. Locking would let an
early bad match ossify and poison the TPS — **don't lock.**

### 1e. R_cand — fixed physical radius, every round
**`R_cand = 200 µm`**, applied as a radius around the *current* (TPS-warped) CZ
position every round (value fixed; the centre re-centres as the warp improves).
This matches your warm-start residual (±200 µm). **Do not** use a k-NN candidate
count and **do not** shrink R_cand per round — a fixed radius is more robust to
locally-uneven warp than k-NN, and the descriptor (soma-print) is insensitive to how
many distractors the radius admits (mutual-best + anchor-vote filter them).

---

## 2. (omitted — no image registration; see §4)

---

## 3. The search structure (your design)

The CZ depth that corresponds to the Xenium top slice is uncertain, and so is the
xy offset. So you **search both** and keep the best result by a GT-free quality
score (§6).

### CZ-slab scan (resolves "which CZ depth = Xenium top slice")
- Build **CZ slabs 40 µm thick**, stepping with **20 µm overlap**, from the top
  **down to 100 µm from the top → 4 slabs** (e.g. 0–40, 20–60, 40–80, 60–100 µm).
- Each slab is thin enough to treat as a 2D xy point set for matching.

### Seed scan per slab (resolves the ±200 µm xy offset)
- **5 seeds per slab:** 1 at the **centre**, 4 at **±200 µm offsets**
  (e.g. ±200 µm in x and in y — a cross). These tile the ±200 µm residual region.
- **Apply the ~0.80 shrinkage** when placing seeds: the CZ FOV is 700 µm in vivo,
  so in the Xenium (shrunk) frame it spans ~560 µm; compute the seed offsets in the
  **Xenium frame** (i.e. scale CZ coordinates by sxy≈0.8 before adding the ±200 µm
  seed translation). Each seed = (sxy≈0.8 scale) + (one of the 5 xy translations).

### Total
4 slabs × 5 seeds = **20 runs of the §1 matcher** per Xenium top-slice. Pick the
single best `(slab, seed)` (§6).

---

## 4. DENSITY MISMATCH — the part you don't know (read carefully)

The two modalities **do not have the same cell density and do not have full cell
correspondence.** 2P sees all GCaMP+ cortical somata; Xenium sees cells that pass
its transcript panel + segmentation. Expect **different densities and only partial
overlap.** This breaks naive approaches. Here is what we learned:

1. **Never warm-start or register on image-intensity / MIP-overlap NCC under
   density mismatch.** Our single biggest failure (subject "782149") was exactly
   this: a *dense* moving modality vs a *sparse* fixed one drove the image-NCC to
   ~0.1, and the registration silently went wrong (and only "worked" earlier because
   it had secretly borrowed a ground-truth scale). **The geometric soma-print +
   seed approach in §1/§3 exists specifically to avoid this.** Do not reintroduce an
   image-NCC step "to refine."

2. **Define the soma-print neighbourhood so density does not distort it.** The
   descriptor compares displacement vectors to neighbours; if one modality is denser,
   its "k nearest" neighbours are physically closer → the vectors have a different
   scale → the descriptor mismatches even for a true pair. Two fixes (use either):
   - **Radius-based neighbours** (preferred under strong mismatch): use all cells
     within a fixed physical radius `R_nbr` (e.g. 60–100 µm) as the neighbour set on
     *both* sides — same physical scale regardless of density.
   - **Asymmetric k-NN**: if you keep count-based neighbours, use **more neighbours
     on the denser modality** (we used 15 on the sparse side, 30 on the dense side).
   Pick `R_nbr` so a typical neighbourhood holds ~10–30 cells on the *sparser* side.

3. **Let mutual-best + anchor-vote extract the common subset; do not force full
   matching.** The denser modality's surplus cells will not win mutual-best (no
   reciprocal partner), and anchor-vote demands neighbour-correspondence
   consistency. So the matcher naturally converges on the *consistent intersection*
   of the two populations. **Accept partial recall** — not every CZ cell has a
   Xenium counterpart and vice-versa. This is correct behaviour, not failure.

4. **If the densities differ by a lot, down-select to a comparable population
   first.** In our pipeline the analog was restricting the fixed modality to a
   quality-filtered subset before matching. For Xenium, consider filtering both
   modalities to a comparable-confidence / comparable-marker subset so the matched
   populations are density-compatible. Match on that subset, then (optionally)
   propagate the warp to all cells.

5. **The TPS only needs the consistent matches.** As long as you get enough
   mutually-consistent, anchor-vote-passing pairs spread across the FOV, the 2D TPS
   warp is well-determined — even if a large fraction of cells in each modality are
   unmatched. Quality of the *accepted set*, not its completeness, is what matters.

---

## 5. Warm-start / scale

- Initial CZ→Xenium scale: **sxy ≈ 0.80** (the ~80 % ex-vivo shrinkage). Apply
  isotropically in xy.
- Each **seed** = this sxy + one of the 5 xy translations (§3). No rotation prior
  assumed; the iterative TPS (and an optional small rigid pre-align inside round 0)
  will absorb modest rotation. If you expect a large rotation between the CZ FOV and
  the Xenium section, add a coarse rotation search to the seed grid.
- **Do not** try to estimate sxy from naïve full-span cell-footprint area ratios
  here — that was unreliable and is what failed for the sparse subject. Use the
  known shrinkage (0.80) as a fixed prior and let the 2D TPS handle residual local
  scale.

> **Cross-platform note — the HCR/CZ production sxy + MIP (promoted 2026-06-04).**
> On the HCR↔CZ z-stack platform, the production scale estimator is the **min-rule
> 2× ¼-FOV** rule (`roi_area_sxy.estimate_sxy_min_rule`):
> `hcr_slab = min(p99(HCR GFP+∩ok∩¼-FOV depth), 2·p99(CZ depth))`,
> `cz_slab = hcr_slab/2` (the CZ slab is HALF the HCR slab — HCR is axially ~2×
> expanded, so capping CZ shallower lowers its median footprint and raises sxy),
> `sxy = sqrt(median HCR max-xsection / median CZ max-xsection)`; the 2× is a
> heuristic, not the measured sz (sz needs a pose → circular). This is what made
> footprint-ratio sxy work where the full-span ratio failed — it recovered the
> sparse/thin subject (782149) GT-free. The registration MIP there was also
> thickened to **80/150 µm** (CZ 0–80, HCR 0–150), because a denser MIP lands the
> rigid/PWR fit for thin top-slab subjects.
>
> **It does NOT transfer directly to Xenium** because Xenium gives a *single thin
> slice*, not a depth profile, so there is no depth band to take a min-rule over
> and no axial-expansion factor to apply. Hence the fixed-0.80 prior above. The
> transferable idea is the **GT-free grid-search fallback**: where the HCR/CZ
> pipeline grid-searches base sxy at `SXY_GRID_SEARCH_OFFSETS =
> (−0.10,−0.05,−0.02,+0.02,+0.05,+0.10)` and keeps the pose that lands (largest
> mutual-best soma-print set / lowest off-centre), the Xenium **seed scan (§3, §6)
> is the same mechanism** — sweep candidate scales/offsets and pick the one whose
> converged accepted set is largest and most stable, never GT. If 0.80 ever proves
> off for a section, widen the seed grid to include scaled variants at those
> offsets and select GT-free exactly as §6 prescribes.

---

## 6. Choosing the best `(slab, seed)` — GT-FREE

You have no ground truth at run time, and **GT-precision is a misleading metric**
anyway (many "non-GT" matches are real). Rank the 20 runs by **GT-free quality**:

- **Primary:** the **size of the converged, mutually-consistent accepted set**
  (more anchor-vote-passing mutual-best matches = better registration). A correct
  (slab, seed) snaps to a large stable set; wrong ones collapse to few.
- **Tie-breakers / sanity:** mean anchor-vote support of accepted pairs; spatial
  spread of accepted pairs across the FOV (a good warp is supported everywhere, not
  one corner); low TPS leave-one-out residual on the accepted set.
- **Never** pick the seed by agreement with any held-out GT — use GT only to
  *validate* the chosen registration afterwards.

A correct registration should also be **stable across adjacent slabs/seeds** (the
two best (slab,seed) should give similar warps); use that as a confidence check.

---

## 7. Parameter summary

```
# 2D soma-print descriptor
neighbour set      : radius-based R_nbr ≈ 60–100 µm (density-robust; see §4.2),
                     or asymmetric k-NN. **SWEEP m_CZ and m_Xenium independently**
                     (per-modality vector counts; densities differ → optima differ).
n_best_vectors n   : **SWEEP** (HCR default 5; do not inherit — see §1a).
# candidate search
R_cand             : 200 µm, fixed, every round (matches ±200 µm residual)
# gate
gate               : within-round anchor-vote; ANCHOR_VOTE_FRAC — START 4/5–5/5 for
                     Xenium (density mismatch ⇒ higher FP risk), relax to 3/5 only if
                     recall too low. Tune on the data (see §1c); 3/5 was our HCR default.
                     OPTIONAL final-round Wang descriptor (re-tested net-POSITIVE at
                     3/5: +19 GT, 0 removed; net-negative only at 5/5; validate; §1c).
mutual_best        : on, every round
# round control
MAX_ROUNDS         : ~5
CONVERGE_REL_DELTA : 0.02
local_flow_rd0     : optional round-0 outlier reject (low impact; skip if simpler)
# search structure
CZ slabs           : 40 µm thick, 20 µm step, 0–100 µm from top  → 4 slabs
seeds per slab     : 5 (centre + 4 at ±200 µm, in the shrunk/Xenium frame)
shrinkage (sxy)    : ≈ 0.80 (ex-vivo / in-vivo)
warp model         : 2D thin-plate spline, re-fit each round, re-evaluation (no locking)
selection          : max converged consistent accepted-set size (GT-free, §6)
```

---

## 8. Lessons / pitfalls carried over (do / don't)

- **DO** use the geometric soma-print + seeds; **DON'T** add an image-NCC/MIP
  overlap step — it fails under density mismatch (our 782149 lesson).
- **DO** anchor-vote *within-round*; **DON'T** use the cross-round (accepted-set)
  version (circular, density-biased, over-conservative).
- **DO** re-evaluate the accepted set each round; **DON'T** lock matches.
- **DO** use a fixed physical `R_cand` (200 µm) every round; **DON'T** use k-NN
  candidates or shrink R_cand per round.
- **DO** make the descriptor neighbourhood physical-radius (or asymmetric) so
  density doesn't distort it; **DON'T** assume equal density / full correspondence.
- **DO** judge registration by accepted-set size + consistency; **DON'T** tune to
  GT-precision (it's a misleading lower bound; non-GT matches are often real).
- **DO** accept partial recall (match the consistent intersection); **DON'T**
  force every cell to match.
- **DO** keep `local_flow` (if used) as a round-0-only outlier reject; **DON'T**
  re-apply it on post-TPS residuals (after the warp the residual is noise — it just
  sheds good pairs).

---

## 9. Suggested validation / logging (port from our pipeline)

For each run log: per-round accepted count, added/removed, anchor-vote support
distribution, and a per-pair trajectory (rounds present, partner switches). For the
chosen registration, overlay the warped CZ-slab centroids on the Xenium top-slice
centroids (this is the honest QC view — analogous to the overlays we used), and
report accepted-set size, spatial coverage, and (if any GT exists) recall/precision
on the GFP+∩ok-equivalent comparable subset — **validation only**.
