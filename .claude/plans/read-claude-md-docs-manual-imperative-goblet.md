# Plan — Revise R1 (coarse affine): scale-free, surface-tilt aligned, graceful degradation

## Context

The grand plan at `/root/capsule/code/docs/07 Grand Plan.md` already exists. The user has now asked to **revise R1** because it relies on benchmark-derived expansion priors (XY ~1.77×, Z ~2.83×), which:

1. Violates `06 Dev Protocol.md`: *"Do not use benchmark data information for developing automatic registration."*
2. Violates the grand plan's own opening rule: *"Do not consider parameter inference from the existing benchmark at all, except when explicitly stated."*
3. Produced a first-pass that fails on 3/6 subjects (767018 136 µm, 767022 158 µm, 782149 372 µm origin error).

User constraints for the revised R1:

- **Only two priors allowed:** 180° XY rotation, and "CZ sits roughly at the XY center of HCR".
- **Nothing** may be assumed about expansion (XY or Z scales).
- **Use the pia surface estimation step**: fit both pia surfaces as planes, match them linearly to recover the residual tilt (small rotation that composes with the 180° XY prior).
- **Depth coverage can differ between CZ and HCR.** The HCR physical section thickness is independent of the expansion factor — a thinner HCR may only capture upper cortical layers, a thicker one may capture more. R1's axial step must handle partial axial overlap, in either direction.
- **Graceful-degradation backup — no separate R1-B candidate.** The "first localize" principle (`CLAUDE.md`) is satisfied by the **sample-specific priors alone**: the 180° rotation, the surface-tilt correction from plane-to-plane alignment, and the CZ-centroid-at-HCR-centroid translation. Those three produce a reliable rotation + translation output that does **not** depend on any scale estimate. When R1's scale search fails on a hard subject, the "sample-specific localisation" above is itself sufficient as R1's output — downstream scale-aware methods (A1, A3) estimate scale internally, and R-series methods (R2/R3) can be seeded from the sample-specific localisation too.

This means R1 has two output tiers produced in the same session:

1. **Minimal output (always reliable):** `R = R_tilt · R_180°XY` + translation `t` from the centroid prior. No scale claim. Derived only from surface fits and centroid geometry.
2. **Extended output (conditional on search confidence):** additionally `S = diag(sx, sy, sz)` from the Z partial-overlap NCC and the XY density-map search. Produced when the per-axis search confidence exceeds a threshold.

Downstream candidates consume whatever R1 produces — minimal when scale confidence is low, extended when it is high. No separate R1-B candidate is introduced.

## Changes to `/root/capsule/code/docs/07 Grand Plan.md`

### 1. Rewrite the R1 block (scale-free, surface-tilt aligned, partial-coverage aware, graceful-degradation)

Replace the current R1 block (around lines 64–77) with:

> **R1 — Scale-free coarse affine with surface-tilt alignment** _(tier 1)_
>
> - **Goal.** Produce a reliable coarse localisation of CZ inside HCR using only (a) the 180° XY rotation prior, (b) the "CZ ≈ HCR XY center" prior, and (c) the pia-surface planes fitted from the existing surface-estimation utilities. The output has two tiers: a **minimal** rotation + translation that is always reliable, and an **extended** rotation + translation + scale that is produced when the scale search is confident. **No benchmark-derived expansion priors.**
> - **Inputs.** CZ + HCR GFP+ centroids; both pia surfaces (foundation 0).
> - **Method sketch.**
>   1. **Pia-plane fit.** Fit a plane through each pia surface (existing `estimate_pia_surface_image_ceiling` for CZ and `estimate_pia_surface_quantile_ceiling` for HCR already produce surface points; feed those to a least-squares plane fit). Extract unit normals `n_cz`, `n_hcr`.
>   2. **Tilt-aligned rotation.** Apply `R_180°XY` to CZ, which rotates `n_cz → R_180°XY · n_cz`. Compute `R_tilt` as the small rotation (Rodrigues, axis = `(R_180°XY · n_cz) × n_hcr`, angle = `acos((R_180°XY · n_cz) · n_hcr)`) that aligns the rotated CZ normal with HCR's normal. Final rotation `R = R_tilt · R_180°XY`.
>   3. **Centroid translation.** After applying `R`, translate so the rotated CZ centroid maps to the HCR centroid in all three axes (Z from pia-alignment, XY from the "roughly at center" prior).
>
>   **End of minimal output.** Steps 1–3 produce `(R, t)` — a sample-specific coarse localisation derived from surfaces and centroids only. This output is always produced and always reliable (modulo a working surface fitter).
>
>   4. **Depth-from-surface** in both modalities — shared pia-anchored axial coordinate, consistent with the aligned surfaces.
>   5. **Z-scale + Z-offset search with partial overlap (1D).** This step does **not** assume CZ and HCR cover the same axial range.
>      - Scan `sz` over a permissive grid (step 0.02). The grid extent is a search-feasibility parameter, **not** a prior on expansion — explicitly documented in the R1 session log.
>      - For each candidate `sz`: rescale the CZ depth-density profile to `sz × z_cz`; slide it across the HCR profile over `tz`.
>      - At each `(sz, tz)`: compute Pearson correlation on the **overlap region only**, with a minimum-overlap floor (e.g. require the overlap to span ≥ 25 % of the shorter profile) to prevent trivial single-peak alignments from scoring high.
>      - Pick `(sz, tz)` maximising the partial-overlap NCC.
>      - This transparently handles the three coverage regimes:
>        - HCR thicker than mapped CZ: overlap = full CZ range.
>        - HCR same as mapped CZ: overlap = full range of both.
>        - HCR thinner than mapped CZ: overlap = top portion of CZ (where HCR captures data); deep CZ cells have no HCR counterpart and are excluded from the Z score.
>   6. **XY-scale + XY-translation joint search.** The user's "CZ ≈ HCR XY center" prior *is* a geometric "CZ ⊂ HCR in XY" constraint, so the `sxy ≤ L_hcr / L_cz` bound is legitimate — it comes from the user's geometric prior, not from benchmark statistics. Scan isotropic `sxy ∈ [0.5, L_hcr / L_cz]` (step ≈ 0.05) and `(tx, ty)` within a ±half-HCR-XY-extent box on a sparse grid. For each `(sxy, tx, ty)`, rasterise Gaussian-blurred 2D density maps on a shared grid and score via NCC / NMI. Pick the best.
>   7. **Anisotropic refinement (optional).** Local search over `(sx, sy)` around the best `sxy` to allow axis-specific scales.
>   8. **Per-axis scale confidence.** Peak-to-RMS ratio of the Z and XY score surfaces. When a component's confidence exceeds a threshold (e.g. 3), it is added to the extended output; when it does not, the component is emitted as `unknown` and downstream methods are responsible for estimating it.
> - **Outputs.**
>   - **Minimal (always):** `(R, t)` from steps 1–3.
>   - **Extended (conditional):** additional `S = diag(sx, sy, sz)` with per-axis confidence. Components with low confidence are marked `unknown`.
>   - A short "coverage regime" note (thinner / equal / thicker HCR) from step 5.
> - **Success metrics.**
>   - Minimal output: localised origin within ≤ 100 µm of manual-landmark-derived affine on held-out subjects; rotation within ±5°.
>   - Extended output (when emitted): in addition to the minimal metric, `(sx, sy, sz)` recovered within ± 20 % of manual-landmark scale.
> - **Dependencies.** Foundation 0 only (surfaces + depth profile). **No benchmark priors.**
> - **Failure modes.**
>   - Plane-fit degeneracy → only occurs if the surface fitter itself fails; both fitters are already validated across the 6 benchmark subjects, so the tilt step is safe in normal operation.
>   - Flat XY score surface on thin / asymmetric HCR (e.g. 782149) → `sxy` emitted as `unknown`; minimal output still valid.
>   - Flat axial partial-overlap NCC when CZ and HCR cortical-layer content is too different → `sz` emitted as `unknown`; minimal output still valid.
>   - In the worst case, every scale component is `unknown` and only `(R, t)` is emitted. That is the intended graceful-degradation — it matches the user-stated "sample-specific condition can be good enough" fallback.
> - **Estimated effort.** 1–1.5 sessions.
> - **Status.** `to_be_revised` — current first-pass (`sessions/04_R1_coarse_align/`, `dev_code/r1_coarse_align.py`) is **superseded** because it used benchmark priors (XY ~1.77× / Z ~2.83× rescale, the GFP+-centroid XY trick, and a `sz ≤ D_hcr / D_cz` bound that assumed full axial coverage). The Z-profile 1D xcorr from the first pass is the only piece worth reusing; tilt alignment, partial-overlap NCC, and graceful-degradation output are new.

### 2. Downstream Dependencies — no text change needed

Every candidate whose Dependencies line mentions `R1` consumes whatever R1 emits. When R1's scale components are `unknown`, the downstream method falls back to internal scale estimation. No `(or R1-B)` suffix is added; the graceful-degradation output is the backup, not a sibling candidate. Specifically:

- **R2, R4, R5** — unchanged; use R1's `(R, t)` at minimum, scale when available.
- **A1 (TEASER++)** — scale-aware by design; when R1's scale is `unknown`, TEASER estimates it. HCR crop uses `(R, t)` + a conservative bounding box around the centroid.
- **A2 (Spectral GM)** — uses R1's `(R, t)` for putative correspondences; when scale is `unknown`, putatives widen slightly but the spectral step still converges.
- **A3 (Fused GW)** — intrinsically invariant to isometry; scale enters via the feature side only, so R1's scale being `unknown` has minimal impact.
- **A4–A8** — all scale-aware or scale-agnostic by design; behave the same way.

### 3. Section 0 — annotate benchmark priors as validation-only

Relabel the "Benchmark priors (hard-codeable for init)" subsection (around lines 37–43) to:

> ### Benchmark priors — **validation reference only**
>
> These numbers exist solely to check algorithm outputs against known benchmark statistics. **They must not seed any algorithm.** Our estimates of scale, rotation, residual, etc., are computed from data geometry; the priors are compared to those estimates only after the fact.
>
> - XY expansion ≈ 1.77× (range 1.64–1.92×).
> - Z expansion ≈ 2.83× (range 2.13–3.58×). Z:XY anisotropy ≈ 1.5×.
> - In-plane rotation ≈ 180° on all 6 subjects.   *(Exception: the 180° rotation is treated as a structural imaging-geometry prior, consistent with the dev protocol — it is a constant of the acquisition setup, not a population statistic.)*
> - 1-NN cell spacing ≈ 20 µm; 5-NN ≈ 33–37 µm.
> - Post-affine landmark residual 16–43 µm RMS — this is the nonrigid warp scale TPS must absorb.

### 4. Section 1 intro update

Line 55 — "Two complementary tracks share the R1 coarse affine **and the G1 review GUI**." — kept as-is. No edit needed; the text already correctly states that both tracks share R1, and R1's graceful-degradation output is consumed transparently by both.

### 5. Section 4 (Recommended session order) update

Under "Shared foundation (both tracks depend on these)" (around lines 321–324), replace with:

> 1. **R1** — scale-free coarse affine with surface-tilt alignment, partial-overlap axial matching, and graceful-degradation output (minimal rotation + translation always; extended scales when confident). Unblocks every downstream candidate; handles thinner or thicker HCR natively; when scale estimation is uncertain, emits `(R, t)` only and leaves scale to downstream scale-aware methods.
> 2. **G1** — minimum review GUI. Required for manual verification/correction of every candidate's output (A-series and Track B alike). Start in parallel with R1.

### 6. Section 6 (Ledger) — no new row

No R1-B row is added; R1 is the only coarse-affine candidate. The existing R1 row is edited to reflect the `to_be_revised` status (see change log).

### 7. Section 7 (Change log) entry

Append:

> - 2026-04-17 — **R1 revised: scale-free, surface-tilt aligned, graceful-degradation.** Only the 180° XY rotation prior, the "CZ ≈ HCR XY center" prior, and the pia-plane fits are used; scales are estimated from data with the axial search using partial-overlap NCC (no `sz ≤ D_hcr / D_cz` bound — HCR section thickness is independent of expansion and the method handles thinner or thicker HCR); residual tilt is recovered by linear plane-to-plane alignment of the pia surfaces; R1 emits minimal `(R, t)` always, adding `S` only when per-axis scale confidence clears threshold (graceful degradation). **No R1-B candidate:** the "first localize" principle (`CLAUDE.md`) is satisfied by the sample-specific priors alone, so when R1's scale search fails the minimal output is itself the backup, consumed transparently by downstream scale-aware methods (A1 / A3 / TEASER-class internals). First-pass implementation (`dev_code/r1_coarse_align.py` / `sessions/04_R1_coarse_align/`) is marked superseded because it used benchmark-derived expansion priors and the coverage-constrained axial bound. Section 0 benchmark priors relabelled as validation-reference only. Downstream Dependencies lines unchanged (they already reference R1 only).

## Files touched during execution

- **EDIT** `/root/capsule/code/docs/07 Grand Plan.md` — changes 1, 3, 5, 7 above.
- **No code touched.** Rewriting `dev_code/r1_coarse_align.py` is the scope of the R1 revision session (opened in `sessions/NN_R1v2_scale_free/`), not of this planning task.
- **No benchmark data read.**

## Verification after edits

1. `Grep` the grand plan for `1.77` and `2.83` — they should appear **only** under Section 0's "validation reference only" heading; nowhere inside any R-/A-block method sketch.
2. `Grep` for `prior anisotropic rescale` — should be absent.
3. `Grep` for `D_hcr / D_cz` (or `D_hcr/D_cz`) — should no longer appear as an `sz` bound anywhere; it only appears in the XY search as `L_hcr / L_cz` (an independent XY geometric constraint from the "CZ at center" prior).
4. `Grep` for `R1-B`, `human anchor`, `quick-anchor`, or `G1 anchor` — should all be absent.
5. R1 method sketch contains:
   - Pia-plane fit + tilt-alignment as steps 1–2.
   - Centroid translation as step 3.
   - Explicit "end of minimal output" marker after step 3.
   - Partial-overlap axial NCC as step 5.
   - XY geometric bound explanation as step 6.
   - Per-axis scale confidence and `unknown` emission semantics as step 8.
6. Section 6 ledger unchanged except R1 row (no R1-B row).
7. Sanity read: a new R1-revision session can start from just 01–07 + the existing foundation utilities — no other context needed; it understands that R1 always outputs `(R, t)` and emits `S` only when confident.

## Self-critique

- **Weakest new assumption:** CZ xy-centroid ≈ HCR xy-centroid. User stated "roughly"; mitigated by searching `(tx, ty)` over a ±half-HCR-extent box when the extended scale search runs, and by the intrinsic scale-aware / scale-invariant behaviour of A1 / A3 when R1 falls back to minimal.
- **Tilt alignment depends on good plane fits.** Both fitters are validated on all 6 benchmark subjects (CZ tilt ≤ 3°, HCR `above_frac ≤ 0.08 %`). If a fitter fails on a new subject, `R_tilt` collapses to identity, which is the conservative default — the minimal output is still `(R_180°, centroid translation)`.
- **Partial-overlap NCC is the key new idea for axial matching.** It transparently handles thinner / equal / thicker HCR. The minimum-overlap floor is the only tuning knob.
- **Graceful degradation replaces a separate R1-B candidate.** The user's insight is that the 180°-rotation prior, the surface-tilt correction, and the centroid translation together *are* the "first localise" step; they are derived from the sample itself (pia surfaces, centroids) and do not need scale to be localisation. When scale estimation fails, the minimal `(R, t)` is still a correct sample-specific localisation. A1 (TEASER-scale-aware) and A3 (Fused GW, isometry-invariant) recover scale from the data themselves; they do not need a separate "R1-B TEASER-scale-aware" preprocessor.
- **Acknowledged risk:** R1's scale search is expected to fail on stress subjects (thin / asymmetric HCR, 782149 / 767022). That is designed-in — the scale components are emitted as `unknown` and A-series methods handle the rest.
- **Alternative considered and rejected — moment / PCA-based scale.** Fragile to outliers; HCR regions are rectangular, not ellipsoidal.
- **Failure of this plan would look like:** A-series methods do not recover scale from the data even after being handed `(R, t)` + `unknown` scale. In that case the benchmark dataset falls back to Track B (R2 + R3 + Q3), which is the grand plan's existing double-track design.
