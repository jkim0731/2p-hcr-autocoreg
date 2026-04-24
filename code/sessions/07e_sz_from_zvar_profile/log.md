# Session 07e — sz from σ_z(d) profile NCC + sxy from xy-kNN

**Verdict: FAIL.** 0/2 primary subjects within ±5 % on both axes.

| subject | sxy (GT)        | sz (GT)        | sxy err   | sz err   | NCC_best |
| ------- | --------------- | -------------- | --------- | -------- | -------- |
| 788406  | 1.349 (1.778)   | 1.946 (2.820)  | −24.1 %   | −31.0 %  | 0.52     |
| 790322  | 2.110 (1.763)   | 1.760 (3.042)  | +19.6 %   | −42.1 %  | 0.68     |

Primary stopping condition: `|err_sxy|≤5 % AND |err_sz|≤5 %` on both
of 788406 + 790322 → **not met**. Session declared failed; no
extension to 755252/767018/767022.

## Hypothesis (at entry)

After 06/07/07b/07c (centroid-count) and 07d (intensity-NCC) all
failed, the hope was that σ_z(d) of centroid positions would be:
1. a *position* statistic, f(d)-invariant (unlike count-density);
2. modality-agnostic (unlike intensity);
3. anchored on a real cortical-depth shape that NCC could align.

If true: xy k-NN for sxy (densities roughly balanced within matched
xy+depth band), σ_z(d) NCC for sz, iterated to fixed point.

## What 07e actually found

### σ_z(d) of centroids is near-constant across the cortical column.

Per-subject σ_z statistics at the final iterate (window 100 µm, stride
25 µm, N_min 50):

| subject | CZ σ_z mean | CZ σ_z std | CZ CV | HCR σ_z mean | HCR σ_z std | HCR CV |
| ------- | ----------- | ---------- | ----- | ------------ | ----------- | ------ |
| 788406  | 28.6 µm     | 1.6 µm     | 5.6 % | 28.3 µm      | 2.9 µm      | 10 %   |
| 790322  | 30.6 µm     | 3.0 µm     | 9.7 % | 30.1 µm      | 3.6 µm      | 12 %   |

CZ σ_z CV ~6 %, HCR σ_z CV ~10 % — both near-flat. The HCR CV is
inflated entirely by a bottom-edge drop (σ_z falls from ~30 µm to
~14 µm at the deepest ~50 µm of tissue) that has no CZ equivalent.
Everywhere else, σ_z(d) is a horizontal line plus shot noise.

Biological read: cortical cells are distributed in columns of
roughly uniform lateral thickness across depth. The z-spread of the
centroid point cloud does not encode a depth-dependent signal that
NCC can exploit to distinguish a 1.5×-stretched profile from a
1.0×-stretched one.

### NCC's peak is at the HCR bottom-edge artifact.

Synthetic control (take real HCR σ_z(d); synthesize CZ by
unstretching HCR by c_true; run the NCC fitter):

| subject | c_true = 1.2         | c_true = 1.5         | c_true = 0.8         |
| ------- | -------------------- | -------------------- | -------------------- |
| 788406  | 1.200 (err 0.0 %, NCC 0.879) | 1.520 (err +1.3 %, NCC 0.920) | 0.810 (err +1.3 %, NCC 0.979) |
| 790322  | 1.210 (err +0.8 %, NCC 0.987) | 1.510 (err +0.7 %, NCC 0.979) | 0.810 (err +1.3 %, NCC 0.986) |

**The fitter is correct.** When CZ and HCR have genuinely similar
shape (they do here — synthetic CZ is literally unstretched HCR), it
recovers c_true within ≤1.3 % at NCC 0.88–0.99.

Real-data NCC peaks at 0.52 (788406) and 0.68 (790322). Peak-to-sd
ratio of the NCC curve: 2.5 for both — i.e. the "argmax" is barely
above the std-dev of NCC values across the c_z grid. The fitter is
finding the best available alignment, but the available signal is
just CZ terminating at ~850 µm vs HCR terminating at ~1100 µm. NCC
prefers whichever c_z puts CZ's termination near HCR's edge-drop,
which is dictated by the CZ imaging depth (a device choice) not by
biological stretch.

### xy k-NN is subject-inconsistently biased.

- 788406: k-NN xy converges to sxy 1.33 (GT 1.78, −24 %).
- 790322: k-NN xy converges to sxy 2.11 (GT 1.76, +20 %).

Two subjects, opposite-sign errors of similar magnitude. This
matches the 06/07b finding: local k-NN on centroid positions is
dominated by local density variations that do not average out in
these sample sizes (N ≈ 700–900 after matched-depth crop), and those
variations differ by subject.

### Iteration does not recover.

788406 iterates five times, oscillates. 790322 declares "converged"
(|c−1| < 0.01 on both axes at iter 4) but converges to the WRONG
scale — the NCC is maximised at c_z ≈ 1 because the current s_z
already placed CZ's termination near HCR's edge drop, not because
the two profiles actually match. The fixed-point condition
`|c − 1| < 0.01` cannot distinguish "correct" from "locally
optimal given a bad seed", because σ_z(d) is invariant to
depth-stretch of the synthetic kind in the approximately flat
regime.

## What this rules out

Every variant of the "match 1D profile along pia depth" framework:
- intensity profile (07d, 2/6 best)
- centroid σ_z profile (07e, 0/2 primary)
- centroid density profile (06, 0/6; see previous sessions)

The information content of any 1D profile along the pia-normal axis
is insufficient to recover sz when the profile has no strong depth
dependence inside the cortical volume.

## Why the depth-invariance of σ_z is actually *expected*

In a cortical column with uniform cell-density lateral spread, σ_z
measured in a sliding depth window is:

  σ_z ≈ sqrt(∫ (z − z̄)² p(z|d) dz)

where `p(z|d)` is the distribution of cell z inside the window at
depth d. If the window is small enough that p is approximately
uniform, σ_z ≈ W/√12 where W is the window width (constant). The
only structure comes from:
- Tissue end (window hits the bottom surface → fewer cells, biased
  sample → σ_z drops).
- Layer-specific changes in cell packing (would broaden/narrow σ_z
  slightly).

Neither gives a reliable modality-matched depth shape.

## Why sxy k-NN is biased with matched-depth+matched-xy crop

The crop equalises the volume both clouds come from, but the CZ and
HCR-GFP+ populations aren't the same selection:
- CZ GCaMP is every cell expressing GCaMP, biased toward the surface
  and toward high-activity regions.
- HCR GFP+ at v2.2 threshold is every cell above the intensity/density
  cutoff in the GFP channel, which contains a modality- and
  subject-specific false-positive tail.

Unequal local densities → k-NN distance ratio ≠ scale ratio. No
amount of crop geometry fixes this — only a sampling-equalised GFP+
set would, and 07b/07c proved that's not achievable with the
available thresholds.

## What *might* still work (not implemented)

1. **2D σ_xy(d) profile.** If σ_x(d) and σ_y(d) have strong
   depth dependence (e.g. because the CZ field-of-view is a rectangle
   that tilts relative to the HCR volume after R1), those could be
   stretchable. Requires the same synthetic-recovery verification
   before investing.

2. **Layer-boundary depth matching.** Segment cortical layers from
   image intensity (DAPI for HCR, baseline GCaMP for CZ), match layer
   boundaries in µm. This is modality-dependent but produces
   depth-landmarks with real scale information. High implementation
   cost; own session.

3. **Cell-count-conservation.** If sxy is well-constrained from a
   surface-projected 2D matching (not yet attempted), and total cell
   count is preserved, then `sz = N_ratio / sxy²`. Sidesteps σ_z
   flatness entirely. Still requires a reliable sxy estimator.

4. **Joint xy-depth NCC on 2D `I(x, d)` / `I(y, d)` intensity
   slabs.** 07d's 1D collapse discarded lateral structure; recovering
   it via a 2D NCC is a natural next step. Separate session.

## Files

```
dev_code/07e_sz_from_zvar_profile.py    estimator + driver
dev_code/07e_synthetic_check.py         synthetic NCC sanity
sessions/07e_sz_from_zvar_profile/
    plan.md
    log.md                              (this file)
    results.json                        per-subject iteration trace
    figures/
        zvar_profile_788406.png
        zvar_profile_790322.png
    notebook.ipynb                      (built via _build_notebook.py)
```

## Self-critique

- **I did not expect σ_z to be this flat.** My plan assumed a
  "characteristic depth shape" without quantifying it in advance.
  A 30-minute pre-check (compute σ_z(d) on HCR GFP+ alone, look at
  it) would have rejected this hypothesis before writing 400 lines
  of estimator code. Next time: front-load a 10-line sanity plot of
  the feature I intend to match before designing the matcher.

- **The synthetic test validates the fitter, not the problem.** I
  designed the synthetic test to verify NCC-on-σ_z machinery; it
  does, cleanly. But it doesn't test the cross-modal assumption that
  CZ σ_z shape tracks HCR σ_z shape modulo stretch. A stronger test
  would have been: compute σ_z on CZ at GT scale, compute σ_z on
  HCR, visually compare before running the NCC. That's the next
  diagnostic I'd run if revisiting this.

- **Iteration-with-damping masks failure.** 790322 "converged" per
  the stopping rule but to the wrong scale. Future estimators should
  require not just `|c−1| < tol` but also NCC peak height vs
  baseline sd (e.g. peak_to_sd > 5) before declaring convergence.
