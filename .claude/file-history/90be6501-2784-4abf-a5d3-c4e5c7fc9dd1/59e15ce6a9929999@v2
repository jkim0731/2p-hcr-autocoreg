# Session 35 — Pia-anchored alignment (negative on 782149)

## Why

S32/S33/S34 established that every centroid-ICP variant in xyz space
fails to find a local minimum at truth on 782149. S32 diagnostics
mentioned a ~12° pia-plane tilt as one feature of that subject.
Hypothesis: reparameterizing z → depth-from-pia in both modalities
pins "pia = 0" in both frames, absorbs tilt, and collapses the
Z-offset dimension. The ICP landscape in (x, y, depth) should have a
cleaner local minimum at truth.

## What was built

`probe_pia.py` on 4 subjects:
  1. Fit CZ pia with `estimate_pia_surface_image_ceiling` (CZ z-stack).
  2. Fit HCR pia with `estimate_pia_surface_image_ceiling` on the
     combined level-4 HCR volume.
  3. Convert all cells to `(x, y, depth_from_pia)`.
  4. 6-seed multi-start ICP × 4 trim levels in pia-anchored space.
  5. Convert winning fit back to xyz (add back `z_hcr_pia(pred_xy)`)
     and evaluate oracle n<50 on GT pairs.
  6. Compare to the same 6-seed × 4-trim multi-start in pure xyz.

## Results

| Subject | HCR tilt | PIA OR n<50 | XYZ OR n<50 | Δ | Notes |
|---------|---------:|------------:|------------:|--:|-------|
| 788406  | 2.8°  | 159 | **191** | −32 | low tilt, baseline already good |
| 755252  | 9.3°  | 170 | **179** | −9  | SS-ranker now picks correct basin (170=170) |
| 767022  | 3.9°  | 0   | 0       |  0  | wrong basin both ways |
| **782149**  | **11.1°** | **0**   | **0**       | **0**  | **primary target still 0** |

### 782149 diagnostics

- HCR tilt 11.1° and CZ tilt 2.0° — the largest tilt delta confirmed.
- CZ depth extent 418 µm; HCR depth extent 788 µm.
- At expected sz ≈ 1.9, scaled CZ spans ~790 µm which matches HCR —
  so the axial range IS compatible after scaling. Tilt alone cannot
  be the blocker.
- After pia-anchoring, ICP converges to a wrong basin for all 24
  (seed × trim) combinations. No change in n_lt50 vs xyz frame.

### Why pia anchoring doesn't help

The wrong basin ICP finds on 782149 is driven by the 3831 HCR GFP+
cells' density distribution — the GT region (~608 paired cells) sits
~335 µm from the HCR-GFP+ centroid (per S32 diagnostic), and the
remaining 3200 cells form a larger density cluster that ICP prefers.
Pia anchoring re-zeros Z but leaves the XY density landscape
unchanged, so the XY component of the wrong basin remains the
lowest-cost match.

The modest benefit on 755252 (SS-ranker now picks correct basin with
n=170 vs S32's wrong-basin pick) suggests pia anchoring helps with
SS-ranker ambiguity when trim and seed combinations produce
near-equivalent recip×unique scores. But it doesn't create new basins.

## Conclusion

- **Pia-anchoring does NOT crack 782149.** Like I2 (S33) and I2-crop
  (S34), it leaves the centroid-ICP objective without a local minimum
  at truth.
- **Do NOT fold pia-anchoring into production.** Hurts 788406 (−32)
  and 755252 (−9); zero marginal gain on stress subjects.
- **782149 is fundamentally outside the reach of centroid-only
  methods.** Recommended next paths:
  - **I3 MI + B-spline deformable** — image-level warp instead of
    centroid ICP. The image signal carries tissue structure that
    centroids discard.
  - **G1 hand-feature GNN** — match cells by F6 feature similarity
    (k-NN angles, depth rank, density) rather than spatial basin
    minimization. Feature matching is density-insensitive.
  - **Surface-to-body envelope coregistration** — align the HCR and
    CZ tissue envelopes (pia + lateral boundaries) as rigid objects,
    then find GFP+ correspondences within the aligned envelope.

## Files

- `probe_pia.py` — main probe.
- `pia_sweep.csv` — full results table.
- `probe.log` — stdout.

## Status

validated — negative result; 782149 unsolved via pia anchoring.
Centroid-only ICP paths now exhausted (S30 rotation, S31 M1 widened,
S32 trim sweep, S33 I2 warm-start, S34 I2-crop, S35 pia-anchored).
Next attempt requires image-level or feature-level matching (S36+).
