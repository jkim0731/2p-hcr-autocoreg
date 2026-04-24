# Session 34 — I2-cropped HCR ICP

## Why

S33 showed I2 produces ~340 µm median-error coarse localization but not
close enough for ICP's 150 µm capture radius. Hypothesis: cropping the
HCR GFP+ cloud to an XY window around I2's estimate prunes the
distracting wrong-basin cells, letting ICP converge on the correct
basin.

## What was built

`probe_crop.py` sweeps W ∈ {400, 600, 800, 1200, ∞} µm on 4 subjects,
for each:
  1. Run I2 → XY centroid estimate.
  2. `hcr_cropped = hcr_gfp[|xy - i2_xy| ≤ W]`.
  3. Seed multi-start ICP at cropped-centroid; sweep trim ∈ {0.4, 0.6,
     0.8, 0.9}; rank by SS `recip × unique_frac`; report OR best
     (n_gt_lt50).

## Results

| Subject | Uncropped OR n<50 | Best-crop OR n<50 | Crop W | Crop trim | Δ |
|---------|------------------:|------------------:|-------:|----------:|--:|
| 788406  | 21 | **69** | 400 | 0.6 | +48 (but S29 prod 185; not fair) |
| **755252** | **179** | **232** | **600** | 0.9 | **+53** |
| 767022  | 0 | 1 | 600 | 0.4 | +1 |
| **782149** | **0** | **0** | — | — | 0 |

### 755252 — real gain

**OR 232 > uncropped 179** with `W=600, trim=0.9, sxy=1.58, sz=2.26`.
SS ranker also picks this basin (SS n=232 at W=600, SS rank=435.96 vs
uncropped 434.25). The SS ranker's margin is small but correctly
identifies the better basin.

### 782149 — still 0

Even at W=400 (tightest crop), n_lt50 = 0. I2 center is at
xyz=(1163, 1159, 505); GT region bbox is x 683-1525, y 951-1704,
z 270-830. So the crop IS centred on the GT region, but ICP still
doesn't converge:

- W=400 (n_hcr=486): the crop includes most but not all GT-paired HCR
  cells (608 cells are in GT region); ICP still finds a wrong basin.
- W=9999 (n_hcr=3831): as before, ICP converges to squeezed basin
  regardless of seed.

Confirms S32's structural finding: for 782149 the centroid-ICP
objective has no local minimum at truth, regardless of HCR cropping.

### 788406 — cropping hurts

The 788406 crop loses HCR cells that legitimately match to CZ at the
correct scale. Lesson: I2-crop is only a win when HCR GFP+ has many
wrong-basin cells outside the CZ's physical extent; 788406 doesn't
have that problem (default ICP works well there).

## Conclusion

- **Do fold I2-crop at W=600 as a multi-start SEED option** — it
  unlocks 755252 (+53 n_lt50) without requiring per-subject tuning,
  because SS ranker correctly prefers the cropped basin when it's
  better.
- **Do NOT reduce HCR to the cropped subset by default** — 788406
  regresses.
- **782149 still needs a different approach** — pia-anchored
  alignment (S35?) or feature-based matching (G1 GNN).

## Files

- `probe_crop.py` — main probe.
- `crop_sweep.csv` — full results table.

## Status

partially validated — 755252 gain confirmed; no follow-through to
production yet. 782149 still unsolved.
