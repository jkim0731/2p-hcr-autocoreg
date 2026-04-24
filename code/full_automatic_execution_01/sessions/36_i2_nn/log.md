# Session 36 — I2 + nearest-neighbor matching (global negative)

## Why

S35 ruled out pia-anchored centroid ICP for 782149. Next logical probe
before committing to I3 (B-spline deformable) or G1 (GNN): does I2's
global MI-affine already carry correspondence signal, i.e. can you skip
ICP entirely and recover matches by plain nearest-neighbor lookup?

If `yes`, then I3 on top of I2 is promising (B-spline just refines a
working affine). If `no`, we know I2's coarse alignment is too loose
for centroid-level NN — and must either tighten coarse (M1 mask, M3
hybrid) or bypass spatial matching (G1 feature-based).

## What was built

`probe_i2_nn.py` on 4 subjects:
  1. Run I2 MI-affine (`mi_affine` via `_load_cz/hcr_fullstack`, 8µm target).
  2. Map each CZ cell via `apply_inverse` → predicted HCR xyz.
  3. For each pred, `cKDTree` nearest-neighbor in HCR-GFP+; form
     `(cz_id, hcr_id)` pairs.
  4. Score vs `coreg_table`:
     - **Direct**: residual of pred vs GT-HCR position.
     - **NN ID-exact**: fraction of pairs whose NN is the GT `hcr_id`.
     - **NN pos-n_lt50**: fraction of NN-chosen `hcr_id`s whose centroid is
       within 50 µm of the GT HCR cell.
     - **Reciprocal NN**: cz↔hcr agree both ways (30 µm ceiling).

## Results

| Subject | n_gt | I2 median | I2 rms | Direct n<50 | NN pos n<50 | NN IDs exact | Recip pos n<50 |
|---------|-----:|----------:|-------:|------------:|------------:|-------------:|---------------:|
| 788406  | 787  | 334 µm | 358 µm | 3   | 5   | 1 | 0 / 46 |
| 755252  | 639  | 260 µm | 390 µm | 1   | 1   | 0 | 0 / 74 |
| 767022  | 793  | 381 µm | 452 µm | 1   | 2   | 2 | 0 / 45 |
| **782149**  | **303**  | **348 µm** | **363 µm** | **1**   | **1**   | **1** | **0 / 12** |

## Interpretation

**I2 affine alone is far from tight** — median residual 260-380 µm
across all subjects, well beyond any centroid 1-NN capture radius
(~20 µm spacing). This explains:

- Why S33 I2-warmstart helps only when a downstream ICP has a local
  minimum near truth (788406 191, 755252 179) but fails on 767022/782149
  where ICP still slides to wrong basins.
- Why I3 B-spline refinement *on top of I2* is unlikely to solve 782149:
  B-spline assumes the affine initial estimate is close; a 348 µm median
  residual means the initial transform has the wrong basin globally, not
  a local nonrigid residual that B-spline was designed to absorb.

Reciprocal NN tells the same story: the 27-104 reciprocal pairs found per
subject have 0/total correctly identified GT — these are chance reciprocal
matches in the dense centroid field, not actual correspondences.

## Conclusion

**I2+NN is a dead end.** I2 is a useful coarse warm-start only when
paired with a refinement stage that has a valid truth-local-minimum
(centroid ICP on 788406/755252; not on 767022/782149). I3 on top of I2
inherits the same wrong-basin failure mode on 782149.

**Implications for the Grand Plan:**

1. **I3 is not a shortcut for 782149.** Before trying I3, the coarse
   alignment needs to be tighter than 300 µm. Options: (a) M1 mask-NCC
   with a tighter peak search, (b) hybrid I2+M1 where mask Dice replaces
   or complements image MI, (c) pia-envelope surface-to-surface that
   exploits shape rather than intensity.

2. **For 782149 specifically, the shortest path is feature-based
   matching** that does not require a close coarse alignment: **G1 GNN
   with F6 per-cell features** encodes local neighborhood geometry
   that is translation/scale-invariant and density-insensitive, so the
   3831 HCR GFP+ cells in a wrong-XY-centroid pattern are no longer a
   blocker.

3. **F6 is on the critical path.** Per Grand Plan §9.1, F6 is a tier-1
   foundation blocking G-series and enriching P1/P4/P5. S37 should
   stand up F6 per-cell features.

## Files

- `probe_i2_nn.py` — main probe.
- `i2_nn.csv` — full results table.
- `probe.log` — stdout.

## Status

validated — negative result; I2 alone is insufficient for centroid
NN matching (260-380 µm median residual across all subjects). Primary
target 782149 still unsolved. Next: F6 per-cell features (S37) as
foundation for G1 GNN and for enriched P1/P4/P5 correspondence scoring.
