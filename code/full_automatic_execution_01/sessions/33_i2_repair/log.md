# Session 33 — I2 SimpleITK MI-affine repair

## Why

S32's negative result identified image-level alignment as one of three
viable paths to crack 782149. S18 had left I2 unimplemented (SimpleITK
not installed at the time). Pre-repair diagnostics showed I2 ran but
`TransformDescriptor.translation` held values of ~(-86, +36) µm — too
small to place CZ inside HCR's ~1000-1500 µm coordinate frame. Applying
the transform via any of three standard Procrustes conventions gave
~2000 µm residuals (random quality).

## What was wrong

Two bugs in `lib/sitk_wrapper.py::mi_affine`:

1. **Center omitted from the exported transform.** SimpleITK's
   `AffineTransform` uses `T(p) = A @ (p - center) + center + translation`.
   The wrapper stored only `(A, t)`, dropping `center` — so the
   downstream `TransformDescriptor` carried only ~2 % of the
   composition's positional information.
2. **Initial translation placed CZ at HCR's origin, not centre.** The
   init translation defaulted to `(0, 0, 0)` while the transform's
   centre was HCR's volume centre. Forward sampling at HCR's centre
   therefore looked up CZ at HCR-centre-coordinates (~1144 µm) — far
   outside CZ's 0–400 µm physical extent. MI couldn't escape this
   initialisation and returned the init unchanged on 782149.

Also in `bench/candidate_impls/_i2_sitk_affine.py`:

3. **Init scale was passed in the wrong axis order** — `(2.8, 1.8, 1.8)`
   sent `sz=2.8` to the X axis. Correct is `(1.8, 1.8, 2.8)` (SITK xyz
   order, sxy on x/y, sz on z).

## What was fixed

- `MIFitResult` now carries `center` (zyx µm) and exposes a correct
  `apply_inverse` helper: `p_fixed = A^-1 @ (p_moving - c - t) + c`.
- `mi_affine` now sets `init.SetTranslation(cz_center_xyz − hcr_center_xyz)`
  so the initial transform places CZ's centre at HCR's centre.
- `_i2_sitk_affine.py` now stores a proper inverse `TransformDescriptor`
  (source anchor = `c + t`, dest anchor = `c`, `R` from SVD of `A^-1`,
  scales from the singular values) and passes `init_scale=(1.8, 1.8, 2.8)`.

## Benchmark after repair (4 subjects, target_um=8.0)

| Subject | I2 median | I2 n<50 | I2-seed+trim n<50 (OR) | Default ICP best n<50 (OR) |
|---------|----------:|--------:|----------------------:|---------------------------:|
| 788406  | 331 µm |  3/1000+ |    87 | **191** (gfp_dz-100, 0.4) |
| 755252  | 262 µm |  1 |   135 | **179** (hcr_gfp, 0.6)    |
| 767022  | 394 µm |  3 |     0 |   0 (hcr_gfp, 0.4; S29 prod ≈ 80 via translation-refine loop) |
| 782149  | 340 µm |  3 |     0 | 0                         |

I2 converges now (was stuck at init on 782149 pre-fix; now 300 iters).

### Did I2 unlock any subject?

- **755252** (SS-ranker issue, not basin issue): I2 seeding with trim=0.6
  gives n=132 where default-single-seed gives n=0. But the 6-seed
  multi-start's best is 179 (hcr_gfp, 0.6), so I2 adds no oracle-best
  gain. SS ranker picks I2's basin (132) over the correct basin (179) —
  persists from S32.
- **788406**: default beats I2-seeded. I2 adds nothing.
- **767022**: neither I2 nor any trim level recovers — matches S32.
  S29 production's ~80 relies on the post-ICP translation refine +
  iterative local refit loop inside `default_warmstart_zyx`, not on
  the raw multi-start.
- **782149**: I2 seed still 0. Image-level alignment places CZ with
  ~340 µm median residual — outside ICP's r_init_um=150 µm capture
  basin. Also consistent with S32 finding that 782149 has no ICP local
  minimum at truth.

## Conclusion

**I2 is now correctly implemented but does NOT crack 782149.** The
image-level MI registration recovers plausible scales (~1.8 XY, ~2.8 Z)
and places CZ roughly in the cortical region, but the ~340 µm residual
is beyond ICP's capture radius and the centroid-ICP objective still has
no local minimum at truth.

782149 remains unsolved by every centroid-based approach tried:
rotation seeds (S30), widened M1 scale grid (S31), trim sweeps (S32),
I2 image-level warm-start (S33).

Next attempt should use **independent geometric priors** beyond GFP+
centroids — i.e. pia-surface anchoring (CZ pia ≈ HCR pia plane after
tilt alignment) reduces the 3D search to 2D XY. S34.

## Files

- `lib/sitk_wrapper.py` — `MIFitResult.center` added; init-translation
  bug fixed; `apply_inverse` helper.
- `bench/candidate_impls/_i2_sitk_affine.py` — axis-order init fix;
  proper inverse TransformDescriptor.
- `sessions/33_i2_repair/probe_i2_raw.py` — pre-fix diagnostic (medians
  ~700 µm confirmed the bug).
- `sessions/33_i2_repair/probe_i2_fixed.py` — post-fix wrapper check.
- `sessions/33_i2_repair/probe_i2_warmstart.py` — I2-only seed ICP.
- `sessions/33_i2_repair/probe_i2_seven_seeds.py` — 6-default + I2
  seeds × 4 trim levels; the fair comparison.
- `sessions/33_i2_repair/warmstart.csv`, `seven_seeds.csv` —
  tabular results.

## Status

validated (I2 now runs correctly; moves to "useful coarse" but not
"unlocking" on any stress subject; no follow-up I2 work planned).
