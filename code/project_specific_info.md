# Project-Specific Information — CTL Inhibitory Cohort

What every new analysis session needs to know. Session-specific detail lives in that
session's folder under `code/sessions/`. Previous version:
`code/archived_files/project_specific_info_archived_260918.md`.

**Subjects (all 6):** 782149, 788406, 790322, 800792, 800995, 804363.
Genotype `Slc32a1-IRES-Cre;jGCaMP8s` — **VGAT, so every imaged cell is inhibitory.** A cell
without a type call is *Untyped*, never "excitatory".

---

## 1. Sessions

| session type | label | notes |
|---|---|---|
| `TRAINING_0_gratings_autorewards_15min` | Pre-training | auto-reward, no task |
| `TRAINING_1_gratings` | Static gratings | passive |
| `TRAINING_2_gratings_flashed` | Flashed gratings | learning the change-detection structure |
| `TRAINING_3_images_A_10uL_reward` | First familiar images | image set A |
| `TRAINING_4_images_A_training` | Image training | active change detection |
| `TRAINING_5_images_A_*` | Handoff | final training before ophys |
| `STAGE_0` | Passive — movie | no task |
| `STAGE_1` | Passive — drifting gratings | has gray-screen **blanks**, which are *not* omissions |
| `OPHYS_1_images_A` | **Familiar** | image set A |
| `OPHYS_4_images_B` | **Novel / Novel+1** | image set B |
| `OPHYS_6_images_B` | **Extinction** | image set B, **no water**; a "reward" event means a correct lick only |
| `CENTER_MOUSEMOTION` | non-task | exclude from analysis |

**Omissions occur only in `OPHYS_*` sessions** (~5 % of non-change flashes). `STAGE_1` blanks
are structurally similar but are not omissions.

### Experience levels — use this table, do not re-derive

`F` = **last** `OPHYS_1` · `N` = **first** `OPHYS_4` · `Np` = **second** `OPHYS_4`.
**`Np` is never `OPHYS_6`.** `OPHYS_6` is Extinction, a separate label (`E`). Getting this
wrong is the most damaging error in this project: `OPHYS_6` has no water, mice stop licking,
so hit-aligned PSTHs come out flat zero.

| subject | F | N | Np |
|---|---|---|---|
| 782149 | 2025-04-16 | 2025-04-28 | 2025-04-29 |
| 788406 | 2025-07-14 | 2025-07-15 | 2025-07-16 |
| 790322 | 2025-07-03 | 2025-07-07 | 2025-07-08 |
| 800792 | 2025-08-15 | 2025-08-18 | 2025-08-19 |
| 800995 | 2025-08-22 | 2025-08-25 | 2025-08-26 |
| 804363 | 2025-08-28 | 2025-08-29 | 2025-09-02 |

Machine-readable: `/scratch/sessions/session18-celltype-heterogeneity/allcoreg/canonical_sessions.csv`.

**Never key a cache by an experience label alone.** Key by `session_key` (`{subject}_{date}`)
and resolve F/N/Np at plot time, or a cache can silently pair with the wrong session.
(`code/matched_cell_index_all_sessions.csv` had this bug and was deleted — see
*Files moved to `archived_files/`* below.)

---

## 2. Coregistration → cell type

Full analysis: `code/sessions/session20-coregistration/REPORT.md`.

### The joined table — start here

**`code/roi_coreg_master_260918.csv`** — one row per segmented ophys ROI (87,535 rows ×
19 columns) across all 6 subjects, 146 sessions, 8 planes, carrying z-stack id, `hcr_id`,
cell type and a one-column `outcome`. Most questions need nothing else.
Columns: `code/sessions/session20-coregistration/roi_coreg_master_COLUMNS.md`.

- **Filter `valid_roi == True` first** — 68,181 of 87,535. The rest touch the motion-correction
  border or are too small and never enter coregistration.
- **Rows are ROIs, not cells.** One cell recurs once per session imaged. Group by `hcr_id`
  (HCR cell) or `ucid` (ROICaT cross-session cell) for unique cells.
- Use **`czstack_id`** (after the ROICaT vote), not `czstack_id_direct` (per-session IoU only).
- 153 rows are `roi_source = "rescued_direct_sept"` and **exist only in this file, not in any
  Code Ocean asset**. Filter `roi_source == "coreg_capsule"` for strictly capsule-derived results.

### Source assets (2026-09-19, subject-clean)

Cell-type yield **43.3 % of valid ROIs**, **all 146 sessions mapped**.

| subject | coreg-id-mapping asset |
|---|---|
| 782149 | `multiplane-ophys_782149_2025-03-28_coreg-id-mapping_2026-09-19_05-47-00` |
| 788406 | `multiplane-ophys_788406_2025-05-29_coreg-id-mapping_2026-09-19_05-50-15` |
| 790322 | `multiplane-ophys_790322_2025-06-11_coreg-id-mapping_2026-09-19_05-53-18` |
| 800792 | `multiplane-ophys_800792_2025-07-22_coreg-id-mapping_2026-09-19_05-50-43` |
| 800995 | `multiplane-ophys_800995_2025-08-05_coreg-id-mapping_2026-09-19_05-46-32` |
| 804363 | `multiplane-ophys_804363_2025-08-12_coreg-id-mapping_2026-09-19_05-46-16` |

Inputs consumed: `code/sessions/session20-coregistration/data_assets_used_rerun_2026-09-19.csv`.
Output columns were renamed at capsule commit `7be40ca`:
`cz_stack_id → initial_matched_czstack_id`, `resolved_cz_stack_id → resolved_czstack_id`.

**When re-running coreg-id-mapping, attach only the subject's own processed sessions**
(1 ROICaT + 1 CTL-fov-to-czstack + 1 autocoreg-qced + N processed). Attaching all six mice
makes provenance unreadable *and* writes the wrong `subject_id` into `data_description.json`.

### Cell types

**Table: `data/hcr-inhibitory-consensus-k18/final_cellxgene_labeled.csv`** — k=18 consensus,
43,328 inhibitory cells. Join on `(mouse, hcr_id)`.

| column | values |
|---|---|
| `subclass` | Pvalb, Sst, Vip, Lamp5, **Sncg**, **Other** — complete, no NaN |
| `cluster` | `Pvalb-1`, `Other-2`, … (18) |
| `cluster_name` | `Pvalb-1  Pthlh`, `Other-1  Chat/Ndnf`, … |

**Do not use `subclass_call`** — a 4-way call that folds Sncg and Other away.
**Do not use the `coreg_*` columns** — stale (1,557 coregistered HCR cells vs 2,508 now).

⚠️ The older `code/archived_files/final_cellxgene_labeled_marina_260913.csv` is the same data with two
columns renamed, so **`subclass` means opposite things in the two files**:

| concept | k18 asset (use this) | marina_260913 copy |
|---|---|---|
| 6-way grouping | **`subclass`** | `block` |
| 4-way call + unassigned | `subclass_call` | `subclass` |

"Cell type assigned" **includes Other and Sncg** (9.3 % of typed ROIs). Four canonical
subclasses only = 39.3 % rather than 43.3 %.

### Files moved to `archived_files/` (2026-09-19) — read this before re-running an old notebook

Two CSVs that older sessions loaded from `code/` now live in `code/archived_files/`:

| file | why |
|---|---|
| `final_cellxgene_labeled_marina_260913.csv` | superseded by the k18 asset above (same data, `block`/`subclass` renamed to `subclass`/`subclass_call`) |
| `matched_cell_index_hcrid_recovered.csv` | built by the hcr_id recovery chain, which is obsolete now that all 146 sessions map |

**Anything still pointing at `code/<file>.csv` will raise `FileNotFoundError`.** These were
left as-is on purpose — they are finished analyses, and their results stand. Known:

- sessions **14, 15, 16, 17, 18, 19** — notebooks and `kernel_features.py` / `build_ve_table.py`
  load the marina CSV
- sessions **12, 14, 15, 16, 17** — reference `matched_cell_index_hcrid_recovered.csv`

If you re-run one of them, repoint it and mind the rename — that file's `block` is the k18
asset's `subclass`, and its `subclass` is `subclass_call`. Reading the k18 `subclass` where the
old code expected `subclass` silently swaps a 6-way grouping for a 4-way call.

`code/matched_cell_index_all_sessions.csv` was **deleted**, not archived — it had the
`Np = OPHYS_6` bug. Use `code/roi_coreg_master_260918.csv` instead.

### Two traps

**The old F/N/N+1 coregistration gradient was an artefact.** On the pre-2026-09-18 tables,
yield rose with experience (F 33.7 %, N 38.8 %, N+1 42.9 %) purely because Familiar sessions
were over-represented among the sessions the run dropped. After the re-run it is flat
(44.7 / 44.6 / 45.1 %). **Any F/N/N+1 comparison computed on the old tables is confounded.**

**ROICaT picks its alignment template by POSITION** (`template_method="sequential"`,
`template=0.5` → the session at the midpoint of the ordered list). Re-running with a different
session set moves it, and if it lands on an outlier the whole plane's alignment fails and the
coreg capsule drops it. After **any** ROICaT re-run:

```bash
python code/sessions/session20-coregistration/scripts/check_roicat_alignment.py <new_roicat_dir> <old_roicat_dir>
```

A plane flipping from mostly-aligned to mostly-failed is this bug, not bad data. Details:
`code/sessions/session20-coregistration/docs/roicat_alignment_template_issue.md`.

---

## 3. Processing (dF/F and events)

Baseline estimation changed. **Old**: 5th percentile of a 60 s long window. **New (`triexp`)**:
multi-exponential fit that ignores extended activity. The new run used the old processed asset
as input through a truncated pipeline fixing only dF/F and the downstream deconvolved events —
so the old asset is findable from the new one's provenance.

**`processing.json` cannot tell the two apart** — the reproc copy carries the old dF/F
parameters. To identify which processing a trace came from, correlate against the GLM asset's
`events_activity_trace_matrix.nc`: r = 1.0000 for new/triexp, r ≈ 0.94–0.96 for old.

---

## 4. GLM: reading kernels for all cells

`kernel_weights_per_cell.csv` covers only tri-session-matched cells and has **no image
kernels**. For all coregistered neurons read the per-session GLM asset:

```python
res = np.load(glob(f'{glm_mount}/glm_results_*_events.npy')[0], allow_pickle=True).item()
W  = res['W_cv'].sel(model='Full').mean('test_fold_ind')     # (weights, cell_roi_id)
ve = res['var_explained_test_cv'].sel(model='Full').mean('test_fold_ind')
```

- `cell_roi_id` is `VISp_<plane>_<idx>` — joins to PSTH cells with no translation. GLM cells
  are a strict subset of QC-valid ROIs.
- `W_cv` dims `(test_fold_ind, model, weights, cell_roi_id)`; no lambda axis. Average the 5 folds.
- Weight names are `hits_0`, `hits_1`, … (**no** `lag`) and sort **lexicographically** — always
  sort lag indices numerically.
- **Lag → time is `lag / fps`** (`fps` = `ophys_frame_rate` from `*activity_trace_info.npy`).
- Kernel spec in `run_params.json`: hits 2.25 s, misses 2.25 s, omissions 3 s, licks 2 s @ −1,
  images 0.75 s. Image kernels are per identity — set **A** (`im061…im085`) in F, set **B**
  (`im000…im106`) in N/Np; average over whichever the session has.
- **fps differs across subjects** (9.48 vs 10.74 Hz), so lag counts differ for the same kernel.
  Interpolate onto a common grid confined to the span *every* session covers, so it never
  extrapolates.
- ⚠️ **`W_cv`'s `weights` coordinate is sorted LEXICOGRAPHICALLY; `design_matrix.nc`'s is
  NUMERIC.** Same 195 names, different order. A positional `X.values @ W.values` therefore
  pairs every regressor with the wrong weight: the reconstructed "fit" has median in-sample
  R² = **−0.55** (the residual has more variance than the data) instead of +0.08. **Always
  align by name** — `W.sel(weights=<design matrix's own name list>)`, which is what the
  pipeline's `glm_fit_tools.get_sessionwise_model_traces` does. Helper:
  `code/sessions/session23-noise-correlation/common/data.py: model_fit()`.
  Code that reads kernels by name (sessions 11/12/16/18/21/22) is unaffected;
  **`session17-signal-noise-corr` is affected and its matrices are invalid.**
- PSTHs (deconvolved event rate) and GLM weights are **different units** — separate y-limits.

---

### Flash-locked latency is a PHASE, not a time (added 2026-09-21, session 22)

Images are flashed every **0.75 s** (measured inter-flash interval 0.7506 s) and the GLM's
image kernel is exactly one cycle long (0 – 0.738 s, 9 bins). A peak at 0.70 s and a peak at
0.05 s are **0.10 s apart on the stimulus cycle**, not 0.65 s apart. Never average, correlate,
regress on, or feed a decoder a raw "time of peak" for anything flash-locked.

Use a **phasor**: `amplitude x (cos th, sin th)` with `th = 2*pi*t/0.75`, or equivalently the real and
imaginary parts of the Fourier coefficient at the flash frequency. Euclidean distance between
phasors is the correct metric, and weighting by amplitude keeps unmodulated cells from voting
with a meaningless phase. For a per-type summary use the **circular mean** and report the
**resultant length** alongside it — Vip's peak phase has resultant 0.43, so its circular mean
is close to meaningless on its own. Helpers: `session22-scalar-features/common/scalars_kernel.py`
(`phasor`, `fourier_coef`, `ring_index`).

Why it matters here, measured: 9.2 % of typed cell-sessions peak in the first or last bin, and
those two groups' mean kernels correlate at **r = 0.96** — the same waveform at opposite ends
of a linear axis. **24.1 %** of all cell pairs are placed further apart by a linear latency than
they are on the cycle. Swapping the phasor for a plain time costs **0.04–0.05 LOSO** on the
image block (session 22 §2). Vip and Sncg sit right on the wrap (circular mean ~0.67 s), so any
linear treatment splits those two populations in half.

**Latencies that stay linear:** time-to-peak of the omission and hit/miss kernels (time since a
one-off event, where the absolute delay is what is meaningful) and everything in the
running / pupil / licks block (not flash-locked). Their *within-cycle* phase can be added as a
separate phasor if the question is about ringing.

There is also an identifiability reason. With flashes tiling the axis at exactly the kernel
length, a kernel and its periodic shift make nearly the same prediction; only changes and
omissions break the tiling. The fitted phase is therefore weakly constrained, and a linear axis
turns that weak constraint into a large numerical difference.

## 5. Figure conventions

**Inhibitory subclass colors — canonical, from `code/spatial_subclass.png`.**
Adopted 2026-09-20. Sampled from that figure's own pixels, so every project figure matches the
spatial-distribution panels. Use these, not the Okabe-Ito set that sessions 14/19/20 used.

| subclass | hex | swatch |
|---|---|---|
| Sst | `#FE9901` | orange |
| Pvalb | `#D93238` | red |
| Lamp5 | `#D98691` | dusty pink |
| Sncg | `#DE7BFB` | magenta |
| Vip | `#A96BC1` | purple |
| Other | `#727272` | grey |

Machine-readable: `code/sessions/session21-celltype-response-specificity/common/palette21.py`
(`SPATIAL_SUBCLASS_COLORS`, merged into `CELLTYPE_COLORS`).

⚠ **Colour-vision caveat, measured not assumed.** Under the OKLab CVD check
(`palette21.validate`, dE x100) this set PASSES the 8 floor for deuteranopia (8.9), protanopia
(10.4) and tritanopia (12.2), but for NORMAL vision two pairs fall below the 15 floor:
**Sncg vs Vip 12.8** and **Lamp5 vs Vip 14.4** — they are adjacent hues in the source figure by
construction. So never let colour alone carry identity for those pairs: pair it with a legend,
a direct label, or a distinct marker/line style. This is a floor for large marks; at small
scatter-point sizes check by eye as well.

**Display order (updated 2026-09-23): shallow → deep mean depth.** Every ordinary plot with
subclass or subtype labels uses **Lamp5 → Sncg → Vip → Sst → Pvalb**. Within each parent,
subtype numbers retain their depth-rank order: Lamp5-1/2/3; Sncg-1/2; Vip-1/2/3/4;
Sst-1/2/3/4; Pvalb-1/2/3/4. Use `palette21.DEPTH_SUBCLASS_ORDER` and
`DEPTH_SUBTYPE_ORDER` (or `P.order_labels()`), rather than an alphabetical or PC1 order.
Colors remain bound to subclass identity, not its display rank.

**Subtype labels include the representative gene except on matched dense matrix axes.** Use the
full consensus-style label `subtype  gene`—for example `Lamp5-3  Npy`—via
`palette21.display_label()` for ordinary labels, legends, and non-matrix plots. When a matrix
has slanted x tick labels and the y axis shows the same cell-type identities, use the compact
subclass--number labels on the **slanted x axis only** (for example `Lamp5-3`) via
`palette21.matrix_label()`; retain the full gene-bearing label on y. A blank gene is retained only where the current consensus
`cluster_name` has none (Sst-3).

**F+N subtype display threshold (updated 2026-09-23).** Matrix, heat-map, and multi-subtype
overlay figures retain every subtype with **n ≥ 15 F+N cells**, in depth order:
Lamp5-1, Lamp5-2, Lamp5-3; Sncg-1, Sncg-2; Vip-1, Vip-2, Vip-3, Vip-4; Sst-1, Sst-2,
Sst-3; Pvalb-1, Pvalb-2, Pvalb-4. This is **15 of 17** observed subtypes. Only Sst-4 (n=5)
and Pvalb-3 (n=7) are omitted. Use `palette21.DISPLAY_SUBTYPE_ORDER`; do not retain an older
"canonical 12/13" list in new or refreshed figures.

**Session-24 main-figure consolidation (2026-09-24).** For every subtype-level main-figure
analysis and label, merge the two Cck/Sncg calls, `Sncg-1` and `Sncg-2`, into one displayed
subtype, **`Sncg`** (n=67 in F+N). This leaves **14 displayed subtypes**. Consolidate
the underlying labels *before* fitting a decoder or aggregating pairwise sums/counts, rather
than merely relabeling separate outputs. Raw consensus labels remain unchanged for provenance
and split-call-specific analyses. Use `palette21.merge_sncg_subtype()` together with
`palette21.DISPLAY_SUBTYPE_ORDER_SNCG_MERGED` for the consolidated deck taxonomy.

**PC1-specific exception.** Section 21 §4 remains an unsupervised MGE↔CGE/PC1 analysis:
its panels that literally show subtype positions along PC1 retain their PC1 sort, because that
sort is the variable being plotted. Their subtype text still uses the full representative-gene
label. `ANALYSIS_ORDER` and `SUBTYPE_ORDER` are therefore analytic PC1 references, not the
default figure display convention.

**⚠ Two subclass palettes — pick by plot type.** The spatial palette above remains the
canonical palette for separated rows, bars, heat-map strips, and legends. For a shared-axis
display—dense per-cell scatter, UMAP/PCA, or traces from multiple subclasses overlaid on the
same axes—use the settled **`palette21.PROJECTION_COLORS`** palette. It was selected by direct
visual review on 2026-09-23 for strong hue and lightness separation; it deliberately does **not**
optimize for colour-vision deficiency.

| subclass | scatter/line colour | hex |
|---|---|---|
| Sst | gold | `#E6B800` |
| Pvalb | bright magenta | `#FF00A8` |
| Lamp5 | bright cyan | `#00C8E8` |
| Vip | bright green | `#00D84A` |
| Sncg | navy | `#142B6F` |
| Other | grey | `#7F7F7F` |

The deciding question is **do different subclasses' marks occupy the same space on the
axes**, not "is this a scatter plot":

| use | palette | why |
|---|---|---|
| a strip plot's jittered columns (each subclass gets its own x-position) | **`SPATIAL_SUBCLASS_COLORS`** | columns never overlap each other — there is no separability problem for the canonical palette to cause |
| stacked traces, one subclass per row/panel | **`SPATIAL_SUBCLASS_COLORS`** | separated by construction |
| bar charts, heat-map row/column strips, legends | **`SPATIAL_SUBCLASS_COLORS`** | one mark per subclass, no overlap |
| a dense per-cell scatter with all subclasses plotted in the same 2-D space | `PROJECTION_COLORS` | dots genuinely overlap |
| several subclass **traces/lines** overlaid on one axes | `PROJECTION_COLORS` | the shared-axis palette makes each trace immediately distinguishable |
| UMAP / PCA projections | `PROJECTION_COLORS` | points genuinely overlap |

Session 23 (`code/sessions/session23-noise-correlation/common/plotting.py`) implements this
as two helpers: `canonical_color()` for the non-overlapping cases and `scatter_color()` for
the overlapping ones; `strip_by_type()` uses `canonical_color()` because a strip plot's
columns never overlap. Getting this backwards was caught 2026-09-21: a first pass used
`PROJECTION_COLORS` for every per-cell plot including non-overlapping strip plots, which
needlessly abandoned the canonical palette everywhere it wasn't required.

Colours are bound to the subclass in both, so the two palettes never disagree about identity,
only about hue. The shared-axis palette was settled 2026-09-23, with the Sst/Lamp5 colors
swapped on 2026-09-24: Sst is gold and Lamp5 is cyan.

**Experience-level colors** (`vba_tools.get_experience_level_colors()`):
Familiar `#0b559f` · Novel `#aa1016` · Novel+1 `#572c92`.

**Flash shading on every flash-aligned axis** (images / omissions / hits / misses), PSTH and
kernel alike, via `plot_flashes_on_trace` from `lamf_analysis.visualization.vba_tools`:

```python
FLASH_CONFIG = {
    'images':    dict(change_time=0, change=None,  omitted=False),   # all gray bands
    'hits':      dict(change_time=0, change=True,  omitted=False),   # blue at t=0
    'misses':    dict(change_time=0, change=True,  omitted=False),
    'omissions': dict(change_time=0, change=None,  omitted=True),    # dashed line at t=0
}
```

Kernel-only regressors (running / pupil / licks) are not flash-aligned — leave them unshaded.
