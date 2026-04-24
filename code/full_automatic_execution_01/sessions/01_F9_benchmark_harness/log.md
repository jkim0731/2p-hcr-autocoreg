# Session 01 — F9 benchmark harness

**Subgoal ID:** F9
**Plan reference:** `07 Grand Plan.md §3 F9`
**Status:** completed

## Goal

A single entrypoint that loads a benchmark subject, invokes a candidate's
`fn(s) -> CoregResult`, scores the returned predictions against
`coreg_table.csv`, and appends a row to `bench_results.csv`.

## Method

`bench/harness.py`:

- `CoregResult(pairs_df, confidence, transform, diagnostics)` dataclass with
  a strict column contract: `cz_id, hcr_id, confidence, cz_x_um, cz_y_um,
  cz_z_um, hcr_x_um, hcr_y_um, hcr_z_um`.
- `TransformDescriptor(R, scales, translation, src_mean, rotation_deg_z, kind)`
  — same row-vector convention as `CoarseAffineV2` / `ProcrustesFit`.
- `@register_candidate("ID")` decorator populates `CANDIDATES: Dict[str, fn]`.
- `bench/candidate_impls/*.py` — one candidate per file. `bench/candidates.py`
  auto-imports all of them on load.
- `run_candidate(candidate_id, subject_id)` returns a flat metrics dict and
  writes three outputs:
  - `bench_out/{id}/{subj}_pairs.csv`
  - `bench_out/{id}/{subj}_diagnostics.json`
  - appends to `bench_out/bench_results.csv`.

`compare_to_gt` emits ID-level recall/precision and centroid-distance
`recall@5um`, `recall@10um`, `recall@20um` (distance is measured from the
predicted HCR centroid to the GT HCR cell's centroid, so wrong-ID-but-near
matches get partial distance credit).

`transform_error_vs_landmarks` evaluates an emitted transform against the
manual landmark set (origin error, rotation-around-z, RMS).

## CLI

`python run_candidate.py <candidate_id> <subject_id|all>`

## Smoke test

A reference candidate `REF_GT` short-circuits by copying the subject's
`coreg_table.csv` as its prediction. Running against 788406:

    [REF_GT:788406] recall=1.00 recall@10um=1.00 n_pred=787 runtime=0.007s

confirms scoring mechanics are correct end-to-end.

## Column conventions consumed from the loader

- `cz_centroids`: columns `cz_id, z_px, y_px, x_px` (NOT `id, z, y, x`).
- `hcr_centroids`: columns `hcr_id, z_px, y_px, x_px`.
- `coreg_table`: columns `cz_id, hcr_id`.

## Deviations from the plan

None — matches Section 3 F9 spec exactly. The CLI module-run warning
forced a small refactor: top-level `run_candidate.py` script dispatches
into `bench.candidates` + `bench.harness.main`, avoiding the
`-m bench.harness` issue of decorators running under a different module
identity.

## Next

S02 — F6 per-cell feature extractor.
