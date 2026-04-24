# S25 — G1-review minimum-viable review GUI (stub)

## Goal
Human-in-the-loop reviewer that consumes a candidate's `CoregResult` and
writes accept/reject/adjust actions to `qc_actions.jsonl` — the training
corpus for the G-series stage-2 fine-tune.

## API (stub implemented)
`gui/review_gui.py`:
- `ReviewSession(s, candidate_result, out_path)` — loads CZ z-stack + HCR
  level-2 into a headless backing store (napari-compatible structure, but
  the test harness uses a CLI interface so no display is required here).
- Actions: `session.accept(pair_idx)`, `session.reject(pair_idx,
  reason=...)`, `session.adjust(pair_idx, new_hcr_id=...)`.
- Writes one JSON per action (fields: `timestamp`, `subject_id`,
  `candidate_id`, `action`, `cz_id`, `hcr_id_before`, `hcr_id_after`,
  `reviewer`, `note`).

## Demo log
Running the stub on a synthetic 10-pair fixture produced
`qc_actions.jsonl` with 10 accept events; the JSON was re-parsed and
reconstituted into an updated `landmarks_qced.csv`.  Manual spot-check:
the CSV preserves the `active=true` flag on all accepted rows.

## Deferred
The napari UI itself is not started here — it requires a display and is
heavy to spin up in auto-mode benchmarking.  The data-layer API is
stable and sufficient for the G-series stage-2 training consumer.

## Files
- `gui/review_gui.py`
