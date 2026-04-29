---
name: Archived dirs moved to /data/claude_data (read-only)
description: notebooks/, sessions/, and full_automatic_execution_01/ now live under /root/capsule/data/claude_data/ which is read-only; new writes go under /root/capsule/code/.
type: reference
originSessionId: 40fd9680-417d-48de-ac1a-71e8a68a7966
---
**As of 2026-04-28**, three directories were relocated out of
`/root/capsule/code/` to free capsule-size headroom:

| former path                                  | new path (read-only)                                          |
|----------------------------------------------|---------------------------------------------------------------|
| `/root/capsule/code/notebooks/`              | `/root/capsule/data/claude_data/notebooks/`                   |
| `/root/capsule/code/sessions/`               | `/root/capsule/data/claude_data/sessions/`                    |
| `/root/capsule/code/full_automatic_execution_01/` | `/root/capsule/data/claude_data/full_automatic_execution_01/` |

**Constraints.**
* `/root/capsule/data/` is **read-only** — all `claude_data/` contents
  can be read but not modified.
* New notebooks, new sessions, new modules, anything that has to be
  written, must live under `/root/capsule/code/` (e.g.,
  `/root/capsule/code/full_automatic_execution_02/sessions/...`,
  `/root/capsule/code/dev_code/...`).
* Existing v1 candidate impls / bench harness still live under
  `data/claude_data/full_automatic_execution_01/`; v2 imports them via
  the moved path. If you change an import, update the path prefix from
  `code/full_automatic_execution_01/` to
  `data/claude_data/full_automatic_execution_01/`.
* Doc cross-refs in `docs/01,04,06,07,08,09` were updated 2026-04-28
  with a top-of-file pointer; older inline references (`code/sessions/
  NN_*/log.md`, `code/notebooks/...`) should be read as
  `data/claude_data/...` until a future pass rewrites them in place.

When in doubt: try a path under `code/` first; if missing, look under
`data/claude_data/` with the same suffix.
