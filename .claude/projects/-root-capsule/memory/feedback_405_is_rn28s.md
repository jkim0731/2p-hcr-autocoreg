---
name: 405 channel is Rn28S, not DAPI / nuclear
description: HCR 405 stains 28S rRNA (cytoplasmic / perinuclear ribosomal RNA), not DNA. Stop calling it DAPI or "nuclear channel".
type: feedback
originSessionId: 0eccb3d7-adef-4802-b154-c928aa0e93fe
---
The HCR 405 channel labels **Rn28S** (28S ribosomal RNA, an HCR probe).
It localises to **cytoplasmic / perinuclear ribosomes**, not DNA in
the nucleus. Do **not** call it DAPI or "nuclear / nucleus channel"
in writing, code comments, docs, memory, or chat.

**Why:** User has corrected this multiple times; the wrong framing
leaks into docs and skews how readers reason about which structure
the 405 signal marks. DAPI binds dsDNA → nucleus only; Rn28S marks
ribosome-rich cytoplasm and perinuclear regions. They make different
predictions about where signal should appear inside a real cell
(uniform cytoplasm-with-rim for Rn28S vs centred punctum for DAPI),
which matters for the v5d ROI-quality features (`c405_core_*`,
`c405_shell_*`, core-vs-shell ratios) and for any pseudo-label
heuristic about "405 should be detectable inside a real cell".

**How to apply:**
- Refer to 405 as "405 (Rn28S)" or "Rn28S channel" in any new
  writing. Never write "DAPI" or "nuclear channel" for 405.
- Heuristics phrased as "no nucleus where one should be" must be
  rephrased to "no Rn28S signal where one should be".
- Authoritative source: `code/docs/01 Data Description.md` and
  `project_S11_v2_features_results.md` (memory) both state
  `405 = Rn28S`.
- When updating docs, check `code/docs/10 Grand Plan v3 — Cell-cell
  matching and QC.md` (already corrected 2026-05-08) and any future
  Grand Plan / S11 docs.
