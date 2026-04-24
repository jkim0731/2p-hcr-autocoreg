"""Render iter05 fitted-surface overlays for the 2 subjects whose PNGs
never wrote because the first iter05 run crashed on 790322 (module
re-exec collided with in-flight edits).

This reuses ``iter05_compute.render_subject`` as-is — the robust-ref
fix to ``tissue_support`` now lands correctly because the code base is
stable.  Writes PNGs only; no CSV (keep iter05_audit.csv for the full
6-subject table by re-running the full script later if needed).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

# Import iter05_compute as a module.
spec = importlib.util.spec_from_file_location(
    "iter05_compute_mod",
    str(ROOT / "code" / "sessions" / "03c_onset_features"
         / "iterations" / "iter05_compute.py"),
)
iter05 = importlib.util.module_from_spec(spec)
sys.modules["iter05_compute_mod"] = iter05
spec.loader.exec_module(iter05)

SUBJECTS = ["788406", "790322"]
for sid in SUBJECTS:
    iter05.render_subject(sid)
