"""Prototype iter07 on 755252 + 790322 only — visual check first."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path("/root/capsule")
sys.path.insert(0, str(ROOT / "code" / "dev_code"))

spec = importlib.util.spec_from_file_location(
    "iter07_mod",
    str(ROOT / "code" / "sessions" / "03c_onset_features"
         / "iterations" / "iter07_compute.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules["iter07_mod"] = mod
spec.loader.exec_module(mod)

for sid in ["755252", "790322"]:
    row = mod.render_subject(sid)
    print("  row:", row)
