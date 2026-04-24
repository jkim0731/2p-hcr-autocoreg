#!/usr/bin/env python
"""Top-level CLI for the F9 benchmark harness.

Usage:
    python run_candidate.py P1 788406
    python run_candidate.py P1 all
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from bench import candidates as _cand  # noqa: F401,E402  (triggers registration)
from bench.harness import main  # noqa: E402


if __name__ == "__main__":
    main(sys.argv[1:])
