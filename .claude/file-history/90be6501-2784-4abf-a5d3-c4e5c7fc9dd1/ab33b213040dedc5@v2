"""F7 calibrators — pickled `CalibrationResult` objects for P1, P4, P6.

Loader helper so callers don't hardcode the path:

    from lib.calibrators import load_calibrator
    cal = load_calibrator("P1")
    p_cal = cal.predict(pairs_df["confidence"].values)

Fit during S55 (see `sessions/55_f7_calibration/fit_calibration.py`) on all 6
benchmark subjects' labeled pairs. Each is a monotone isotonic map from the
method's raw confidence score to P(pair_correct | conf).

Training Brier on pooled 6-subject data:
  P1 — 0.157 (base rate 0.206)
  P4 — 0.125 (base rate 0.150) — largest improvement (raw Brier 0.47-0.64 → 0.07-0.24 LOSO)
  P6 — 0.149 (base rate 0.185)

Calibrators are useful for GUI thresholding and any downstream code that
needs to act on absolute probabilities (e.g. P@95 precision thresholds).
They do NOT improve C5's ensemble r@20 — per-density priority dispatch is
already within 0.001 of the oracle. See S55 log.md.
"""
from __future__ import annotations

import pickle
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent


def load_calibrator(method_id: str):
    """Load pickled `CalibrationResult` for method_id ∈ {'P1', 'P4', 'P6'}."""
    mid = method_id.lower()
    path = _CALIB_DIR / f"{mid}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No calibrator at {path}. Run S55 to fit.")
    with open(path, "rb") as f:
        return pickle.load(f)
