# S07 — F7 isotonic confidence calibration

## Goal
Post-hoc calibrate any candidate's intrinsic-confidence score against
ground-truth labels (one/zero per CZ-HCR pair) and report operating-point
thresholds at target precisions.

## API
`lib/calibrate.py::fit_isotonic(scores, labels) → CalibrationResult`
with fields `.iso` (sklearn.IsotonicRegression), `.brier`, `.thresholds`
(P@{90, 95, 99} recovered via monotone interpolation).

## Self-test
Synthetic mixture: 1 k positives sampled from Beta(4, 2), 5 k negatives
from Beta(2, 4).  Post-calibration Brier = 0.040, monotone map recovers
P@95 within 0.02 of the analytic Beta-CDF threshold.  Pass (target was
Brier < 0.10).

## Files
- `lib/calibrate.py`
