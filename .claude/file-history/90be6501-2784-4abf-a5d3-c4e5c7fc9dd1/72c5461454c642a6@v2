"""F7 — Confidence calibration.

Isotonic calibration of candidate per-pair scores to GT labels.  Pure
post-hoc monotone remap — does not replace intrinsic confidence; rather
produces a probability scale for GUI thresholding and reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


@dataclass
class CalibrationResult:
    iso: object = None
    brier: float = float("nan")
    thresholds: dict = field(default_factory=dict)  # {"p_at_0.95_precision": ..., etc}

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.iso is None:
            return np.asarray(scores, dtype=float)
        return np.asarray(self.iso.predict(scores), dtype=float)


def fit_isotonic(scores: Iterable[float], labels: Iterable[int]) -> CalibrationResult:
    scores = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(labels), dtype=int)
    if not _HAS_SKLEARN:
        return CalibrationResult()
    if len(scores) < 5 or len(np.unique(labels)) < 2:
        return CalibrationResult()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, labels)
    probs = iso.predict(scores)
    brier = float(brier_score_loss(labels, probs))

    thresholds = {}
    # Precision-at-score walk
    order = np.argsort(-scores)  # descending
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    prec = tp / np.maximum(tp + fp, 1)
    for target in (0.90, 0.95, 0.99):
        idx = np.where(prec >= target)[0]
        if len(idx):
            k = idx[-1]
            thresholds[f"score_at_p{int(target*100)}"] = float(scores[order][k])
            thresholds[f"prob_at_p{int(target*100)}"] = float(probs[order][k])

    return CalibrationResult(iso=iso, brier=brier, thresholds=thresholds)


def _selftest():
    rng = np.random.default_rng(0)
    n = 500
    scores = np.concatenate([rng.normal(0.8, 0.2, n // 2), rng.normal(0.2, 0.2, n // 2)])
    labels = np.concatenate([np.ones(n // 2), np.zeros(n // 2)]).astype(int)
    r = fit_isotonic(scores, labels)
    print("F7 selftest brier", r.brier, "thresholds", r.thresholds)


if __name__ == "__main__":
    _selftest()
