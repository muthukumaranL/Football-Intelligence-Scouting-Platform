"""Audit whether transfer-success probabilities match observed frequencies.

Classification rank metrics such as ROC-AUC do not show whether a predicted
70% chance actually succeeds about 70% of the time. This helper reports Brier
score plus equal-width reliability bins for scouting decision support.
"""
from __future__ import annotations

import numpy as np


def calibration_audit(y_true, probabilities, bins: int = 10) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be aligned non-empty vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    rows = []
    weighted_gap = 0.0
    for idx in range(bins):
        mask = assignments == idx
        if not np.any(mask):
            continue
        mean_probability = float(np.mean(p[mask]))
        observed_rate = float(np.mean(y[mask]))
        count = int(np.sum(mask))
        gap = abs(mean_probability - observed_rate)
        weighted_gap += gap * count / y.size
        rows.append({
            "bin": idx + 1,
            "count": count,
            "mean_probability": mean_probability,
            "observed_success_rate": observed_rate,
            "calibration_gap": gap,
        })

    brier = float(np.mean((p - y) ** 2))
    return {"brier_score": brier, "expected_calibration_error": weighted_gap, "bins": rows}


if __name__ == "__main__":
    outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    predicted = [0.82, 0.35, 0.72, 0.65, 0.28, 0.42, 0.77, 0.21, 0.61, 0.88]
    print(calibration_audit(outcomes, predicted, bins=5))
