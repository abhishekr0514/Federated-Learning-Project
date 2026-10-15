"""Calibration metrics for binary research evaluation (Prompt 12A)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class CalibrationInputError(ValueError):
    pass


def _validate_binary_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if y_true.size == 0 or y_prob.size == 0:
        raise CalibrationInputError("Empty inputs.")
    if y_true.shape != y_prob.shape:
        raise CalibrationInputError("y_true and y_prob must have the same shape.")
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if not np.isfinite(y_prob).all():
        raise CalibrationInputError("Probabilities must be finite.")
    if np.min(y_prob) < 0.0 or np.max(y_prob) > 1.0:
        raise CalibrationInputError("Probabilities must lie in [0, 1].")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise CalibrationInputError("Labels must be binary 0/1.")
    return y_true, y_prob


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    ECE with uniform probability bins on [0, 1].

    Empty bins contribute 0 weight; documented in returned summary.
    """
    if n_bins < 1:
        raise CalibrationInputError("n_bins must be >= 1.")
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_summary: list[dict[str, Any]] = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= low) & (y_prob < high)
        else:
            mask = (y_prob >= low) & (y_prob <= high)
        count = int(mask.sum())
        if count == 0:
            bins_summary.append(
                {
                    "bin": i,
                    "range": [float(low), float(high)],
                    "count": 0,
                    "mean_predicted": None,
                    "fraction_positive": None,
                    "empty": True,
                }
            )
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (count / n) * abs(frac_pos - mean_pred)
        bins_summary.append(
            {
                "bin": i,
                "range": [float(low), float(high)],
                "count": count,
                "mean_predicted": mean_pred,
                "fraction_positive": frac_pos,
                "empty": False,
            }
        )

    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "n_samples": n,
        "binning": "uniform [0,1]",
        "reliability_bins": bins_summary,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    return {
        "brier_score": brier_score(y_true, y_prob),
        "expected_calibration_error": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "note": (
            "Research calibration only; small smoke samples are not clinically meaningful."
        ),
    }
