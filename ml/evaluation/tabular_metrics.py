"""
Evaluation metrics and validation-only threshold selection for tabular baseline.

Protocol: docs/research/TABULAR_BASELINE_PROTOCOL_v1.md
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ml.evaluation.calibration import compute_calibration_metrics

CANDIDATE_THRESHOLDS = tuple(round(x * 0.05, 2) for x in range(1, 20))


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        n1 = int(np.sum(y_true == 1))
        n0 = int(np.sum(y_true == 0))
        if n0 == 0 or n1 == 0:
            return float("nan")
        order = np.argsort(y_score)
        sorted_scores = y_score[order]
        unique_scores, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
        tie_offsets = np.cumsum(np.concatenate(([0], counts[:-1])))
        avg_rank_per_unique = tie_offsets + (counts - 1) / 2.0
        ranks = avg_rank_per_unique[inverse]
        orig_ranks = np.empty_like(ranks)
        orig_ranks[order] = ranks
        r1 = float(np.sum(orig_ranks[y_true == 1]))
        u = r1 - (n1 * (n1 - 1)) / 2.0
        return float(u / (n0 * n1))


def _compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        n1 = int(np.sum(y_true == 1))
        if n1 == 0:
            return float("nan")
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        tp_cumsum = np.cumsum(y_sorted == 1)
        k = np.arange(1, len(y_sorted) + 1)
        precisions = tp_cumsum / k
        # AP is weighted mean of precisions at each relevant threshold
        relevant_precisions = precisions[y_sorted == 1]
        return float(np.sum(relevant_precisions) / n1)


def compute_confusion_and_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true_b = y_true.astype(int)
    y_pred_b = y_pred.astype(int)

    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "youden_j": float(sensitivity + specificity - 1.0),
    }


def find_best_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidate_thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
    fallback_threshold: float = 0.5,
) -> tuple[float, float, list[dict[str, Any]]]:
    """
    Select decision threshold on validation set using Youden's J (sensitivity + specificity - 1).
    Frozen for test evaluation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_threshold = fallback_threshold
    best_j = -float("inf")
    grid_results = []

    for t in candidate_thresholds:
        y_pred = (y_prob >= t).astype(int)
        rates = compute_confusion_and_rates(y_true, y_pred)
        j = rates["youden_j"]
        grid_results.append({
            "threshold": t,
            "youden_j": j,
            "sensitivity": rates["sensitivity"],
            "specificity": rates["specificity"],
            "f1": rates["f1"],
        })
        if j > best_j:
            best_j = j
            best_threshold = t
        elif abs(j - best_j) < 1e-6 and abs(t - 0.5) < abs(best_threshold - 0.5):
            # Tie-breaker: choose threshold closest to 0.5
            best_threshold = t

    if best_j == -float("inf") or math.isnan(best_j):
        best_threshold = fallback_threshold
        best_j = 0.0

    return float(best_threshold), float(best_j), grid_results


def compute_tabular_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute full Prompt 17 metrics:
    Primary: AUROC
    Secondary: AUPRC, precision, F1, sensitivity, specificity, confusion matrix, Brier, ECE.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities contain non-finite values.")
    y_prob = np.clip(y_prob, 0.0, 1.0)

    y_pred = (y_prob >= threshold).astype(int)
    rates = compute_confusion_and_rates(y_true, y_pred)
    cm = rates["confusion_matrix"]
    n = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
    accuracy = float((cm["tp"] + cm["tn"]) / n) if n else float("nan")
    balanced_accuracy = float((rates["sensitivity"] + rates["specificity"]) / 2.0)

    out: dict[str, Any] = {
        "applied_threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "n_samples": int(n),
        "positive_rate": float(np.mean(y_true)) if n else float("nan"),
        "auroc": _compute_auroc(y_true, y_prob),
        "auprc": _compute_auprc(y_true, y_prob),
        "precision": rates["precision"],
        "f1": rates["f1"],
        "sensitivity": rates["sensitivity"],
        "recall": rates["recall"],
        "specificity": rates["specificity"],
        "confusion_matrix": rates["confusion_matrix"],
        "youden_j": rates["youden_j"],
    }

    try:
        calib = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins)
        out["brier_score"] = calib["brier_score"]
        out["ece"] = calib["expected_calibration_error"]["ece"]
        out["calibration_details"] = calib["expected_calibration_error"]
    except Exception as exc:
        out["brier_score"] = float(np.mean((y_prob - y_true) ** 2))
        out["ece"] = float("nan")
        out["calibration_details"] = {"error": str(exc)}

    return out
