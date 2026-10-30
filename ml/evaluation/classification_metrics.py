"""Binary classification metrics for held-out evaluation reports (Class 0=NORM, Class 1=MI/STTC)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_LABELS = ("NORM", "MI/STTC")
POSITIVE_CLASS_INDEX = 1
POSITIVE_CLASS_LABEL = "MI/STTC"


def _nan_reason_unavailable() -> str:
    return "unavailable"


def compute_full_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score_positive: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Full metric bundle for binary proxy task.

    y_score_positive: P(class 1 = MI/STTC), used for AUROC / AUPRC — not max-softmax confidence.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    n = int(y_true_arr.size)
    if n == 0:
        return {
            "n_samples": 0,
            "status": "empty",
            "reason": "No evaluation examples.",
        }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    correct = int((y_true_arr == y_pred_arr).sum())
    incorrect = n - correct

    uniq = np.unique(y_true_arr)
    both_classes_present = uniq.size >= 2

    per_class_precision = precision_score(y_true_arr, y_pred_arr, labels=[0, 1], average=None, zero_division=0)
    per_class_recall = recall_score(y_true_arr, y_pred_arr, labels=[0, 1], average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=[0, 1], average=None, zero_division=0)
    supports = [int((y_true_arr == c).sum()) for c in (0, 1)]

    out: Dict[str, Any] = {
        "n_samples": n,
        "n_correct": correct,
        "n_incorrect": incorrect,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "confusion_matrix_raw": {
            "labels_order": list(CLASS_LABELS),
            "rows_actual": list(CLASS_LABELS),
            "cols_predicted": list(CLASS_LABELS),
            "matrix": cm.tolist(),
        },
        "per_class": {
            CLASS_LABELS[0]: {
                "precision": float(per_class_precision[0]),
                "recall": float(per_class_recall[0]),
                "f1": float(per_class_f1[0]),
                "support": supports[0],
            },
            CLASS_LABELS[1]: {
                "precision": float(per_class_precision[1]),
                "recall": float(per_class_recall[1]),
                "f1": float(per_class_f1[1]),
                "support": supports[1],
            },
        },
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "class_1_precision": float(per_class_precision[1]),
        "class_1_recall_sensitivity": float(per_class_recall[1]),
        "class_1_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan,
        "class_0_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else math.nan,
        "class_1_npv": float(tn / (tn + fn)) if (tn + fn) > 0 else math.nan,
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred_arr)) if both_classes_present else math.nan,
        "cohen_kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr)) if both_classes_present else math.nan,
        "classification_report_text": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=[0, 1],
            target_names=list(CLASS_LABELS),
            digits=4,
            zero_division=0,
        ),
        "positive_class_index": POSITIVE_CLASS_INDEX,
        "positive_class_label": POSITIVE_CLASS_LABEL,
        "score_semantics": "softmax_probability_class_1_for_auroc",
        "confidence_note": (
            "GAT uses argmax for predicted class; reported confidence in API is max(softmax) "
            "and is uncalibrated — not used as P(MI/STTC) for these metrics."
        ),
    }

    if y_score_positive is not None:
        scores = np.asarray(y_score_positive, dtype=float)
        if scores.shape[0] != n:
            raise ValueError("y_score_positive length must match y_true")
        if both_classes_present and np.isfinite(scores).all():
            out["auroc"] = float(roc_auc_score(y_true_arr, scores))
            out["average_precision"] = float(average_precision_score(y_true_arr, scores))
            fpr, tpr, _ = roc_curve(y_true_arr, scores, pos_label=1)
            prec, rec, _ = precision_recall_curve(y_true_arr, scores, pos_label=1)
            out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            out["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
        else:
            reason = "only one class in y_true" if not both_classes_present else "non-finite scores"
            out["auroc"] = math.nan
            out["average_precision"] = math.nan
            out["auroc_unavailable_reason"] = reason
            out["average_precision_unavailable_reason"] = reason
    else:
        out["auroc"] = math.nan
        out["average_precision"] = math.nan
        out["auroc_unavailable_reason"] = "probability scores not provided"
        out["average_precision_unavailable_reason"] = "probability scores not provided"

    if not both_classes_present:
        out["mcc_unavailable_reason"] = "only one class present in y_true"
        out["cohen_kappa_unavailable_reason"] = "only one class present in y_true"

    class_counts: Dict[str, int] = {}
    for c in (0, 1):
        class_counts[str(c)] = int((y_true_arr == c).sum())
    out["class_counts"] = class_counts
    return out


def metrics_comparison_row(model_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten selected metrics for CSV / comparison table."""
    if metrics.get("n_samples", 0) == 0:
        return {"model_id": model_id, "status": "unavailable"}
    return {
        "model_id": model_id,
        "n_samples": metrics["n_samples"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "class_1_precision": metrics["class_1_precision"],
        "class_1_recall_sensitivity": metrics["class_1_recall_sensitivity"],
        "class_0_specificity": metrics.get("class_0_specificity"),
        "auroc": metrics.get("auroc"),
        "average_precision": metrics.get("average_precision"),
        "mcc": metrics.get("mcc"),
    }


def absolute_metric_differences(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    *,
    keys: Tuple[str, ...] = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "class_1_precision",
        "class_1_recall_sensitivity",
        "auroc",
        "average_precision",
        "mcc",
    ),
) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    for k in keys:
        a, b = row_a.get(k), row_b.get(k)
        if a is None or b is None:
            continue
        if isinstance(a, float) and (math.isnan(a) or math.isnan(b)):
            continue
        diffs[k] = float(b) - float(a)
    return diffs
