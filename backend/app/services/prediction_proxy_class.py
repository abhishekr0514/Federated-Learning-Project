"""Resolve research proxy class labels from persisted prediction rows."""

from __future__ import annotations

from typing import Any, Dict, Optional

MODEL_FAMILY_GAT_ECG_REPORT_FUSION = "MedShield-GAT-ECG-Report-Fusion"
MODEL_FAMILY_GAT_MULTIMODAL = "MedShield-GAT-Multimodal"
MODEL_FAMILY_ECG_REPORT_FUSION = "MedShield-ECG-Report-Fusion"

LEGACY_PROXY_CLASS_LABEL = "Legacy (class not stored)"
GAT_LEGACY_REPLAY_CLASS_INDEX = -1
GAT_LEGACY_REPLAY_CLASS_LABEL = LEGACY_PROXY_CLASS_LABEL

PROXY_CLASS_LABELS = {0: "NORM", 1: "MI/STTC"}


def uses_gat_max_softmax_confidence(model_family: str) -> bool:
    """GAT paths store max(softmax); class cannot be inferred from confidence alone."""
    return model_family in (
        MODEL_FAMILY_GAT_ECG_REPORT_FUSION,
        MODEL_FAMILY_GAT_MULTIMODAL,
    )


def _class_from_index(class_index: int) -> Dict[str, Any]:
    label = PROXY_CLASS_LABELS[class_index]
    return {
        "predicted_class_index": class_index,
        "predicted_class_label": label,
        "predicted_proxy_class": label,
        "proxy_class_status": "STORED",
    }


def resolve_patient_portal_proxy_class(
    *,
    confidence: Optional[float],
    predicted_class_index: Optional[int],
    model_family: str,
    class_meaning_map: Dict[str, str],
) -> Dict[str, Any]:
    """Patient history/chart payload fields for proxy class (never guess GAT class from confidence)."""
    if predicted_class_index is not None and predicted_class_index in PROXY_CLASS_LABELS:
        payload = _class_from_index(predicted_class_index)
        payload["class_meaning"] = class_meaning_map.get(payload["predicted_class_label"])
        return payload

    if model_family == MODEL_FAMILY_ECG_REPORT_FUSION and confidence is not None:
        idx = 1 if confidence >= 0.5 else 0
        payload = _class_from_index(idx)
        payload["proxy_class_status"] = "DERIVED_FROM_ECG_PROBABILITY"
        payload["class_meaning"] = class_meaning_map.get(payload["predicted_class_label"])
        return payload

    if uses_gat_max_softmax_confidence(model_family) or predicted_class_index is None:
        return {
            "predicted_class_index": None,
            "predicted_class_label": None,
            "predicted_proxy_class": LEGACY_PROXY_CLASS_LABEL,
            "proxy_class_status": "LEGACY_UNAVAILABLE",
            "class_meaning": None,
        }

    return {
        "predicted_class_index": None,
        "predicted_class_label": None,
        "predicted_proxy_class": LEGACY_PROXY_CLASS_LABEL,
        "proxy_class_status": "LEGACY_UNAVAILABLE",
        "class_meaning": None,
    }


def gat_idempotent_replay_class(row_predicted_class_index: Optional[int]) -> tuple[int, str]:
    """GAT evaluate idempotent replay: use stored argmax; never derive from max-softmax confidence."""
    if row_predicted_class_index is not None and row_predicted_class_index in PROXY_CLASS_LABELS:
        return row_predicted_class_index, PROXY_CLASS_LABELS[row_predicted_class_index]
    return GAT_LEGACY_REPLAY_CLASS_INDEX, GAT_LEGACY_REPLAY_CLASS_LABEL


def ecg_idempotent_replay_class(
    row_predicted_class_index: Optional[int],
    confidence: Optional[float],
) -> tuple[int, str]:
    """ECG+report replay: stored index first, else P(MI/STTC) threshold (unchanged semantics)."""
    if row_predicted_class_index is not None and row_predicted_class_index in PROXY_CLASS_LABELS:
        return row_predicted_class_index, PROXY_CLASS_LABELS[row_predicted_class_index]
    prob = confidence or 0.0
    idx = 1 if prob >= 0.5 else 0
    return idx, PROXY_CLASS_LABELS[idx]
