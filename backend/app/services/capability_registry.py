"""Single source of truth for platform subsystem capability states."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

from backend.app.core.config import settings


def _optional_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _model_dir() -> Path:
    root = Path(settings.MODEL_DIR)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root
    return root


def _fusion_readiness():
    from ml.inference.fusion_runtime import get_fusion_inference_runtime

    runtime = get_fusion_inference_runtime(
        _model_dir(),
        inference_enabled=settings.INFERENCE_ENABLED,
        reset=True,
    )
    return runtime.assess_readiness()


def get_clinician_workspace_capabilities() -> Dict[str, str]:
    readiness = _fusion_readiness()

    if _optional_import("flwr"):
        flower = "INFRASTRUCTURE_READY"
    else:
        flower = "MISSING_DEPENDENCY"

    dice = "NOT_IMPLEMENTED"
    if not _optional_import("dice_ml"):
        dice = "MISSING_DEPENDENCY"

    dowhy = "NOT_IMPLEMENTED"
    if not _optional_import("dowhy"):
        dowhy = "MISSING_DEPENDENCY"

    if not settings.INFERENCE_ENABLED:
        inference = "NOT_ENABLED"
    elif not readiness.dependencies.get("torch") or not readiness.dependencies.get("torch_geometric"):
        inference = "MISSING_DEPENDENCY"
    elif readiness.ready:
        inference = "AVAILABLE"
    else:
        inference = "NOT_IMPLEMENTED"

    return {
        "clinical_ai_inference": inference,
        "multimodal_gat_fusion": inference,
        "dice_counterfactuals": dice,
        "dowhy_causal": dowhy,
        "flower_federated": flower,
        "clinical_decision_support": "NOT_IMPLEMENTED",
        "document_ingestion": "AVAILABLE",
        "privacy_masking_pipeline": "AVAILABLE",
        "ecg_ml_encoder": "NOT_IMPLEMENTED",
        "clinical_text_encoder": "NOT_IMPLEMENTED",
    }


def get_patient_portal_capabilities() -> Dict[str, str]:
    clin = get_clinician_workspace_capabilities()
    return {
        "ai_assisted_assessment": "NOT_ENABLED",
        "ecg_analysis": clin["ecg_ml_encoder"],
        "explanation_summaries": clin["dice_counterfactuals"],
        "causal_insights": clin["dowhy_causal"],
        "clinical_decision_support": clin["clinical_decision_support"],
        "patient_messaging": "NOT_IMPLEMENTED",
    }


def get_platform_capability_report() -> Dict[str, Any]:
    readiness = _fusion_readiness()
    caps = get_clinician_workspace_capabilities()
    return {
        "capabilities": caps,
        "diagnostics": {
            "inference_api_wired": True,
            "inference_enabled_flag": settings.INFERENCE_ENABLED,
            "inference_ready": readiness.ready,
            "inference_blockers": readiness.blockers,
            "model_artifacts_in_model_dir": readiness.checkpoint_present,
            "manifest_present": readiness.manifest_present,
            "checkpoint_trained": readiness.checkpoint_trained,
            "model_loaded": readiness.model_loaded,
            "flower_server_configured": bool(settings.FLOWER_SERVER_ADDRESS),
            "optional_ml_packages": {
                "torch": _optional_import("torch"),
                "transformers": _optional_import("transformers"),
                "torch_geometric": _optional_import("torch_geometric"),
                "dice_ml": _optional_import("dice_ml"),
                "dowhy": _optional_import("dowhy"),
                "flwr": _optional_import("flwr"),
            },
        },
    }
