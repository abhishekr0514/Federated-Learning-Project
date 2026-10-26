"""Single source of truth for platform subsystem capability states."""

from __future__ import annotations

import importlib.util
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from backend.app.core.config import settings
from backend.app.services.capability_cache import get_cached_capability_bundle

logger = logging.getLogger(__name__)


def _optional_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ecg_report_readiness(*, verify_forward_pass: bool = False) -> dict[str, Any]:
    from backend.app.services.ecg_report_inference_service import ECGReportInferenceService

    return ECGReportInferenceService.get_readiness(verify_forward_pass=verify_forward_pass)


def _gat_fusion_readiness() -> dict[str, Any]:
    from backend.app.services.gat_fusion_service import GATFusionService

    return GATFusionService.get_readiness()


def _capability_from_readiness(
    *,
    enabled_flag: bool,
    ready: bool,
    blockers: list[str],
    disabled_label: str = "NOT_ENABLED",
) -> str:
    if not enabled_flag:
        return disabled_label
    if not _optional_import("torch"):
        return "MISSING_DEPENDENCY"
    if ready:
        return "AVAILABLE"
    if blockers:
        return "CONFIGURATION_REQUIRED"
    return "UNAVAILABLE"


def _detail(summary: str, reason: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": summary,
        "reason": reason,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    return payload


def _build_capability_details() -> Dict[str, Dict[str, Any]]:
    t0 = time.perf_counter()
    ecg = _ecg_report_readiness(verify_forward_pass=False)
    logger.debug("capability_timing ecg_readiness_ms=%.1f", (time.perf_counter() - t0) * 1000)
    t1 = time.perf_counter()
    gat = _gat_fusion_readiness()
    logger.debug("capability_timing gat_readiness_ms=%.1f", (time.perf_counter() - t1) * 1000)
    flower = __import__(
        "backend.app.services.federated_status_service",
        fromlist=["get_flower_capability_status"],
    ).get_flower_capability_status()
    causal = __import__(
        "backend.app.services.causal_analysis_service",
        fromlist=["CausalAnalysisService"],
    ).CausalAnalysisService.get_readiness(build_cohort=False)

    ecg_summary = _capability_from_readiness(
        enabled_flag=settings.ECG_REPORT_INFERENCE_ENABLED,
        ready=bool(ecg.get("ready")),
        blockers=list(ecg.get("blockers") or []),
    )
    gat_summary = _capability_from_readiness(
        enabled_flag=settings.FUSION_INFERENCE_ENABLED,
        ready=bool(gat.get("ready")),
        blockers=list(gat.get("blockers") or []),
    )

    dice_summary = "NOT_IMPLEMENTED"
    dice_reason = "DiCE counterfactual pipeline and API are not implemented (no trained explainer model wired)."
    if not _optional_import("dice_ml"):
        dice_summary = "MISSING_DEPENDENCY"
        dice_reason = "dice_ml package not installed; feature remains not implemented."

    return {
        "ecg_report_inference": _detail(
            ecg_summary,
            "; ".join(ecg.get("blockers") or [])
            or (
                "ECG + report encoder configured; weights load on first inference."
                if not ecg.get("runtime_initialized")
                else "ECG + report encoder ready for inference."
            ),
            ready=bool(ecg.get("ready")),
            enabled=settings.ECG_REPORT_INFERENCE_ENABLED,
            blockers=ecg.get("blockers") or [],
            runtime_initialized=bool(ecg.get("runtime_initialized")),
        ),
        "multimodal_gat_fusion": _detail(
            gat_summary,
            "; ".join(gat.get("blockers") or [])
            or (
                "Two-modality GAT fusion configured; weights load on first inference."
                if not gat.get("model_loaded")
                else "Two-modality GAT fusion ready."
            ),
            ready=bool(gat.get("ready")),
            enabled=settings.FUSION_INFERENCE_ENABLED,
            blockers=gat.get("blockers") or [],
            embedding_contract=gat.get("embedding_contracts"),
            runtime_initialized=bool(gat.get("model_loaded")),
        ),
        "flower_federated": _detail(
            flower["summary"],
            flower["reason"],
            **{k: flower[k] for k in flower if k not in ("summary", "reason")},
        ),
        "dice_counterfactuals": _detail(dice_summary, dice_reason),
        "dowhy_causal": _detail(
            causal["summary"],
            causal["reason"],
            blockers=causal.get("blockers") or [],
            cohort_id=causal.get("cohort_id"),
        ),
        "clinical_decision_support": _detail(
            "NOT_IMPLEMENTED",
            "Clinical decision support rules engine is not implemented.",
        ),
    }


def _summaries_from_details(details: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    ecg = details["ecg_report_inference"]["status"]
    gat = details["multimodal_gat_fusion"]["status"]
    flower = details["flower_federated"]["status"]
    dice = details["dice_counterfactuals"]["status"]
    dowhy = details["dowhy_causal"]["status"]
    return {
        "ecg_report_inference": ecg,
        "clinical_ai_inference": ecg,
        "multimodal_gat_fusion": gat,
        "dice_counterfactuals": dice,
        "dowhy_causal": dowhy,
        "flower_federated": flower,
        "clinical_decision_support": "NOT_IMPLEMENTED",
        "document_ingestion": "AVAILABLE",
        "privacy_masking_pipeline": "AVAILABLE",
        "ecg_ml_encoder": ecg,
        "clinical_text_encoder": ecg,
    }


def _build_capability_bundle() -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    details = _build_capability_details()
    return details, _summaries_from_details(details)


def get_capability_details() -> Dict[str, Dict[str, Any]]:
    details, _ = get_cached_capability_bundle(_build_capability_bundle)
    return details


def get_clinician_workspace_capabilities() -> Dict[str, str]:
    _, caps = get_cached_capability_bundle(_build_capability_bundle)
    return caps


def get_patient_portal_capabilities() -> Dict[str, str]:
    clin = get_clinician_workspace_capabilities()
    return {
        "ai_assisted_assessment": "NOT_ENABLED",
        "ecg_analysis": clin["ecg_report_inference"],
        "explanation_summaries": clin["dice_counterfactuals"],
        "causal_insights": clin["dowhy_causal"],
        "clinical_decision_support": clin["clinical_decision_support"],
        "patient_messaging": "NOT_IMPLEMENTED",
    }


def get_platform_capability_report() -> Dict[str, Any]:
    details, caps = get_cached_capability_bundle(_build_capability_bundle)
    ecg = details.get("ecg_report_inference") or {}
    gat = details.get("multimodal_gat_fusion") or {}
    return {
        "capabilities": caps,
        "capability_details": details,
        "diagnostics": {
            "ecg_report_inference_wired": True,
            "ecg_report_inference_enabled": settings.ECG_REPORT_INFERENCE_ENABLED,
            "ecg_report_ready": bool(ecg.get("ready")),
            "ecg_report_blockers": ecg.get("blockers") or [],
            "gat_fusion_wired": True,
            "fusion_inference_enabled_flag": settings.FUSION_INFERENCE_ENABLED,
            "gat_fusion_ready": bool(gat.get("ready")),
            "gat_fusion_blockers": gat.get("blockers") or [],
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
