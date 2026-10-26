"""Capability registry vs clinician/hospital surfaces — consistency regression."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.app.services.capability_cache import invalidate_capability_cache
from backend.app.services.capability_registry import (
    get_capability_details,
    get_clinician_workspace_capabilities,
)


@pytest.fixture(autouse=True)
def _fresh_capability_cache():
    invalidate_capability_cache()
    yield
    invalidate_capability_cache()


def _assert_consistent(details, clin, hospital):
    assert details["ecg_report_inference"]["status"] == clin["ecg_report_inference"]
    assert details["multimodal_gat_fusion"]["status"] == clin["multimodal_gat_fusion"]
    assert details["dowhy_causal"]["status"] == clin["dowhy_causal"]
    assert details["flower_federated"]["status"] == clin["flower_federated"]
    assert details["dice_counterfactuals"]["status"] == clin["dice_counterfactuals"]
    assert hospital["dowhy_causal"] == clin["dowhy_causal"]
    assert hospital["flower_federated"] == clin["flower_federated"]


def _patches(*, ecg_ready, gat_ready, gat_enabled, causal_summary, dowhy_ok, flower_payload):
    from backend.app.core.config import settings

    return (
        patch.object(settings, "ECG_REPORT_INFERENCE_ENABLED", True),
        patch.object(settings, "FUSION_INFERENCE_ENABLED", gat_enabled),
        patch.object(settings, "CAUSAL_ANALYSIS_ENABLED", True),
        patch(
            "backend.app.services.ecg_report_inference_service.ECGReportInferenceService.get_readiness",
            return_value={"ready": ecg_ready, "blockers": [] if ecg_ready else ["mock"]},
        ),
        patch(
            "backend.app.services.gat_fusion_service.GATFusionService.get_readiness",
            return_value={"ready": gat_ready, "blockers": [] if gat_ready else ["mock"]},
        ),
        patch(
            "backend.app.services.causal_analysis_service.CausalAnalysisService.get_readiness",
            return_value={
                "summary": causal_summary,
                "reason": "test",
                "blockers": [],
                "cohort_id": "gat_ecg_report_ptbxl_research_v1",
            },
        ),
        patch(
            "backend.app.services.causal_analysis_service._dowhy_available",
            lambda: dowhy_ok,
        ),
        patch(
            "backend.app.services.federated_status_service.get_flower_capability_status",
            return_value=flower_payload,
        ),
    )


def _hospital_caps():
    from backend.app.services.capability_registry import get_platform_capability_report

    return get_platform_capability_report()["capabilities"]


def test_ecg_inference_ready(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", True)
    patches = _patches(
        ecg_ready=True,
        gat_ready=True,
        gat_enabled=True,
        causal_summary="CAUSAL_DESIGN_REQUIRED",
        dowhy_ok=True,
        flower_payload={"summary": "DISABLED_BY_CONFIGURATION", "reason": "off"},
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        _assert_consistent(
            get_capability_details(),
            get_clinician_workspace_capabilities(),
            _hospital_caps(),
        )


def test_gat_checkpoint_missing_maps_configuration_required(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", True)
    patches = _patches(
        ecg_ready=True,
        gat_ready=False,
        gat_enabled=True,
        causal_summary="GAT_MODEL_NOT_READY",
        dowhy_ok=True,
        flower_payload={"summary": "MISSING_DEPENDENCY", "reason": "no flwr"},
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        details = get_capability_details()
        clin = get_clinician_workspace_capabilities()
        assert details["multimodal_gat_fusion"]["status"] == "CONFIGURATION_REQUIRED"
        _assert_consistent(details, clin, _hospital_caps())


def test_dowhy_missing_dependency(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", False)
    patches = _patches(
        ecg_ready=True,
        gat_ready=False,
        gat_enabled=False,
        causal_summary="MISSING_DEPENDENCY",
        dowhy_ok=False,
        flower_payload={"summary": "DISABLED_BY_CONFIGURATION", "reason": "off"},
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        details = get_capability_details()
        assert details["dowhy_causal"]["status"] == "MISSING_DEPENDENCY"
        _assert_consistent(details, get_clinician_workspace_capabilities(), _hospital_caps())


def test_flower_running_status(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FLOWER_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr(settings, "FLOWER_TRAINING_ENABLED", True)
    monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", True)
    patches = _patches(
        ecg_ready=True,
        gat_ready=True,
        gat_enabled=True,
        causal_summary="CAUSAL_DESIGN_REQUIRED",
        dowhy_ok=True,
        flower_payload={
            "summary": "RUNNING",
            "reason": "in progress",
            "lifecycle": "RUNNING",
            "training_round_active": True,
            "orchestration_enabled": True,
            "training_enabled": True,
            "dependency_installed": True,
            "security_configured": True,
        },
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        details = get_capability_details()
        assert details["flower_federated"]["status"] == "RUNNING"
        _assert_consistent(details, get_clinician_workspace_capabilities(), _hospital_caps())


def test_dice_still_not_implemented(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", True)
    patches = _patches(
        ecg_ready=True,
        gat_ready=True,
        gat_enabled=True,
        causal_summary="CAUSAL_DESIGN_REQUIRED",
        dowhy_ok=True,
        flower_payload={"summary": "COMPLETED", "reason": "done", "lifecycle": "COMPLETED"},
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        details = get_capability_details()
        assert details["dice_counterfactuals"]["status"] in ("NOT_IMPLEMENTED", "MISSING_DEPENDENCY")
        _assert_consistent(details, get_clinician_workspace_capabilities(), _hospital_caps())
