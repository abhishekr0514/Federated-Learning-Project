"""Capability detail map aligns summary status with readiness sources."""

from unittest.mock import patch

from backend.app.services.capability_registry import get_capability_details, get_clinician_workspace_capabilities


@patch(
    "backend.app.services.causal_analysis_service.CausalAnalysisService.get_readiness",
    return_value={
        "summary": "CAUSAL_DESIGN_REQUIRED",
        "reason": "test",
        "blockers": [],
        "cohort_id": "gat_ecg_report_ptbxl_research_v1",
    },
)
def test_capability_details_keys_match_summary(_mock_causal):
    details = get_capability_details()
    caps = get_clinician_workspace_capabilities()
    assert details["multimodal_gat_fusion"]["status"] == caps["multimodal_gat_fusion"]
    assert details["ecg_report_inference"]["status"] == caps["ecg_report_inference"]
    assert "reason" in details["flower_federated"]
    assert details["dice_counterfactuals"]["status"] in (
        "NOT_IMPLEMENTED",
        "MISSING_DEPENDENCY",
    )
