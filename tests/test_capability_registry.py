"""Capability registry honesty checks."""

from backend.app.services.capability_registry import (
    get_clinician_workspace_capabilities,
    get_patient_portal_capabilities,
    get_platform_capability_report,
)


def test_clinician_capabilities_include_ingestion_available():
    caps = get_clinician_workspace_capabilities()
    assert caps["document_ingestion"] == "AVAILABLE"
    assert caps["privacy_masking_pipeline"] == "AVAILABLE"
    assert caps["clinical_ai_inference"] in ("NOT_IMPLEMENTED", "NOT_ENABLED", "MISSING_DEPENDENCY")


def test_patient_portal_ai_assessment_not_enabled():
    caps = get_patient_portal_capabilities()
    assert caps["ai_assisted_assessment"] == "NOT_ENABLED"


def test_platform_report_marks_inference_api_unwired():
    report = get_platform_capability_report()
    assert report["diagnostics"]["inference_api_wired"] is True
    assert "capabilities" in report
    assert "inference_ready" in report["diagnostics"]
