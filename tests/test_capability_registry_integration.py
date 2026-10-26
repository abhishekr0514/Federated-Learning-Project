"""Capability registry reflects ECG-report and GAT readiness (not legacy 3-mod gate only)."""

import pytest

from backend.app.services.capability_cache import invalidate_capability_cache
from backend.app.services.capability_registry import get_clinician_workspace_capabilities, get_platform_capability_report


@pytest.fixture(autouse=True)
def _fresh_capability_cache():
    invalidate_capability_cache()
    yield
    invalidate_capability_cache()


def test_capability_keys_include_ecg_and_gat():
    caps = get_clinician_workspace_capabilities()
    assert "ecg_report_inference" in caps
    assert "multimodal_gat_fusion" in caps
    assert caps["clinical_decision_support"] == "NOT_IMPLEMENTED"


def test_platform_report_diagnostics_shape():
    report = get_platform_capability_report()
    diag = report["diagnostics"]
    assert "ecg_report_ready" in diag
    assert "gat_fusion_ready" in diag
    assert "ecg_report_blockers" in diag
