"""Capability registry must not repeat expensive work per HTTP request."""

from __future__ import annotations

from unittest.mock import patch

from backend.app.services import capability_cache
from backend.app.services.capability_registry import get_platform_capability_report


def test_platform_report_builds_details_once_per_cache_window():
    capability_cache.invalidate_capability_cache()
    calls = {"n": 0}

    def fake_build():
        calls["n"] += 1
        details = {
            "ecg_report_inference": {"status": "AVAILABLE", "ready": True, "reason": "ok", "blockers": []},
            "multimodal_gat_fusion": {"status": "AVAILABLE", "ready": True, "reason": "ok", "blockers": []},
            "flower_federated": {"status": "DISABLED_BY_CONFIGURATION", "reason": "off"},
            "dice_counterfactuals": {"status": "NOT_IMPLEMENTED", "reason": "n/a"},
            "dowhy_causal": {"status": "CAUSAL_DESIGN_REQUIRED", "reason": "design"},
            "clinical_decision_support": {"status": "NOT_IMPLEMENTED", "reason": "n/a"},
        }
        caps = {"multimodal_gat_fusion": "AVAILABLE", "ecg_report_inference": "AVAILABLE"}
        return details, caps

    with patch(
        "backend.app.services.capability_registry._build_capability_bundle",
        side_effect=fake_build,
    ):
        get_platform_capability_report()
        get_platform_capability_report()
    assert calls["n"] == 1


def test_concurrent_capability_reports_share_one_build():
    capability_cache.invalidate_capability_cache()
    calls = {"n": 0}

    def fake_build():
        calls["n"] += 1
        details = {
            "ecg_report_inference": {"status": "AVAILABLE", "ready": True, "reason": "ok", "blockers": []},
            "multimodal_gat_fusion": {"status": "AVAILABLE", "ready": True, "reason": "ok", "blockers": []},
            "flower_federated": {"status": "DISABLED_BY_CONFIGURATION", "reason": "off"},
            "dice_counterfactuals": {"status": "NOT_IMPLEMENTED", "reason": "n/a"},
            "dowhy_causal": {"status": "CAUSAL_DESIGN_REQUIRED", "reason": "design"},
            "clinical_decision_support": {"status": "NOT_IMPLEMENTED", "reason": "n/a"},
        }
        caps = {"multimodal_gat_fusion": "AVAILABLE"}
        return details, caps

    import threading

    with patch(
        "backend.app.services.capability_registry._build_capability_bundle",
        side_effect=fake_build,
    ):
        barrier = threading.Barrier(3)

        def worker():
            barrier.wait()
            get_platform_capability_report()

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    assert calls["n"] == 1


def test_ecg_readiness_default_skips_forward_pass():
    from backend.app.services.ecg_report_inference_service import ECGReportInferenceService

    class R:
        def to_safe_dict(self):
            return {"ready": True, "blockers": []}

    runtime = patch("backend.app.services.ecg_report_inference_service.get_ecg_report_runtime").start()
    runtime.return_value.assess_readiness.return_value = R()
    try:
        ECGReportInferenceService.get_readiness()
        runtime.return_value.assess_readiness.assert_called_once_with(verify_forward_pass=False)
    finally:
        patch.stopall()
