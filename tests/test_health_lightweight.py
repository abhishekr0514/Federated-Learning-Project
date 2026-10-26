"""`/health` must stay fast — ML readiness belongs on /api/v1/capabilities."""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_does_not_call_ecg_readiness():
    client = TestClient(app)
    with patch(
        "backend.app.services.ecg_report_inference_service.ECGReportInferenceService.get_readiness",
    ) as get_ready:
        resp = client.get("/health")
        get_ready.assert_not_called()
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_completes_within_budget():
    client = TestClient(app)
    start = time.perf_counter()
    resp = client.get("/health")
    elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert elapsed < 5.0, f"/health took {elapsed:.1f}s"
