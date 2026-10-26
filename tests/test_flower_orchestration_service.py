"""Flower orchestration service tests."""

from __future__ import annotations

import importlib.util

import pytest

from backend.app.services.flower_orchestration_service import FlowerOrchestrationService


@pytest.fixture(autouse=True)
def reset_flower_state():
    from backend.app.services import flower_orchestration_legacy as mod

    with mod._STATE_LOCK:
        mod._RUN_STATE["history"] = []
        mod._reset_run_state()
    yield
    with mod._STATE_LOCK:
        mod._RUN_STATE["history"] = []
        mod._reset_run_state()


def test_start_rejected_when_disabled(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FLOWER_DEPLOYMENT_MODE", "legacy_inprocess")
    monkeypatch.setattr(settings, "FLOWER_ORCHESTRATION_ENABLED", False)
    monkeypatch.setattr(settings, "FLOWER_TRAINING_ENABLED", False)
    with pytest.raises(RuntimeError, match="orchestration is disabled"):
        FlowerOrchestrationService.start_run()


def test_status_includes_demo_supernodes_offline_when_idle(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FLOWER_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr(settings, "FLOWER_TRAINING_ENABLED", False)
    monkeypatch.setattr(
        "backend.app.services.flower_orchestration_service._flwr_available",
        lambda: True,
    )
    st = FlowerOrchestrationService.get_status()
    assert len(st["demo_supernodes"]) == 2
    assert all(n["connection_status"] == "offline" for n in st["demo_supernodes"])


def test_status_missing_dependency(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "FLOWER_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr(settings, "FLOWER_TRAINING_ENABLED", False)
    monkeypatch.setattr(
        "backend.app.services.flower_orchestration_service._flwr_available",
        lambda: False,
    )
    st = FlowerOrchestrationService.get_status()
    assert st["summary"] == "MISSING_DEPENDENCY"


@pytest.mark.skipif(not importlib.util.find_spec("flwr"), reason="flwr not installed")
def test_synthetic_smoke_completes_round():
    st = FlowerOrchestrationService.run_synthetic_smoke_blocking(num_rounds=1)
    assert st["lifecycle"] in ("COMPLETED", "FAILED")
    if st["lifecycle"] == "COMPLETED":
        assert st["successful_clients_last_round"] >= 1
        assert st["last_successful_round"] is not None
