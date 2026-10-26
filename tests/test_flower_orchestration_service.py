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


def test_demo_supernodes_completed_run_shows_participated_not_live():
    snap = {"processes": [], "lifecycle": "COMPLETED", "run_id": "test-run"}
    file_state = {
        "lifecycle": "COMPLETED",
        "total_rounds": 2,
        "current_round": 2,
        "initial_parameter_hash": "aaa",
        "global_parameter_hash": "bbb",
        "registered_nodes": [
            {"node_id": 101, "status": "online", "owner": "none"},
            {"node_id": 102, "status": "online", "owner": "none"},
        ],
        "round_history": [{"round": 1, "run_id": "r1"}, {"round": 2, "run_id": "r1"}],
    }
    nodes = FlowerOrchestrationService._demo_supernodes(
        snap=snap, file_state=file_state, lifecycle="COMPLETED"
    )
    assert len(nodes) == 2
    assert all(not n["process_alive"] for n in nodes)
    assert all(n["connection_status"] == "offline" for n in nodes)
    assert all(n["participation_status"] == "completed_run" for n in nodes)


def test_status_post_run_processes_stopped(monkeypatch):
    from backend.app.core.config import settings

    file_state = {
        "lifecycle": "COMPLETED",
        "total_rounds": 2,
        "current_round": 2,
        "initial_parameter_hash": "aaa",
        "global_parameter_hash": "bbb",
        "registered_nodes": [
            {"node_id": 101, "status": "online"},
            {"node_id": 102, "status": "online"},
        ],
        "round_history": [{"round": 1, "run_id": "r1"}, {"round": 2, "run_id": "r1"}],
    }
    snap = {"processes": [], "lifecycle": "COMPLETED", "file_state": file_state}

    monkeypatch.setattr(settings, "FLOWER_ORCHESTRATION_ENABLED", True)
    monkeypatch.setattr(settings, "FLOWER_TRAINING_ENABLED", True)
    monkeypatch.setattr(settings, "FLOWER_REQUIRE_TLS", False)
    monkeypatch.setattr(
        "backend.app.services.flower_orchestration_service._flwr_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "backend.app.services.flower_orchestration_service.FlowerSupervisor.get_snapshot",
        lambda: snap,
    )
    monkeypatch.setattr(
        "backend.app.services.flower_orchestration_service.FlowerSupervisor.security_readiness",
        lambda: (True, []),
    )

    st = FlowerOrchestrationService.get_status()
    assert st["summary"] == "COMPLETED"
    assert st["post_run_processes_stopped"] is True
    assert st["demo_nodes_live_count"] == 0
    assert st["demo_nodes_participated_count"] == 2
    assert "stopped by the supervisor" in st["reason"]


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
