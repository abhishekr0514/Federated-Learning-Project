"""
Real Flower SuperLink E2E (TLS + SuperNode auth + 2 clients).

Run manually:
  FLOWER_E2E=1 pytest tests/test_flower_superlink_e2e.py -v --timeout=600

Requires: openssl, ssh-keygen, flwr 1.30, dev certs in deployment/flower/dev/
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("FLOWER_E2E") != "1", reason="Set FLOWER_E2E=1 to run real SuperLink E2E")
def test_superlink_synthetic_federation_aggregates():
    cert_dir = REPO / "deployment" / "flower" / "dev"
    if not (cert_dir / "ca.pem").is_file():
        ps1 = REPO / "scripts" / "flower" / "generate_dev_materials.ps1"
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            check=True,
            cwd=str(REPO),
        )

    from backend.app.core.config import settings
    from backend.app.services.flower_supervisor import FlowerSupervisor

    # Enable flags only inside this test process (does not mutate .env on disk)
    settings.FLOWER_ORCHESTRATION_ENABLED = True
    settings.FLOWER_TRAINING_ENABLED = True
    settings.FLOWER_REQUIRE_TLS = True
    settings.FLOWER_DEPLOYMENT_MODE = "superlink"

    ok, blockers = FlowerSupervisor.security_readiness()
    assert ok, blockers

    start = FlowerSupervisor.start(num_rounds=2)
    assert "run_id" in start

    deadline = time.monotonic() + 480.0
    lifecycle = "STARTING"
    while time.monotonic() < deadline:
        snap = FlowerSupervisor.get_snapshot()
        lifecycle = snap.get("lifecycle") or snap.get("file_state", {}).get("lifecycle")
        if lifecycle in ("COMPLETED", "FAILED", "SERVER_STOPPED"):
            break
        time.sleep(2.0)

    snap = FlowerSupervisor.get_snapshot()
    state = snap.get("file_state") or {}
    assert lifecycle == "COMPLETED", snap.get("last_error") or state
    assert state.get("initial_parameter_hash")
    assert state.get("global_parameter_hash")
    assert state.get("initial_parameter_hash") != state.get("global_parameter_hash")
    assert int(state.get("successful_clients_last_round") or 0) >= 2
