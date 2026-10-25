"""API tests for sanitized tabular federated experiment status."""

from __future__ import annotations

import json
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import Base, User

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api_client(tmp_path, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.services import federated_experiment_service as svc

    run_dir = tmp_path / "tabular_cardio_runs" / "demo_run"
    run_dir.mkdir(parents=True)
    report = {
        "experiment": "real_tabular_federated_cardio",
        "config": {
            "federation_id": "medshield.fl.tabular_cardio.v1",
            "task_id": "medshield.tabular.cardio_survey.baseline",
            "architecture_id": "medshield.tabular.cardio_survey.classifier",
            "seed": 42,
            "num_rounds": 2,
            "max_federated_train_rows": 100,
            "participation": {"num_clients": 2},
            "local_training": {"local_epochs": 1},
            "aggregation_strategy": "fedavg",
        },
        "partition_plan": {
            "client_train_counts": [50, 50],
            "client_train_id_hashes": [["abc123"]],
            "simulation_disclaimer": "simulated",
        },
        "round_history": [{"round": 1, "mean_client_loss": 0.5}],
        "evaluation": {"test": {"auroc": 0.7}, "validation": {"auroc": 0.69}},
        "privacy_note": "not dp",
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(svc, "_federated_runs_root", lambda: tmp_path / "tabular_cardio_runs")

    mem_url = "sqlite+aiosqlite:///file:memdb_fed_api?mode=memory&cache=shared&uri=true"
    engine = create_async_engine(mem_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    admin_id = uuid.UUID("00000000-0000-0000-0000-000000000099")
    async with Session() as session:
        session.add(
            User(
                id=admin_id,
                email="fedadmin@test.org",
                hashed_password=hash_password("x"),
                is_active=True,
                is_super_admin=True,
            )
        )
        await session.commit()

    token, _ = create_access_token(user_id=str(admin_id))
    headers = {"Authorization": f"Bearer {token}"}

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), headers

    app.dependency_overrides.pop(get_db, None)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_tabular_experiment_latest_sanitized(api_client):
    ac, headers = api_client
    res = await ac.get("/api/v1/federated/experiments/tabular/latest", headers=headers)
    assert res.status_code == 200
    data = res.json()["data"]
    blob = json.dumps(data)
    assert "patient_id" not in blob
    assert "client_train_id_hashes" not in blob
    assert "abc123" not in blob
    assert data["status"] == "AVAILABLE"
    assert data["report"]["evaluation_summary"]["test_auroc"] == 0.7
    assert "artifact_paths" not in blob
    assert "data/processed" not in blob
    assert "global_state_ndarrays" not in blob
    assert "parameter_schema_signature" not in blob
    assert "client_train_id_hashes" not in blob


@pytest.mark.anyio
async def test_tabular_experiment_requires_auth(api_client):
    ac, _headers = api_client
    res = await ac.get("/api/v1/federated/experiments/tabular/latest")
    assert res.status_code in (401, 403)
