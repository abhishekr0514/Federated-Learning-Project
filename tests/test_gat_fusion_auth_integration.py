"""JWT-protected GAT fusion API tests with isolated in-memory SQLite (no live PostgreSQL)."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import Base, Role, User

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_gat_fusion?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
def reset_fusion_runtime_singleton():
    import ml.inference.fusion_runtime as fusion_runtime

    fusion_runtime._RUNTIME = None
    yield
    fusion_runtime._RUNTIME = None


@pytest.fixture(autouse=True)
async def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    app.dependency_overrides.pop(get_db, None)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def _emb(seed: float = 0.01) -> list[float]:
    return [seed * (i % 17 - 8) for i in range(128)]


@pytest.mark.anyio
async def test_gat_readiness_contract():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/gat-fusion/readiness")
    assert res.status_code == 200
    body = res.json()
    assert body["embedding_contracts"] == {"ecg": 128, "clinical_text": 128}
    assert body["architecture_id"] == "medshield.diagnostic_net.gat_ecg_report"
    assert "tabular" not in body["embedding_contracts"]


@pytest.mark.anyio
async def test_gat_evaluate_unauthenticated():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/gat-fusion/evaluate",
            json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
        )
    assert res.status_code == 401


@pytest.mark.anyio
async def test_gat_evaluate_invalid_credentials_rejected():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        user = User(
            id=uuid.uuid4(),
            email="dr.gat@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        login = await ac.post(
            "/api/v1/auth/login",
            json={"email": "dr.gat@metro.org", "password": "WrongPassword123!"},
        )
    assert login.status_code == 401


@pytest.mark.anyio
async def test_auth_login_success_and_gat_evaluate():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        user = User(
            id=uuid.uuid4(),
            email="dr.gat@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        login = await ac.post(
            "/api/v1/auth/login",
            json={"email": "dr.gat@metro.org", "password": "ValidPassword123!"},
        )
        assert login.status_code == 200
        token = login.json()["data"]["access_token"]
        assert token

        with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
            res = await ac.post(
                "/api/v1/gat-fusion/evaluate",
                headers={"Authorization": f"Bearer {token}"},
                json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
            )

    if res.status_code == 503:
        pytest.skip("GAT artifacts not present or inference not ready in this environment")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "EVALUATED"
    assert body["architecture_id"] == "medshield.diagnostic_net.gat_ecg_report"
    assert body["modalities_used"] == ["ecg", "clinical_text"]
    assert 0.0 <= body["confidence_uncalibrated"] <= 1.0
    assert body["predicted_class_index"] in (0, 1)
    assert body["is_research_only"] is True
    assert body["proxy_task"] is True
    assert "research" in body["disclaimer"].lower()


@pytest.mark.anyio
async def test_gat_evaluate_rejects_bad_dimensions():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        user = User(
            id=uuid.uuid4(),
            email="dr.dim@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.commit()
        token, _ = create_access_token(str(user.id))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/gat-fusion/evaluate",
            headers={"Authorization": f"Bearer {token}"},
            json={"ecg_embedding": [0.0] * 64, "text_embedding": _emb()},
        )
    assert res.status_code == 422


@pytest.mark.anyio
async def test_gat_evaluate_rejects_non_finite():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        user = User(
            id=uuid.uuid4(),
            email="dr.nan@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.commit()
        token, _ = create_access_token(str(user.id))

    import json

    bad = _emb()
    bad[3] = float("nan")
    payload = json.dumps(
        {"ecg_embedding": bad, "text_embedding": _emb()},
        allow_nan=True,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/gat-fusion/evaluate",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            content=payload,
        )
    assert res.status_code == 422


@pytest.mark.anyio
async def test_gat_evaluate_blocked_when_inference_disabled():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        user = User(
            id=uuid.uuid4(),
            email="dr.off@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.commit()
        token, _ = create_access_token(str(user.id))

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                "/api/v1/gat-fusion/evaluate",
                headers={"Authorization": f"Bearer {token}"},
                json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
            )
    assert res.status_code == 503
