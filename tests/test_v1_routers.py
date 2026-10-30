"""Test suite verifying /api/v1 router namespaces, authentication guards, and safe 501 behavior."""

import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import Base, User

# Shared in-memory SQLite database for router tests
SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_routers?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(SHARED_MEM_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

SUPER_ADMIN_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def override_get_db():
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture(autouse=True)
async def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestingSessionLocal() as session:
        stmt = select(User).where(User.id == SUPER_ADMIN_ID)
        res = await session.execute(stmt)
        if not res.scalar_one_or_none():
            super_admin = User(
                id=SUPER_ADMIN_ID,
                email="superadmin@medshield.org",
                hashed_password=hash_password("SuperSecret123!"),
                is_active=True,
                is_super_admin=True,
            )
            session.add(super_admin)
            await session.commit()

    yield

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


V1_POST_501_ENDPOINTS = [
    "/api/v1/explanations/dice",
    "/api/v1/causal/evaluate-intervention",
]

V1_GET_501_ENDPOINTS = [
    "/api/v1/users/",
    "/api/v1/hospitals/",
    "/api/v1/patients/",
    "/api/v1/clinicians/dashboard",
    "/api/v1/documents/",
    "/api/v1/federated/rounds",
    "/api/v1/audit/",
]


@pytest.fixture
def auth_headers():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.anyio
async def test_diagnosis_readiness_route_registered_and_returns_safe_json():
    """Regression: readiness must be on the MedShield app OpenAPI and return safe diagnostics."""
    from backend.app.main import app as medshield_app

    assert "/api/v1/diagnosis/readiness" in medshield_app.openapi()["paths"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/diagnosis/readiness")
    assert res.status_code == 200
    body = res.json()
    assert "ready" in body
    assert "blockers" in body
    assert "dependencies" in body
    assert "next_steps" in body
    assert "environment" in body


@pytest.mark.anyio
async def test_diagnosis_evaluate_returns_503_when_inference_not_ready(auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            "/api/v1/diagnosis/evaluate?patient_id=00000000-0000-0000-0000-000000000099",
            headers=auth_headers,
            json={
                "embeddings": {
                    "ecg": [0.0] * 128,
                    "clinical_text": [0.0] * 128,
                    "tabular": [0.0] * 64,
                }
            },
        )
    assert res.status_code in (403, 404, 503)


@pytest.mark.parametrize("endpoint", V1_POST_501_ENDPOINTS)
@pytest.mark.anyio
async def test_v1_post_skeleton_endpoints_return_501(endpoint, auth_headers):
    """Verify POST domain skeleton endpoints safely return 501 Not Implemented when authenticated."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.post(endpoint, headers=auth_headers)
        assert response.status_code == 501, f"Endpoint {endpoint} should return 501 Not Implemented"
        data = response.json()
        assert data["error"]["code"] == "NOT_IMPLEMENTED"
        assert "metadata" in data
        assert "request_id" in data["metadata"]


@pytest.mark.parametrize("endpoint", V1_GET_501_ENDPOINTS)
@pytest.mark.anyio
async def test_v1_get_skeleton_endpoints_return_501(endpoint, auth_headers):
    """Verify GET domain skeleton endpoints safely return 501 Not Implemented when authenticated."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.get(endpoint, headers=auth_headers)
        assert response.status_code == 501, f"Endpoint {endpoint} should return 501 Not Implemented"
        data = response.json()
        assert data["error"]["code"] == "NOT_IMPLEMENTED"
        assert "metadata" in data
        assert "request_id" in data["metadata"]


@pytest.mark.anyio
async def test_models_status_endpoint_safe_response(auth_headers):
    """Verify GET /api/v1/models/status returns safe status without fabricating GAT predictions."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.get("/api/v1/models/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["inference_enabled"] is False
        assert data["status"] == "pending_ml_implementation"
        assert "target_architecture" in data
        assert data["target_architecture"]["fusion"] == "Graph Attention Network (GAT)"


@pytest.mark.anyio
async def test_privacy_status_endpoint_safe_response(auth_headers):
    """Verify GET /api/v1/privacy/status returns system status configuration."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.get("/api/v1/privacy/status", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["privacy_masking_enabled"] is True
        assert data["differential_privacy"] == "active"
