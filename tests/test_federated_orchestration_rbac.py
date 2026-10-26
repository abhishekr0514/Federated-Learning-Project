"""RBAC for Flower SuperLink orchestration API (federated:monitor)."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import UserRole, create_access_token, hash_password
from backend.app.main import app
from backend.app.models import Base, Role, User


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def rbac_client():
    mem_url = "sqlite+aiosqlite:///file:memdb_fed_rbac?mode=memory&cache=shared&uri=true"
    engine = create_async_engine(mem_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with Session() as session:
        super_role = Role(id=uuid.uuid4(), name=UserRole.SUPER_ADMIN.value, description="super")
        clin_role = Role(id=uuid.uuid4(), name=UserRole.CLINICIAN.value, description="clin")
        session.add_all([super_role, clin_role])

        super_id = uuid.UUID("00000000-0000-0000-0000-000000000101")
        clin_id = uuid.UUID("00000000-0000-0000-0000-000000000102")
        super_user = User(
            id=super_id,
            email="admin@medshield.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            is_super_admin=True,
        )
        super_user.roles.append(super_role)
        clin_user = User(
            id=clin_id,
            email="dr_metro@metro.org",
            hashed_password=hash_password("ValidPassword123!"),
            is_active=True,
            is_super_admin=False,
        )
        clin_user.roles.append(clin_role)
        session.add_all([super_user, clin_user])
        await session.commit()

    super_token, _ = create_access_token(user_id=str(super_id))
    clin_token, _ = create_access_token(user_id=str(clin_id))
    headers = {
        "super": {"Authorization": f"Bearer {super_token}"},
        "clinician": {"Authorization": f"Bearer {clin_token}"},
    }

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), headers

    app.dependency_overrides.pop(get_db, None)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_super_admin_can_read_federated_status(rbac_client):
    ac, headers = rbac_client
    res = await ac.get("/api/v1/federated/status", headers=headers["super"])
    assert res.status_code == 200
    data = res.json()["data"]
    assert "lifecycle" in data
    assert "demo_supernodes" in data


@pytest.mark.anyio
async def test_clinician_denied_federated_status(rbac_client):
    ac, headers = rbac_client
    res = await ac.get("/api/v1/federated/status", headers=headers["clinician"])
    assert res.status_code == 403
    body = res.json()
    assert "federated:monitor" in str(body)


@pytest.mark.anyio
async def test_super_admin_can_list_rounds_clinician_denied(rbac_client):
    ac, headers = rbac_client
    ok = await ac.get("/api/v1/federated/rounds", headers=headers["super"])
    assert ok.status_code == 200
    denied = await ac.get("/api/v1/federated/rounds", headers=headers["clinician"])
    assert denied.status_code == 403
