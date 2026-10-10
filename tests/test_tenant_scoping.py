"""Integration tests for Tenant Scoping, Patient Ownership, and API Authorization."""

import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import (
    Base,
    User,
    Role,
    Hospital,
    HospitalMembership,
    Patient,
)

# Shared in-memory SQLite database URI
SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_tenant?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(SHARED_MEM_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


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
async def prepare_db():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


@pytest.mark.anyio
async def test_unauthenticated_request_fails_with_401():
    """Verify requesting a protected endpoint without Bearer token returns 401 Unauthorized."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.get("/api/v1/hospitals/", headers={"Authorization": ""})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.anyio
async def test_invalid_bearer_token_fails_with_401():
    """Verify requesting a protected endpoint with malformed token returns 401 Unauthorized."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        headers = {"Authorization": "Bearer invalid_jwt_token_string"}
        response = await ac.get("/api/v1/hospitals/", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.anyio
async def test_authenticated_user_access_validation():
    """Verify authenticated access token for valid non-deleted user."""
    async with TestingSessionLocal() as session:
        user = User(
            id=uuid.uuid4(),
            email="test_user@medshield.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        user_id = user.id

    token, _ = create_access_token(user_id=str(user_id))
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        response = await ac.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["email"] == "test_user@medshield.org"
        assert data["is_active"] is True


@pytest.mark.anyio
async def test_hospital_admin_cross_tenant_isolation():
    """Verify Hospital Admin cannot access another hospital's resources."""
    async with TestingSessionLocal() as session:
        role_admin = Role(id=uuid.uuid4(), name="HOSPITAL_ADMIN", description="Hospital Administrator")
        hosp_a = Hospital(id=uuid.uuid4(), code="HOSP-A", name="Hospital A")
        hosp_b = Hospital(id=uuid.uuid4(), code="HOSP-B", name="Hospital B")
        session.add_all([role_admin, hosp_a, hosp_b])
        await session.commit()

        admin_a = User(
            id=uuid.uuid4(),
            email="admin_a@hosp_a.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        admin_a.roles.append(role_admin)
        session.add(admin_a)
        await session.commit()

        membership_a = HospitalMembership(id=uuid.uuid4(), user_id=admin_a.id, hospital_id=hosp_a.id, role_in_hospital="HOSPITAL_ADMIN")
        session.add(membership_a)
        await session.commit()

        token_a, _ = create_access_token(user_id=str(admin_a.id))
        hosp_a_id = str(hosp_a.id)
        hosp_b_id = str(hosp_b.id)

    headers = {"Authorization": f"Bearer {token_a}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        # Access Hosp A -> authorized (returns 501 Not Implemented)
        res_a = await ac.get(f"/api/v1/hospitals/{hosp_a_id}", headers=headers)
        assert res_a.status_code == 501

        # Access Hosp B -> denied (returns 404 Not Found)
        res_b = await ac.get(f"/api/v1/hospitals/{hosp_b_id}", headers=headers)
        assert res_b.status_code == 404


@pytest.mark.anyio
async def test_patient_cross_ownership_isolation():
    """Verify Patient user cannot access another patient's resources."""
    async with TestingSessionLocal() as session:
        role_patient = Role(id=uuid.uuid4(), name="PATIENT", description="Patient User")
        session.add(role_patient)
        await session.commit()

        user_p1 = User(
            id=uuid.uuid4(),
            email="patient1@medshield.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_p2 = User(
            id=uuid.uuid4(),
            email="patient2@medshield.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_p1.roles.append(role_patient)
        user_p2.roles.append(role_patient)
        session.add_all([user_p1, user_p2])
        await session.commit()

        patient1 = Patient(id=uuid.uuid4(), opaque_patient_id="PATIENT_HASH_1", user_id=user_p1.id)
        patient2 = Patient(id=uuid.uuid4(), opaque_patient_id="PATIENT_HASH_2", user_id=user_p2.id)
        session.add_all([patient1, patient2])
        await session.commit()

        token_p1, _ = create_access_token(user_id=str(user_p1.id))
        p1_id = str(patient1.id)
        p2_id = str(patient2.id)

    headers = {"Authorization": f"Bearer {token_p1}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        # Patient 1 accessing Patient 1 -> authorized (returns 501 Not Implemented)
        res_own = await ac.get(f"/api/v1/patients/{p1_id}", headers=headers)
        assert res_own.status_code == 501

        # Patient 1 accessing Patient 2 -> denied (returns 404 Not Found)
        res_other = await ac.get(f"/api/v1/patients/{p2_id}", headers=headers)
        assert res_other.status_code == 404
