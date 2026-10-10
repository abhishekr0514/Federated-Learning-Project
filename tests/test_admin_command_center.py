import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from backend.app.main import app
from backend.app.models import Base, User, Role, Hospital, SecurityEvent
from backend.app.core.security import UserRole, hash_password, create_access_token
from backend.app.api.dependencies import get_db

SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_admin?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(SHARED_MEM_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

SUPER_ADMIN_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
HOSP_ADMIN_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
CLINICIAN_ID = uuid.UUID("00000000-0000-0000-0000-000000000003")
PATIENT_ID = uuid.UUID("00000000-0000-0000-0000-000000000004")

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

            hosp_admin = User(
                id=HOSP_ADMIN_ID,
                email="hospadmin@medshield.org",
                hashed_password=hash_password("HospAdminSecret123!"),
                is_active=True,
                is_super_admin=False,
            )

            clinician = User(
                id=CLINICIAN_ID,
                email="clinician@medshield.org",
                hashed_password=hash_password("ClinicianSecret123!"),
                is_active=True,
                is_super_admin=False,
            )

            patient = User(
                id=PATIENT_ID,
                email="patient@medshield.org",
                hashed_password=hash_password("PatientSecret123!"),
                is_active=True,
                is_super_admin=False,
            )

            session.add_all([super_admin, hosp_admin, clinician, patient])

            hospital = Hospital(
                id=uuid.uuid4(),
                name="St. Jude Medical Node",
                code="HOSP-STJUDE",
                is_active=True,
            )
            session.add(hospital)

            sec_event = SecurityEvent(
                id=uuid.uuid4(),
                actor_id=SUPER_ADMIN_ID,
                event_name="LOGIN_SUCCESS",
                severity="INFO",
                request_id="req-test-123",
            )
            session.add(sec_event)

            await session.commit()

    yield
    app.dependency_overrides.pop(get_db, None)

@pytest.mark.anyio
async def test_admin_overview():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["total_hospitals"] == 1
        assert data["active_users"] == 4

@pytest.mark.anyio
async def test_admin_system_health():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/system-health", headers=headers)
        assert res.status_code == 200
        health = res.json()["data"]
        assert health["api_health"]["status"] == "HEALTHY"

@pytest.mark.anyio
async def test_admin_hospitals():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/hospitals", headers=headers)
        assert res.status_code == 200
        hosps = res.json()["data"]
        assert hosps["count"] == 1

@pytest.mark.anyio
async def test_admin_federated():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/federated", headers=headers)
        assert res.status_code == 200
        fed = res.json()["data"]
        assert fed["status"] == "NOT_ENABLED"

@pytest.mark.anyio
async def test_admin_security():
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/security", headers=headers)
        assert res.status_code == 200
        sec = res.json()["data"]
        print("SECURITY DEBUG DATA:", sec)
        assert sec["event_counts"]["LOGIN_SUCCESS"] >= 1

@pytest.mark.anyio
async def test_non_super_admin_roles_rejected():
    for user_id in [HOSP_ADMIN_ID, CLINICIAN_ID, PATIENT_ID]:
        token, _ = create_access_token(user_id=str(user_id))
        headers = {"Authorization": f"Bearer {token}"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/v1/admin/overview", headers=headers)
            assert res.status_code == 403

@pytest.mark.anyio
async def test_unauthenticated_request_rejected():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/admin/overview")
        assert res.status_code == 401
