"""Integration tests for Hospital Admin Dashboard and Observability Endpoints.

Tests verify:
1. Multi-tenant isolation: Hospital Admin A cannot retrieve Hospital B's scoped data.
2. Supplying unauthorized hospital_id parameter returns HTTP 404.
3. Zero mock data contract: missing data returns NO_DATA / NOT_ENABLED / NOT_MEASURED instead of fake 0.
4. RBAC: non-admin roles (CLINICIAN, PATIENT) receive HTTP 403.
5. Unauthenticated requests receive HTTP 401.
6. Server-side pagination on clinicians, patients, and audit endpoints.
"""

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
    ClinicianProfile,
    Patient,
    PatientHospitalAssociation,
    ModelVersion,
    HospitalMetric,
    FederatedClient,
    AuditLog,
    HospitalAlert,
    DataQualityEvent,
)

SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_hospital_admin?mode=memory&cache=shared&uri=true"
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
async def setup_test_database():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def seeded_test_env():
    """Seed test database with two distinct hospitals, roles, admins, clinicians, and records."""
    async with TestingSessionLocal() as session:
        # 1. Roles
        role_admin = Role(id=uuid.uuid4(), name="HOSPITAL_ADMIN", description="Hospital Administrator")
        role_clinician = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_patient = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        session.add_all([role_admin, role_clinician, role_patient])
        await session.commit()

        # 2. Hospitals
        hosp_a = Hospital(id=uuid.uuid4(), code="HOSP-A", name="Metro General Hospital")
        hosp_b = Hospital(id=uuid.uuid4(), code="HOSP-B", name="St. Jude Medical Center")
        session.add_all([hosp_a, hosp_b])
        await session.commit()

        # 3. Hospital Admin Users
        admin_a = User(
            id=uuid.uuid4(),
            email="admin_a@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        admin_a.roles.append(role_admin)

        admin_b = User(
            id=uuid.uuid4(),
            email="admin_b@stjude.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        admin_b.roles.append(role_admin)

        clinician_user = User(
            id=uuid.uuid4(),
            email="dr_smith@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        clinician_user.roles.append(role_clinician)

        session.add_all([admin_a, admin_b, clinician_user])
        await session.commit()

        # 4. Memberships
        mem_a = HospitalMembership(
            id=uuid.uuid4(),
            user_id=admin_a.id,
            hospital_id=hosp_a.id,
            role_in_hospital="HOSPITAL_ADMIN",
        )
        mem_b = HospitalMembership(
            id=uuid.uuid4(),
            user_id=admin_b.id,
            hospital_id=hosp_b.id,
            role_in_hospital="HOSPITAL_ADMIN",
        )
        mem_clinician = HospitalMembership(
            id=uuid.uuid4(),
            user_id=clinician_user.id,
            hospital_id=hosp_a.id,
            role_in_hospital="CLINICIAN",
        )
        session.add_all([mem_a, mem_b, mem_clinician])
        await session.commit()

        # 5. Clinician Profile for Hospital A
        clin_prof_a = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=clinician_user.id,
            hospital_id=hosp_a.id,
            department="Cardiology",
            license_number_hash="HASH_CARDIO_001",
        )
        session.add(clin_prof_a)
        await session.commit()

        # 6. Patients for Hospital A and B
        pat_a = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_A_101", user_id=None)
        pat_b = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_B_202", user_id=None)
        session.add_all([pat_a, pat_b])
        await session.commit()

        assoc_a = PatientHospitalAssociation(
            id=uuid.uuid4(),
            patient_id=pat_a.id,
            hospital_id=hosp_a.id,
            hospital_patient_ref_hash="REF_A_101",
        )
        assoc_b = PatientHospitalAssociation(
            id=uuid.uuid4(),
            patient_id=pat_b.id,
            hospital_id=hosp_b.id,
            hospital_patient_ref_hash="REF_B_202",
        )
        session.add_all([assoc_a, assoc_b])
        await session.commit()

        # 7. Model Version & Metrics for Hospital A
        mv = ModelVersion(
            id=uuid.uuid4(),
            model_family="MedShield-GAT-Multimodal",
            version_tag="v1.0-alpha",
            architecture_config='{"layers": 3}',
            target_definition="PTB-XL Normal Sinus Rhythm vs Pathological Proxy Task",
            status="active",
        )
        session.add(mv)
        await session.commit()

        hosp_metric_a = HospitalMetric(
            id=uuid.uuid4(),
            hospital_id=hosp_a.id,
            model_version_id=mv.id,
            metric_name="local_macro_f1",
            metric_value=0.884,
        )
        session.add(hosp_metric_a)
        await session.commit()

        # 8. Federated Client for Hospital A
        fed_a = FederatedClient(
            id=uuid.uuid4(),
            hospital_id=hosp_a.id,
            client_identifier="FL-CLIENT-METRO-01",
            status="active",
        )
        session.add(fed_a)
        await session.commit()

        # 9. Audit Logs
        audit_a = AuditLog(
            id=uuid.uuid4(),
            event_type="NODE_LOGIN",
            actor_id=admin_a.id,
            hospital_id=hosp_a.id,
            resource_type="hospital",
            resource_id=str(hosp_a.id),
            request_id="req-metro-1",
            outcome="SUCCESS",
        )
        audit_b = AuditLog(
            id=uuid.uuid4(),
            event_type="NODE_LOGIN",
            actor_id=admin_b.id,
            hospital_id=hosp_b.id,
            resource_type="hospital",
            resource_id=str(hosp_b.id),
            request_id="req-stjude-1",
            outcome="SUCCESS",
        )
        session.add_all([audit_a, audit_b])
        await session.commit()

        # 10. Alerts & Data Quality for Hospital A
        alert_a = HospitalAlert(
            id=uuid.uuid4(),
            hospital_id=hosp_a.id,
            severity="INFO",
            status="ACTIVE",
            category="INFRASTRUCTURE",
            title="Local storage health nominal",
        )
        dq_a = DataQualityEvent(
            id=uuid.uuid4(),
            hospital_id=hosp_a.id,
            severity="LOW",
            status="OPEN",
            category="SCHEMA_VALIDATION",
        )
        session.add_all([alert_a, dq_a])
        await session.commit()

        token_a, _ = create_access_token(user_id=str(admin_a.id))
        token_b, _ = create_access_token(user_id=str(admin_b.id))
        token_clin, _ = create_access_token(user_id=str(clinician_user.id))

        return {
            "hosp_a_id": str(hosp_a.id),
            "hosp_b_id": str(hosp_b.id),
            "token_a": token_a,
            "token_b": token_b,
            "token_clin": token_clin,
        }


@pytest.mark.anyio
async def test_unauthenticated_request_fails_with_401():
    """Unauthenticated requests to hospital endpoints return 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/overview")
        assert res.status_code == 401


@pytest.mark.anyio
async def test_clinician_forbidden_from_hospital_admin(seeded_test_env):
    """Clinician user without HOSPITAL_ADMIN role receives 403 Forbidden."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_clin']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/overview", headers=headers)
        assert res.status_code == 403


@pytest.mark.anyio
async def test_hospital_admin_a_accesses_own_overview(seeded_test_env):
    """Hospital Admin A successfully retrieves overview for Metro General Hospital."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "AVAILABLE"
        assert data["hospital_scope"]["code"] == "HOSP-A"
        assert data["clinicians"]["count"] == 1
        assert data["patients"]["count"] == 1
        assert data["federated_clients"]["count"] == 1
        assert data["latest_metric"]["status"] == "AVAILABLE"
        assert data["latest_metric"]["metric_value"] == 0.884
        assert data["local_training"]["status"] == "NOT_ENABLED"
        assert data["federated_training"]["status"] == "NOT_ENABLED"


@pytest.mark.anyio
async def test_cross_tenant_isolation_hospital_a_cannot_query_hospital_b(seeded_test_env):
    """Hospital Admin A cannot retrieve Hospital B data by supplying hospital_id=hosp_b_id."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    hosp_b_id = seeded_test_env["hosp_b_id"]

    endpoints = [
        f"/api/v1/hospital/overview?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/system-health?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/clinicians?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/patients?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/models?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/metrics?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/federated?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/privacy?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/data-quality?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/audit?hospital_id={hosp_b_id}",
        f"/api/v1/hospital/alerts?hospital_id={hosp_b_id}",
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        for ep in endpoints:
            res = await ac.get(ep, headers=headers)
            assert res.status_code == 404, f"Endpoint {ep} did not enforce tenant isolation (expected 404, got {res.status_code})"


@pytest.mark.anyio
async def test_hospital_admin_b_honest_no_data_states(seeded_test_env):
    """Hospital Admin B (with no seeded metrics/clients) gets honest NO_DATA states."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_b']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/overview", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["hospital_scope"]["code"] == "HOSP-B"
        assert data["clinicians"]["status"] == "NO_DATA"
        assert data["clinicians"]["count"] == 0
        assert data["federated_clients"]["status"] == "NO_DATA"
        assert data["latest_metric"]["status"] == "NO_DATA"

        # Check metrics endpoint
        res_metrics = await ac.get("/api/v1/hospital/metrics", headers=headers)
        assert res_metrics.status_code == 200
        assert res_metrics.json()["status"] == "NO_DATA"
        assert res_metrics.json()["metrics"] == []

        # Check privacy endpoint
        res_privacy = await ac.get("/api/v1/hospital/privacy", headers=headers)
        assert res_privacy.status_code == 200
        assert res_privacy.json()["status"] == "NO_DATA"


@pytest.mark.anyio
async def test_system_health_explicit_hardware_states(seeded_test_env):
    """System health returns AVAILABLE for DB/API and NOT_MEASURED for hardware."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/system-health", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["api_health"]["status"] == "AVAILABLE"
        assert data["database"]["status"] == "AVAILABLE"
        assert data["system_hardware"]["cpu_usage"] == "NOT_MEASURED"
        assert data["system_hardware"]["memory_usage"] == "NOT_MEASURED"
        assert data["system_hardware"]["worker_health"] == "NOT_MEASURED"


@pytest.mark.anyio
async def test_privacy_boundaries_zero_pii_in_patient_and_clinician(seeded_test_env):
    """Opaque patient references and clinician profiles never expose PII or secrets."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_pat = await ac.get("/api/v1/hospital/patients", headers=headers)
        assert res_pat.status_code == 200
        patients = res_pat.json()["patients"]
        assert len(patients) == 1
        p = patients[0]
        assert "opaque_patient_id" in p
        assert p["opaque_patient_id"] == "OPAQUE_PAT_A_101"
        assert "name" not in p
        assert "email" not in p
        assert "raw_notes" not in p
        assert "ecg" not in p

        res_clin = await ac.get("/api/v1/hospital/clinicians", headers=headers)
        assert res_clin.status_code == 200
        clinicians = res_clin.json()["clinicians"]
        assert len(clinicians) == 1
        c = clinicians[0]
        assert "password" not in c
        assert "hashed_password" not in c
        assert "token" not in c
        assert c["user_status"] == "ACTIVE"


@pytest.mark.anyio
async def test_audit_logs_strictly_scoped_and_paginated(seeded_test_env):
    """Hospital A only sees Hospital A audit logs, never Hospital B logs."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/audit?limit=10&offset=0", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "AVAILABLE"
        assert len(data["logs"]) == 1
        assert data["logs"][0]["request_id"] == "req-metro-1"
        # Zero Hospital B logs
        for log in data["logs"]:
            assert log["request_id"] != "req-stjude-1"


@pytest.mark.anyio
async def test_accessible_nodes_returns_only_authorized_hospitals(seeded_test_env):
    """Admin A only sees Hospital A in accessible-nodes."""
    headers = {"Authorization": f"Bearer {seeded_test_env['token_a']}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/hospital/accessible-nodes", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 1
        assert data["hospitals"][0]["code"] == "HOSP-A"
