"""Integration and Scientific Provenance Tests for Patient Portal.

Validates all Prompt 09 mandatory requirements:
1. Missing authentication -> HTTP 401 Unauthorized.
2. Non-patient user (Clinician, Hospital Admin) -> HTTP 403 Forbidden.
3. Patient A can access Patient A's own records -> HTTP 200 OK.
4. Patient A cannot access Patient B's records -> HTTP 404 Not Found.
5. Patient A cannot access another hospital's patient -> HTTP 404 Not Found.
6. Changing patient_id in URL cannot bypass ownership -> HTTP 404 Not Found.
7. No other patient data appears in the response.
8. Zero secrets, tokens, passwords, or credentials appear.
9. Zero raw ECG waveforms or voltage arrays appear.
10. Zero raw clinical notes or unmasked text appear.
11. No diagnosis is fabricated when no prediction exists (NO_DATA / NOT_ENABLED).
12. No confidence or accuracy is fabricated.
13. Prediction confidence is strictly labeled model confidence (never model accuracy).
14. AI assessment unavailable state (NOT_ENABLED) is correctly returned.
15. Mandatory safety wording is present whenever AI assessment results are returned.
16. Clinician/admin cannot accidentally inherit the patient portal's ownership context.
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
    MedicalDocument,
    DocumentVersion,
    DocumentProcessingJob,
    PrivacyMaskingResult,
    ECGRecord,
    ClinicalObservation,
    ModelVersion,
    DiagnosisPrediction,
)

SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_patient_portal?mode=memory&cache=shared&uri=true"
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
async def seeded_patient_env():
    """Seed test database with hospitals, multiple patient users, clinician, admin, and clinical records."""
    async with TestingSessionLocal() as session:
        # 1. Roles
        role_patient = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        role_clinician = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_admin = Role(id=uuid.uuid4(), name="HOSPITAL_ADMIN", description="Hospital Administrator")
        session.add_all([role_patient, role_clinician, role_admin])
        await session.commit()

        # 2. Hospitals
        hosp_a = Hospital(id=uuid.uuid4(), code="HOSP-A", name="Metro General Hospital")
        hosp_b = Hospital(id=uuid.uuid4(), code="HOSP-B", name="St. Jude Medical Center")
        session.add_all([hosp_a, hosp_b])
        await session.commit()

        # 3. Users
        # Patient A (Hospital A)
        user_pat_a = User(
            id=uuid.uuid4(),
            email="patient_a@example.com",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_pat_a.roles.append(role_patient)

        # Patient B (Hospital A)
        user_pat_b = User(
            id=uuid.uuid4(),
            email="patient_b@example.com",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_pat_b.roles.append(role_patient)

        # Patient C (Hospital B)
        user_pat_c = User(
            id=uuid.uuid4(),
            email="patient_c@example.com",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_pat_c.roles.append(role_patient)

        # Clinician (No patient role)
        user_clin = User(
            id=uuid.uuid4(),
            email="dr_metro@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_clin.roles.append(role_clinician)

        # Hospital Admin (No patient role)
        user_admin = User(
            id=uuid.uuid4(),
            email="admin_metro@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_admin.roles.append(role_admin)

        session.add_all([user_pat_a, user_pat_b, user_pat_c, user_clin, user_admin])
        await session.commit()

        # 4. Patient Entities linked to User IDs
        pat_a = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_A_111", user_id=user_pat_a.id)
        pat_b = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_B_222", user_id=user_pat_b.id)
        pat_c = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_C_333", user_id=user_pat_c.id)
        session.add_all([pat_a, pat_b, pat_c])
        await session.commit()

        assoc_a = PatientHospitalAssociation(id=uuid.uuid4(), patient_id=pat_a.id, hospital_id=hosp_a.id)
        assoc_b = PatientHospitalAssociation(id=uuid.uuid4(), patient_id=pat_b.id, hospital_id=hosp_a.id)
        assoc_c = PatientHospitalAssociation(id=uuid.uuid4(), patient_id=pat_c.id, hospital_id=hosp_b.id)
        session.add_all([assoc_a, assoc_b, assoc_c])
        await session.commit()

        # 5. Records for Patient A
        doc_a = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=pat_a.id,
            hospital_id=hosp_a.id,
            local_document_uri="local://metro/storage/notes/doc_a.bin",
            document_type="cardiology_consult",
            status="completed",
        )
        session.add(doc_a)
        await session.commit()

        mask_a = PrivacyMaskingResult(
            id=uuid.uuid4(),
            document_id=doc_a.id,
            masking_method="BioClinicalBERT-PII-Masker",
            entities_detected_count=2,
            sanitization_status="sanitized",
        )
        ecg_a = ECGRecord(
            id=uuid.uuid4(),
            patient_id=pat_a.id,
            hospital_id=hosp_a.id,
            local_storage_ref="local://metro/ecg/ecg_a.mat",
            sampling_rate_hz=500,
            duration_seconds=10.0,
            lead_count=12,
            dataset_source="PTB-XL-Proxy",
        )
        obs_a = ClinicalObservation(
            id=uuid.uuid4(),
            patient_id=pat_a.id,
            hospital_id=hosp_a.id,
            observation_type="blood_pressure",
            feature_schema_version="v1.0",
            local_ref="local://metro/obs/obs_a.parquet",
        )
        session.add_all([mask_a, ecg_a, obs_a])
        await session.commit()

        # 6. Distinct Records for Patient B
        doc_b = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=pat_b.id,
            hospital_id=hosp_a.id,
            local_document_uri="local://metro/storage/notes/doc_b.bin",
            document_type="neurology_consult",
            status="completed",
        )
        session.add(doc_b)
        await session.commit()

        # Tokens
        token_a, _ = create_access_token(user_id=str(user_pat_a.id))
        token_b, _ = create_access_token(user_id=str(user_pat_b.id))
        token_c, _ = create_access_token(user_id=str(user_pat_c.id))
        token_clin, _ = create_access_token(user_id=str(user_clin.id))
        token_admin, _ = create_access_token(user_id=str(user_admin.id))

        return {
            "pat_a_id": str(pat_a.id),
            "pat_b_id": str(pat_b.id),
            "pat_c_id": str(pat_c.id),
            "token_a": token_a,
            "token_b": token_b,
            "token_c": token_c,
            "token_clin": token_clin,
            "token_admin": token_admin,
        }


@pytest.mark.anyio
async def test_unauthenticated_patient_request_returns_401():
    """Unauthenticated requests to patient portal endpoints return HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/patient/overview")
        assert res.status_code == 401


@pytest.mark.anyio
async def test_non_patient_roles_receive_403(seeded_patient_env):
    """Clinician and Hospital Admin receive HTTP 403 Forbidden on patient portal endpoints."""
    headers_clin = {"Authorization": f"Bearer {seeded_patient_env['token_clin']}"}
    headers_admin = {"Authorization": f"Bearer {seeded_patient_env['token_admin']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_clin = await ac.get("/api/v1/patient/overview", headers=headers_clin)
        assert res_clin.status_code == 403

        res_admin = await ac.get("/api/v1/patient/overview", headers=headers_admin)
        assert res_admin.status_code == 403


@pytest.mark.anyio
async def test_patient_can_access_own_overview_and_records(seeded_patient_env):
    """Patient A can access their own health overview, records, reports, and diagnosis status."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}
    pat_a_id = seeded_patient_env["pat_a_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Overview
        res_ov = await ac.get("/api/v1/patient/overview", headers=headers)
        assert res_ov.status_code == 200
        ov = res_ov.json()
        assert ov["patient"]["opaque_id"] == "OPAQUE_PAT_A_111"
        assert ov["summary"]["records_count"] == 3  # 1 doc + 1 ecg + 1 obs
        assert ov["summary"]["reports_count"] == 1
        assert ov["summary"]["health_status"] == "AVAILABLE"
        assert ov["assessment"]["status"] in ("NOT_ENABLED", "NO_DATA", "RESULTS_ON_FILE")
        assert ov["results"]["status"] in ("NO_DATA", "AVAILABLE")
        assert ov["results"]["stored_count"] == 0
        assert ov["capabilities"]["patient_messaging"] == "AVAILABLE"
        assert ov["capabilities"]["clinician_messaging"] == "AVAILABLE"

        # Records
        res_rec = await ac.get("/api/v1/patient/records", headers=headers)
        assert res_rec.status_code == 200
        recs = res_rec.json()
        assert recs["status"] == "AVAILABLE"
        assert recs["total_count"] == 3
        types = [r["record_type"] for r in recs["records"]]
        assert "Medical Document" in types
        assert "Cardiac Recording" in types
        assert "Health Observation" in types

        # Scoped Records with matching self ID
        res_rec_scoped = await ac.get(f"/api/v1/patient/records/{pat_a_id}", headers=headers)
        assert res_rec_scoped.status_code == 200

        # Reports
        res_rep = await ac.get("/api/v1/patient/reports", headers=headers)
        assert res_rep.status_code == 200
        reps = res_rep.json()
        assert reps["status"] == "AVAILABLE"
        assert reps["total_count"] == 1
        assert reps["reports"][0]["status"] == "Privacy Sanitized"

        # Diagnosis (none present -> honest NO_DATA / NOT_ENABLED)
        res_diag = await ac.get("/api/v1/patient/diagnosis", headers=headers)
        assert res_diag.status_code == 200
        diag = res_diag.json()
        assert diag["status"] == "NO_DATA"
        assert diag["assessment_status"] == "NOT_ENABLED"
        assert diag["prediction"] is None
        assert "AI-assisted assessment is not currently available" in diag["safety_wording"]


@pytest.mark.anyio
async def test_patient_cannot_access_other_patient_returns_404(seeded_patient_env):
    """Patient A cannot access Patient B's records or diagnosis (returns HTTP 404 Not Found)."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}
    pat_b_id = seeded_patient_env["pat_b_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_rec = await ac.get(f"/api/v1/patient/records/{pat_b_id}", headers=headers)
        assert res_rec.status_code == 404

        res_diag = await ac.get(f"/api/v1/patient/diagnosis/{pat_b_id}", headers=headers)
        assert res_diag.status_code == 404


@pytest.mark.anyio
async def test_patient_cannot_access_other_hospital_patient_returns_404(seeded_patient_env):
    """Patient A cannot access Patient C (Hospital B) records (returns HTTP 404 Not Found)."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}
    pat_c_id = seeded_patient_env["pat_c_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get(f"/api/v1/patient/records/{pat_c_id}", headers=headers)
        assert res.status_code == 404


@pytest.mark.anyio
async def test_changing_patient_id_in_url_returns_404(seeded_patient_env):
    """Arbitrary UUID in patient route returns HTTP 404 Not Found."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}
    fake_id = str(uuid.uuid4())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_rec = await ac.get(f"/api/v1/patient/records/{fake_id}", headers=headers)
        assert res_rec.status_code == 404

        res_diag = await ac.get(f"/api/v1/patient/diagnosis/{fake_id}", headers=headers)
        assert res_diag.status_code == 404


@pytest.mark.anyio
async def test_no_other_patient_data_leaks_in_responses(seeded_patient_env):
    """Patient A responses contain zero mentions of Patient B identifiers or records."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        for ep in ["/api/v1/patient/overview", "/api/v1/patient/records", "/api/v1/patient/reports"]:
            res = await ac.get(ep, headers=headers)
            assert res.status_code == 200
            assert "OPAQUE_PAT_B_222" not in res.text
            assert "neurology_consult" not in res.text


@pytest.mark.anyio
async def test_privacy_invariants_across_all_patient_responses(seeded_patient_env):
    """Verify zero secrets, raw waveforms, unmasked text, or internal PII in responses."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        endpoints = [
            "/api/v1/patient/overview",
            "/api/v1/patient/records",
            "/api/v1/patient/reports",
            "/api/v1/patient/diagnosis",
        ]
        forbidden_keys = [
            "password",
            "hashed_password",
            "token",
            "jwt",
            "refresh_token",
            "secret",
            "ssn",
            "mrn",
            "raw_notes",
            "raw_waveform",
            "voltage_array",
            "samples",
            "loss",
            "gradient",
            "logits",
        ]

        for ep in endpoints:
            res = await ac.get(ep, headers=headers)
            assert res.status_code == 200
            text = res.text.lower()
            for key in forbidden_keys:
                assert f'"{key}"' not in text, f"Forbidden key '{key}' found in response from {ep}"


@pytest.mark.anyio
async def test_empty_patient_records_displays_honest_no_data(seeded_patient_env):
    """Patient C with zero records receives canonical NO_DATA states and no fabricated 'Healthy' claims."""
    headers = {"Authorization": f"Bearer {seeded_patient_env['token_c']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_ov = await ac.get("/api/v1/patient/overview", headers=headers)
        assert res_ov.status_code == 200
        ov = res_ov.json()
        assert ov["summary"]["records_count"] == 0
        assert ov["summary"]["health_status"] == "NO_DATA"
        assert ov["summary"]["health_message"] == "No health information is currently available."
        # Must not state "Healthy" merely because data is missing
        assert "healthy" not in res_ov.text.lower()

        res_rec = await ac.get("/api/v1/patient/records", headers=headers)
        assert res_rec.status_code == 200
        rec = res_rec.json()
        assert rec["status"] == "NO_DATA"
        assert rec["total_count"] == 0
        assert rec["message"] == "No records are currently available."


@pytest.mark.anyio
async def test_genuine_prediction_surfaces_model_confidence_not_accuracy(seeded_patient_env):
    """When a genuine prediction exists, surfaces model confidence and safety wording; never accuracy."""
    pat_a_id = uuid.UUID(seeded_patient_env["pat_a_id"])

    async with TestingSessionLocal() as session:
        from sqlalchemy import select
        hosp = (await session.execute(select(Hospital))).scalars().first()
        model_ver = ModelVersion(
            id=uuid.uuid4(),
            version_tag="v1.0.0-gat-proxy",
            architecture_config='{"fusion": "GAT"}',
            target_definition="Elevated Arrhythmia Risk",
            status="experimental",
        )
        session.add(model_ver)
        await session.commit()

        pred = DiagnosisPrediction(
            id=uuid.uuid4(),
            patient_id=pat_a_id,
            hospital_id=hosp.id,
            model_version_id=model_ver.id,
            prediction_type="multimodal_risk",
            target_definition="Elevated Arrhythmia Risk",
            confidence=0.892,
            status="completed",
            provenance="GAT-Prototype-v0.1",
        )
        session.add(pred)
        await session.commit()

    headers = {"Authorization": f"Bearer {seeded_patient_env['token_a']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/patient/diagnosis", headers=headers)
        assert res.status_code == 200
        diag = res.json()
        assert diag["status"] == "AVAILABLE"
        assert diag["assessment_status"] == "RESULTS_ON_FILE"
        assert diag["prediction"] is not None
        assert diag["prediction"]["model_confidence"] == "89.2%"
        assert "model accuracy" not in res.text.lower()
        assert "diagnostic accuracy" not in res.text.lower()
        # Verify mandatory medical safety wording
        assert "This result was generated with AI-assisted clinical decision support" in diag["safety_wording"]
        assert "Please discuss your result with your qualified healthcare professional" in diag["safety_wording"]
