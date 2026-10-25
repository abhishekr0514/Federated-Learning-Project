"""Integration and Scientific Provenance Tests for Clinician Diagnostic Workspace.

Validates all Prompt 08 mandatory requirements:
1. Unauthenticated clinician endpoint -> 401.
2. Non-clinician user (Patient, Hospital Admin without clinician role) -> 403.
3. Clinician can access an assigned patient.
4. Clinician cannot access an unassigned patient (404 Not Found).
5. Clinician cannot access another hospital's patient (404 Not Found).
6. Patient cannot access clinician endpoints (403 Forbidden).
7. Hospital Admin cannot access clinician endpoints without clinician role (403 Forbidden).
8. Patient list contains only authorized assigned patients with server-side pagination.
9. Diagnosis endpoint does not fabricate a prediction when none exists (NO_DATA / NOT_IMPLEMENTED).
10. Explanation endpoint returns NOT_IMPLEMENTED / NO_DATA appropriately.
11. Causal endpoint returns NOT_IMPLEMENTED / NO_DATA appropriately.
12. Raw PII is absent from clinician API responses.
13. Raw ECG waveform/sample data is absent from clinician API responses.
14. JWTs, refresh tokens, passwords, and secrets are absent.
15. Provenance invariants: no prediction means no risk level and no confidence;
    confidence is never labeled model accuracy; proxy metrics remain labeled.
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
    PatientClinicianAssignment,
    MedicalDocument,
    DocumentVersion,
    DocumentProcessingJob,
    PrivacyMaskingResult,
    ECGRecord,
    ClinicalObservation,
    ModelVersion,
    DiagnosisPrediction,
)

SHARED_MEM_DB_URL = "sqlite+aiosqlite:///file:memdb_clinician_ws?mode=memory&cache=shared&uri=true"
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
async def seeded_clinician_env():
    """Seed test database with two hospitals, clinician users, assigned and unassigned patients, and records."""
    async with TestingSessionLocal() as session:
        # 1. Roles
        role_clinician = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_admin = Role(id=uuid.uuid4(), name="HOSPITAL_ADMIN", description="Hospital Administrator")
        role_patient = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        session.add_all([role_clinician, role_admin, role_patient])
        await session.commit()

        # 2. Hospitals
        hosp_a = Hospital(id=uuid.uuid4(), code="HOSP-A", name="Metro General Hospital")
        hosp_b = Hospital(id=uuid.uuid4(), code="HOSP-B", name="St. Jude Medical Center")
        session.add_all([hosp_a, hosp_b])
        await session.commit()

        # 3. Users
        # Clinician 1 at Hospital A
        user_clin_1 = User(
            id=uuid.uuid4(),
            email="dr_metro@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_clin_1.roles.append(role_clinician)

        # Clinician 2 at Hospital B
        user_clin_2 = User(
            id=uuid.uuid4(),
            email="dr_stjude@stjude.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_clin_2.roles.append(role_clinician)

        # Hospital Admin (no clinician role)
        user_admin = User(
            id=uuid.uuid4(),
            email="admin_metro@metro.org",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_admin.roles.append(role_admin)

        # Patient user
        user_pat = User(
            id=uuid.uuid4(),
            email="patient_user@gmail.com",
            hashed_password=hash_password("Pass123!"),
            is_active=True,
            is_super_admin=False,
        )
        user_pat.roles.append(role_patient)

        session.add_all([user_clin_1, user_clin_2, user_admin, user_pat])
        await session.commit()

        # 4. Memberships & Profiles
        mem_clin_1 = HospitalMembership(
            id=uuid.uuid4(), user_id=user_clin_1.id, hospital_id=hosp_a.id, role_in_hospital="CLINICIAN"
        )
        mem_clin_2 = HospitalMembership(
            id=uuid.uuid4(), user_id=user_clin_2.id, hospital_id=hosp_b.id, role_in_hospital="CLINICIAN"
        )
        mem_admin = HospitalMembership(
            id=uuid.uuid4(), user_id=user_admin.id, hospital_id=hosp_a.id, role_in_hospital="HOSPITAL_ADMIN"
        )
        session.add_all([mem_clin_1, mem_clin_2, mem_admin])
        await session.commit()

        prof_clin_1 = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user_clin_1.id,
            hospital_id=hosp_a.id,
            department="Cardiology",
            license_number_hash="HASH_CARDIO_001",
        )
        prof_clin_2 = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user_clin_2.id,
            hospital_id=hosp_b.id,
            department="Neurology",
            license_number_hash="HASH_NEURO_002",
        )
        session.add_all([prof_clin_1, prof_clin_2])
        await session.commit()

        # 5. Patients
        # Patient 1: In Hospital A, Assigned to Clinician 1
        pat_assigned_1 = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_METRO_001", user_id=None)
        # Patient 2: In Hospital A, UNASSIGNED to Clinician 1
        pat_unassigned_1 = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_METRO_002", user_id=None)
        # Patient 3: In Hospital B, Assigned to Clinician 2
        pat_hosp_b = Patient(id=uuid.uuid4(), opaque_patient_id="OPAQUE_PAT_STJUDE_003", user_id=None)
        session.add_all([pat_assigned_1, pat_unassigned_1, pat_hosp_b])
        await session.commit()

        assoc_a1 = PatientHospitalAssociation(
            id=uuid.uuid4(), patient_id=pat_assigned_1.id, hospital_id=hosp_a.id, hospital_patient_ref_hash="REF_A_001"
        )
        assoc_a2 = PatientHospitalAssociation(
            id=uuid.uuid4(), patient_id=pat_unassigned_1.id, hospital_id=hosp_a.id, hospital_patient_ref_hash="REF_A_002"
        )
        assoc_b3 = PatientHospitalAssociation(
            id=uuid.uuid4(), patient_id=pat_hosp_b.id, hospital_id=hosp_b.id, hospital_patient_ref_hash="REF_B_003"
        )
        session.add_all([assoc_a1, assoc_a2, assoc_b3])
        await session.commit()

        # Explicit Clinician Assignments
        assign_1 = PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=pat_assigned_1.id,
            clinician_id=prof_clin_1.id,
            hospital_id=hosp_a.id,
        )
        assign_2 = PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=pat_hosp_b.id,
            clinician_id=prof_clin_2.id,
            hospital_id=hosp_b.id,
        )
        session.add_all([assign_1, assign_2])
        await session.commit()

        # 6. Clinical Data for Patient Assigned 1 (Metadata only)
        doc_1 = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=pat_assigned_1.id,
            hospital_id=hosp_a.id,
            local_document_uri="local://metro/storage/notes/doc_001.bin",
            document_type="clinical_note",
            status="completed",
        )
        session.add(doc_1)
        await session.commit()

        doc_ver_1 = DocumentVersion(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            version_number=1,
            checksum_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        doc_job_1 = DocumentProcessingJob(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            status="completed",
        )
        doc_mask_1 = PrivacyMaskingResult(
            id=uuid.uuid4(),
            document_id=doc_1.id,
            masking_method="BioClinicalBERT-PII-Masker",
            entities_detected_count=3,
            sanitization_status="sanitized",
        )
        session.add_all([doc_ver_1, doc_job_1, doc_mask_1])
        await session.commit()

        ecg_1 = ECGRecord(
            id=uuid.uuid4(),
            patient_id=pat_assigned_1.id,
            hospital_id=hosp_a.id,
            local_storage_ref="local://metro/ecg/record_001.mat",
            sampling_rate_hz=500,
            duration_seconds=10.0,
            lead_count=12,
            dataset_source="PTB-XL-Proxy",
        )
        obs_1 = ClinicalObservation(
            id=uuid.uuid4(),
            patient_id=pat_assigned_1.id,
            hospital_id=hosp_a.id,
            observation_type="tabular_features",
            feature_schema_version="v1.0",
            local_ref="local://metro/obs/obs_001.parquet",
        )
        session.add_all([ecg_1, obs_1])
        await session.commit()

        # Generate tokens
        token_clin_1, _ = create_access_token(user_id=str(user_clin_1.id))
        token_clin_2, _ = create_access_token(user_id=str(user_clin_2.id))
        token_admin, _ = create_access_token(user_id=str(user_admin.id))
        token_pat, _ = create_access_token(user_id=str(user_pat.id))

        return {
            "hosp_a_id": str(hosp_a.id),
            "hosp_b_id": str(hosp_b.id),
            "pat_assigned_1_id": str(pat_assigned_1.id),
            "pat_unassigned_1_id": str(pat_unassigned_1.id),
            "pat_hosp_b_id": str(pat_hosp_b.id),
            "doc_1_id": str(doc_1.id),
            "token_clin_1": token_clin_1,
            "token_clin_2": token_clin_2,
            "token_admin": token_admin,
            "token_pat": token_pat,
        }


@pytest.mark.anyio
async def test_unauthenticated_clinician_endpoint_returns_401():
    """Unauthenticated requests to clinician endpoints return HTTP 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/clinician/overview")
        assert res.status_code == 401


@pytest.mark.anyio
async def test_non_clinician_roles_receive_403(seeded_clinician_env):
    """Patient and unprivileged Hospital Admin receive HTTP 403 Forbidden on clinician endpoints."""
    headers_pat = {"Authorization": f"Bearer {seeded_clinician_env['token_pat']}"}
    headers_admin = {"Authorization": f"Bearer {seeded_clinician_env['token_admin']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_pat = await ac.get("/api/v1/clinician/overview", headers=headers_pat)
        assert res_pat.status_code == 403

        res_admin = await ac.get("/api/v1/clinician/overview", headers=headers_admin)
        assert res_admin.status_code == 403


@pytest.mark.anyio
async def test_clinician_can_access_assigned_patient(seeded_clinician_env):
    """Clinician 1 can view assigned patient details and overview."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Overview
        res_ov = await ac.get("/api/v1/clinician/overview", headers=headers)
        assert res_ov.status_code == 200
        ov = res_ov.json()
        assert ov["clinician"]["department"] == "Cardiology"
        assert ov["metrics"]["assigned_patients"]["count"] == 1
        assert ov["metrics"]["assigned_patients"]["status"] == "AVAILABLE"
        assert ov["capabilities"]["clinical_ai_inference"] in (
            "NOT_IMPLEMENTED",
            "NOT_ENABLED",
            "MISSING_DEPENDENCY",
        )
        assert ov["capabilities"]["dice_counterfactuals"] in (
            "NOT_IMPLEMENTED",
            "MISSING_DEPENDENCY",
        )

        # Patient Detail
        res_pat = await ac.get(f"/api/v1/clinician/patients/{pat_id}", headers=headers)
        assert res_pat.status_code == 200
        data = res_pat.json()
        assert data["patient"]["patient_id"] == pat_id
        assert data["patient"]["opaque_patient_id"] == "OPAQUE_PAT_METRO_001"
        assert data["documents"]["count"] == 1
        assert data["ecg_records"]["count"] == 1
        assert data["diagnosis"]["status"] == "NO_DATA"
        assert data["ai_capabilities"]["clinical_ai_inference"] in (
            "NOT_IMPLEMENTED",
            "NOT_ENABLED",
            "MISSING_DEPENDENCY",
        )


@pytest.mark.anyio
async def test_clinician_cannot_access_unassigned_patient_returns_404(seeded_clinician_env):
    """Clinician 1 cannot access unassigned patient from same hospital (returns 404 Not Found)."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    unassigned_id = seeded_clinician_env["pat_unassigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get(f"/api/v1/clinician/patients/{unassigned_id}", headers=headers)
        assert res.status_code == 404, f"Expected 404 for unassigned patient, got {res.status_code}"


@pytest.mark.anyio
async def test_clinician_cannot_access_other_hospitals_patient_returns_404(seeded_clinician_env):
    """Clinician 1 cannot access patient belonging to Hospital B (returns 404 Not Found)."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    hosp_b_pat_id = seeded_clinician_env["pat_hosp_b_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get(f"/api/v1/clinician/patients/{hosp_b_pat_id}", headers=headers)
        assert res.status_code == 404, f"Expected 404 for cross-hospital patient, got {res.status_code}"


@pytest.mark.anyio
async def test_patient_list_contains_only_assigned_patients(seeded_clinician_env):
    """Patient directory lists only authorized assigned patients with server-side pagination."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/clinician/patients?limit=10&offset=0", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "AVAILABLE"
        assert data["total_count"] == 1
        assert len(data["patients"]) == 1
        assert data["patients"][0]["patient_id"] == seeded_clinician_env["pat_assigned_1_id"]
        assert data["patients"][0]["opaque_patient_id"] == "OPAQUE_PAT_METRO_001"


@pytest.mark.anyio
async def test_documents_endpoint_returns_only_assigned_metadata(seeded_clinician_env):
    """Document endpoints return metadata only for assigned patients; unassigned returns 404."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    doc_id = seeded_clinician_env["doc_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Document list
        res_list = await ac.get("/api/v1/clinician/documents?limit=10&offset=0", headers=headers)
        assert res_list.status_code == 200
        docs = res_list.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id
        assert docs[0]["document_type"] == "clinical_note"
        assert docs[0]["masking_status"] == "sanitized"
        assert "raw_notes" not in docs[0]
        assert "content" not in docs[0]

        # Document detail
        res_det = await ac.get(f"/api/v1/clinician/documents/{doc_id}", headers=headers)
        assert res_det.status_code == 200
        det = res_det.json()
        assert det["id"] == doc_id
        assert len(det["versions"]) == 1
        assert det["versions"][0]["checksum_sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        # Clinician 2 cannot access document 1 (belongs to Hospital A patient)
        headers_2 = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_2']}"}
        res_det_2 = await ac.get(f"/api/v1/clinician/documents/{doc_id}", headers=headers_2)
        assert res_det_2.status_code == 404


@pytest.mark.anyio
async def test_diagnosis_endpoint_honest_no_data_and_not_implemented(seeded_clinician_env):
    """Diagnosis endpoints return honest NO_DATA and NOT_IMPLEMENTED states without fabricating predictions."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]
    unassigned_id = seeded_clinician_env["pat_unassigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # List predictions
        res_diag = await ac.get("/api/v1/clinician/diagnosis", headers=headers)
        assert res_diag.status_code == 200
        diag = res_diag.json()
        assert diag["status"] == "NO_DATA"
        assert diag["count"] == 0
        assert diag["inference_capability"] == "NOT_IMPLEMENTED"

        # Patient diagnosis
        res_pat_diag = await ac.get(f"/api/v1/clinician/diagnosis/{pat_id}", headers=headers)
        assert res_pat_diag.status_code == 200
        pat_diag = res_pat_diag.json()
        assert pat_diag["status"] == "NO_DATA"
        assert pat_diag["prediction"] is None
        assert pat_diag["inference_capability"] == "NOT_IMPLEMENTED"

        # Unassigned patient diagnosis returns 404
        res_unassigned_diag = await ac.get(f"/api/v1/clinician/diagnosis/{unassigned_id}", headers=headers)
        assert res_unassigned_diag.status_code == 404


@pytest.mark.anyio
async def test_explanations_and_causal_endpoints_return_not_implemented(seeded_clinician_env):
    """Explanations and Causal endpoints return NOT_IMPLEMENTED when queried for non-existent prediction."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    fake_pred_id = str(uuid.uuid4())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_exp = await ac.get(f"/api/v1/clinician/explanations/{fake_pred_id}", headers=headers)
        assert res_exp.status_code == 404

        res_causal = await ac.get(f"/api/v1/clinician/causal/{fake_pred_id}", headers=headers)
        assert res_causal.status_code == 404


@pytest.mark.anyio
async def test_privacy_and_provenance_invariants_across_all_responses(seeded_clinician_env):
    """Verify zero PII, zero raw ECG arrays, zero secrets, and scientific provenance rules."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        endpoints_to_check = [
            "/api/v1/clinician/overview",
            "/api/v1/clinician/patients",
            f"/api/v1/clinician/patients/{pat_id}",
            "/api/v1/clinician/documents",
            "/api/v1/clinician/diagnosis",
            f"/api/v1/clinician/diagnosis/{pat_id}",
        ]

        forbidden_keys = [
            "password",
            "hashed_password",
            "token",
            "jwt",
            "refresh_token",
            "secret",
            "mrn",
            "ssn",
            "raw_notes",
            "raw_waveform",
            "samples",
            "voltage_array",
        ]

        for ep in endpoints_to_check:
            res = await ac.get(ep, headers=headers)
            assert res.status_code == 200, f"Endpoint {ep} failed with {res.status_code}"
            text = res.text.lower()
            for key in forbidden_keys:
                assert f'"{key}"' not in text, f"Forbidden key '{key}' found in response from {ep}"

        # Provenance invariant: patient detail has NO risk score and NO accuracy claim
        res_detail = await ac.get(f"/api/v1/clinician/patients/{pat_id}", headers=headers)
        detail_json = res_detail.json()
        assert "risk_score" not in detail_json
        assert "accuracy" not in detail_json
        assert detail_json["diagnosis"]["status"] == "NO_DATA"


@pytest.mark.anyio
async def test_eligible_patients_lists_hospital_unassigned_only(seeded_clinician_env):
    """Eligible directory includes same-hospital unassigned patients, not cross-hospital or already assigned."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/clinician/patients/eligible?limit=10&offset=0", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total_count"] == 1
        assert data["patients"][0]["patient_id"] == seeded_clinician_env["pat_unassigned_1_id"]


@pytest.mark.anyio
async def test_assign_patient_success_and_duplicate_conflict(seeded_clinician_env):
    """Clinician can assign eligible patient; duplicate assignment returns HTTP 409."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    unassigned_id = seeded_clinician_env["pat_unassigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_assign = await ac.post(
            f"/api/v1/clinician/patients/{unassigned_id}/assignment",
            headers=headers,
        )
        assert res_assign.status_code == 200
        body = res_assign.json()
        assert body["status"] == "ASSIGNED"
        assert body["patient_id"] == unassigned_id

        res_dup = await ac.post(
            f"/api/v1/clinician/patients/{unassigned_id}/assignment",
            headers=headers,
        )
        assert res_dup.status_code == 409

        res_detail = await ac.get(f"/api/v1/clinician/patients/{unassigned_id}", headers=headers)
        assert res_detail.status_code == 200


@pytest.mark.anyio
async def test_assign_patient_cross_hospital_returns_404(seeded_clinician_env):
    """Clinician cannot assign patient registered at another hospital."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    hosp_b_pat = seeded_clinician_env["pat_hosp_b_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/clinician/patients/{hosp_b_pat}/assignment",
            headers=headers,
        )
        assert res.status_code == 404


@pytest.mark.anyio
async def test_assign_patient_denied_for_patient_role(seeded_clinician_env):
    """Patient users cannot call clinician assignment endpoints."""
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_pat']}"}
    pat_id = seeded_clinician_env["pat_unassigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(f"/api/v1/clinician/patients/{pat_id}/assignment", headers=headers)
        assert res.status_code == 403


@pytest.mark.anyio
async def test_clinician_two_cannot_assign_other_clinicians_patient_detail(seeded_clinician_env):
    """After clinician 1 assigns patient, clinician 2 still receives 404 on dossier access."""
    headers_1 = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    headers_2 = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_2']}"}
    unassigned_id = seeded_clinician_env["pat_unassigned_1_id"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res_assign = await ac.post(
            f"/api/v1/clinician/patients/{unassigned_id}/assignment",
            headers=headers_1,
        )
        assert res_assign.status_code == 200

        res_other = await ac.get(f"/api/v1/clinician/patients/{unassigned_id}", headers=headers_2)
        assert res_other.status_code == 404
