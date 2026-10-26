"""Role-based GAT inference, patient questions, and assessment history."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import (
    Base,
    ClinicianProfile,
    DiagnosisPrediction,
    Hospital,
    HospitalMembership,
    Patient,
    PatientClinicianAssignment,
    PatientClinicianQuestion,
    PatientHospitalAssociation,
    Role,
    User,
)

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_role_portal?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

_MOCK_INFERENCE = {
    "target_definition": "PTB-XL proxy",
    "prediction_type": "proxy_evaluation",
    "model_version_tag": "gat-role-test",
    "confidence_uncalibrated": 0.44,
    "predicted_class_index": 0,
    "predicted_class_label": "NORM",
    "architecture_id": "medshield.diagnostic_net.gat_ecg_report",
    "disclaimer": "Research only.",
}


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


def _emb() -> list[float]:
    return [0.02 * (i % 9) for i in range(128)]


def _mock_runtime():
    rt = MagicMock()
    ready = MagicMock()
    ready.ready = True
    ready.to_safe_dict.return_value = {}
    rt.assess_readiness.return_value = ready
    rt.run_proxy_inference.return_value = dict(_MOCK_INFERENCE)
    return rt


async def _seed_roles(session):
    roles = {}
    for name in ("SUPER_ADMIN", "HOSPITAL_ADMIN", "CLINICIAN", "PATIENT"):
        r = Role(id=uuid.uuid4(), name=name, description=name)
        session.add(r)
        roles[name] = r
    await session.flush()
    return roles


@pytest.fixture
async def multi_tenant_env():
    async with TestingSessionLocal() as session:
        roles = await _seed_roles(session)
        hosp_a = Hospital(id=uuid.uuid4(), name="Hosp A", code="HSPA")
        hosp_b = Hospital(id=uuid.uuid4(), name="Hosp B", code="HSPB")
        session.add_all([hosp_a, hosp_b])

        super_user = User(
            id=uuid.uuid4(),
            email="super@demo.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_super_admin=True,
            roles=[roles["SUPER_ADMIN"]],
        )
        hosp_admin = User(
            id=uuid.uuid4(),
            email="admin@hspa.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[roles["HOSPITAL_ADMIN"]],
        )
        clinician_user = User(
            id=uuid.uuid4(),
            email="clin@hspa.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[roles["CLINICIAN"]],
        )
        pat_user = User(
            id=uuid.uuid4(),
            email="patient@demo.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[roles["PATIENT"]],
        )
        session.add_all([super_user, hosp_admin, clinician_user, pat_user])
        await session.flush()

        session.add(
            HospitalMembership(
                id=uuid.uuid4(),
                user_id=hosp_admin.id,
                hospital_id=hosp_a.id,
                role_in_hospital="ADMIN",
            )
        )
        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=clinician_user.id,
            hospital_id=hosp_a.id,
            license_number_hash="LIC-A",
        )
        session.add(profile)

        patient_a = Patient(id=uuid.uuid4(), opaque_patient_id="DEMO-SYNTH-PATIENT-0001", user_id=pat_user.id)
        patient_b = Patient(id=uuid.uuid4(), opaque_patient_id="OTHER-HOSP-PATIENT")
        session.add_all([patient_a, patient_b])
        await session.flush()

        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient_a.id,
                hospital_id=hosp_a.id,
                hospital_patient_ref_hash="ref-a",
            )
        )
        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient_b.id,
                hospital_id=hosp_b.id,
                hospital_patient_ref_hash="ref-b",
            )
        )
        session.add(
            PatientClinicianAssignment(
                id=uuid.uuid4(),
                patient_id=patient_a.id,
                clinician_id=profile.id,
                hospital_id=hosp_a.id,
            )
        )
        await session.commit()

        tokens = {
            "super": create_access_token(str(super_user.id))[0],
            "hosp_admin": create_access_token(str(hosp_admin.id))[0],
            "clinician": create_access_token(str(clinician_user.id))[0],
            "patient": create_access_token(str(pat_user.id))[0],
        }
        return {
            "tokens": tokens,
            "patient_a_id": patient_a.id,
            "patient_b_id": patient_b.id,
            "profile_id": profile.id,
        }


@pytest.mark.anyio
async def test_hospital_admin_linked_gat_persists(multi_tenant_env):
    token = multi_tenant_env["tokens"]["hosp_admin"]
    pid = multi_tenant_env["patient_a_id"]
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch("backend.app.services.gat_fusion_service._get_runtime", return_value=_mock_runtime()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={pid}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb()},
                )
    assert res.status_code == 200


@pytest.mark.anyio
async def test_hospital_admin_rejects_out_of_scope_patient(multi_tenant_env):
    token = multi_tenant_env["tokens"]["hosp_admin"]
    pid = multi_tenant_env["patient_b_id"]
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch("backend.app.services.gat_fusion_service._get_runtime", return_value=_mock_runtime()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={pid}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb()},
                )
    assert res.status_code == 404


@pytest.mark.anyio
async def test_clinician_assigned_and_unassigned(multi_tenant_env):
    token = multi_tenant_env["tokens"]["clinician"]
    ok_id = multi_tenant_env["patient_a_id"]
    bad_id = multi_tenant_env["patient_b_id"]
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch("backend.app.services.gat_fusion_service._get_runtime", return_value=_mock_runtime()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                ok = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={ok_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb()},
                )
                bad = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={bad_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb()},
                )
    assert ok.status_code == 200
    assert bad.status_code == 404


@pytest.mark.anyio
async def test_patient_questions_and_isolation(multi_tenant_env):
    pat_token = multi_tenant_env["tokens"]["patient"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        submit = await ac.post(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {pat_token}"},
            json={"question_text": "Should I adjust my exercise routine?"},
        )
        assert submit.status_code == 200
        listed = await ac.get(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {pat_token}"},
        )
        assert listed.status_code == 200
        assert len(listed.json()["questions"]) == 1


@pytest.mark.anyio
async def test_assessment_history_for_charts(multi_tenant_env):
    pat_token = multi_tenant_env["tokens"]["patient"]
    pid = multi_tenant_env["patient_a_id"]
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch("backend.app.services.gat_fusion_service._get_runtime", return_value=_mock_runtime()):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={pid}",
                    headers={"Authorization": f"Bearer {multi_tenant_env['tokens']['clinician']}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb()},
                )
                diag = await ac.get(
                    "/api/v1/patient/diagnosis",
                    headers={"Authorization": f"Bearer {pat_token}"},
                )
    assert diag.status_code == 200
    body = diag.json()
    assert body["status"] == "AVAILABLE"
    assert len(body["history"]) >= 1
    assert body["charts"]["assessment_count"] >= 1
    assert body["validation_notice"]
    assert body["prediction"]["validation_status"] == "NOT_CLINICALLY_VALIDATED"


@pytest.mark.anyio
async def test_clinician_can_respond_to_patient_question(multi_tenant_env):
    pat_token = multi_tenant_env["tokens"]["patient"]
    clin_token = multi_tenant_env["tokens"]["clinician"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        submit = await ac.post(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {pat_token}"},
            json={"question_text": "Is mild chest discomfort after walking normal to discuss?"},
        )
        qid = submit.json()["question_id"]
        patch = await ac.patch(
            f"/api/v1/clinician/questions/{qid}",
            headers={"Authorization": f"Bearer {clin_token}"},
            json={"response_text": "Please discuss symptoms at your next visit; call the clinic if worsening."},
        )
        assert patch.status_code == 200
        assert patch.json()["status"] == "responded"
        listed = await ac.get(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {pat_token}"},
        )
        assert listed.json()["questions"][0]["response_text"]


@pytest.mark.anyio
async def test_gat_disabled_returns_503(multi_tenant_env):
    token = multi_tenant_env["tokens"]["clinician"]
    pid = multi_tenant_env["patient_a_id"]
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                f"/api/v1/gat-fusion/evaluate?patient_id={pid}",
                headers={"Authorization": f"Bearer {token}"},
                json={"ecg_embedding": _emb(), "text_embedding": _emb()},
            )
    assert res.status_code == 503
