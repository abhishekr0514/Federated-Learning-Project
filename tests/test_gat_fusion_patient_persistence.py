"""Patient-linked GAT fusion evaluation: persistence, isolation, and research-proxy behavior."""

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
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
    Role,
    User,
)

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_gat_persist?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

_MOCK_INFERENCE = {
    "target_definition": "PTB-XL MI/STTC vs NORM (research proxy)",
    "prediction_type": "proxy_evaluation",
    "model_version_tag": "gat-test-v1",
    "confidence_uncalibrated": 0.62,
    "predicted_class_index": 1,
    "predicted_class_label": "MI/STTC",
    "architecture_id": "medshield.diagnostic_net.gat_ecg_report",
    "research_only": True,
    "proxy_task": True,
    "disclaimer": "Uncalibrated research proxy output (two-modality GAT).",
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


def _emb(seed: float = 0.01) -> list[float]:
    return [seed * (i % 17 - 8) for i in range(128)]


def _mock_runtime_ready():
    runtime = MagicMock()
    readiness = MagicMock()
    readiness.ready = True
    readiness.to_safe_dict.return_value = {}
    runtime.assess_readiness.return_value = readiness
    runtime.run_proxy_inference.return_value = dict(_MOCK_INFERENCE)
    return runtime


@pytest.fixture
async def clinical_assignment_env():
    async with TestingSessionLocal() as session:
        role_clin = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_pat = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        session.add_all([role_clin, role_pat])
        hosp = Hospital(id=uuid.uuid4(), name="Metro", code="METRO")
        session.add(hosp)

        clinician_user = User(
            id=uuid.uuid4(),
            email="dr.gat.persist@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role_clin],
        )
        session.add(clinician_user)
        await session.flush()

        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=clinician_user.id,
            hospital_id=hosp.id,
            license_number_hash="LIC-GAT",
        )
        session.add(profile)

        pat_a_user = User(
            id=uuid.uuid4(),
            email="patient.gat.a@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role_pat],
        )
        pat_b_user = User(
            id=uuid.uuid4(),
            email="patient.gat.b@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role_pat],
        )
        session.add_all([pat_a_user, pat_b_user])
        await session.flush()

        patient_a = Patient(
            id=uuid.uuid4(),
            user_id=pat_a_user.id,
            opaque_patient_id="OPAQUE-GAT-A",
        )
        patient_b = Patient(
            id=uuid.uuid4(),
            user_id=pat_b_user.id,
            opaque_patient_id="OPAQUE-GAT-B",
        )
        session.add_all([patient_a, patient_b])
        await session.flush()

        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient_a.id,
                hospital_id=hosp.id,
                hospital_patient_ref_hash="hash-a",
            )
        )
        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient_b.id,
                hospital_id=hosp.id,
                hospital_patient_ref_hash="hash-b",
            )
        )
        session.add(
            PatientClinicianAssignment(
                id=uuid.uuid4(),
                patient_id=patient_a.id,
                clinician_id=profile.id,
                hospital_id=hosp.id,
            )
        )
        await session.commit()

        clinician_token, _ = create_access_token(str(clinician_user.id))
        patient_a_token, _ = create_access_token(str(pat_a_user.id))
        patient_b_token, _ = create_access_token(str(pat_b_user.id))

    return {
        "clinician_token": clinician_token,
        "patient_a_token": patient_a_token,
        "patient_b_token": patient_b_token,
        "patient_a_id": patient_a.id,
        "patient_b_id": patient_b.id,
    }


@pytest.mark.anyio
async def test_gat_research_proxy_does_not_persist(clinical_assignment_env):
    token = clinical_assignment_env["clinician_token"]
    patient_id = clinical_assignment_env["patient_a_id"]

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime_ready(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    "/api/v1/gat-fusion/evaluate",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "proxy-only-key",
                    },
                )
                assert res.status_code == 200
                assert res.json()["patient_id"] is None
                assert res.json()["proxy_task"] is True

                async with TestingSessionLocal() as session:
                    count = (
                        await session.execute(select(DiagnosisPrediction))
                    ).scalars().all()
                    assert len(count) == 0

                res_linked = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={patient_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "linked-key",
                    },
                )
                assert res_linked.status_code == 200
                assert res_linked.json()["patient_id"] == str(patient_id)
                assert res_linked.json()["proxy_task"] is False


@pytest.mark.anyio
async def test_gat_patient_linked_persists_and_patient_portal_sees_result(clinical_assignment_env):
    token = clinical_assignment_env["clinician_token"]
    patient_a_token = clinical_assignment_env["patient_a_token"]
    patient_id = clinical_assignment_env["patient_a_id"]

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime_ready(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={patient_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "persist-key-1",
                    },
                )
                assert res.status_code == 200
                prediction_id = res.json()["prediction_id"]

                async with TestingSessionLocal() as session:
                    row = (
                        await session.execute(
                            select(DiagnosisPrediction).where(
                                DiagnosisPrediction.id == uuid.UUID(prediction_id)
                            )
                        )
                    ).scalar_one()
                    assert row.patient_id == patient_id
                    assert row.status == "evaluated"
                    assert row.provenance == "idempotency:persist-key-1"

                portal = await ac.get(
                    "/api/v1/patient/diagnosis",
                    headers={"Authorization": f"Bearer {patient_a_token}"},
                )
                assert portal.status_code == 200
                body = portal.json()
                assert body["status"] == "AVAILABLE"
                assert body["prediction"]["prediction_id"] == prediction_id


@pytest.mark.anyio
async def test_gat_patient_isolation_other_patient_sees_no_data(clinical_assignment_env):
    token = clinical_assignment_env["clinician_token"]
    patient_a_id = clinical_assignment_env["patient_a_id"]
    patient_b_token = clinical_assignment_env["patient_b_token"]

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime_ready(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={patient_a_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                    },
                )
                portal_b = await ac.get(
                    "/api/v1/patient/diagnosis",
                    headers={"Authorization": f"Bearer {patient_b_token}"},
                )
                assert portal_b.status_code == 200
                assert portal_b.json()["status"] == "NO_DATA"


@pytest.mark.anyio
async def test_gat_unassigned_patient_returns_404(clinical_assignment_env):
    token = clinical_assignment_env["clinician_token"]
    patient_b_id = clinical_assignment_env["patient_b_id"]

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime_ready(),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={patient_b_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
                )
                assert res.status_code == 404


@pytest.mark.anyio
async def test_gat_inference_disabled_returns_503(clinical_assignment_env):
    token = clinical_assignment_env["clinician_token"]
    patient_id = clinical_assignment_env["patient_a_id"]

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.post(
                f"/api/v1/gat-fusion/evaluate?patient_id={patient_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
            )
        assert res.status_code == 503
