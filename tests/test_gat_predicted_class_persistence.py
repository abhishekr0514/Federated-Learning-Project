"""GAT predicted_class_index persistence, idempotent replay, and patient portal history."""

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
    ModelVersion,
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
    Role,
    User,
)
from backend.app.services.prediction_proxy_class import (
    LEGACY_PROXY_CLASS_LABEL,
    MODEL_FAMILY_ECG_REPORT_FUSION,
    MODEL_FAMILY_GAT_ECG_REPORT_FUSION,
    resolve_patient_portal_proxy_class,
)
from backend.app.services.gat_fusion_service import GATFusionService

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_gat_class?mode=memory&cache=shared&uri=true"
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


def _mock_runtime(inference: dict):
    runtime = MagicMock()
    readiness = MagicMock()
    readiness.ready = True
    readiness.to_safe_dict.return_value = {}
    runtime.assess_readiness.return_value = readiness
    runtime.run_proxy_inference.return_value = dict(inference)
    return runtime


def _inference(*, class_index: int, confidence: float) -> dict:
    label = "MI/STTC" if class_index == 1 else "NORM"
    return {
        "target_definition": "PTB-XL proxy",
        "prediction_type": "proxy_evaluation",
        "model_version_tag": "gat-class-test",
        "confidence_uncalibrated": confidence,
        "predicted_class_index": class_index,
        "predicted_class_label": label,
        "architecture_id": "medshield.diagnostic_net.gat_ecg_report",
        "disclaimer": "Research only.",
    }


@pytest.fixture
async def env():
    async with TestingSessionLocal() as session:
        role_clin = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        role_pat = Role(id=uuid.uuid4(), name="PATIENT", description="Patient")
        session.add_all([role_clin, role_pat])
        hosp = Hospital(id=uuid.uuid4(), name="Metro", code="METRO-CLS")
        session.add(hosp)
        clinician_user = User(
            id=uuid.uuid4(),
            email="dr.class@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role_clin],
        )
        pat_user = User(
            id=uuid.uuid4(),
            email="patient.class@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role_pat],
        )
        session.add_all([clinician_user, pat_user])
        await session.flush()
        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=clinician_user.id,
            hospital_id=hosp.id,
            license_number_hash="LIC-CLS",
        )
        session.add(profile)
        patient = Patient(
            id=uuid.uuid4(),
            user_id=pat_user.id,
            opaque_patient_id="OPAQUE-CLASS",
        )
        session.add(patient)
        await session.flush()
        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient.id,
                hospital_id=hosp.id,
                hospital_patient_ref_hash="hash",
            )
        )
        session.add(
            PatientClinicianAssignment(
                id=uuid.uuid4(),
                patient_id=patient.id,
                clinician_id=profile.id,
                hospital_id=hosp.id,
            )
        )
        await session.commit()
        clin_token, _ = create_access_token(str(clinician_user.id))
        pat_token, _ = create_access_token(str(pat_user.id))
    return {
        "clinician_token": clin_token,
        "patient_token": pat_token,
        "patient_id": patient.id,
    }


@pytest.mark.anyio
async def test_gat_norm_high_confidence_persisted_and_portal_shows_norm(env):
    """NORM (0) with confidence > 0.5 must not be mislabeled MI/STTC."""
    inference = _inference(class_index=0, confidence=0.683)
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime(inference),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={env['patient_id']}",
                    headers={"Authorization": f"Bearer {env['clinician_token']}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "norm-high-conf",
                    },
                )
                assert res.status_code == 200
                body = res.json()
                assert body["predicted_class_index"] == 0
                assert body["predicted_class_label"] == "NORM"
                assert body["confidence_uncalibrated"] == 0.683

                async with TestingSessionLocal() as session:
                    row = (
                        await session.execute(select(DiagnosisPrediction))
                    ).scalar_one()
                    assert row.predicted_class_index == 0
                    assert row.confidence == 0.683

                portal = await ac.get(
                    "/api/v1/patient/diagnosis",
                    headers={"Authorization": f"Bearer {env['patient_token']}"},
                )
                pred = portal.json()["prediction"]
                assert pred["predicted_proxy_class"] == "NORM"
                assert pred["predicted_class_index"] == 0
                assert pred["model_confidence_raw"] == 0.683


@pytest.mark.anyio
async def test_gat_mi_high_confidence_persisted(env):
    inference = _inference(class_index=1, confidence=0.91)
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime(inference),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={env['patient_id']}",
                    headers={"Authorization": f"Bearer {env['clinician_token']}"},
                    json={"ecg_embedding": _emb(), "text_embedding": _emb(0.02)},
                )
                assert res.json()["predicted_class_index"] == 1
                assert res.json()["predicted_class_label"] == "MI/STTC"


@pytest.mark.anyio
async def test_gat_idempotent_replay_preserves_class_and_confidence(env):
    inference = _inference(class_index=0, confidence=0.72)
    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        with patch(
            "backend.app.services.gat_fusion_service._get_runtime",
            return_value=_mock_runtime(inference),
        ) as get_rt:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                first = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={env['patient_id']}",
                    headers={"Authorization": f"Bearer {env['clinician_token']}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "idem-class",
                    },
                )
                assert first.status_code == 200
                second = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={env['patient_id']}",
                    headers={"Authorization": f"Bearer {env['clinician_token']}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(0.02),
                        "idempotency_key": "idem-class",
                    },
                )
                assert second.status_code == 200
                assert second.json()["predicted_class_index"] == 0
                assert second.json()["predicted_class_label"] == "NORM"
                assert second.json()["confidence_uncalibrated"] == 0.72
                assert get_rt.return_value.run_proxy_inference.call_count == 1


def test_legacy_gat_row_does_not_guess_class_from_confidence():
    proxy = resolve_patient_portal_proxy_class(
        confidence=0.85,
        predicted_class_index=None,
        model_family=MODEL_FAMILY_GAT_ECG_REPORT_FUSION,
        class_meaning_map={},
    )
    assert proxy["predicted_proxy_class"] == LEGACY_PROXY_CLASS_LABEL
    assert proxy["predicted_class_index"] is None
    assert proxy["proxy_class_status"] == "LEGACY_UNAVAILABLE"


def test_ecg_probability_semantics_unchanged_for_legacy_rows():
    proxy = resolve_patient_portal_proxy_class(
        confidence=0.35,
        predicted_class_index=None,
        model_family=MODEL_FAMILY_ECG_REPORT_FUSION,
        class_meaning_map={},
    )
    assert proxy["predicted_proxy_class"] == "NORM"
    assert proxy["predicted_class_index"] == 0
    assert proxy["proxy_class_status"] == "DERIVED_FROM_ECG_PROBABILITY"


def test_gat_idempotent_replay_helper_legacy():
    row = DiagnosisPrediction(
        id=uuid.uuid4(),
        patient_id=uuid.uuid4(),
        hospital_id=uuid.uuid4(),
        model_version_id=uuid.uuid4(),
        target_definition="t",
        prediction_type="proxy_evaluation",
        provenance="legacy",
        confidence=0.9,
        predicted_class_index=None,
        status="evaluated",
    )
    resp = GATFusionService._to_response(row, result_architecture_id=None)
    assert resp.predicted_class_index == -1
    assert "Legacy" in resp.predicted_class_label
    assert resp.confidence_uncalibrated == 0.9
