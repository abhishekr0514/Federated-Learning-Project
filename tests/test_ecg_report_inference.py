"""Test suite for the two-modality ECG + clinical report inference service and API."""

from __future__ import annotations

import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import (
    Base,
    ClinicianProfile,
    Hospital,
    HospitalMembership,
    Patient,
    PatientClinicianAssignment,
    Role,
    User,
)
from ml.inference.ecg_report_runtime import (
    ECGReportInferenceRuntime,
    get_ecg_report_runtime,
)
from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding
from ml.preprocessing.ecg_normalization import ECGNormalizationStats

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_ecg_report_test?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
async def setup_test_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client():
    return TestClient(app)


def make_valid_ecg() -> list[list[float]]:
    # (12, 1000) small amplitude normal ECG signal
    rng = np.random.default_rng(42)
    sig = rng.normal(0.0, 0.15, size=(12, 1000)).astype(np.float32)
    return sig.tolist()


# =========================================================================
# 1. ECG Signal and Preprocessing Tests
# =========================================================================

def test_normalization_stats_loading():
    stats = ECGNormalizationStats.load("models/ecg_report_fusion/normalization.json")
    assert stats.mean.shape == (12,)
    assert stats.std.shape == (12,)
    assert (stats.std > 0).all()

    arr = np.zeros((12, 1000), dtype=np.float32)
    normed = stats.transform(arr)
    assert normed.shape == (12, 1000)
    assert np.isfinite(normed).all()


def test_normalization_rejects_wrong_shape():
    stats = ECGNormalizationStats.load("models/ecg_report_fusion/normalization.json")
    with pytest.raises(ValueError, match=r"Expected \(12, 1000\)"):
        stats.transform(np.zeros((10, 1000)))


def test_runtime_rejects_wrong_ecg_shape():
    rt = get_ecg_report_runtime("models/ecg_report_fusion")
    with pytest.raises(ValueError, match=r"ECG array must have shape \(12, 1000\)"):
        rt.run_inference(np.zeros((8, 1000)), "Normal sinus rhythm")


def test_runtime_rejects_nan_inf_ecg():
    rt = get_ecg_report_runtime("models/ecg_report_fusion")
    bad_ecg = np.zeros((12, 1000), dtype=np.float32)
    bad_ecg[0, 50] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        rt.run_inference(bad_ecg, "Normal sinus rhythm")

    bad_ecg[0, 50] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        rt.run_inference(bad_ecg, "Normal sinus rhythm")


def test_runtime_rejects_empty_clinical_text():
    rt = get_ecg_report_runtime("models/ecg_report_fusion")
    sig = make_valid_ecg()
    with pytest.raises(ValueError, match="non-empty"):
        rt.run_inference(sig, "")
    with pytest.raises(ValueError, match="non-empty"):
        rt.run_inference(sig, "   \n\t  ")


def test_privacy_masking_applied():
    raw_text = "Patient John Doe with email jdoe@hospital.org presented with chest tightness."
    masked = mask_clinical_text_for_encoding(raw_text)
    assert "jdoe@hospital.org" not in masked
    assert "chest tightness" in masked


# =========================================================================
# 2. Forward Pass and Runtime Tests
# =========================================================================

def test_runtime_readiness_and_forward_pass():
    rt = get_ecg_report_runtime("models/ecg_report_fusion")
    readiness = rt.assess_readiness()
    assert readiness.ready is True
    assert readiness.manifest_present is True
    assert readiness.checkpoint_present is True
    assert readiness.checkpoint_trained is True
    assert readiness.normalization_present is True
    assert readiness.architecture_id == "medshield.ecg_report_fusion.v1"

    sig = make_valid_ecg()
    result = rt.run_inference(sig, "Normal sinus rhythm. Normal ECG.")
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_class_index"] in (0, 1)
    assert result["predicted_class_label"] in ("NORM", "MI/STTC")
    assert result["architecture_id"] == "medshield.ecg_report_fusion.v1"
    assert result["is_research_only"] is True
    assert "EXPERIMENTAL RESEARCH" in result["disclaimer"]


# =========================================================================
# 3. API Integration Tests (FastAPI Client)
# =========================================================================

def test_api_readiness_endpoint(client):
    res = client.get("/api/v1/ecg-report/readiness")
    assert res.status_code == 200
    data = res.json()
    assert data["ready"] is True
    assert data["architecture_id"] == "medshield.ecg_report_fusion.v1"
    assert data["manifest_present"] is True
    assert data["checkpoint_present"] is True
    assert data["normalization_present"] is True
    assert "blockers" in data
    # Safe response: no absolute filesystem paths
    assert "C:\\" not in str(data)


def test_api_evaluate_unauthenticated(client):
    payload = {
        "ecg_leads": make_valid_ecg(),
        "report_text": "Normal sinus rhythm.",
    }
    res = client.post("/api/v1/ecg-report/evaluate", json=payload)
    assert res.status_code == 401


@pytest.mark.anyio
async def test_api_evaluate_invalid_shape():
    async with TestingSessionLocal() as session:
        clinician_role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(clinician_role)
        user = User(
            id=uuid.uuid4(),
            email="dr.invalid@hospital.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[clinician_role],
        )
        session.add(user)
        await session.commit()
        token, _ = create_access_token(str(user.id))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "ecg_leads": [[0.0] * 500 for _ in range(12)],  # 500 instead of 1000
            "report_text": "Normal sinus rhythm.",
        }
        res = await ac.post("/api/v1/ecg-report/evaluate", json=payload, headers=headers)
        assert res.status_code == 422


@pytest.mark.anyio
async def test_api_evaluate_authenticated_standalone():
    # Setup test clinician
    async with TestingSessionLocal() as session:
        clinician_role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(clinician_role)
        hosp = Hospital(id=uuid.uuid4(), name="St Jude", code="STJ01")
        session.add(hosp)
        user = User(
            id=uuid.uuid4(),
            email="dr.smith@hospital.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[clinician_role],
        )
        session.add(user)
        await session.commit()

        token, _ = create_access_token(str(user.id))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "ecg_leads": make_valid_ecg(),
            "report_text": "Sinus bradycardia, otherwise normal ECG tracing.",
        }
        res = await ac.post("/api/v1/ecg-report/evaluate", json=payload, headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "EVALUATED"
        assert 0.0 <= body["probability"] <= 1.0
        assert body["predicted_class_index"] in (0, 1)
        assert body["predicted_class_label"] in ("NORM", "MI/STTC")
        assert body["architecture_id"] == "medshield.ecg_report_fusion.v1"
        assert body["is_research_only"] is True
        assert "EXPERIMENTAL RESEARCH" in body["disclaimer"]
        assert body["patient_id"] is None
        # Verify no raw report text leaked in response
        assert "dr.smith" not in str(body)


@pytest.mark.anyio
async def test_api_evaluate_with_assigned_patient():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        hosp = Hospital(id=uuid.uuid4(), name="General Hosp", code="GH01")
        session.add(hosp)
        user = User(
            id=uuid.uuid4(),
            email="dr.assigned@hosp.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.flush()

        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            hospital_id=hosp.id,
            license_number_hash="LIC-12345",
        )
        session.add(profile)

        pat_user = User(
            id=uuid.uuid4(),
            email="patient.a@hosp.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
        )
        session.add(pat_user)
        await session.flush()

        patient = Patient(
            id=uuid.uuid4(),
            user_id=pat_user.id,
            opaque_patient_id="OPAQUE-PAT-1001",
        )
        session.add(patient)
        await session.flush()

        assign = PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=patient.id,
            clinician_id=profile.id,
            hospital_id=hosp.id,
        )
        session.add(assign)
        await session.commit()

        token, _ = create_access_token(str(user.id))
        patient_id = patient.id

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "ecg_leads": make_valid_ecg(),
            "report_text": "Normal sinus rhythm, unconfirmed report.",
            "patient_id": str(patient_id),
            "idempotency_key": "test-key-12345",
        }
        res = await ac.post(
            f"/api/v1/ecg-report/evaluate?patient_id={patient_id}",
            json=payload,
            headers=headers,
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "EVALUATED"
        assert body["patient_id"] == str(patient_id)
        assert 0.0 <= body["probability"] <= 1.0

        # Test idempotency replay
        res_replay = await ac.post(
            f"/api/v1/ecg-report/evaluate?patient_id={patient_id}",
            json=payload,
            headers=headers,
        )
        assert res_replay.status_code == 200
        replay_body = res_replay.json()
        assert replay_body["prediction_id"] == body["prediction_id"]
