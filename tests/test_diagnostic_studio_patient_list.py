"""Diagnostic Studio patient list ordering, labels, and UUID submission regression."""

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
    DiagnosisPrediction,
    Hospital,
    Patient,
    PatientHospitalAssociation,
    Role,
    User,
)
from backend.app.utils.patient_reference_display import DEMO_SYNTHETIC_OPAQUE_PREFIX

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_diag_studio_pts?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

DEMO_PATIENT_ID = uuid.UUID("a1b2c3d4-0005-4000-8000-000000000001")
DEMO_OPAQUE = "DEMO-SYNTH-PATIENT-0001"
HASH_OPAQUE = "c89a71239bf049b1a520fa981e7d34107298319f030f81d86d5268c07e594d21"
HASH_PATIENT_ID = uuid.UUID("40004d41-e306-459d-abb6-c76dcec54add")

_MOCK_INFERENCE = {
    "target_definition": "PTB-XL MI/STTC vs NORM (research proxy)",
    "prediction_type": "proxy_evaluation",
    "model_version_tag": "gat-test-v1",
    "confidence_uncalibrated": 0.55,
    "predicted_class_index": 0,
    "predicted_class_label": "NORM",
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
async def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    app.dependency_overrides.pop(get_db, None)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def super_admin_and_patients():
    async with TestingSessionLocal() as session:
        role_admin = Role(id=uuid.uuid4(), name="SUPER_ADMIN", description="Super Admin")
        session.add(role_admin)
        hosp = Hospital(id=uuid.uuid4(), name="Demo Metro", code="DEMO-METRO")
        session.add(hosp)

        admin = User(
            id=uuid.uuid4(),
            email="super.admin@demo.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            is_super_admin=True,
            roles=[role_admin],
        )
        session.add(admin)
        await session.flush()

        hash_patient = Patient(id=HASH_PATIENT_ID, opaque_patient_id=HASH_OPAQUE)
        demo_patient = Patient(id=DEMO_PATIENT_ID, opaque_patient_id=DEMO_OPAQUE)
        session.add_all([hash_patient, demo_patient])
        await session.flush()

        for pat in (hash_patient, demo_patient):
            session.add(
                PatientHospitalAssociation(
                    id=uuid.uuid4(),
                    patient_id=pat.id,
                    hospital_id=hosp.id,
                    hospital_patient_ref_hash=f"ref-{pat.opaque_patient_id[:8]}",
                )
            )
        await session.commit()
        token, _ = create_access_token(str(admin.id))

    return {"super_admin_token": token}


def _emb() -> list[float]:
    return [0.01 * (i % 11) for i in range(128)]


def _mock_runtime_ready():
    runtime = MagicMock()
    readiness = MagicMock()
    readiness.ready = True
    readiness.to_safe_dict.return_value = {}
    runtime.assess_readiness.return_value = readiness
    runtime.run_proxy_inference.return_value = dict(_MOCK_INFERENCE)
    return runtime


@pytest.mark.anyio
async def test_super_admin_lists_demo_synthetic_first_with_stable_uuid(super_admin_and_patients):
    token = super_admin_and_patients["super_admin_token"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get(
            "/api/v1/clinician/patients?limit=50&offset=0",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert res.status_code == 200
    patients = res.json()["patients"]
    assert len(patients) >= 2
    assert patients[0]["opaque_patient_id"] == DEMO_OPAQUE
    assert patients[0]["patient_id"] == str(DEMO_PATIENT_ID)
    assert patients[0]["is_demo_synthetic"] is True
    assert DEMO_OPAQUE in patients[0]["display_label"]
    assert DEMO_SYNTHETIC_OPAQUE_PREFIX in patients[0]["opaque_patient_id"]

    by_opaque = {p["opaque_patient_id"]: p for p in patients}
    assert by_opaque[DEMO_OPAQUE]["patient_id"] == str(DEMO_PATIENT_ID)
    assert by_opaque[HASH_OPAQUE]["patient_id"] == str(HASH_PATIENT_ID)
    assert "…" in by_opaque[HASH_OPAQUE]["display_label"] or len(HASH_OPAQUE) > 36


@pytest.mark.anyio
async def test_gat_evaluate_using_demo_synthetic_list_entry_persists_stable_uuid(
    super_admin_and_patients,
):
    """Selecting DEMO-SYNTH-PATIENT-0001 must submit patients.id a1b2c3d4-…0001."""
    token = super_admin_and_patients["super_admin_token"]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        listing = await ac.get(
            "/api/v1/clinician/patients?limit=50&offset=0",
            headers={"Authorization": f"Bearer {token}"},
        )
        demo_row = next(
            p for p in listing.json()["patients"] if p["opaque_patient_id"] == DEMO_OPAQUE
        )
        submitted_patient_id = demo_row["patient_id"]
        assert submitted_patient_id == str(DEMO_PATIENT_ID)

        with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
            with patch(
                "backend.app.services.gat_fusion_service._get_runtime",
                return_value=_mock_runtime_ready(),
            ):
                eval_res = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={submitted_patient_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": _emb(),
                        "text_embedding": _emb(),
                        "idempotency_key": "demo-synth-0001-regression",
                    },
                )
        assert eval_res.status_code == 200
        prediction_id = eval_res.json()["prediction_id"]

        async with TestingSessionLocal() as session:
            row = (
                await session.execute(
                    select(DiagnosisPrediction).where(
                        DiagnosisPrediction.id == uuid.UUID(prediction_id)
                    )
                )
            ).scalar_one()
            assert row.patient_id == DEMO_PATIENT_ID
            assert row.patient_id != HASH_PATIENT_ID
