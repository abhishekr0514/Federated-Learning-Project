"""Demo clinician hospital alignment after postgres dev + synthetic seed interaction."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path
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
    Hospital,
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
    Role,
    User,
)
from backend.app.services.clinician_service import ClinicianService

_SEED_PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_synthetic.py"
_spec = importlib.util.spec_from_file_location("seed_demo_synthetic", _SEED_PATH)
assert _spec and _spec.loader
seed_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_mod)

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_demo_align?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)

DEMO_PATIENT_UUID = uuid.UUID("a1b2c3d4-0005-4000-8000-000000000001")
DEMO_HOSPITAL_UUID = uuid.UUID("a1b2c3d4-0001-4000-8000-000000000001")

_MOCK_INFERENCE = {
    "target_definition": "PTB-XL proxy",
    "model_version_tag": "gat-align-test",
    "confidence_uncalibrated": 0.51,
    "predicted_class_index": 1,
    "predicted_class_label": "MI/STTC",
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
async def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    app.dependency_overrides.pop(get_db, None)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_ensure_clinician_profile_realigns_from_legacy_hosp_metro():
    """Simulates seed_postgres_dev profile on HOSP-METRO, then synthetic seed realigns to DEMO-METRO."""
    metro_def = seed_mod.DEMO_HOSPITALS[0]
    clin_def = seed_mod.DEMO_CLINICIANS[0]

    async with TestingSessionLocal() as session:
        legacy_hosp = Hospital(id=uuid.uuid4(), code="HOSP-METRO", name="Metro General Hospital")
        demo_hosp = Hospital(
            id=metro_def["stable_uuid"],
            code=metro_def["code"],
            name=metro_def["name"],
        )
        session.add_all([legacy_hosp, demo_hosp])
        user = User(
            id=clin_def["user_stable_uuid"],
            email=clin_def["email"],
            hashed_password=hash_password("Password123!"),
            is_active=True,
        )
        session.add(user)
        await session.flush()
        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            hospital_id=legacy_hosp.id,
            license_number_hash="DEV_HASH_LIC_METRO_001",
        )
        session.add(profile)
        await session.commit()

        async with session.begin():
            realigned = await seed_mod._ensure_clinician_profile(
                session, clin_def, user, demo_hosp
            )
        assert realigned.hospital_id == demo_hosp.id
        assert realigned.license_number_hash == clin_def["license_hash"]


@pytest.mark.anyio
async def test_metro_clinician_sees_demo_patient_and_gat_scope():
    async with TestingSessionLocal() as session:
        role = Role(id=uuid.uuid4(), name="CLINICIAN", description="Clinician")
        session.add(role)
        demo_hosp = Hospital(
            id=DEMO_HOSPITAL_UUID,
            code="DEMO-METRO",
            name="[SYNTHETIC] Demo Metro Medical Center",
        )
        other_hosp = Hospital(id=uuid.uuid4(), code="HOSP-OTHER", name="Other")
        session.add_all([demo_hosp, other_hosp])

        user = User(
            id=uuid.uuid4(),
            email="dr_metro@metro.org",
            hashed_password=hash_password("Password123!"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        await session.flush()

        profile = ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            hospital_id=demo_hosp.id,
            license_number_hash="DEMO-LIC-METRO-001",
        )
        session.add(profile)

        demo_patient = Patient(
            id=DEMO_PATIENT_UUID,
            opaque_patient_id="DEMO-SYNTH-PATIENT-0001",
        )
        other_patient = Patient(id=uuid.uuid4(), opaque_patient_id="OTHER-PATIENT")
        session.add_all([demo_patient, other_patient])
        await session.flush()

        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=demo_patient.id,
                hospital_id=demo_hosp.id,
                hospital_patient_ref_hash="ref-demo-1",
            )
        )
        session.add(
            PatientClinicianAssignment(
                id=uuid.uuid4(),
                patient_id=demo_patient.id,
                clinician_id=profile.id,
                hospital_id=demo_hosp.id,
            )
        )
        await session.commit()

        listing = await ClinicianService.get_assigned_patients(session, profile)
        ids = {p["patient_id"] for p in listing["patients"]}
        assert str(DEMO_PATIENT_UUID) in ids
        assert str(other_patient.id) not in ids

        token, _ = create_access_token(str(user.id))

    with patch("backend.app.services.gat_fusion_service.settings.FUSION_INFERENCE_ENABLED", True):
        rt = MagicMock()
        ready = MagicMock()
        ready.ready = True
        ready.to_safe_dict.return_value = {}
        rt.assess_readiness.return_value = ready
        rt.run_proxy_inference.return_value = dict(_MOCK_INFERENCE)
        with patch("backend.app.services.gat_fusion_service._get_runtime", return_value=rt):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                ok = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={DEMO_PATIENT_UUID}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": [0.01] * 128,
                        "text_embedding": [0.02] * 128,
                    },
                )
                blocked = await ac.post(
                    f"/api/v1/gat-fusion/evaluate?patient_id={other_patient.id}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "ecg_embedding": [0.01] * 128,
                        "text_embedding": [0.02] * 128,
                    },
                )
    assert ok.status_code == 200
    assert blocked.status_code == 404
