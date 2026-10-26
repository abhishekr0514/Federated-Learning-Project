"""Patient questions API: empty list, auth, and isolation."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_db
from backend.app.core.security import create_access_token, hash_password
from backend.app.main import app
from backend.app.models import (
    Base,
    Hospital,
    Patient,
    PatientHospitalAssociation,
    Role,
    User,
)

TEST_DB_URL = "sqlite+aiosqlite:///file:memdb_patient_questions?mode=memory&cache=shared&uri=true"
async_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


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


async def _seed_patient(*, email: str = "pat_q@test.org") -> tuple[str, uuid.UUID]:
    async with TestingSessionLocal() as session:
        hospital = Hospital(
            id=uuid.uuid4(),
            name=f"Test Hospital {uuid.uuid4().hex[:6]}",
            code=f"TST-{uuid.uuid4().hex[:4]}",
            region="test",
        )
        session.add(hospital)
        role = (await session.execute(select(Role).where(Role.name == "PATIENT"))).scalar_one_or_none()
        if role is None:
            role = Role(id=uuid.uuid4(), name="PATIENT", description="PATIENT")
            session.add(role)
            await session.flush()
        user = User(
            id=uuid.uuid4(),
            email=email,
            hashed_password=hash_password("testpass123"),
            is_active=True,
            roles=[role],
        )
        session.add(user)
        patient = Patient(
            id=uuid.uuid4(),
            user_id=user.id,
            opaque_patient_id=f"opaque-{uuid.uuid4().hex[:8]}",
        )
        session.add(patient)
        session.add(
            PatientHospitalAssociation(
                id=uuid.uuid4(),
                patient_id=patient.id,
                hospital_id=hospital.id,
                hospital_patient_ref_hash="ref-hash",
            )
        )
        await session.commit()
        token, _jti = create_access_token(str(user.id))
        return token, patient.id


@pytest.mark.anyio
async def test_list_questions_empty_returns_200_no_data():
    token, _ = await _seed_patient()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "NO_DATA"
    assert body["questions"] == []
    assert len(body["suggested_templates"]) >= 1
    assert body["emergency_notice"]


@pytest.mark.anyio
async def test_list_questions_requires_patient_role():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/patient/questions")
    assert res.status_code in (401, 403)


@pytest.mark.anyio
async def test_list_questions_patient_isolation():
    token_a, _ = await _seed_patient(email="pat_a_q@test.org")
    token_b, _ = await _seed_patient(email="pat_b_q@test.org")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        submit = await ac.post(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"question_text": "Question visible only to patient A"},
        )
        assert submit.status_code == 200
        list_b = await ac.get(
            "/api/v1/patient/questions",
            headers={"Authorization": f"Bearer {token_b}"},
        )
    assert list_b.status_code == 200
    assert list_b.json()["questions"] == []
