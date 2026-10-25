"""Clinician profile guard (get_current_clinician_context) regression tests."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.api.dependencies import get_current_clinician_context
from backend.app.core.errors import ForbiddenError
from backend.app.core.security import UserRole, hash_password
from backend.app.models import Base
from backend.app.models.identity import ClinicianProfile, Hospital, Role, User

MEM_URL = "sqlite+aiosqlite:///file:memdb_clin_profile?mode=memory&cache=shared&uri=true"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_session():
    engine = create_async_engine(MEM_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


async def _role(session: AsyncSession, name: str) -> Role:
    role = Role(id=uuid.uuid4(), name=name, description=name)
    session.add(role)
    await session.flush()
    return role


@pytest.mark.anyio
async def test_clinician_without_profile_forbidden(db_session: AsyncSession):
    role = await _role(db_session, UserRole.CLINICIAN.value)
    user = User(
        id=uuid.uuid4(),
        email="noclin@metro.org",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    user.roles.append(role)
    db_session.add(user)
    await db_session.commit()

    async def _current_user():
        return user

    with pytest.raises(ForbiddenError, match="Clinician Profile"):
        await get_current_clinician_context(db=db_session, current_user=user)


@pytest.mark.anyio
async def test_clinician_with_profile_passes_guard(db_session: AsyncSession):
    role = await _role(db_session, UserRole.CLINICIAN.value)
    hospital = Hospital(id=uuid.uuid4(), code="HOSP-T", name="Test Hospital", is_active=True)
    user = User(
        id=uuid.uuid4(),
        email="withprof@metro.org",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    user.roles.append(role)
    profile = ClinicianProfile(
        id=uuid.uuid4(),
        user_id=user.id,
        hospital_id=hospital.id,
        department="Cardiology",
    )
    db_session.add_all([hospital, user, profile])
    await db_session.commit()

    out_user, out_profile = await get_current_clinician_context(db=db_session, current_user=user)
    assert out_user.id == user.id
    assert out_profile is not None
    assert out_profile.user_id == user.id


@pytest.mark.anyio
async def test_clinician_profile_unique_per_user(db_session: AsyncSession):
    """Schema: one profile row per user (unique user_id on clinician_profiles)."""
    hospital = Hospital(id=uuid.uuid4(), code="HOSP-U", name="Unique Hospital", is_active=True)
    user = User(
        id=uuid.uuid4(),
        email="unique@metro.org",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    db_session.add_all([hospital, user])
    await db_session.flush()
    db_session.add(
        ClinicianProfile(
            id=uuid.uuid4(),
            user_id=user.id,
            hospital_id=hospital.id,
        )
    )
    await db_session.commit()
    count = await db_session.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == user.id)
    )
    assert len(count.scalars().all()) == 1
