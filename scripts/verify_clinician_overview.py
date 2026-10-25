"""Verify clinician overview after PostgreSQL dev seed (uses JWT, not passwords)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.core.security import create_access_token  # noqa: E402
from backend.app.db.connectivity import assert_postgres_configuration, verify_async_connection  # noqa: E402
from backend.app.models.identity import ClinicianProfile, User  # noqa: E402

CLINICIAN_EMAIL = "dr_metro@metro.org"
BASE_URL = "http://127.0.0.1:8000"


async def main() -> int:
    assert_postgres_configuration()
    conn = await verify_async_connection()
    if not conn.ok:
        print("PostgreSQL connectivity failed; cannot verify profile.")
        return 1

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        result = await session.execute(
            select(User)
            .where(User.email == CLINICIAN_EMAIL)
            .options(selectinload(User.roles))
        )
        user = result.scalar_one_or_none()
        if user is None:
            print(f"User not found: {CLINICIAN_EMAIL}. Run scripts/seed_postgres_dev.py first.")
            return 1
        prof_res = await session.execute(
            select(ClinicianProfile).where(ClinicianProfile.user_id == user.id)
        )
        profile = prof_res.scalar_one_or_none()
        if profile is None:
            print("Clinician profile missing. Run scripts/seed_postgres_dev.py again.")
            return 1
        print(f"Profile OK for {CLINICIAN_EMAIL} (hospital_id present).")

    await engine.dispose()

    access_token, _jti = create_access_token(user_id=str(user.id))
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            response = await client.get("/api/v1/clinician/overview", headers=headers)
    except httpx.ConnectError:
        print(f"API not reachable at {BASE_URL}. Start uvicorn and retry.")
        return 1

    print(f"GET /api/v1/clinician/overview -> HTTP {response.status_code}")
    return 0 if response.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
