"""Verify patient portal HTTP responses (JWT from DB, no secrets printed)."""

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
from backend.app.models.identity import User  # noqa: E402
from backend.app.models.patient import Patient  # noqa: E402

PATIENT_EMAIL = "patient_alice@example.com"
BASE_URL = "http://127.0.0.1:8000"


async def main() -> int:
    assert_postgres_configuration()
    if not (await verify_async_connection()).ok:
        print("PostgreSQL unavailable.")
        return 1

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        user = (
            await session.execute(
                select(User).where(User.email == PATIENT_EMAIL).options(selectinload(User.roles))
            )
        ).scalar_one_or_none()
        if user is None:
            print(f"Run seed first: missing {PATIENT_EMAIL}")
            return 1
        patient = (
            await session.execute(select(Patient).where(Patient.user_id == user.id))
        ).scalar_one_or_none()
        print(f"patient_profile_linked: {patient is not None}")

    await engine.dispose()

    token, _ = create_access_token(user_id=str(user.id))
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=15.0) as client:
            for path in (
                "/api/v1/patient/overview",
                "/api/v1/patient/records",
                "/api/v1/patient/reports",
                "/api/v1/patient/diagnosis",
            ):
                res = await client.get(path, headers=headers)
                print(f"GET {path} -> HTTP {res.status_code}")
                if path.endswith("/overview") and res.status_code == 200:
                    body = res.json()
                    summary = body.get("summary", {})
                    print(
                        f"  records_count={summary.get('records_count')} "
                        f"reports_count={summary.get('reports_count')} "
                        f"health_status={summary.get('health_status')}"
                    )
                    print(f"  assessment.status={body.get('assessment', {}).get('status')}")
        return 0
    except httpx.ConnectError:
        print(f"API not reachable at {BASE_URL}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
