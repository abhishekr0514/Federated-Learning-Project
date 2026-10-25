"""Verify clinician patient assignment flow against local API (JWT from DB, no password print)."""

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
from backend.app.models.patient import Patient  # noqa: E402

CLINICIAN_EMAIL = "dr_metro@metro.org"
PATIENT_EMAIL = "patient_alice@example.com"
BASE_URL = "http://127.0.0.1:8000"


async def main() -> int:
    assert_postgres_configuration()
    conn = await verify_async_connection()
    if not conn.ok:
        print("PostgreSQL connectivity failed; cannot verify assignment flow.")
        return 1

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        clin_res = await session.execute(
            select(User)
            .where(User.email == CLINICIAN_EMAIL)
            .options(selectinload(User.roles))
        )
        clinician = clin_res.scalar_one_or_none()
        if clinician is None:
            print(f"Clinician not found: {CLINICIAN_EMAIL}. Run scripts/seed_postgres_dev.py first.")
            return 1

        prof_res = await session.execute(
            select(ClinicianProfile).where(ClinicianProfile.user_id == clinician.id)
        )
        if prof_res.scalar_one_or_none() is None:
            print("Clinician profile missing. Re-run scripts/seed_postgres_dev.py.")
            return 1

        pat_user_res = await session.execute(select(User).where(User.email == PATIENT_EMAIL))
        pat_user = pat_user_res.scalar_one_or_none()
        if pat_user is None:
            print(f"Patient user not found: {PATIENT_EMAIL}. Run scripts/seed_postgres_dev.py first.")
            return 1

        patient_res = await session.execute(select(Patient).where(Patient.user_id == pat_user.id))
        patient = patient_res.scalar_one_or_none()
        if patient is None:
            print("Patient portal link missing. Re-run scripts/seed_postgres_dev.py.")
            return 1

        patient_id = str(patient.id)

    await engine.dispose()

    access_token, _jti = create_access_token(user_id=str(clinician.id))
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=15.0) as client:
            eligible = await client.get("/api/v1/clinician/patients/eligible", headers=headers)
            print(f"GET /api/v1/clinician/patients/eligible -> HTTP {eligible.status_code}")

            listed = await client.get("/api/v1/clinician/patients", headers=headers)
            print(f"GET /api/v1/clinician/patients -> HTTP {listed.status_code}")

            if listed.status_code == 200 and listed.json().get("total_count", 0) == 0:
                assign = await client.post(
                    f"/api/v1/clinician/patients/{patient_id}/assignment",
                    headers=headers,
                )
                print(f"POST /api/v1/clinician/patients/{{id}}/assignment -> HTTP {assign.status_code}")
                if assign.status_code not in (200, 409):
                    return 1

            listed_after = await client.get("/api/v1/clinician/patients", headers=headers)
            print(f"GET /api/v1/clinician/patients (after assign) -> HTTP {listed_after.status_code}")
            if listed_after.status_code != 200:
                return 1
            count = listed_after.json().get("total_count", 0)
            print(f"Assigned patient count: {count}")
            return 0 if count >= 1 else 1
    except httpx.ConnectError:
        print(f"API not reachable at {BASE_URL}. Start uvicorn and retry.")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
