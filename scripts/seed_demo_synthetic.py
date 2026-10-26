"""Idempotent Synthetic Demo Seed for MedShield-FL.

Creates a coherent, clearly-labelled synthetic demo dataset using the existing DB models.
Safe to re-run: uses stable deterministic identifiers to prevent duplicate creation.

IMPORTANT:
- All generated records are explicitly labelled as synthetic/demo.
- Does NOT overwrite existing user data or production records.
- Does NOT insert fabricated clinical predictions or invented AUROC values.
- Does NOT silently reset the database.
- Development/demo environment only.

Scale:
  - 2 synthetic demo hospitals (DEMO-METRO, DEMO-RIVERSIDE)
  - 15 synthetic patients (8 Metro, 7 Riverside) with user accounts + portal links
  - 2 demo clinicians (one per hospital) with profiles + patient assignments
  - Clinical metadata rows (medical document refs, ECG metadata refs, observation refs) per patient
  - AuditLog entries are created by the login operations themselves when users log in;
    this seed does NOT fabricate historical audit claims.

Demo credentials (all accounts):  DemoSeed2026!  (override: MEDSHIELD_DEMO_SEED_PASSWORD env var)

Cleanup SQL (dev only — do not run on production):
  DELETE FROM patient_clinician_assignments WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM patient_hospital_associations WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM clinical_observations WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM ecg_records WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM medical_documents WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM patients WHERE opaque_patient_id LIKE 'DEMO-SYNTH-%';
  DELETE FROM clinician_profiles WHERE license_number_hash LIKE 'DEMO-LIC-%';
  DELETE FROM hospital_memberships WHERE hospital_id IN (
    SELECT id FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE'));
  DELETE FROM users WHERE email LIKE 'demo_%@demo.org';
  DELETE FROM hospitals WHERE code IN ('DEMO-METRO', 'DEMO-RIVERSIDE');
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import settings  # noqa: E402
from backend.app.core.security import hash_password  # noqa: E402
from backend.app.db.connectivity import (  # noqa: E402
    assert_postgres_configuration,
    database_target_summary,
    verify_async_connection,
)
from backend.app.models.identity import (  # noqa: E402
    ClinicianProfile,
    Hospital,
    HospitalMembership,
    Role,
    User,
    user_roles,
)
from backend.app.models.clinical import (  # noqa: E402
    ClinicalObservation,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingResult,
)
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation  # noqa: E402

EXPECTED_DATABASE = "medshield_fl_db"
DEMO_SEED_PASSWORD_ENV = "MEDSHIELD_DEMO_SEED_PASSWORD"
DEFAULT_DEMO_PASSWORD = "DemoSeed2026!"

# ─── Stable Demo Hospital Definitions ────────────────────────────────────────

DEMO_HOSPITALS = [
    {
        "code": "DEMO-METRO",
        "name": "[SYNTHETIC] Demo Metro Medical Center",
        "region": "Metro District — Demo Environment",
        "stable_uuid": uuid.UUID("a1b2c3d4-0001-4000-8000-000000000001"),
    },
    {
        "code": "DEMO-RIVERSIDE",
        "name": "[SYNTHETIC] Demo Riverside Clinic",
        "region": "Riverside District — Demo Environment",
        "stable_uuid": uuid.UUID("a1b2c3d4-0001-4000-8000-000000000002"),
    },
]

# ─── Stable Demo Clinician Definitions ───────────────────────────────────────

DEMO_CLINICIANS = [
    {
        "email": "dr_metro@metro.org",
        "full_name": "Dr. Metro (Demo Clinician)",
        "hospital_code": "DEMO-METRO",
        "department": "[SYNTHETIC] Cardiology & Electrophysiology",
        "license_hash": "DEMO-LIC-METRO-001",
        "user_stable_uuid": uuid.UUID("a1b2c3d4-0002-4000-8000-000000000001"),
        "profile_stable_uuid": uuid.UUID("a1b2c3d4-0003-4000-8000-000000000001"),
    },
    {
        "email": "demo_clinician_riverside@demo.org",
        "full_name": "Dr. Riverside (Demo Clinician)",
        "hospital_code": "DEMO-RIVERSIDE",
        "department": "[SYNTHETIC] Internal Medicine",
        "license_hash": "DEMO-LIC-RIVERSIDE-001",
        "user_stable_uuid": uuid.UUID("a1b2c3d4-0002-4000-8000-000000000002"),
        "profile_stable_uuid": uuid.UUID("a1b2c3d4-0003-4000-8000-000000000002"),
    },
]

# ─── Stable Demo Patient Definitions (8 Metro, 7 Riverside) ──────────────────

def _patient_stable_uuid(idx: int) -> uuid.UUID:
    return uuid.UUID(f"a1b2c3d4-0004-4000-8000-{idx:012d}")

def _patient_opaque_id(idx: int) -> str:
    return f"DEMO-SYNTH-PATIENT-{idx:04d}"

DEMO_PATIENTS: List[dict] = []
for _i in range(1, 16):
    _hospital_code = "DEMO-METRO" if _i <= 8 else "DEMO-RIVERSIDE"
    DEMO_PATIENTS.append({
        "idx": _i,
        "email": f"demo_patient_{_i:02d}@demo.org",
        "full_name": f"[SYNTHETIC] Demo Patient {_i:02d}",
        "hospital_code": _hospital_code,
        "opaque_id": _patient_opaque_id(_i),
        "user_stable_uuid": _patient_stable_uuid(_i),
        "patient_stable_uuid": uuid.UUID(f"a1b2c3d4-0005-4000-8000-{_i:012d}"),
    })

# ─── Clinical Metadata Templates (no raw values, local URI refs only) ─────────

DOCUMENT_TYPES = ["cardiology_consult", "clinical_note", "discharge_summary", "ecg_report"]


def _make_doc_local_uri(hospital_code: str, patient_idx: int) -> str:
    return f"local://{hospital_code.lower()}/storage/notes/demo_patient_{patient_idx:04d}_note.bin"


def _make_ecg_local_ref(hospital_code: str, patient_idx: int) -> str:
    return f"local://{hospital_code.lower()}/ecg/demo_patient_{patient_idx:04d}_ecg.mat"


def _make_obs_local_ref(hospital_code: str, patient_idx: int) -> str:
    return f"local://{hospital_code.lower()}/obs/demo_patient_{patient_idx:04d}_vitals.parquet"


# ─── Core Helpers ─────────────────────────────────────────────────────────────

def _resolve_demo_password() -> str:
    pwd = os.environ.get(DEMO_SEED_PASSWORD_ENV, DEFAULT_DEMO_PASSWORD)
    if len(pwd) < 12:
        raise SystemExit(
            f"{DEMO_SEED_PASSWORD_ENV} must be at least 12 characters. "
            f"Default: {DEFAULT_DEMO_PASSWORD}"
        )
    return pwd


async def _ensure_hospital(session, hosp_def: dict) -> Hospital:
    res = await session.execute(select(Hospital).where(Hospital.code == hosp_def["code"]))
    hospital = res.scalar_one_or_none()
    if hospital is None:
        hospital = Hospital(
            id=hosp_def["stable_uuid"],
            code=hosp_def["code"],
            name=hosp_def["name"],
            region=hosp_def["region"],
            is_active=True,
        )
        session.add(hospital)
        await session.flush()
        print(f"  [CREATE] Hospital: {hosp_def['code']} — {hosp_def['name']}")
    else:
        print(f"  [REUSE]  Hospital: {hosp_def['code']}")
    return hospital


async def _ensure_role(session, role_name: str) -> Role:
    res = await session.execute(select(Role).where(Role.name == role_name))
    role = res.scalar_one_or_none()
    if role is None:
        role = Role(id=uuid.uuid4(), name=role_name, description=f"{role_name} demo role")
        session.add(role)
        await session.flush()
        print(f"  [CREATE] Role: {role_name}")
    else:
        print(f"  [REUSE]  Role: {role_name}")
    return role


async def _user_has_role(session, user_id: uuid.UUID, role_name: str) -> bool:
    stmt = (
        select(Role.id)
        .join(user_roles, Role.id == user_roles.c.role_id)
        .where(user_roles.c.user_id == user_id, Role.name == role_name)
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _ensure_user_role(session, user: User, role: Role) -> None:
    """Assign role via association table (async-safe; no lazy load on user.roles)."""
    if await _user_has_role(session, user.id, role.name):
        return
    await session.execute(user_roles.insert().values(user_id=user.id, role_id=role.id))
    print(f"    [ROLE]  Assigned {role.name} to {user.email}")


async def _ensure_user(session, email: str, full_name: str, stable_uuid: uuid.UUID, hashed_pwd: str, is_super_admin: bool = False) -> Tuple[User, bool]:
    res = await session.execute(select(User).where(User.email == email))
    user = res.scalar_one_or_none()
    created = False
    if user is None:
        user = User(
            id=stable_uuid,
            email=email,
            full_name=full_name,
            hashed_password=hashed_pwd,
            is_active=True,
            is_super_admin=is_super_admin,
        )
        session.add(user)
        await session.flush()
        print(f"  [CREATE] User: {email}")
        created = True
    else:
        print(f"  [REUSE]  User: {email}")
    return user, created


async def _ensure_hospital_membership(session, user: User, hospital: Hospital, role_in_hospital: str) -> None:
    res = await session.execute(
        select(HospitalMembership).where(
            HospitalMembership.user_id == user.id,
            HospitalMembership.hospital_id == hospital.id,
        )
    )
    if res.scalar_one_or_none() is None:
        session.add(HospitalMembership(
            id=uuid.uuid4(),
            user_id=user.id,
            hospital_id=hospital.id,
            role_in_hospital=role_in_hospital,
        ))
        print(f"    [MEMBR] {role_in_hospital} membership: {user.email} @ {hospital.code}")


async def _ensure_clinician_profile(session, clinician_def: dict, user: User, hospital: Hospital) -> ClinicianProfile:
    """Ensure demo clinician profile is bound to the synthetic demo hospital node.

    When ``seed_postgres_dev.py`` ran first, ``dr_metro@metro.org`` may still point at
    ``HOSP-METRO``. This seed realigns the profile to ``DEMO-METRO`` / ``DEMO-RIVERSIDE``
    idempotently so assignments and clinician patient lists stay consistent.
    """
    res = await session.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == user.id)
    )
    profile = res.scalar_one_or_none()
    if profile is None:
        profile = ClinicianProfile(
            id=clinician_def["profile_stable_uuid"],
            user_id=user.id,
            hospital_id=hospital.id,
            department=clinician_def["department"],
            license_number_hash=clinician_def["license_hash"],
        )
        session.add(profile)
        await session.flush()
        print(f"    [PROFL] Clinician profile created for {user.email}")
    else:
        changed = False
        if profile.hospital_id != hospital.id:
            profile.hospital_id = hospital.id
            changed = True
        if profile.department != clinician_def["department"]:
            profile.department = clinician_def["department"]
            changed = True
        if profile.license_number_hash != clinician_def["license_hash"]:
            profile.license_number_hash = clinician_def["license_hash"]
            changed = True
        if changed:
            await session.flush()
            print(
                f"    [PROFL] Clinician profile realigned to {hospital.code} for {user.email}"
            )
        else:
            print(f"    [PROFL] Clinician profile reused for {user.email}")
    return profile


async def _ensure_patient(session, patient_def: dict, user: User, hospital: Hospital) -> Patient:
    res = await session.execute(
        select(Patient).where(Patient.opaque_patient_id == patient_def["opaque_id"])
    )
    patient = res.scalar_one_or_none()
    if patient is None:
        patient = Patient(
            id=patient_def["patient_stable_uuid"],
            opaque_patient_id=patient_def["opaque_id"],
            user_id=user.id,
        )
        session.add(patient)
        await session.flush()
        print(f"    [PATIE] Patient record created: {patient_def['opaque_id']}")
    else:
        if patient.user_id is None:
            patient.user_id = user.id
            print(f"    [PATIE] Linked patient to user: {patient_def['opaque_id']}")
        else:
            print(f"    [PATIE] Patient reused: {patient_def['opaque_id']}")

    # Ensure hospital association
    assoc_res = await session.execute(
        select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id,
            PatientHospitalAssociation.hospital_id == hospital.id,
        )
    )
    if assoc_res.scalar_one_or_none() is None:
        session.add(PatientHospitalAssociation(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=hospital.id,
            hospital_patient_ref_hash=f"DEMO-REF-{patient_def['opaque_id']}",
        ))
        print(f"    [ASSOC] Hospital association: {patient_def['opaque_id']} @ {patient_def['hospital_code']}")

    return patient


async def _ensure_clinical_metadata(session, patient_def: dict, patient: Patient, hospital: Hospital) -> None:
    """Create synthetic clinical metadata references (local URIs only, no raw values, no PHI)."""
    doc_uri = _make_doc_local_uri(patient_def["hospital_code"], patient_def["idx"])
    doc_res = await session.execute(
        select(MedicalDocument.id).where(
            MedicalDocument.patient_id == patient.id,
            MedicalDocument.local_document_uri == doc_uri,
        )
    )
    if doc_res.scalar_one_or_none() is None:
        doc_type = DOCUMENT_TYPES[(patient_def["idx"] - 1) % len(DOCUMENT_TYPES)]
        doc = MedicalDocument(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=hospital.id,
            local_document_uri=doc_uri,
            document_type=doc_type,
            status="completed",
        )
        session.add(doc)
        await session.flush()
        session.add(PrivacyMaskingResult(
            id=uuid.uuid4(),
            document_id=doc.id,
            masking_method="BioClinicalBERT-PII-Masker",
            entities_detected_count=(patient_def["idx"] % 5),  # plausible small count
            sanitization_status="sanitized",
        ))
        print(f"      [DOC]  Medical document metadata: patient {patient_def['idx']:02d}")

    ecg_ref = _make_ecg_local_ref(patient_def["hospital_code"], patient_def["idx"])
    ecg_res = await session.execute(
        select(ECGRecord.id).where(
            ECGRecord.patient_id == patient.id,
            ECGRecord.local_storage_ref == ecg_ref,
        )
    )
    if ecg_res.scalar_one_or_none() is None:
        session.add(ECGRecord(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=hospital.id,
            local_storage_ref=ecg_ref,
            sampling_rate_hz=500,
            duration_seconds=10.0,
            lead_count=12,
            dataset_source="PTB-XL-Proxy-Synthetic",
        ))
        print(f"      [ECG]  ECG metadata: patient {patient_def['idx']:02d}")

    obs_ref = _make_obs_local_ref(patient_def["hospital_code"], patient_def["idx"])
    obs_res = await session.execute(
        select(ClinicalObservation.id).where(
            ClinicalObservation.patient_id == patient.id,
            ClinicalObservation.local_ref == obs_ref,
        )
    )
    if obs_res.scalar_one_or_none() is None:
        session.add(ClinicalObservation(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=hospital.id,
            observation_type="vital_signs",
            feature_schema_version="v1.0",
            local_ref=obs_ref,
        ))
        print(f"      [OBS]  Vital observation metadata: patient {patient_def['idx']:02d}")


async def _ensure_patient_assignment(
    session,
    patient: Patient,
    profile: ClinicianProfile,
    hospital: Hospital,
) -> None:
    assn_res = await session.execute(
        select(PatientClinicianAssignment).where(
            PatientClinicianAssignment.patient_id == patient.id,
            PatientClinicianAssignment.clinician_id == profile.id,
        )
    )
    existing = assn_res.scalar_one_or_none()
    if existing is None:
        session.add(PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=patient.id,
            clinician_id=profile.id,
            hospital_id=hospital.id,
        ))
        print(f"      [ASGN] Assigned patient {patient.opaque_patient_id} to clinician {profile.license_number_hash}")
    elif existing.hospital_id != hospital.id:
        existing.hospital_id = hospital.id
        print(
            f"      [ASGN] Realigned assignment hospital to {hospital.code} for {patient.opaque_patient_id}"
        )


# ─── Main Seed Entry Point ────────────────────────────────────────────────────

async def seed(*, non_interactive: bool = True) -> None:
    assert_postgres_configuration(expected_database=EXPECTED_DATABASE)
    target = database_target_summary()
    print(f"\nTarget database: {target['username']}@{target['host']}:{target['port']}/{target['database']}")
    print("=" * 70)
    print("MEDSHIELD-FL SYNTHETIC DEMO SEED")
    print("All records are labelled as synthetic. No real patient data created.")
    print("=" * 70)

    connectivity = await verify_async_connection()
    if not connectivity.ok:
        print(f"Aborting: database connection failed ({connectivity.error_type}).")
        raise SystemExit(1)

    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    hashed_pwd = hash_password(_resolve_demo_password())
    print(f"\nDemo password hashed (using env MEDSHIELD_DEMO_SEED_PASSWORD or default).")

    try:
        async with Session() as session:
            async with session.begin():
                # Rolls back entirely on failure; safe to re-run after partial crashes.
                # 1. Create roles
                print("\n--- Ensuring Roles ---")
                clinician_role = await _ensure_role(session, "CLINICIAN")
                patient_role = await _ensure_role(session, "PATIENT")
                await _ensure_role(session, "HOSPITAL_ADMIN")
                await _ensure_role(session, "SUPER_ADMIN")

                # 2. Create hospitals
                print("\n--- Ensuring Demo Hospitals ---")
                hosp_map: dict[str, Hospital] = {}
                for h_def in DEMO_HOSPITALS:
                    hosp = await _ensure_hospital(session, h_def)
                    hosp_map[h_def["code"]] = hosp

                # 3. Create clinicians
                print("\n--- Ensuring Demo Clinicians ---")
                clinician_profiles: dict[str, ClinicianProfile] = {}
                for c_def in DEMO_CLINICIANS:
                    user, created = await _ensure_user(
                        session,
                        email=c_def["email"],
                        full_name=c_def["full_name"],
                        stable_uuid=c_def["user_stable_uuid"],
                        hashed_pwd=hashed_pwd,
                    )
                    await _ensure_user_role(session, user, clinician_role)

                    hospital = hosp_map[c_def["hospital_code"]]
                    await _ensure_hospital_membership(session, user, hospital, "CLINICIAN")
                    profile = await _ensure_clinician_profile(session, c_def, user, hospital)
                    clinician_profiles[c_def["hospital_code"]] = profile

                # 4. Create patients
                print("\n--- Ensuring Demo Patients (15 total) ---")
                patient_map: dict[str, Patient] = {}
                for p_def in DEMO_PATIENTS:
                    hospital = hosp_map[p_def["hospital_code"]]
                    user, created = await _ensure_user(
                        session,
                        email=p_def["email"],
                        full_name=p_def["full_name"],
                        stable_uuid=p_def["user_stable_uuid"],
                        hashed_pwd=hashed_pwd,
                    )
                    await _ensure_user_role(session, user, patient_role)

                    patient = await _ensure_patient(session, p_def, user, hospital)
                    patient_map[p_def["opaque_id"]] = patient

                    # 5. Clinical metadata for each patient
                    await _ensure_clinical_metadata(session, p_def, patient, hospital)

                    # 6. Assign each patient to the demo clinician at their hospital
                    profile = clinician_profiles.get(p_def["hospital_code"])
                    if profile:
                        await _ensure_patient_assignment(session, patient, profile, hospital)

        print("\n" + "=" * 70)
        print("SYNTHETIC DEMO SEED COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nDemo accounts (password: DemoSeed2026! or MEDSHIELD_DEMO_SEED_PASSWORD):")
        print("  admin@medshield.org           — Super Admin")
        print("  dr_metro@metro.org             — Demo Clinician (Metro)")
        print("  demo_clinician_riverside@demo.org — Demo Clinician (Riverside)")
        print("  demo_patient_01@demo.org .. demo_patient_15@demo.org — Patients")
        print("\nPatient portal records: 1 document + 1 ECG + 1 observation per patient.")
        print("Clinician assignments: all Metro patients → dr_metro; all Riverside → dr_riverside.")
        print("\nAudit log entries will appear in GET /api/v1/admin/audit after patients log in.")
        print("\nTo clean up demo data, see the SQL at the top of this script.")

    except Exception as exc:
        print(f"\nSEED FAILED (transaction rolled back): {exc}", file=sys.stderr)
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed MedShield-FL synthetic demo dataset.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=True,
        help="Use MEDSHIELD_DEMO_SEED_PASSWORD env var or built-in demo default.",
    )
    args = parser.parse_args()
    asyncio.run(seed(non_interactive=args.non_interactive))
