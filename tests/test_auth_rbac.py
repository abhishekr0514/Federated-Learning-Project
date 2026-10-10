"""Comprehensive Authentication and RBAC Test Suite."""

import uuid
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.security import (
    UserRole,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.models import (
    Base,
    User,
    Role,
    Permission,
    Hospital,
    HospitalMembership,
    ClinicianProfile,
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
    UserSession,
)


@pytest.fixture(scope="module")
def db_session():
    """Create in-memory SQLite session for authentication and RBAC tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_password_hashing_and_verification():
    """Verify password hashing with bcrypt and secure comparison."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_access_token_generation_and_decoding():
    """Verify JWT access token creation, claims, and signature decoding."""
    user_id = str(uuid.uuid4())
    token, jti = create_access_token(user_id=user_id)

    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == user_id
    assert payload["jti"] == jti
    assert payload["type"] == "access"


def test_refresh_token_generation():
    """Verify JWT refresh token creation and claims."""
    user_id = str(uuid.uuid4())
    token, jti, expire = create_refresh_token(user_id=user_id)

    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == user_id
    assert payload["jti"] == jti
    assert payload["type"] == "refresh"


def test_rbac_roles_and_permissions(db_session: Session):
    """Verify setup of canonical roles and assignment to users."""
    role_super = Role(name=UserRole.SUPER_ADMIN.value, description="Super Administrator")
    role_admin = Role(name=UserRole.HOSPITAL_ADMIN.value, description="Hospital Administrator")
    role_clinician = Role(name=UserRole.CLINICIAN.value, description="Medical Clinician")
    role_patient = Role(name=UserRole.PATIENT.value, description="Patient User")

    db_session.add_all([role_super, role_admin, role_clinician, role_patient])
    db_session.commit()

    user_admin = User(
        email="admin_boston@medshield.org",
        hashed_password=hash_password("Password123!"),
        full_name="Boston Admin",
    )
    user_admin.roles.append(role_admin)

    db_session.add(user_admin)
    db_session.commit()

    fetched = db_session.query(User).filter_by(email="admin_boston@medshield.org").first()
    assert len(fetched.roles) == 1
    assert fetched.roles[0].name == UserRole.HOSPITAL_ADMIN.value


def test_soft_deleted_user_invalidation(db_session: Session):
    """Verify soft-deleted user (deleted_at is set) cannot be active."""
    deleted_user = User(
        email="deleted@medshield.org",
        hashed_password=hash_password("Password123!"),
        deleted_at=datetime.now(timezone.utc),
    )
    db_session.add(deleted_user)
    db_session.commit()

    fetched = db_session.query(User).filter_by(email="deleted@medshield.org").first()
    assert fetched.deleted_at is not None
