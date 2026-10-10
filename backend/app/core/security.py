"""Security primitives, JWT token handling, password hashing, and RBAC definitions."""

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import bcrypt
from jose import JWTError, jwt

from backend.app.core.config import settings


class UserRole(str, Enum):
    """Canonical Role-Based Access Control (RBAC) Roles."""

    SUPER_ADMIN = "SUPER_ADMIN"
    HOSPITAL_ADMIN = "HOSPITAL_ADMIN"
    CLINICIAN = "CLINICIAN"
    PATIENT = "PATIENT"


class SystemPermission(str, Enum):
    """System Permission Capabilities."""

    # Super Admin Permissions
    GLOBAL_ADMIN = "global:admin"
    HOSPITAL_MANAGE = "hospital:manage"
    FEDERATED_MONITOR = "federated:monitor"
    GLOBAL_MODEL_MANAGE = "global_model:manage"
    SECURITY_MONITOR = "security:monitor"
    AUDIT_READ = "audit:read"
    GLOBAL_METRICS_READ = "global_metrics:read"

    # Hospital Admin Permissions
    HOSPITAL_CONFIG = "hospital:config"
    CLINICIAN_MANAGE = "clinician:manage"
    LOCAL_PATIENT_MANAGE = "local_patient:manage"
    LOCAL_TRAINING_MONITOR = "local_training:monitor"
    LOCAL_METRICS_READ = "local_metrics:read"
    LOCAL_AUDIT_READ = "local_audit:read"

    # Clinician Permissions
    CLINICAL_READ = "clinical:read"
    CLINICAL_WRITE = "clinical:write"
    DIAGNOSIS_READ = "diagnosis:read"
    EXPLANATION_READ = "explanation:read"
    CAUSAL_READ = "causal:read"

    # Patient Permissions
    OWN_PROFILE_READ = "own_profile:read"
    OWN_RECORDS_READ = "own_records:read"
    OWN_DIAGNOSIS_READ = "own_diagnosis:read"


ROLE_PERMISSIONS: Dict[UserRole, List[SystemPermission]] = {
    UserRole.SUPER_ADMIN: [
        SystemPermission.GLOBAL_ADMIN,
        SystemPermission.HOSPITAL_MANAGE,
        SystemPermission.FEDERATED_MONITOR,
        SystemPermission.GLOBAL_MODEL_MANAGE,
        SystemPermission.SECURITY_MONITOR,
        SystemPermission.AUDIT_READ,
        SystemPermission.GLOBAL_METRICS_READ,
    ],
    UserRole.HOSPITAL_ADMIN: [
        SystemPermission.HOSPITAL_CONFIG,
        SystemPermission.CLINICIAN_MANAGE,
        SystemPermission.LOCAL_PATIENT_MANAGE,
        SystemPermission.LOCAL_TRAINING_MONITOR,
        SystemPermission.LOCAL_METRICS_READ,
        SystemPermission.LOCAL_AUDIT_READ,
    ],
    UserRole.CLINICIAN: [
        SystemPermission.CLINICAL_READ,
        SystemPermission.CLINICAL_WRITE,
        SystemPermission.DIAGNOSIS_READ,
        SystemPermission.EXPLANATION_READ,
        SystemPermission.CAUSAL_READ,
    ],
    UserRole.PATIENT: [
        SystemPermission.OWN_PROFILE_READ,
        SystemPermission.OWN_RECORDS_READ,
        SystemPermission.OWN_DIAGNOSIS_READ,
    ],
}


def hash_password(password: str) -> str:
    """Hash plain text password using bcrypt."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    pwd_bytes = plain_password.encode("utf-8")[:72]
    hashed_bytes = hashed_password.encode("utf-8")
    try:
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)
    except Exception:
        return False


def create_access_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Encode JWT access token payload with signature. Returns (token, jti)."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode: Dict[str, Any] = {
        "sub": str(user_id),
        "jti": jti,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": settings.PROJECT_NAME,
    }
    if extra_claims:
        to_encode.update(extra_claims)

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt, jti


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, str, datetime]:
    """Encode JWT refresh token payload with signature. Returns (token, jti, expires_at)."""
    jti = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))

    to_encode: Dict[str, Any] = {
        "sub": str(user_id),
        "jti": jti,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": settings.PROJECT_NAME,
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded_jwt, jti, expire


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT access or refresh token signature."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError:
        return None
