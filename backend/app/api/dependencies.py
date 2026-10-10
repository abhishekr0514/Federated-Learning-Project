"""FastAPI Dependency Injection for DB Sessions and RBAC/Tenant Authorization Guards."""

import uuid
from typing import AsyncGenerator, Callable, List, Optional, Tuple
from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.errors import AuthenticationAppError, ForbiddenError, NotFoundError
from backend.app.core.request_context import get_request_id
from backend.app.core.security import SystemPermission, UserRole, decode_token
from backend.app.db.session import AsyncSessionLocal
from backend.app.models.identity import ClinicianProfile, Hospital, HospitalMembership, User
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation

security_bearer = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields a request-scoped async SQLAlchemy database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_request_id_dependency(x_request_id: Optional[str] = Header(None)) -> str:
    """Dependency to retrieve the current request ID."""
    return get_request_id() or x_request_id or "unknown"


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> User:
    """Extract Bearer token, validate signature, and return active non-deleted User."""
    if not auth or not auth.credentials:
        raise AuthenticationAppError("Authentication token missing")

    payload = decode_token(auth.credentials)
    if not payload or payload.get("type") != "access":
        raise AuthenticationAppError("Invalid or expired access token")

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise AuthenticationAppError("Malformed access token payload")

    try:
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        raise AuthenticationAppError("Invalid user identifier claim")

    stmt = select(User).where(User.id == user_uuid, User.deleted_at.is_(None)).options(selectinload(User.roles))
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise AuthenticationAppError("Authenticated user account not found")

    if not user.is_active:
        raise AuthenticationAppError("User account is deactivated")

    return user


def require_role(*required_roles: UserRole) -> Callable:
    """Dependency guard requiring user to possess one of the specified canonical roles or Super Admin."""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.is_super_admin:
            return current_user

        user_role_names = [r.name for r in current_user.roles]
        required_role_values = [r.value for r in required_roles]

        if any(r in required_role_values for r in user_role_names):
            return current_user

        raise ForbiddenError(f"User role does not satisfy authorization requirement: {required_role_values}")

    return role_checker


def require_permission(*required_permissions: SystemPermission) -> Callable:
    """Dependency guard requiring user to possess explicit system permission capabilities."""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.is_super_admin:
            return current_user

        # Fetch permissions assigned to user's roles
        user_role_names = [r.name for r in current_user.roles]
        
        # Super Admin override check
        if UserRole.SUPER_ADMIN.value in user_role_names or UserRole.SUPER_ADMIN in user_role_names:
            return current_user

        from backend.app.core.security import ROLE_PERMISSIONS, UserRole
        
        granted_permissions = set()
        for role_name in user_role_names:
            try:
                role_enum = UserRole(role_name)
                for perm in ROLE_PERMISSIONS.get(role_enum, []):
                    granted_permissions.add(perm.value if isinstance(perm, SystemPermission) else perm)
            except ValueError:
                continue

        req_perm_values = [p.value for p in required_permissions]
        if any(p in granted_permissions for p in req_perm_values):
            return current_user

        raise ForbiddenError(f"User lacks required system permissions: {req_perm_values}")

    return permission_checker


async def verify_hospital_access(
    hospital_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    """Verify authenticated user is authorized to access hospital-scoped resources."""
    if current_user.is_super_admin:
        return current_user

    stmt = select(HospitalMembership).where(
        HospitalMembership.user_id == current_user.id,
        HospitalMembership.hospital_id == hospital_id,
    )
    result = await db.execute(stmt)
    membership = result.scalar_one_or_none()

    if not membership:
        # Raise 404 to avoid leaking hospital existence across tenant boundaries
        raise NotFoundError(f"Hospital node resource not accessible")

    return current_user


async def get_authorized_hospital_context(
    hospital_id: Optional[uuid.UUID] = Query(None, description="Optional hospital ID for multi-hospital admins"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.HOSPITAL_ADMIN, UserRole.SUPER_ADMIN)),
) -> Hospital:
    """Resolve and enforce the authorized hospital scope for hospital administration endpoints.
    
    Tenant Isolation Invariants:
    1. Caller must possess active HOSPITAL_ADMIN or SUPER_ADMIN role.
    2. Single-hospital admin: automatically bound to their authorized hospital.
    3. Multi-hospital admin: client-supplied hospital_id is validated against authorized memberships.
    4. Supplying an unauthorized hospital ID returns HTTP 404 to prevent cross-tenant enumeration.
    """
    if current_user.is_super_admin:
        if hospital_id:
            hosp_res = await db.execute(select(Hospital).where(Hospital.id == hospital_id))
            hosp = hosp_res.scalar_one_or_none()
            if not hosp:
                raise NotFoundError("Hospital node resource not accessible")
            return hosp
        else:
            hosp_res = await db.execute(select(Hospital).where(Hospital.is_active.isnot(False)).limit(1))
            hosp = hosp_res.scalar_one_or_none()
            if not hosp:
                raise NotFoundError("No active hospital node found in system")
            return hosp

    # Fetch all active hospital memberships for current user
    stmt = (
        select(HospitalMembership)
        .options(selectinload(HospitalMembership.hospital))
        .where(HospitalMembership.user_id == current_user.id)
    )
    res = await db.execute(stmt)
    memberships = res.scalars().all()

    if not memberships:
        raise ForbiddenError("User has no authorized hospital memberships")

    authorized_hospital_ids = {m.hospital_id: m.hospital for m in memberships if m.hospital}

    if hospital_id:
        if hospital_id not in authorized_hospital_ids:
            # Raise 404 to avoid leaking hospital existence across tenant boundaries
            raise NotFoundError("Hospital node resource not accessible")
        return authorized_hospital_ids[hospital_id]

    # For single-hospital admin or when no hospital_id param is provided, pick the primary/first membership
    primary_membership = memberships[0]
    if not primary_membership.hospital:
        # Load hospital directly if not eager loaded
        hosp_res = await db.execute(select(Hospital).where(Hospital.id == primary_membership.hospital_id))
        primary_hosp = hosp_res.scalar_one_or_none()
        if not primary_hosp:
            raise NotFoundError("Hospital node resource not accessible")
        return primary_hosp
    return primary_membership.hospital


async def verify_patient_access(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    """Verify authenticated user (Patient self, assigned Clinician, or Hospital Admin) access."""
    if current_user.is_super_admin:
        return current_user

    user_role_names = [r.name for r in current_user.roles]

    # 1. Patient Ownership Check
    stmt_patient = select(Patient).where(Patient.id == patient_id)
    res_patient = await db.execute(stmt_patient)
    patient = res_patient.scalar_one_or_none()

    if not patient:
        raise NotFoundError("Patient reference resource not accessible")

    if patient.user_id == current_user.id:
        return current_user

    # 2. Clinician Assignment / Hospital Admin Check
    if UserRole.CLINICIAN.value in user_role_names or UserRole.HOSPITAL_ADMIN.value in user_role_names:
        # Check clinician assignment
        stmt_clinician = select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        res_clinician = await db.execute(stmt_clinician)
        clinician = res_clinician.scalar_one_or_none()

        if clinician:
            # Check explicit assignment
            stmt_assign = select(PatientClinicianAssignment).where(
                PatientClinicianAssignment.patient_id == patient_id,
                PatientClinicianAssignment.clinician_id == clinician.id,
            )
            res_assign = await db.execute(stmt_assign)
            if res_assign.scalar_one_or_none():
                return current_user

            # Check hospital association parity
            stmt_assoc = select(PatientHospitalAssociation).where(
                PatientHospitalAssociation.patient_id == patient_id,
                PatientHospitalAssociation.hospital_id == clinician.hospital_id,
            )
            res_assoc = await db.execute(stmt_assoc)
            if res_assoc.scalar_one_or_none():
                return current_user

        # Check Hospital Admin membership parity
        stmt_memberships = select(HospitalMembership.hospital_id).where(HospitalMembership.user_id == current_user.id)
        res_memberships = await db.execute(stmt_memberships)
        user_hospital_ids = res_memberships.scalars().all()

        if user_hospital_ids:
            stmt_hosp_assoc = select(PatientHospitalAssociation).where(
                PatientHospitalAssociation.patient_id == patient_id,
                PatientHospitalAssociation.hospital_id.in_(user_hospital_ids),
            )
            res_hosp_assoc = await db.execute(stmt_hosp_assoc)
            if res_hosp_assoc.scalar_one_or_none():
                return current_user

    raise NotFoundError("Patient reference resource not accessible")


async def get_current_clinician_context(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.SUPER_ADMIN)),
) -> Tuple[User, Optional[ClinicianProfile]]:
    """Enforce caller possesses active CLINICIAN (or SUPER_ADMIN) role and resolve ClinicianProfile."""
    if current_user.is_super_admin:
        stmt = select(ClinicianProfile).limit(1)
        res = await db.execute(stmt)
        profile = res.scalar_one_or_none()
        return current_user, profile

    stmt = select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
    res = await db.execute(stmt)
    profile = res.scalar_one_or_none()

    if not profile:
        raise ForbiddenError("User does not possess an active Clinician Profile")

    return current_user, profile


async def get_authorized_clinician_patient(
    patient_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    clinician_ctx: Tuple[User, Optional[ClinicianProfile]] = Depends(get_current_clinician_context),
) -> Patient:
    """Enforce that the requested patient reference is explicitly assigned to the calling clinician.
    
    Returns HTTP 404 for unassigned patients or patients from another hospital to prevent
    cross-clinician and cross-tenant resource enumeration.
    """
    current_user, profile = clinician_ctx

    if current_user.is_super_admin:
        stmt = select(Patient).where(Patient.id == patient_id)
        res = await db.execute(stmt)
        patient = res.scalar_one_or_none()
        if not patient:
            raise NotFoundError("Patient reference resource not accessible")
        return patient

    # Explicit clinician assignment check matching clinician ID and hospital ID
    stmt_assign = select(PatientClinicianAssignment).where(
        PatientClinicianAssignment.patient_id == patient_id,
        PatientClinicianAssignment.clinician_id == profile.id,
        PatientClinicianAssignment.hospital_id == profile.hospital_id,
    )
    res_assign = await db.execute(stmt_assign)
    assignment = res_assign.scalar_one_or_none()

    if not assignment:
        raise NotFoundError("Patient reference resource not accessible")

    stmt_patient = select(Patient).where(Patient.id == patient_id)
    res_patient = await db.execute(stmt_patient)
    patient = res_patient.scalar_one_or_none()

    if not patient:
        raise NotFoundError("Patient reference resource not accessible")

    return patient


async def get_current_patient_context(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.PATIENT, UserRole.SUPER_ADMIN)),
) -> Tuple[User, Patient]:
    """Enforce caller possesses active PATIENT role and resolve linked Patient entity.

    Ensures clinicians/admins without an explicit patient account link cannot
    accidentally inherit patient portal context.
    """
    stmt = select(Patient).where(Patient.user_id == current_user.id)
    res = await db.execute(stmt)
    patient = res.scalar_one_or_none()

    if not patient:
        raise NotFoundError("Patient reference resource not accessible")

    return current_user, patient


async def get_authorized_patient_self(
    patient_id: uuid.UUID,
    patient_ctx: Tuple[User, Patient] = Depends(get_current_patient_context),
) -> Patient:
    """Enforce that the requested patient resource matches the authenticated patient's own ID.

    Returns HTTP 404 for any mismatched UUID to prevent cross-patient traversal
    and resource enumeration.
    """
    _, patient = patient_ctx
    if patient.id != patient_id:
        raise NotFoundError("Patient reference resource not accessible")

    return patient

