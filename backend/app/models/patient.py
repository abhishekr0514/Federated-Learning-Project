"""Privacy-safe Opaque Patient Domain Models and Assignments.

CRITICAL PRIVACY BOUNDARY:
This central database schema stores NO raw Patient Identifiable Information (PII) or Protected Health Information (PHI).
Raw patient identities, names, government IDs, and clinical contact details remain strictly inside local hospital boundaries.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.guid import GUID


class Patient(Base):
    """Centralized Opaque Patient Entity.

    Contains strictly opaque reference hashes without central PII/PHI storage.
    Optionally associated with a Patient User Account (`user_id`).
    """

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    opaque_patient_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), unique=True, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user: Mapped[Optional["User"]] = relationship("User")
    hospital_associations: Mapped[List["PatientHospitalAssociation"]] = relationship(
        "PatientHospitalAssociation", back_populates="patient", cascade="all, delete-orphan"
    )
    clinician_assignments: Mapped[List["PatientClinicianAssignment"]] = relationship(
        "PatientClinicianAssignment", back_populates="patient", cascade="all, delete-orphan"
    )


class PatientHospitalAssociation(Base):
    """Association linking an opaque central Patient reference to a hospital node."""

    __tablename__ = "patient_hospital_associations"
    __table_args__ = (
        UniqueConstraint("patient_id", "hospital_id", name="uq_patient_hospital_assoc"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_patient_ref_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient", back_populates="hospital_associations")
    hospital: Mapped["Hospital"] = relationship("Hospital")


class PatientClinicianAssignment(Base):
    """Assignment linking a Patient to an authorized Clinician profile within a hospital."""

    __tablename__ = "patient_clinician_assignments"
    __table_args__ = (
        UniqueConstraint("patient_id", "clinician_id", name="uq_patient_clinician_assignment"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    clinician_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("clinician_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient", back_populates="clinician_assignments")
    clinician: Mapped["ClinicianProfile"] = relationship("ClinicianProfile")
    hospital: Mapped["Hospital"] = relationship("Hospital")
