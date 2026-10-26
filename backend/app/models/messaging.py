"""Patient-to-clinician question workflow (non-emergency, async)."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.guid import GUID


class PatientClinicianQuestion(Base):
    """Secure patient question with optional clinician response."""

    __tablename__ = "patient_clinician_questions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("hospitals.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    author_user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    assigned_clinician_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        GUID(), ForeignKey("clinician_profiles.id", ondelete="SET NULL"), nullable=True, index=True
    )

    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="submitted")
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship("Patient")
    hospital: Mapped["Hospital"] = relationship("Hospital")
    assigned_clinician: Mapped[Optional["ClinicianProfile"]] = relationship("ClinicianProfile")
