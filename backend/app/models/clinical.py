"""Clinical Data Reference and Metadata Models.

PRIVACY ARCHITECTURE:
Centralized database stores clinical METADATA and local storage URIs only.
Raw clinical notes, unmasked text documents, and raw ECG signal arrays remain inside local hospital node storage.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.guid import GUID


class MedicalDocument(Base):
    """Clinical document reference entity."""

    __tablename__ = "medical_documents"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="RESTRICT"), nullable=False, index=True)
    
    local_document_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    document_type: Mapped[str] = mapped_column(String(100), nullable=False, default="clinical_note")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient")
    hospital: Mapped["Hospital"] = relationship("Hospital")
    versions: Mapped[List["DocumentVersion"]] = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    jobs: Mapped[List["DocumentProcessingJob"]] = relationship("DocumentProcessingJob", back_populates="document", cascade="all, delete-orphan")
    masking_results: Mapped[List["PrivacyMaskingResult"]] = relationship("PrivacyMaskingResult", back_populates="document", cascade="all, delete-orphan")


class DocumentVersion(Base):
    """Clinical document version metadata."""

    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version_number", name="uq_doc_version"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("medical_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    document: Mapped["MedicalDocument"] = relationship("MedicalDocument", back_populates="versions")


class DocumentProcessingJob(Base):
    """Tracking entity for background text processing and privacy sanitization jobs."""

    __tablename__ = "document_processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("medical_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    document: Mapped["MedicalDocument"] = relationship("MedicalDocument", back_populates="jobs")


class PrivacyMaskingResult(Base):
    """Metadata result of text privacy sanitization."""

    __tablename__ = "privacy_masking_results"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("medical_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    masking_method: Mapped[str] = mapped_column(String(100), nullable=False, default="BioClinicalBERT-PII-Masker")
    entities_detected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sanitization_status: Mapped[str] = mapped_column(String(50), nullable=False, default="sanitized")

    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    document: Mapped["MedicalDocument"] = relationship("MedicalDocument", back_populates="masking_results")
    entity_counts: Mapped[List["PrivacyMaskingEntityCount"]] = relationship(
        "PrivacyMaskingEntityCount", back_populates="masking_result", cascade="all, delete-orphan"
    )


class PrivacyMaskingEntityCount(Base):
    """Aggregate entity counts by PII category for a masking result.
    
    Zero raw values, zero sensitive spans stored.
    """

    __tablename__ = "privacy_masking_entity_counts"
    __table_args__ = (
        UniqueConstraint("masking_result_id", "entity_category", name="uq_masking_entity_category"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    masking_result_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("privacy_masking_results.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_category: Mapped[str] = mapped_column(String(50), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    masking_result: Mapped["PrivacyMaskingResult"] = relationship("PrivacyMaskingResult", back_populates="entity_counts")


class ECGRecord(Base):
    """ECG metadata reference entity.
    
    Contains acquisition metadata and local hospital storage URI. Zero raw signal arrays centrally stored.
    """

    __tablename__ = "ecg_records"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="RESTRICT"), nullable=False, index=True)

    local_storage_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    sampling_rate_hz: Mapped[int] = mapped_column(Integer, nullable=False, default=500)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    lead_count: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    dataset_source: Mapped[str] = mapped_column(String(100), nullable=False, default="PTB-XL-Proxy")

    acquired_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient")
    hospital: Mapped["Hospital"] = relationship("Hospital")


class ClinicalObservation(Base):
    """Tabular clinical observation metadata reference entity."""

    __tablename__ = "clinical_observations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="RESTRICT"), nullable=False, index=True)

    observation_type: Mapped[str] = mapped_column(String(100), nullable=False, default="tabular_features")
    feature_schema_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0")
    local_ref: Mapped[str] = mapped_column(String(512), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient")
    hospital: Mapped["Hospital"] = relationship("Hospital")
