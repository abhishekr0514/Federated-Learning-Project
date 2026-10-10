"""Federated Learning Network Models.

PRIVACY ARCHITECTURE:
Federated Learning models track training campaign metadata, round parameters, and hospital client updates.
No raw patient records, ECG signals, clinical notes, or client datasets are stored in federated tables.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.guid import GUID


class FederatedRound(Base):
    """Federated Learning Training Round Entity."""

    __tablename__ = "federated_rounds"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    round_number: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")  # pending, active, aggregated, failed
    min_clients_required: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    model_version: Mapped["ModelVersion"] = relationship("ModelVersion")
    client_updates: Mapped[List["FederatedClientUpdate"]] = relationship("FederatedClientUpdate", back_populates="round", cascade="all, delete-orphan")


class FederatedClient(Base):
    """Registered Federated Client (Hospital Node) Entity."""

    __tablename__ = "federated_clients"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False, index=True)
    client_identifier: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")  # active, offline, revoked

    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    hospital: Mapped["Hospital"] = relationship("Hospital")
    updates: Mapped[List["FederatedClientUpdate"]] = relationship("FederatedClientUpdate", back_populates="client", cascade="all, delete-orphan")


class FederatedClientUpdate(Base):
    """Metadata for local model update submitted by a hospital client in a federated round."""

    __tablename__ = "federated_client_updates"
    __table_args__ = (
        UniqueConstraint("round_id", "client_id", name="uq_round_client_update"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    round_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("federated_rounds.id", ondelete="CASCADE"), nullable=False, index=True)
    client_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("federated_clients.id", ondelete="CASCADE"), nullable=False, index=True)

    update_status: Mapped[str] = mapped_column(String(50), nullable=False, default="submitted")  # submitted, accepted, rejected
    local_sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    training_duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weights_reference_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    round: Mapped["FederatedRound"] = relationship("FederatedRound", back_populates="client_updates")
    client: Mapped["FederatedClient"] = relationship("FederatedClient", back_populates="updates")
