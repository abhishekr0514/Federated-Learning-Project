"""ML Model Registry, Evaluation Metrics, Predictions, and Explanations Models."""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.guid import GUID


class ModelVersion(Base):
    """Registered Neural Model Architecture Version."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    model_family: Mapped[str] = mapped_column(String(100), nullable=False, default="MedShield-GAT-Multimodal")
    version_tag: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    architecture_config: Mapped[str] = mapped_column(String(1024), nullable=False)
    target_definition: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="PTB-XL Normal Sinus Rhythm (SNOMED 426783006) vs Pathological Superclasses Proxy Task",
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="experimental")  # experimental, staged, active, deprecated
    artifact_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    metrics: Mapped[List["ModelMetric"]] = relationship("ModelMetric", back_populates="model_version", cascade="all, delete-orphan")


class GlobalModel(Base):
    """Global Model registry container."""

    __tablename__ = "global_models"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    active_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    active_version: Mapped[Optional["ModelVersion"]] = relationship("ModelVersion")


class ModelMetric(Base):
    """Global evaluation metric record."""

    __tablename__ = "model_metrics"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)

    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    evaluation_split: Mapped[str] = mapped_column(String(50), nullable=False, default="test_split")
    dataset_version: Mapped[str] = mapped_column(String(100), nullable=False, default="PTB-XL-v1.0.3")
    task_target: Mapped[str] = mapped_column(String(255), nullable=False)
    provenance: Mapped[str] = mapped_column(String(255), nullable=False, default="empirical_offline_eval")

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    model_version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="metrics")


class HospitalMetric(Base):
    """Per-hospital local evaluation metric record."""

    __tablename__ = "hospital_metrics"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False, index=True)

    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    hospital: Mapped["Hospital"] = relationship("Hospital")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion")


class DiagnosisPrediction(Base):
    """Multimodal diagnosis prediction telemetry record.
    
    Contains explicit target definitions and prediction types.
    Does NOT imply clinical diagnosis or fake ground truth.
    """

    __tablename__ = "diagnosis_predictions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    hospital_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("hospitals.id", ondelete="RESTRICT"), nullable=False, index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False, index=True)

    target_definition: Mapped[str] = mapped_column(String(255), nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(50), nullable=False, default="proxy_evaluation")
    provenance: Mapped[str] = mapped_column(String(255), nullable=False, default="gat_inference_pipeline")
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="disabled")  # disabled, evaluated, pending

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    patient: Mapped["Patient"] = relationship("Patient")
    hospital: Mapped["Hospital"] = relationship("Hospital")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion")
    explanations: Mapped[List["PredictionExplanation"]] = relationship("PredictionExplanation", back_populates="prediction", cascade="all, delete-orphan")
    causal_analyses: Mapped[List["CausalAnalysis"]] = relationship("CausalAnalysis", back_populates="prediction", cascade="all, delete-orphan")


class PredictionExplanation(Base):
    """Explainable AI (DiCE Counterfactual) explanation record."""

    __tablename__ = "prediction_explanations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("diagnosis_predictions.id", ondelete="CASCADE"), nullable=False, index=True)

    method: Mapped[str] = mapped_column(String(100), nullable=False, default="DiCE-Counterfactual")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    prediction: Mapped["DiagnosisPrediction"] = relationship("DiagnosisPrediction", back_populates="explanations")
    counterfactuals: Mapped[List["CounterfactualExplanation"]] = relationship("CounterfactualExplanation", back_populates="explanation", cascade="all, delete-orphan")


class CounterfactualExplanation(Base):
    """Counterfactual feature perturbation details."""

    __tablename__ = "counterfactual_explanations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    explanation_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("prediction_explanations.id", ondelete="CASCADE"), nullable=False, index=True)

    feature_perturbation_summary: Mapped[str] = mapped_column(String(1024), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    explanation: Mapped["PredictionExplanation"] = relationship("PredictionExplanation", back_populates="counterfactuals")


class CausalAnalysis(Base):
    """Causal inference (DoWhy DAG & Interventions) record."""

    __tablename__ = "causal_analyses"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("diagnosis_predictions.id", ondelete="CASCADE"), nullable=False, index=True)

    method: Mapped[str] = mapped_column(String(100), nullable=False, default="DoWhy-DAG-Intervention")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    prediction: Mapped["DiagnosisPrediction"] = relationship("DiagnosisPrediction", back_populates="causal_analyses")
