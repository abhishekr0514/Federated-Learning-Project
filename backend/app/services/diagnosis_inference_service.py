"""Clinician-scoped multimodal proxy inference and prediction persistence."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.errors import ConflictError, InferenceNotReadyError, ValidationAppError
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.ml_models import DiagnosisPrediction, ModelVersion
from backend.app.models.patient import Patient
from backend.app.schemas.diagnosis_inference import DiagnosisEvaluateRequest, DiagnosisEvaluateResponse
from backend.app.services.prediction_proxy_class import gat_idempotent_replay_class
from ml.inference.fusion_runtime import get_fusion_inference_runtime
from ml.inference.setup_diagnostics import build_next_steps, environment_summary


def _model_dir() -> Path:
    root = Path(settings.MODEL_DIR)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root
    return root


class DiagnosisInferenceService:
    @staticmethod
    def get_readiness() -> Dict[str, Any]:
        runtime = get_fusion_inference_runtime(
            _model_dir(),
            inference_enabled=settings.INFERENCE_ENABLED,
            reset=True,
        )
        readiness = runtime.assess_readiness()
        payload = readiness.to_safe_dict()
        model_dir = _model_dir()
        payload["environment"] = environment_summary()
        payload["next_steps"] = build_next_steps(
            inference_enabled=settings.INFERENCE_ENABLED,
            readiness_blockers=readiness.blockers,
            model_dir=model_dir,
            manifest_present=readiness.manifest_present,
            checkpoint_present=readiness.checkpoint_present,
        )
        payload["model_dir_configured"] = settings.MODEL_DIR
        payload["inference_enabled"] = settings.INFERENCE_ENABLED
        return payload

    @staticmethod
    async def evaluate_for_assigned_patient(
        db: AsyncSession,
        *,
        user: User,
        profile: ClinicianProfile,
        patient: Patient,
        body: DiagnosisEvaluateRequest,
        patient_id: uuid.UUID,
    ) -> DiagnosisEvaluateResponse:
        if patient.id != patient_id:
            raise ValidationAppError("patient_id must match the authorized patient")

        runtime = get_fusion_inference_runtime(
            _model_dir(),
            inference_enabled=settings.INFERENCE_ENABLED,
        )
        readiness = runtime.assess_readiness()
        if not readiness.ready:
            raise InferenceNotReadyError(
                "Multimodal inference is not ready for safe execution.",
                details=readiness.to_safe_dict(),
            )

        if body.idempotency_key:
            existing = await DiagnosisInferenceService._find_idempotent_prediction(
                db, patient.id, body.idempotency_key
            )
            if existing:
                return DiagnosisInferenceService._to_response(existing)

        try:
            result = runtime.run_proxy_inference(
                ecg=body.embeddings.ecg,
                clinical_text=body.embeddings.clinical_text,
                tabular=body.embeddings.tabular,
            )
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc
        except RuntimeError as exc:
            raise InferenceNotReadyError(str(exc)) from exc

        model_version = await DiagnosisInferenceService._get_or_create_model_version(
            db,
            version_tag=result["model_version_tag"],
            target_definition=result["target_definition"],
            artifact_uri=str(_model_dir() / "medshield_fusion_manifest.json"),
        )

        provenance = (
            f"idempotency:{body.idempotency_key}"
            if body.idempotency_key
            else "gat_inference_pipeline"
        )

        if body.idempotency_key:
            dup = await DiagnosisInferenceService._find_idempotent_prediction(
                db, patient.id, body.idempotency_key
            )
            if dup:
                return DiagnosisInferenceService._to_response(dup)

        prediction = DiagnosisPrediction(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=profile.hospital_id,
            model_version_id=model_version.id,
            target_definition=result["target_definition"],
            prediction_type=result["prediction_type"],
            provenance=provenance,
            confidence=float(result["confidence_uncalibrated"]),
            predicted_class_index=int(result["predicted_class_index"]),
            status="evaluated",
        )
        db.add(prediction)
        await db.flush()

        return DiagnosisEvaluateResponse(
            status="EVALUATED",
            prediction_id=prediction.id,
            patient_id=patient.id,
            target_definition=prediction.target_definition,
            prediction_type=prediction.prediction_type,
            model_version_tag=model_version.version_tag,
            confidence_uncalibrated=prediction.confidence or 0.0,
            predicted_class_index=int(result["predicted_class_index"]),
            disclaimer=result["disclaimer"],
        )

    @staticmethod
    async def _find_idempotent_prediction(
        db: AsyncSession,
        patient_id: uuid.UUID,
        idempotency_key: str,
    ) -> Optional[DiagnosisPrediction]:
        stmt = select(DiagnosisPrediction).where(
            DiagnosisPrediction.patient_id == patient_id,
            DiagnosisPrediction.provenance == f"idempotency:{idempotency_key}",
        )
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    @staticmethod
    async def _get_or_create_model_version(
        db: AsyncSession,
        *,
        version_tag: str,
        target_definition: str,
        artifact_uri: str,
    ) -> ModelVersion:
        stmt = select(ModelVersion).where(ModelVersion.version_tag == version_tag)
        res = await db.execute(stmt)
        existing = res.scalar_one_or_none()
        if existing:
            return existing
        version = ModelVersion(
            id=uuid.uuid4(),
            model_family="MedShield-GAT-Multimodal",
            version_tag=version_tag,
            architecture_config="medshield.diagnostic_net.gat_fusion@1.0.0",
            target_definition=target_definition[:255],
            status="experimental",
            artifact_uri=artifact_uri[:512],
        )
        db.add(version)
        await db.flush()
        return version

    @staticmethod
    def _to_response(row: DiagnosisPrediction) -> DiagnosisEvaluateResponse:
        class_idx, _label = gat_idempotent_replay_class(row.predicted_class_index)
        return DiagnosisEvaluateResponse(
            status="EVALUATED",
            prediction_id=row.id,
            patient_id=row.patient_id,
            target_definition=row.target_definition,
            prediction_type=row.prediction_type,
            model_version_tag="unknown",
            confidence_uncalibrated=row.confidence or 0.0,
            predicted_class_index=class_idx,
            disclaimer="Returned from idempotent replay; class index not re-materialized.",
        )
