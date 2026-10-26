"""Service layer for operational two-modality ECG + clinical report evaluation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.errors import ConflictError, InferenceNotReadyError, ValidationAppError
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.ml_models import DiagnosisPrediction, ModelVersion
from backend.app.models.patient import Patient
from backend.app.schemas.ecg_report_inference import (
    ECGReportEmbedRequest,
    ECGReportEmbedResponse,
    ECGReportEvaluateRequest,
    ECGReportEvaluateResponse,
    ECGReportReadinessResponse,
)
from backend.app.services.prediction_proxy_class import ecg_idempotent_replay_class
from ml.inference.ecg_report_runtime import get_ecg_report_runtime

logger = logging.getLogger(__name__)


def _model_dir() -> Path:
    root = Path(settings.ECG_REPORT_MODEL_DIR)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root
    return root


class ECGReportInferenceService:
    @staticmethod
    def get_readiness(*, verify_forward_pass: bool = False) -> Dict[str, Any]:
        """Assess readiness safely without leaking internal paths or secrets."""
        runtime = get_ecg_report_runtime(
            _model_dir(),
            enabled=settings.ECG_REPORT_INFERENCE_ENABLED,
        )
        readiness = runtime.assess_readiness(verify_forward_pass=verify_forward_pass)
        payload = readiness.to_safe_dict()
        payload["environment"] = settings.ENVIRONMENT
        return payload

    @staticmethod
    def initialize_runtime() -> None:
        """Explicitly warm-up / load runtime weights into memory."""
        runtime = get_ecg_report_runtime(
            _model_dir(),
            enabled=settings.ECG_REPORT_INFERENCE_ENABLED,
        )
        runtime.ensure_loaded()

    @staticmethod
    async def embed_modalities(*, body: ECGReportEmbedRequest) -> ECGReportEmbedResponse:
        """Produce 128-d ECG and text embeddings via the authorized encoder checkpoint."""
        runtime = get_ecg_report_runtime(
            _model_dir(),
            enabled=settings.ECG_REPORT_INFERENCE_ENABLED,
        )
        readiness = runtime.assess_readiness()
        if not readiness.ready:
            if readiness.manifest_present and readiness.checkpoint_present:
                try:
                    runtime.ensure_loaded()
                    readiness = runtime.assess_readiness()
                except Exception as exc:
                    logger.error("ECG + report runtime load failed: %s", type(exc).__name__)
                    raise InferenceNotReadyError(
                        "ECG + report encoder is not ready for safe execution.",
                        details=readiness.to_safe_dict(),
                    ) from exc
            if not readiness.ready:
                raise InferenceNotReadyError(
                    "ECG + report encoder is not ready for safe execution.",
                    details=readiness.to_safe_dict(),
                )

        try:
            encoded = runtime.extract_encoder_embeddings(
                signal_mv=body.ecg_leads,
                report_text=body.report_text,
            )
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc
        except RuntimeError as exc:
            raise InferenceNotReadyError(str(exc)) from exc
        except Exception as exc:
            logger.error("Encoder embedding extraction failed: %s", type(exc).__name__)
            raise ValidationAppError("Encoder embedding extraction failed.") from exc

        return ECGReportEmbedResponse(
            status="ENCODED",
            ecg_embedding=encoded["ecg_embedding"],
            text_embedding=encoded["text_embedding"],
            architecture_id=encoded["architecture_id"],
            model_version_tag=encoded["model_version_tag"],
            privacy_validation_status=encoded["privacy_validation_status"],
        )

    @staticmethod
    async def evaluate(
        db: Optional[AsyncSession],
        *,
        body: ECGReportEvaluateRequest,
        user: Optional[User] = None,
        profile: Optional[ClinicianProfile] = None,
        patient: Optional[Patient] = None,
        hospital_id: Optional[uuid.UUID] = None,
    ) -> ECGReportEvaluateResponse:
        """
        Evaluate 12-lead ECG and clinical report.
        If patient and db session are provided, verifies clinician scope and persists prediction.
        """
        runtime = get_ecg_report_runtime(
            _model_dir(),
            enabled=settings.ECG_REPORT_INFERENCE_ENABLED,
        )

        readiness = runtime.assess_readiness()
        if not readiness.ready:
            # Try to load if model not yet in memory but artifacts are present
            if readiness.manifest_present and readiness.checkpoint_present:
                try:
                    runtime.ensure_loaded()
                    readiness = runtime.assess_readiness()
                except Exception as exc:
                    logger.error("ECG + report runtime load failed: %s", type(exc).__name__)
                    raise InferenceNotReadyError(
                        "ECG + report model is not ready for safe execution.",
                        details=readiness.to_safe_dict(),
                    ) from exc
            if not readiness.ready:
                raise InferenceNotReadyError(
                    "ECG + report model is not ready for safe execution.",
                    details=readiness.to_safe_dict(),
                )

        # Idempotency check if patient and key are provided
        if patient and db and body.idempotency_key:
            existing = await ECGReportInferenceService._find_idempotent_prediction(
                db, patient.id, body.idempotency_key
            )
            if existing:
                return ECGReportInferenceService._to_response(existing)

        # Execute forward pass
        try:
            result = runtime.run_inference(
                signal_mv=body.ecg_leads,
                report_text=body.report_text,
            )
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc
        except RuntimeError as exc:
            raise InferenceNotReadyError(str(exc)) from exc
        except Exception as exc:
            logger.error("Inference execution failed: %s", type(exc).__name__)
            raise ValidationAppError("Model evaluation failed on provided inputs.") from exc

        prediction_id = uuid.uuid4()

        # If patient-associated, persist to database with clinical provenance
        if patient and db:
            resolved_hospital_id = hospital_id
            if resolved_hospital_id is None:
                resolved_hospital_id = profile.hospital_id if profile else None
            if resolved_hospital_id is None:
                from backend.app.models.patient import PatientHospitalAssociation

                assoc_stmt = (
                    select(PatientHospitalAssociation)
                    .where(PatientHospitalAssociation.patient_id == patient.id)
                    .limit(1)
                )
                assoc = (await db.execute(assoc_stmt)).scalar_one_or_none()
                if not assoc:
                    raise ValidationAppError(
                        "Patient reference is not associated with a hospital node; cannot persist prediction."
                    )
                resolved_hospital_id = assoc.hospital_id

            model_version = await ECGReportInferenceService._get_or_create_model_version(
                db,
                version_tag=result["model_version_tag"],
                target_definition=result["target_definition"],
                artifact_uri=str(_model_dir() / "manifest.json"),
            )

            provenance = (
                f"idempotency:{body.idempotency_key}"
                if body.idempotency_key
                else result["provenance"]
            )

            prediction = DiagnosisPrediction(
                id=prediction_id,
                patient_id=patient.id,
                hospital_id=resolved_hospital_id,
                model_version_id=model_version.id,
                target_definition=result["target_definition"],
                prediction_type="proxy_evaluation",
                provenance=provenance,
                confidence=float(result["probability"]),
                predicted_class_index=int(result["predicted_class_index"]),
                status="evaluated",
            )
            db.add(prediction)
            await db.flush()

        return ECGReportEvaluateResponse(
            status="EVALUATED",
            prediction_id=prediction_id,
            patient_id=patient.id if patient else None,
            probability=float(result["probability"]),
            predicted_class_index=int(result["predicted_class_index"]),
            predicted_class_label=str(result["predicted_class_label"]),
            target_definition=result["target_definition"],
            architecture_id=result["architecture_id"],
            model_version_tag=result["model_version_tag"],
            is_research_only=True,
            disclaimer=result["disclaimer"],
            provenance=result["provenance"],
            privacy_validation_status="VALIDATED",
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
            model_family="MedShield-ECG-Report-Fusion",
            version_tag=version_tag,
            architecture_config="medshield.ecg_report_fusion.v1@1.0.0",
            target_definition=target_definition[:255],
            status="experimental",
            artifact_uri=artifact_uri[:512],
        )
        db.add(version)
        await db.flush()
        return version

    @staticmethod
    def _to_response(row: DiagnosisPrediction) -> ECGReportEvaluateResponse:
        prob = row.confidence or 0.0
        class_idx, label = ecg_idempotent_replay_class(row.predicted_class_index, row.confidence)
        return ECGReportEvaluateResponse(
            status="EVALUATED",
            prediction_id=row.id,
            patient_id=row.patient_id,
            probability=float(prob),
            predicted_class_index=class_idx,
            predicted_class_label=label,
            target_definition=row.target_definition,
            architecture_id="medshield.ecg_report_fusion.v1",
            model_version_tag="ecg-report-fusion-v1-ptbxl",
            is_research_only=True,
            disclaimer=(
                "EXPERIMENTAL RESEARCH PROXY OUTPUT ONLY. Replayed from idempotent execution."
            ),
            provenance=row.provenance,
            privacy_validation_status="VALIDATED",
        )
