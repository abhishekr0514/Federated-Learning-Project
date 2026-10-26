"""Two-modality GAT ECG + clinical report fusion service."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.errors import InferenceNotReadyError, ValidationAppError
from backend.app.models.identity import ClinicianProfile
from backend.app.models.ml_models import DiagnosisPrediction, ModelVersion
from backend.app.models.patient import Patient, PatientHospitalAssociation
from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)
from backend.app.services.prediction_proxy_class import gat_idempotent_replay_class
from ml.fusion.pipeline import MultimodalGATPipeline
from ml.inference.fusion_runtime import get_fusion_inference_runtime

logger = logging.getLogger(__name__)


def _fusion_model_dir() -> Path:
    root = Path(settings.FUSION_MODEL_DIR)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root
    return root


def _get_runtime():
    return get_fusion_inference_runtime(
        _fusion_model_dir(),
        inference_enabled=settings.FUSION_INFERENCE_ENABLED,
    )


class GATFusionService:
    @staticmethod
    def initialize_runtime() -> None:
        """Eager-load GAT weights when inference is enabled (optional warm-up)."""
        if not settings.FUSION_INFERENCE_ENABLED:
            return
        runtime = _get_runtime()
        readiness = runtime.assess_readiness()
        if not readiness.ready:
            logger.info("GAT fusion warm-up skipped: %s", "; ".join(readiness.blockers[:3]))
            return
        try:
            runtime.run_proxy_inference(
                ecg=[0.0] * 128,
                clinical_text=[0.0] * 128,
            )
            logger.info("GAT fusion runtime warm-up complete.")
        except Exception as exc:
            logger.warning("GAT fusion warm-up failed: %s", type(exc).__name__)

    @staticmethod
    def get_readiness() -> Dict[str, Any]:
        pipeline_report = MultimodalGATPipeline.assess_gat_pipeline_readiness()
        runtime = _get_runtime()
        runtime_readiness = runtime.assess_readiness()

        blockers = list(pipeline_report.blockers) + list(runtime_readiness.blockers)
        if not settings.FUSION_INFERENCE_ENABLED:
            if "FUSION_INFERENCE_ENABLED is false" not in blockers:
                blockers.append("FUSION_INFERENCE_ENABLED is false")

        ready = (
            pipeline_report.linkage_gate_allowed
            and runtime_readiness.ready
            and settings.FUSION_INFERENCE_ENABLED
        )

        payload = GATFusionReadinessResponse(
            ready=ready,
            architecture_id=pipeline_report.architecture_id,
            architecture_version=pipeline_report.architecture_version,
            embedding_contracts=pipeline_report.embedding_contracts,
            manifest_present=runtime_readiness.manifest_present,
            checkpoint_present=runtime_readiness.checkpoint_present,
            checkpoint_trained=runtime_readiness.checkpoint_trained,
            model_loaded=runtime_readiness.model_loaded,
            pairing_gate_allowed=pipeline_report.linkage_gate_allowed,
            pairing_gate_verdict=pipeline_report.linkage_gate_verdict,
            inference_enabled=settings.FUSION_INFERENCE_ENABLED,
            blockers=blockers if not ready else [],
            required_to_unblock=list(pipeline_report.required_to_unblock),
            environment=settings.ENVIRONMENT,
        )
        return payload.model_dump()

    @staticmethod
    async def evaluate(
        db: Optional[AsyncSession],
        *,
        body: GATFusionEvaluateRequest,
        profile: Optional[ClinicianProfile] = None,
        patient: Optional[Patient] = None,
        hospital_id: Optional[uuid.UUID] = None,
    ) -> GATFusionEvaluateResponse:
        runtime = _get_runtime()
        readiness = runtime.assess_readiness()
        if not readiness.ready:
            raise InferenceNotReadyError(
                "Two-modality GAT fusion is not ready.",
                details=readiness.to_safe_dict(),
            )

        if patient and db and body.idempotency_key:
            existing = await GATFusionService._find_idempotent_prediction(
                db, patient.id, body.idempotency_key
            )
            if existing:
                return GATFusionService._to_response(existing, result_architecture_id=None)

        try:
            result = runtime.run_proxy_inference(
                ecg=body.ecg_embedding,
                clinical_text=body.text_embedding,
            )
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc
        except RuntimeError as exc:
            raise InferenceNotReadyError(str(exc)) from exc
        except Exception as exc:
            logger.error("GAT fusion inference failed: %s", type(exc).__name__)
            raise ValidationAppError("GAT fusion evaluation failed.") from exc

        prediction_id = uuid.uuid4()
        architecture_id = str(
            result.get("architecture_id", "medshield.diagnostic_net.gat_ecg_report")
        )

        if patient and db:
            if hospital_id is None:
                hospital_id = await GATFusionService._resolve_hospital_id(db, patient, profile)
            model_version = await GATFusionService._get_or_create_model_version(
                db,
                version_tag=result["model_version_tag"],
                target_definition=result["target_definition"],
                artifact_uri=str(_fusion_model_dir() / "medshield_fusion_manifest.json"),
            )
            provenance = (
                f"idempotency:{body.idempotency_key}"
                if body.idempotency_key
                else "gat_fusion_inference_pipeline"
            )
            prediction = DiagnosisPrediction(
                id=prediction_id,
                patient_id=patient.id,
                hospital_id=hospital_id,
                model_version_id=model_version.id,
                target_definition=result["target_definition"],
                prediction_type="proxy_evaluation",
                provenance=provenance,
                confidence=float(result["confidence_uncalibrated"]),
                predicted_class_index=int(result["predicted_class_index"]),
                status="evaluated",
            )
            db.add(prediction)
            await db.flush()

        return GATFusionEvaluateResponse(
            status="EVALUATED",
            prediction_id=prediction_id,
            patient_id=patient.id if patient else None,
            confidence_uncalibrated=float(result["confidence_uncalibrated"]),
            predicted_class_index=int(result["predicted_class_index"]),
            predicted_class_label=str(result["predicted_class_label"]),
            target_definition=result["target_definition"],
            architecture_id=architecture_id,
            model_version_tag=result["model_version_tag"],
            disclaimer=result["disclaimer"],
            proxy_task=patient is None,
        )

    @staticmethod
    async def _resolve_hospital_id(
        db: AsyncSession,
        patient: Patient,
        profile: Optional[ClinicianProfile],
    ) -> uuid.UUID:
        if profile is not None:
            return profile.hospital_id
        stmt = (
            select(PatientHospitalAssociation)
            .where(PatientHospitalAssociation.patient_id == patient.id)
            .limit(1)
        )
        assoc = (await db.execute(stmt)).scalar_one_or_none()
        if not assoc:
            raise ValidationAppError(
                "Patient reference is not associated with a hospital node; cannot persist prediction."
            )
        return assoc.hospital_id

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
            model_family="MedShield-GAT-ECG-Report-Fusion",
            version_tag=version_tag,
            architecture_config="medshield.diagnostic_net.gat_ecg_report@1.0.0",
            target_definition=target_definition[:255],
            status="experimental",
            artifact_uri=artifact_uri[:512],
        )
        db.add(version)
        await db.flush()
        return version

    @staticmethod
    def _to_response(
        row: DiagnosisPrediction,
        *,
        result_architecture_id: Optional[str],
    ) -> GATFusionEvaluateResponse:
        prob = row.confidence or 0.0
        class_idx, label = gat_idempotent_replay_class(row.predicted_class_index)
        return GATFusionEvaluateResponse(
            status="EVALUATED",
            prediction_id=row.id,
            patient_id=row.patient_id,
            confidence_uncalibrated=float(prob),
            predicted_class_index=class_idx,
            predicted_class_label=label,
            target_definition=row.target_definition,
            architecture_id=result_architecture_id or "medshield.diagnostic_net.gat_ecg_report",
            model_version_tag="gat-ecg-report-fusion",
            disclaimer=(
                "EXPERIMENTAL RESEARCH PROXY OUTPUT ONLY. Replayed from idempotent execution."
            ),
            proxy_task=False,
        )
