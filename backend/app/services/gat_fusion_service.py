"""Two-modality GAT ECG + clinical report fusion service."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.core.errors import InferenceNotReadyError, ValidationAppError
from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)
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
        *,
        body: GATFusionEvaluateRequest,
        patient_id: Optional[uuid.UUID] = None,
    ) -> GATFusionEvaluateResponse:
        runtime = _get_runtime()
        readiness = runtime.assess_readiness()
        if not readiness.ready:
            raise InferenceNotReadyError(
                "Two-modality GAT fusion is not ready.",
                details=readiness.to_safe_dict(),
            )

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

        return GATFusionEvaluateResponse(
            status="EVALUATED",
            prediction_id=uuid.uuid4(),
            patient_id=patient_id,
            confidence_uncalibrated=float(result["confidence_uncalibrated"]),
            predicted_class_index=int(result["predicted_class_index"]),
            predicted_class_label=str(result["predicted_class_label"]),
            target_definition=result["target_definition"],
            architecture_id=str(result.get("architecture_id", "medshield.diagnostic_net.gat_ecg_report")),
            model_version_tag=result["model_version_tag"],
            disclaimer=result["disclaimer"],
        )
