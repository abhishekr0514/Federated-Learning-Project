"""Service layer for the three-modality GAT fusion proxy inference endpoint.

Architecture: medshield.diagnostic_net.gat_fusion
Research proxy task: Binary CVD classification from fused ECG + text + tabular embeddings.
All outputs are experimental and must not be used for clinical diagnosis.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.errors import InferenceNotReadyError, ValidationAppError
from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)
from ml.fusion.pipeline import GATReadinessReport, MultimodalGATPipeline
from ml.inference.fusion_runtime import (
    FusionInferenceRuntime,
    get_fusion_inference_runtime,
)

logger = logging.getLogger(__name__)


def _fusion_model_dir() -> Path:
    root = Path(settings.FUSION_MODEL_DIR)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = repo_root / root
    return root


def _get_runtime() -> FusionInferenceRuntime:
    return get_fusion_inference_runtime(
        _fusion_model_dir(),
        inference_enabled=settings.FUSION_INFERENCE_ENABLED,
    )


class GATFusionService:
    """Service for three-modality GAT fusion proxy evaluation."""

    @staticmethod
    def get_readiness() -> Dict[str, Any]:
        """
        Return combined GAT pipeline readiness diagnostics.

        Merges pipeline-level linkage gate checks with runtime-level model readiness.
        Does not expose filesystem paths or internal secrets.
        """
        # 1. Pipeline-level gate assessment (data linkage, architecture contracts)
        pipeline_report: GATReadinessReport = MultimodalGATPipeline.assess_gat_pipeline_readiness()

        # 2. Runtime-level model readiness (manifest, checkpoint file, torch)
        runtime = _get_runtime()
        runtime_readiness = runtime.assess_readiness()

        blockers: List[str] = list(pipeline_report.blockers) + list(runtime_readiness.blockers)

        payload = GATFusionReadinessResponse(
            # If the GAT model is not yet trained, it can never be "ready" for production,
            # but we still surface runtime state truthfully for diagnostics.
            ready=False,
            architecture_id=pipeline_report.architecture_id,
            architecture_version=pipeline_report.architecture_version,
            embedding_contracts=pipeline_report.embedding_contracts,
            manifest_present=runtime_readiness.manifest_present,
            checkpoint_present=runtime_readiness.checkpoint_present,
            checkpoint_trained=runtime_readiness.checkpoint_trained,
            model_loaded=runtime_readiness.model_loaded,
            linkage_gate_allowed=pipeline_report.linkage_gate_allowed,
            linkage_gate_verdict=pipeline_report.linkage_gate_verdict,
            blockers=blockers,
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
        """
        Run a three-modality GAT fusion forward pass on pre-computed embedding vectors.

        At least one of ecg_embedding, text_embedding, or tabular_embedding must be
        provided. Missing modalities are zero-masked inside the GAT model.

        The model must carry FUSION_INFERENCE_ENABLED=true AND a valid trained checkpoint
        in FUSION_MODEL_DIR for this call to succeed. Currently the GAT model is
        architecture-only (untrained); the runtime will raise InferenceNotReadyError
        unless a verified trained checkpoint is loaded.
        """
        runtime = _get_runtime()
        readiness = runtime.assess_readiness()

        if not readiness.ready:
            raise InferenceNotReadyError(
                "Three-modality GAT fusion model is not ready. "
                "Blockers: " + "; ".join(readiness.blockers),
                details=readiness.to_safe_dict(),
            )

        # Track which modalities are present
        modalities_used: List[str] = []
        if body.ecg_embedding is not None:
            modalities_used.append("ecg")
        if body.text_embedding is not None:
            modalities_used.append("clinical_text")
        if body.tabular_embedding is not None:
            modalities_used.append("tabular")

        try:
            result = runtime.run_proxy_inference(
                ecg=body.ecg_embedding or [0.0] * 128,
                clinical_text=body.text_embedding or [0.0] * 128,
                tabular=body.tabular_embedding or [0.0] * 64,
            )
        except ValueError as exc:
            raise ValidationAppError(str(exc)) from exc
        except RuntimeError as exc:
            raise InferenceNotReadyError(str(exc)) from exc
        except Exception as exc:
            logger.error("GAT fusion inference failed: %s", type(exc).__name__)
            raise ValidationAppError("GAT fusion model evaluation failed on provided inputs.") from exc

        return GATFusionEvaluateResponse(
            status="EVALUATED",
            prediction_id=uuid.uuid4(),
            patient_id=patient_id,
            confidence_uncalibrated=float(result["confidence_uncalibrated"]),
            predicted_class_index=int(result["predicted_class_index"]),
            predicted_class_label=str(result.get("predicted_class_label", str(result["predicted_class_index"]))),
            target_definition=result["target_definition"],
            architecture_id="medshield.diagnostic_net.gat_fusion",
            model_version_tag=result["model_version_tag"],
            modalities_used=modalities_used,
            is_research_only=True,
            proxy_task=True,
            disclaimer=result["disclaimer"],
        )
