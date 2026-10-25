"""
Three-modality GAT fusion pipeline definitions and contracts (Prompt 15 / Prompt 16 / Path B).

Combines:
1. ECG: 12-lead (12, 1000) mV waveform -> ECGEmbeddingProvider (128-d)
2. Clinical text: Masked clinical note/report -> ClinicalTextEncodingPipeline (128-d)
3. Tabular: 10 structured features -> NeuralTabularEncoder (64-d)
Through MedShieldDiagnosticNet GAT fusion (3 nodes -> 2 GATConv layers -> binary classification).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ml.fusion.errors import FusionInputError
from ml.inference.constants import ECG_EMBED_DIM, TABULAR_EMBED_DIM, TEXT_EMBED_DIM
from ml.models.gnn_fusion import (
    ARCHITECTURE_ID,
    ARCHITECTURE_VERSION,
    FusionModelInput,
    MedShieldDiagnosticNet,
    ModalityIndex,
    NUM_MODALITIES,
)
from ml.training.linkage_gate import evaluate_three_modality_gate, LinkageGateResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalityContract:
    name: str
    dimension: int
    input_format: str
    preprocessing: str
    missing_allowed: bool


MODALITY_CONTRACTS: dict[str, ModalityContract] = {
    "ecg": ModalityContract(
        name="ecg",
        dimension=ECG_EMBED_DIM,  # 128
        input_format="12 leads x 1000 samples @ 100Hz in mV",
        preprocessing="Per-lead z-score normalization using training-fitted stats",
        missing_allowed=True,
    ),
    "clinical_text": ModalityContract(
        name="clinical_text",
        dimension=TEXT_EMBED_DIM,  # 128
        input_format="Unstructured clinical text string",
        preprocessing="Privacy-first regex PII masking + BioClinicalBERT CLS projection",
        missing_allowed=True,
    ),
    "tabular": ModalityContract(
        name="tabular",
        dimension=TABULAR_EMBED_DIM,  # 64
        input_format="10 structured features (medshield.tabular.cardio_survey v1.0.0)",
        preprocessing="Median imputation + standard scaling + NeuralTabularEncoder",
        missing_allowed=True,
    ),
}


@dataclass
class GATReadinessReport:
    """Safe diagnostics describing three-modality GAT pipeline readiness."""

    ready: bool
    architecture_id: str = ARCHITECTURE_ID
    architecture_version: str = ARCHITECTURE_VERSION
    embedding_contracts: dict[str, int] = field(
        default_factory=lambda: {
            "ecg": ECG_EMBED_DIM,
            "clinical_text": TEXT_EMBED_DIM,
            "tabular": TABULAR_EMBED_DIM,
        }
    )
    linkage_gate_allowed: bool = False
    linkage_gate_verdict: str = "BLOCKED"
    blockers: list[str] = field(default_factory=list)
    required_to_unblock: list[str] = field(default_factory=list)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "architecture_id": self.architecture_id,
            "architecture_version": self.architecture_version,
            "embedding_contracts": dict(self.embedding_contracts),
            "linkage_gate_allowed": self.linkage_gate_allowed,
            "linkage_gate_verdict": self.linkage_gate_verdict,
            "blockers": list(self.blockers),
            "required_to_unblock": list(self.required_to_unblock),
        }


class MultimodalGATPipeline:
    """
    Manages multimodal ingestion, embedding contract validation, missingness handling,
    and forward propagation through MedShieldDiagnosticNet.
    """

    def __init__(
        self,
        gat_model: MedShieldDiagnosticNet | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gat_model = gat_model
        if self.gat_model is not None:
            self.gat_model.to(self.device)
            self.gat_model.eval()

    @staticmethod
    def assess_gat_pipeline_readiness() -> GATReadinessReport:
        """
        Assess whether three-modality GAT training and inference can be executed safely.
        """
        gate = evaluate_three_modality_gate()
        blockers: list[str] = []
        required: list[str] = []

        if not gate.allowed:
            blockers.append(
                f"Data linkage gate: {gate.verdict} — No cross-cohort patient linkage between PTB-XL, cardio_train, and NBME."
            )
            required.append(
                "Authorized multimodal dataset with shared subject/encounter keys across all 3 modalities (e.g. MIMIC-IV-ECG + MIMIC-IV)."
            )
            required.append("Clinically verified unified CVD target label across modalities.")
            required.append("Patient-disjoint train/validation/test split protocol on linked records.")

        blockers.append("GAT model weights not trained on linked data (research architecture only).")
        required.append("Execution of supervised multimodal training with verified loss convergence.")

        return GATReadinessReport(
            ready=False,
            linkage_gate_allowed=gate.allowed,
            linkage_gate_verdict=gate.verdict,
            blockers=blockers,
            required_to_unblock=required,
        )

    def build_fusion_input(
        self,
        *,
        ecg_emb: Sequence[float] | None = None,
        text_emb: Sequence[float] | None = None,
        tabular_emb: Sequence[float] | None = None,
    ) -> FusionModelInput:
        """
        Build a single-instance (batch=1) FusionModelInput with validated contracts and masks.
        """
        available = [ecg_emb is not None, text_emb is not None, tabular_emb is not None]
        if not any(available):
            raise FusionInputError("At least one modality must be available for multimodal evaluation.")

        # Validate dimensions and values
        ecg_tensor = None
        if ecg_emb is not None:
            if len(ecg_emb) != ECG_EMBED_DIM:
                raise FusionInputError(f"ECG embedding dimension must be {ECG_EMBED_DIM}, got {len(ecg_emb)}")
            arr = np.asarray(ecg_emb, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise FusionInputError("ECG embedding contains non-finite values.")
            ecg_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        text_tensor = None
        if text_emb is not None:
            if len(text_emb) != TEXT_EMBED_DIM:
                raise FusionInputError(f"Text embedding dimension must be {TEXT_EMBED_DIM}, got {len(text_emb)}")
            arr = np.asarray(text_emb, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise FusionInputError("Text embedding contains non-finite values.")
            text_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        tab_tensor = None
        if tabular_emb is not None:
            if len(tabular_emb) != TABULAR_EMBED_DIM:
                raise FusionInputError(f"Tabular embedding dimension must be {TABULAR_EMBED_DIM}, got {len(tabular_emb)}")
            arr = np.asarray(tabular_emb, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                raise FusionInputError("Tabular embedding contains non-finite values.")
            tab_tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        available_mask = torch.tensor([available], dtype=torch.bool, device=self.device)
        quality_mask = available_mask.clone()

        return FusionModelInput(
            ecg=ecg_tensor,
            clinical_text=text_tensor,
            tabular=tab_tensor,
            available=available_mask,
            quality_ok=quality_mask,
        )
