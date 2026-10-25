"""Two-modality ECG + clinical report fusion (distinct from three-modality GAT)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.models.clinical_text_encoder import ClinicalTextEncoder, ClinicalTextEncoderConfig
from ml.models.ecg_encoder import ECGEncoder

ARCHITECTURE_ID = "medshield.ecg_report_fusion.v1"
ARCHITECTURE_VERSION = "1.0.0"
FUSION_INPUT_DIM = 256


@dataclass
class ECGReportFusionConfig:
    architecture_id: str = ARCHITECTURE_ID
    architecture_version: str = ARCHITECTURE_VERSION
    ecg_embed_dim: int = 128
    text_embed_dim: int = 128
    trained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ECGReportFusionClassifier(nn.Module):
    """ECG encoder + masked BioClinicalBERT text encoder → concat → binary logit."""

    def __init__(
        self,
        text_encoder: ClinicalTextEncoder,
        *,
        config: ECGReportFusionConfig | None = None,
        ecg_encoder: ECGEncoder | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ECGReportFusionConfig()
        self.ecg_encoder = ecg_encoder or ECGEncoder()
        self.text_encoder = text_encoder
        self.head = nn.Linear(FUSION_INPUT_DIM, 1)

    def forward(
        self,
        ecg: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ecg_emb = self.ecg_encoder(ecg)
        text_emb = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([ecg_emb, text_emb], dim=-1)
        logit = self.head(fused).squeeze(-1)
        return logit, ecg_emb, text_emb


def save_ecg_report_checkpoint(
    path: Path | str,
    model: ECGReportFusionClassifier,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    cfg = model.config
    payload = {
        "architecture_id": cfg.architecture_id,
        "architecture_version": cfg.architecture_version,
        "model_config": cfg.to_dict(),
        "framework": {"pytorch": torch.__version__},
        "trained": cfg.trained,
        "experiment": "two_modality_ecg_report_ptbxl",
        "state_dict": model.state_dict(),
        "metadata": extra_metadata or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_ecg_report_checkpoint(
    path: Path | str,
    text_encoder: ClinicalTextEncoder,
    *,
    map_location: str | torch.device = "cpu",
) -> ECGReportFusionClassifier:
    p = Path(path)
    try:
        payload = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=map_location)
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"Incompatible architecture_id: {payload.get('architecture_id')}")
    cfg = ECGReportFusionConfig(**payload.get("model_config", {}))
    model = ECGReportFusionClassifier(text_encoder, config=cfg)
    model.load_state_dict(payload["state_dict"])
    return model
