"""
Two-modality GAT fusion over ECG + clinical-report embeddings (128 + 128).

Graph: two nodes (ECG, clinical_text), fully connected directed edges including self-loops.
Two GATConv layers (4 heads, then 1 head) per MODEL_CONTRACT spirit; concat disabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

from ml.fusion.errors import FusionCheckpointError, FusionInputError

ARCHITECTURE_ID = "medshield.diagnostic_net.gat_ecg_report"
ARCHITECTURE_VERSION = "1.0.0"

ECG_EMBED_DIM = 128
TEXT_EMBED_DIM = 128
NUM_MODALITIES = 2
MODALITY_ORDER = ("ecg", "clinical_text")
NODE_DIM = 128
FLAT_FUSION_DIM = NUM_MODALITIES * NODE_DIM
HIDDEN_DIM = 128
NUM_CLASSES = 2

_FC_EDGE_INDEX = torch.tensor(
    [[i, j] for i in range(NUM_MODALITIES) for j in range(NUM_MODALITIES)],
    dtype=torch.long,
).t().contiguous()


@dataclass
class GATECGReportConfig:
    architecture_id: str = ARCHITECTURE_ID
    architecture_version: str = ARCHITECTURE_VERSION
    gat1_heads: int = 4
    gat2_heads: int = 1
    hidden_dropout: float = 0.3
    trained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GATECGReportInput:
    ecg: torch.Tensor
    clinical_text: torch.Tensor

    def __post_init__(self) -> None:
        if self.ecg.shape != self.clinical_text.shape:
            raise FusionInputError("ECG and clinical_text embeddings must share batch shape.")
        if self.ecg.dim() != 2 or self.ecg.size(-1) != ECG_EMBED_DIM:
            raise FusionInputError(f"ECG must be [B, {ECG_EMBED_DIM}].")
        if self.clinical_text.size(-1) != TEXT_EMBED_DIM:
            raise FusionInputError(f"clinical_text must be [B, {TEXT_EMBED_DIM}].")
        if not torch.isfinite(self.ecg).all() or not torch.isfinite(self.clinical_text).all():
            raise FusionInputError("Non-finite embedding values.")


def _build_batched_graph(node_tensor: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, f = node_tensor.shape
    x = node_tensor.reshape(b * n, f)
    edge = edge_index.to(node_tensor.device)
    batched = torch.cat([edge + off * n for off in range(b)], dim=1)
    return x, batched


class MedShieldGATECGReportNet(nn.Module):
    def __init__(self, config: GATECGReportConfig | None = None) -> None:
        super().__init__()
        self.config = config or GATECGReportConfig()
        if self.config.gat1_heads != 4 or self.config.gat2_heads != 1:
            raise ValueError("GAT head configuration must be (4, 1).")
        self.proj_ecg = nn.Linear(ECG_EMBED_DIM, NODE_DIM)
        self.proj_text = nn.Linear(TEXT_EMBED_DIM, NODE_DIM)
        self.gat1 = GATConv(NODE_DIM, NODE_DIM, heads=4, concat=False, dropout=0.0)
        self.gat2 = GATConv(NODE_DIM, NODE_DIM, heads=1, concat=False, dropout=0.0)
        self.classifier_hidden = nn.Sequential(
            nn.Linear(FLAT_FUSION_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.config.hidden_dropout),
        )
        self.classifier_out = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        self.register_buffer("fc_edge_index", _FC_EDGE_INDEX, persistent=False)

    def build_graph_nodes(self, ecg: torch.Tensor, clinical_text: torch.Tensor) -> torch.Tensor:
        n0 = self.proj_ecg(ecg)
        n1 = self.proj_text(clinical_text)
        nodes = torch.stack((n0, n1), dim=1)
        if nodes.shape != (ecg.size(0), NUM_MODALITIES, NODE_DIM):
            raise FusionInputError(f"Expected nodes [B, 2, 128], got {tuple(nodes.shape)}.")
        return nodes

    def forward_gat_stack(self, nodes: torch.Tensor) -> torch.Tensor:
        x, edge_index = _build_batched_graph(nodes, self.fc_edge_index)
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        b = nodes.size(0)
        return x.reshape(b, NUM_MODALITIES, NODE_DIM)

    def forward(self, fusion_input: GATECGReportInput) -> torch.Tensor:
        ecg = fusion_input.ecg
        text = fusion_input.clinical_text
        dev = next(self.parameters()).device
        if ecg.device != dev:
            ecg = ecg.to(dev)
            text = text.to(dev)
        nodes = self.build_graph_nodes(ecg, text)
        nodes = self.forward_gat_stack(nodes)
        flat = nodes.reshape(nodes.size(0), FLAT_FUSION_DIM)
        hidden = self.classifier_hidden(flat)
        logits = self.classifier_out(hidden)
        if not torch.isfinite(logits).all():
            raise FusionInputError("Non-finite logits from GAT head.")
        return logits


def save_gat_ecg_report_checkpoint(path: Path | str, model: MedShieldGATECGReportNet, *, extra: dict | None = None) -> None:
    cfg = model.config
    payload = {
        "architecture_id": cfg.architecture_id,
        "architecture_version": cfg.architecture_version,
        "model_config": cfg.to_dict(),
        "modality_order": list(MODALITY_ORDER),
        "input_dims": {"ecg": ECG_EMBED_DIM, "clinical_text": TEXT_EMBED_DIM},
        "framework": {"pytorch": torch.__version__},
        "trained": cfg.trained,
        "state_dict": model.state_dict(),
        "metadata": extra or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_gat_ecg_report_checkpoint(path: Path | str, *, map_location: str | torch.device = "cpu") -> MedShieldGATECGReportNet:
    p = Path(path)
    if not p.is_file():
        raise FusionCheckpointError(f"Missing checkpoint: {p}")
    try:
        payload = torch.load(p, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location=map_location)
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise FusionCheckpointError(f"Incompatible architecture_id: {payload.get('architecture_id')}")
    cfg = GATECGReportConfig(**payload.get("model_config", {}))
    model = MedShieldGATECGReportNet(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def checkpoint_summary(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "architecture_id": payload.get("architecture_id"),
        "architecture_version": payload.get("architecture_version"),
        "trained": payload.get("trained", False),
        "modality_order": payload.get("modality_order"),
    }
