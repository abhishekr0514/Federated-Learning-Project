"""
MedShieldDiagnosticNet — GAT multimodal fusion (research architecture only).

Implements docs/MODEL_CONTRACT.md §4. Not trained; not for clinical diagnosis or production inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for GAT fusion. Install with: pip install -e '.[ml]'"
    ) from exc

from ml.fusion.errors import FusionCheckpointError, FusionInputError

ARCHITECTURE_ID = "medshield.diagnostic_net.gat_fusion"
ARCHITECTURE_VERSION = "1.0.0"

ECG_EMBED_DIM = 128
TEXT_EMBED_DIM = 128
TABULAR_EMBED_DIM = 64
NODE_DIM = 128
NUM_MODALITIES = 3
FLAT_FUSION_DIM = NUM_MODALITIES * NODE_DIM  # 384
HIDDEN_DIM = 128
NUM_CLASSES = 2

MODALITY_ORDER: tuple[str, ...] = ("ecg", "clinical_text", "tabular")

# Fully connected directed edges over 3 nodes (including self-loops).
_FC_EDGE_INDEX = torch.tensor(
    [[i, j] for i in range(NUM_MODALITIES) for j in range(NUM_MODALITIES)],
    dtype=torch.long,
).t().contiguous()


class ModalityIndex(IntEnum):
    ECG = 0
    CLINICAL_TEXT = 1
    TABULAR = 2


@dataclass
class FusionModelInput:
    """Batch of encoder embeddings plus explicit availability and quality masks."""

    ecg: torch.Tensor | None
    clinical_text: torch.Tensor | None
    tabular: torch.Tensor | None
    available: torch.Tensor
    quality_ok: torch.Tensor

    def __post_init__(self) -> None:
        if self.available.shape != self.quality_ok.shape:
            raise FusionInputError("available and quality_ok must have the same shape.")
        if self.available.dim() != 2 or self.available.size(-1) != NUM_MODALITIES:
            raise FusionInputError(f"Masks must be [B, {NUM_MODALITIES}].")


@dataclass
class FusionResearchOutput:
    """Research-only forward result; softmax confidence is uncalibrated."""

    logits: torch.Tensor | None
    probabilities: torch.Tensor | None
    predicted_class: torch.Tensor | None
    confidence: torch.Tensor | None
    modality_used: torch.Tensor
    available: torch.Tensor
    quality_ok: torch.Tensor
    complete: torch.Tensor
    incomplete_reasons: list[list[str]]
    uncalibrated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "logits": None if self.logits is None else self.logits.detach().cpu().tolist(),
            "probabilities": None
            if self.probabilities is None
            else self.probabilities.detach().cpu().tolist(),
            "predicted_class": None
            if self.predicted_class is None
            else self.predicted_class.detach().cpu().tolist(),
            "confidence": None if self.confidence is None else self.confidence.detach().cpu().tolist(),
            "modality_used": self.modality_used.detach().cpu().tolist(),
            "complete": self.complete.detach().cpu().tolist(),
            "uncalibrated": self.uncalibrated,
            "incomplete_reasons": self.incomplete_reasons,
        }


@dataclass
class FusionModelConfig:
    architecture_id: str = ARCHITECTURE_ID
    architecture_version: str = ARCHITECTURE_VERSION
    modality_order: tuple[str, ...] = MODALITY_ORDER
    ecg_embed_dim: int = ECG_EMBED_DIM
    text_embed_dim: int = TEXT_EMBED_DIM
    tabular_embed_dim: int = TABULAR_EMBED_DIM
    node_dim: int = NODE_DIM
    gat1_heads: int = 4
    gat2_heads: int = 1
    hidden_dropout: float = 0.3
    ecg_protocol_version: str | None = None
    tabular_schema_version: str | None = None
    text_encoder_version: str | None = None
    trained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _batch_size_from_input(fusion_input: FusionModelInput) -> int:
    for tensor in (fusion_input.ecg, fusion_input.clinical_text, fusion_input.tabular):
        if tensor is not None:
            return int(tensor.size(0))
    return int(fusion_input.available.size(0))


def _validate_embedding(name: str, tensor: torch.Tensor | None, expected_last: int, batch: int) -> None:
    if tensor is None:
        raise FusionInputError(f"{name} embedding tensor is required when modality is marked available.")
    if tensor.dim() != 2 or tensor.size(0) != batch or tensor.size(1) != expected_last:
        raise FusionInputError(
            f"{name} must have shape ({batch}, {expected_last}), got {tuple(tensor.shape)}."
        )
    if not torch.isfinite(tensor).all():
        raise FusionInputError(f"{name} contains non-finite values.")


def _build_batched_graph(node_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """node_tensor [B, 3, 128] -> PyG batched node features and edges."""
    b, n, f = node_tensor.shape
    x = node_tensor.reshape(b * n, f)
    edge_index = _FC_EDGE_INDEX.to(node_tensor.device)
    offsets = torch.arange(b, device=node_tensor.device) * n
    batched_edges = torch.cat(
        [edge_index + off for off in offsets],
        dim=1,
    )
    batch = torch.arange(b, device=node_tensor.device).repeat_interleave(n)
    return x, batched_edges, batch


def _modality_tensors_present(fusion_input: FusionModelInput, batch: int, device: torch.device) -> torch.Tensor:
    present = torch.zeros(batch, NUM_MODALITIES, dtype=torch.bool, device=device)
    if fusion_input.ecg is not None and fusion_input.ecg.size(0) == batch:
        present[:, ModalityIndex.ECG] = True
    if fusion_input.clinical_text is not None and fusion_input.clinical_text.size(0) == batch:
        present[:, ModalityIndex.CLINICAL_TEXT] = True
    if fusion_input.tabular is not None and fusion_input.tabular.size(0) == batch:
        present[:, ModalityIndex.TABULAR] = True
    return present


def _unsupported_reasons(
    available: torch.Tensor,
    quality_ok: torch.Tensor,
) -> list[list[str]]:
    b = available.size(0)
    reasons: list[list[str]] = []
    for i in range(b):
        row: list[str] = []
        for j, name in enumerate(MODALITY_ORDER):
            if available[i, j].item() == 0:
                row.append(f"{name}:unavailable")
            elif quality_ok[i, j].item() == 0:
                row.append(f"{name}:quality_rejected")
            # tensor presence checked separately via modality_used
        reasons.append(row)
    return reasons


def validate_full_modality_batch(fusion_input: FusionModelInput) -> None:
    """
    Full GAT fusion requires all modalities available and quality-accepted for every sample.

    Missing-modality imputation is not implemented (no contract-approved strategy).
    """
    batch = _batch_size_from_input(fusion_input)
    if fusion_input.available.size(0) != batch:
        raise FusionInputError("Batch size mismatch between embeddings and availability mask.")

    avail = fusion_input.available.to(dtype=torch.bool)
    quality = fusion_input.quality_ok.to(dtype=torch.bool)

    if not avail.all() or not quality.all():
        reasons = _unsupported_reasons(avail, quality)
        raise FusionInputError(
            "Unsupported modality pattern: full three-modality fusion requires all modalities "
            f"available and quality_ok. Details: {reasons}"
        )

    _validate_embedding("ecg", fusion_input.ecg, ECG_EMBED_DIM, batch)
    _validate_embedding("clinical_text", fusion_input.clinical_text, TEXT_EMBED_DIM, batch)
    _validate_embedding("tabular", fusion_input.tabular, TABULAR_EMBED_DIM, batch)


class MedShieldDiagnosticNet(nn.Module):
    """
    Contracted GAT fusion head over precomputed modality embeddings.

    Node 0: ECG, Node 1: clinical text, Node 2: tabular (after Linear(64,128) projection).
    """

    def __init__(self, config: FusionModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionModelConfig()
        if self.config.gat1_heads != 4 or self.config.gat2_heads != 1:
            raise ValueError("MODEL_CONTRACT requires GAT heads (4, 1).")

        self.proj_ecg = nn.Linear(ECG_EMBED_DIM, NODE_DIM)
        self.proj_text = nn.Linear(TEXT_EMBED_DIM, NODE_DIM)
        self.proj_tabular = nn.Linear(TABULAR_EMBED_DIM, NODE_DIM)

        self.gat1 = GATConv(
            in_channels=NODE_DIM,
            out_channels=NODE_DIM,
            heads=self.config.gat1_heads,
            concat=False,
            dropout=0.0,
        )
        self.gat2 = GATConv(
            in_channels=NODE_DIM,
            out_channels=NODE_DIM,
            heads=self.config.gat2_heads,
            concat=False,
            dropout=0.0,
        )
        self.classifier_hidden = nn.Sequential(
            nn.Linear(FLAT_FUSION_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.config.hidden_dropout),
        )
        self.classifier_out = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def build_graph_nodes(
        self,
        ecg: torch.Tensor,
        clinical_text: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        n0 = self.proj_ecg(ecg)
        n1 = self.proj_text(clinical_text)
        n2 = self.proj_tabular(tabular)
        nodes = torch.stack((n0, n1, n2), dim=1)
        if nodes.shape != (ecg.size(0), NUM_MODALITIES, NODE_DIM):
            raise FusionInputError(f"Expected graph nodes [B, 3, 128], got {tuple(nodes.shape)}.")
        return nodes

    def forward_gat_stack(self, nodes: torch.Tensor) -> torch.Tensor:
        x, edge_index, _batch = _build_batched_graph(nodes)
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        b = nodes.size(0)
        return x.view(b, NUM_MODALITIES, NODE_DIM)

    def forward_head(self, nodes_after_gat: torch.Tensor) -> torch.Tensor:
        flat = nodes_after_gat.reshape(nodes_after_gat.size(0), FLAT_FUSION_DIM)
        hidden = self.classifier_hidden(flat)
        logits = self.classifier_out(hidden)
        if not torch.isfinite(logits).all():
            raise FusionInputError("Non-finite logits from fusion head.")
        return logits

    def forward(self, fusion_input: FusionModelInput) -> FusionResearchOutput:
        validate_full_modality_batch(fusion_input)
        batch = _batch_size_from_input(fusion_input)
        avail = fusion_input.available.to(dtype=torch.bool)
        quality = fusion_input.quality_ok.to(dtype=torch.bool)
        present = _modality_tensors_present(fusion_input, batch, avail.device)
        modality_used = avail & quality & present

        ecg = fusion_input.ecg
        clinical_text = fusion_input.clinical_text
        tabular = fusion_input.tabular
        assert ecg is not None and clinical_text is not None and tabular is not None

        nodes = self.build_graph_nodes(ecg, clinical_text, tabular)
        nodes = self.forward_gat_stack(nodes)
        logits = self.forward_head(nodes)

        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all():
            raise FusionInputError("Non-finite probabilities.")
        confidence, predicted = probs.max(dim=-1)

        complete = torch.ones(batch, dtype=torch.bool, device=logits.device)
        return FusionResearchOutput(
            logits=logits,
            probabilities=probs,
            predicted_class=predicted,
            confidence=confidence,
            modality_used=modality_used,
            available=avail,
            quality_ok=quality,
            complete=complete,
            incomplete_reasons=[[] for _ in range(batch)],
            uncalibrated=True,
        )

    def describe_modality_status(self, fusion_input: FusionModelInput) -> FusionResearchOutput:
        """
        Report availability/quality without running GAT when fusion is unsupported.
        Never marks missing modalities as used.
        """
        batch = _batch_size_from_input(fusion_input)
        avail = fusion_input.available.to(dtype=torch.bool)
        quality = fusion_input.quality_ok.to(dtype=torch.bool)
        present = _modality_tensors_present(fusion_input, batch, avail.device)
        modality_used = avail & quality & present
        reasons = _unsupported_reasons(avail, quality)
        for i in range(batch):
            for j, name in enumerate(MODALITY_ORDER):
                if avail[i, j] and quality[i, j] and not present[i, j]:
                    reasons[i].append(f"{name}:embedding_missing")
        complete = modality_used.all(dim=1)
        return FusionResearchOutput(
            logits=None,
            probabilities=None,
            predicted_class=None,
            confidence=None,
            modality_used=modality_used,
            available=avail,
            quality_ok=quality,
            complete=complete,
            incomplete_reasons=reasons,
            uncalibrated=True,
        )


def save_fusion_checkpoint(
    path: Path,
    model: MedShieldDiagnosticNet,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    cfg = model.config
    payload = {
        "architecture_id": cfg.architecture_id,
        "architecture_version": cfg.architecture_version,
        "model_config": cfg.to_dict(),
        "modality_order": list(MODALITY_ORDER),
        "input_dims": {
            "ecg": ECG_EMBED_DIM,
            "clinical_text": TEXT_EMBED_DIM,
            "tabular": TABULAR_EMBED_DIM,
        },
        "node_dim": NODE_DIM,
        "framework": {"pytorch": torch.__version__},
        "trained": cfg.trained,
        "state_dict": model.state_dict(),
        "metadata": extra_metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_fusion_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> MedShieldDiagnosticNet:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise FusionCheckpointError(
            f"Incompatible architecture_id: {payload.get('architecture_id')!r}"
        )
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise FusionCheckpointError(
            f"Incompatible architecture_version: {payload.get('architecture_version')!r}"
        )
    cfg_dict = payload.get("model_config") or {}
    cfg = FusionModelConfig(**{k: v for k, v in cfg_dict.items() if k in FusionModelConfig.__dataclass_fields__})
    model = MedShieldDiagnosticNet(cfg)
    state = payload.get("state_dict")
    if state is None:
        raise FusionCheckpointError("Checkpoint missing state_dict.")
    model.load_state_dict(state, strict=strict)
    return model


def checkpoint_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "architecture_id": payload.get("architecture_id"),
        "architecture_version": payload.get("architecture_version"),
        "trained": payload.get("trained", False),
        "modality_order": payload.get("modality_order"),
        "pytorch": payload.get("framework", {}).get("pytorch"),
    }
