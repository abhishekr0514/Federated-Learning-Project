"""Federated PTB-XL ECG + clinical report adapter (paired modality)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from fl.client.adapter import TaskClientAdapter
from fl.ecg_report.trainable import (
    apply_trainable_scope,
    merge_trainable_into_model,
    trainable_state_dict,
)
from fl.errors import ClientLocalDataError
from fl.parameters.schema import ParameterSchema, schema_from_named_arrays
from fl.parameters.serialize import state_dict_to_ndarrays
from ml.datasets.ptbxl_ecg_report import PTBXLECGReportDataset
from ml.models.ecg_report_fusion import ARCHITECTURE_ID, ARCHITECTURE_VERSION, ECGReportFusionClassifier
from ml.training.train_ecg_report_fusion import freeze_bert, make_collate_fn, train_one_epoch

TEXT_MAX_LEN = 256


def ecg_report_federation_readiness(
    *,
    ptbxl_root: Path | str = Path("data/raw/ptb-xl"),
    encoder_manifest_path: Path | str = Path("models/ecg_report_fusion/manifest.json"),
) -> dict[str, Any]:
    issues: list[str] = []
    root = Path(ptbxl_root)
    manifest = Path(encoder_manifest_path)
    if not root.is_dir():
        issues.append(f"PTB-XL root missing: {root}")
    if not manifest.is_file():
        issues.append(f"Encoder manifest missing: {manifest}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        issues.append(f"PyTorch required: {exc}")
    return {
        "ready": not issues,
        "issues": issues,
        "federation_id": "medshield.fl.ecg_report.v1",
    }


class ECGReportFederationAdapter(TaskClientAdapter):
    """
    Local client for paired ECG+report records.
    Returns only federated trainable parameters (default: fusion head).
    """

    def __init__(
        self,
        *,
        train_records: list[dict[str, Any]],
        model: ECGReportFusionClassifier,
        text_pipeline,
        normalization,
        ptbxl_root: Path,
        trainable_names: list[str],
        federated_trainable_scope: str,
        client_index: int = 0,
        num_clients: int = 2,
        local_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device_name: str = "auto",
        encoder_fingerprint: str = "",
    ) -> None:
        self._train_records = train_records
        self._model = model
        self._text_pipeline = text_pipeline
        self._normalization = normalization
        self._ptbxl_root = Path(ptbxl_root)
        self._trainable_names = trainable_names
        self._scope = federated_trainable_scope
        self._client_index = client_index
        self._num_clients = num_clients
        self._local_epochs = local_epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._device_name = device_name
        self._encoder_fingerprint = encoder_fingerprint
        self._frozen_encoder_snapshot: dict[str, np.ndarray] | None = None
        self._device = None
        self._loader = None
        self._schema: ParameterSchema | None = None

    @property
    def federation_id(self) -> str:
        return "medshield.fl.ecg_report.v1"

    @property
    def task_id(self) -> str:
        return "medshield.ecg_report.ptbxl.proxy_superclass"

    @property
    def architecture_id(self) -> str:
        return ARCHITECTURE_ID

    @property
    def architecture_version(self) -> str:
        return ARCHITECTURE_VERSION

    def _ensure_torch(self):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise ClientLocalDataError("PyTorch required for ECG+report federation") from exc

        if self._device is None:
            if self._device_name == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self._device_name)
            self._model.to(self._device)

        if self._frozen_encoder_snapshot is None:
            snap: dict[str, np.ndarray] = {}
            for name, param in self._model.named_parameters():
                if name not in self._trainable_names:
                    snap[name] = param.detach().cpu().numpy().copy()
            self._frozen_encoder_snapshot = snap

        if self._loader is None and self._train_records:
            ds = PTBXLECGReportDataset(self._train_records, self._ptbxl_root, self._normalization)
            collate = make_collate_fn(self._text_pipeline.tokenizer, TEXT_MAX_LEN)
            self._loader = DataLoader(ds, batch_size=self._batch_size, shuffle=True, collate_fn=collate)

        return torch, nn, DataLoader

    def _assert_frozen_unchanged(self) -> None:
        if self._frozen_encoder_snapshot is None:
            return
        for name, expected in self._frozen_encoder_snapshot.items():
            current = self._model.state_dict()[name].detach().cpu().numpy()
            if not np.allclose(current, expected, rtol=0, atol=0):
                raise ClientLocalDataError(f"Frozen parameter changed during local training: {name}")

    def parameter_schema(self) -> ParameterSchema:
        if self._schema is None:
            td = trainable_state_dict(self._model, self._trainable_names)
            arrays, names = state_dict_to_ndarrays(td)
            pairs = list(zip(names, arrays))
            self._schema = schema_from_named_arrays(
                pairs,
                architecture_id=self.architecture_id,
                architecture_version=self.architecture_version,
            )
        return self._schema

    def get_parameters(self) -> tuple[list[np.ndarray], list[str]]:
        self._ensure_torch()
        td = trainable_state_dict(self._model, self._trainable_names)
        return state_dict_to_ndarrays(td)

    def set_parameters(self, arrays: list[np.ndarray], names: list[str]) -> None:
        self._ensure_torch()
        merge_trainable_into_model(self._model, arrays, names, self._trainable_names)
        self._assert_frozen_unchanged()

    def num_local_examples(self) -> int:
        return len(self._train_records)

    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        torch, nn, _ = self._ensure_torch()
        if not self._train_records:
            arrays, names = self.get_parameters()
            return arrays, 0, {"loss": 0.0}

        epochs = int(config.get("local_epochs", self._local_epochs))
        lr = float(config.get("learning_rate", self._lr))
        weight_decay = float(config.get("weight_decay", self._weight_decay))

        if epochs <= 0:
            arrays, _ = self.get_parameters()
            return arrays, self.num_local_examples(), {"loss": 0.0}

        optimizer = torch.optim.Adam(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        self._model.train()
        total_loss = 0.0
        n = 0
        for _ in range(epochs):
            loss = train_one_epoch(self._model, self._loader, optimizer, criterion, self._device)
            total_loss += loss * self.num_local_examples()
            n += self.num_local_examples()

        self._assert_frozen_unchanged()
        avg_loss = float(total_loss / max(n, 1))
        arrays, _names = self.get_parameters()
        return arrays, self.num_local_examples(), {"loss": avg_loss}


def load_encoder_bundle(repo_root: Path, manifest_path: Path, device):
    import json

    import torch

    from ml.models.clinical_text_encoder import ClinicalTextEncoderConfig
    from ml.models.ecg_report_fusion import load_ecg_report_checkpoint
    from ml.preprocessing.ecg_normalization import ECGNormalizationStats
    from ml.text.clinical_text_pipeline import ClinicalTextEncodingPipeline

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ckpt = repo_root / manifest["checkpoint_path"]
    norm_path = repo_root / manifest["normalization_path"]
    if manifest.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Incompatible encoder manifest architecture_id")
    if not ckpt.is_file() or not norm_path.is_file():
        raise FileNotFoundError("Encoder checkpoint or normalization missing")

    text_cfg = ClinicalTextEncoderConfig(max_sequence_length=TEXT_MAX_LEN)
    pipe, _ = ClinicalTextEncodingPipeline.from_pretrained(config=text_cfg, device=device)
    model = load_ecg_report_checkpoint(ckpt, pipe.encoder, map_location="cpu")
    model.to(device)
    model.eval()
    freeze_bert(model)
    if hasattr(pipe, "encoder"):
        pipe.encoder.to(device)
    norm = ECGNormalizationStats.load(norm_path)
    import hashlib

    fp = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    return model, pipe, norm, manifest, fp


def build_client_model_template(
    encoder_model: ECGReportFusionClassifier,
    scope: str,
) -> tuple[ECGReportFusionClassifier, list[str], list[str]]:
    model = copy.deepcopy(encoder_model)
    trainable, frozen = apply_trainable_scope(model, scope)
    return model, trainable, frozen
