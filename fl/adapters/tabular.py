"""Tabular federation adapter (cardio_train.csv + TabularBinaryClassifier)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fl.client.adapter import TaskClientAdapter
from fl.errors import ClientLocalDataError
from fl.parameters.schema import ParameterSchema, try_schema_from_torch_state_dict
from fl.parameters.serialize import ndarrays_to_state_dict, state_dict_to_ndarrays
from fl.registry import TABULAR_CARDIO_V1
from ml.datasets.cardio_tabular import CardioTabularDataset
from ml.tabular.preprocessing import TabularPreprocessor

DEFAULT_DATA_PATH = Path("data/datasets/cardio_train.csv")


def tabular_federation_readiness(data_path: Path | str = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Readiness probe without constructing a full client partition."""
    path = Path(data_path)
    issues: list[str] = []
    if not path.is_file():
        issues.append(f"Missing dataset: {path}")
    try:
        import torch  # noqa: F401
        from ml.models.tabular_encoder import TabularBinaryClassifier

        TabularBinaryClassifier()
    except ImportError as exc:
        issues.append(f"PyTorch/tabular model unavailable: {exc}")
    return {"ready": not issues, "issues": issues, "federation_id": TABULAR_CARDIO_V1.federation_id}

# Frozen protocol hyperparameters (TABULAR_BASELINE_PROTOCOL_v1.md)
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


class TabularFederationAdapter(TaskClientAdapter):
    """
    Federated tabular client: local partition + frozen preprocessor + TabularBinaryClassifier.
    """

    def __init__(
        self,
        *,
        train_records: list[dict[str, Any]],
        preprocessor: TabularPreprocessor,
        data_path: Path | str = DEFAULT_DATA_PATH,
        client_index: int = 0,
        num_clients: int = 2,
        local_epochs: int = 1,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        device_name: str = "auto",
    ) -> None:
        self._train_records = train_records
        self._preprocessor = preprocessor
        self._data_path = Path(data_path)
        self._client_index = client_index
        self._num_clients = num_clients
        self._local_epochs = local_epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._device_name = device_name
        self._model = None
        self._device = None
        self._loader = None

    @property
    def federation_id(self) -> str:
        return TABULAR_CARDIO_V1.federation_id

    @property
    def task_id(self) -> str:
        return TABULAR_CARDIO_V1.task_id

    @property
    def architecture_id(self) -> str:
        return TABULAR_CARDIO_V1.architecture_id

    @property
    def architecture_version(self) -> str:
        return TABULAR_CARDIO_V1.architecture_version

    def _ensure_torch(self):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader

            from ml.models.tabular_encoder import TabularBinaryClassifier
        except ImportError as exc:
            raise ClientLocalDataError("PyTorch required for tabular federation adapter") from exc

        if self._device is None:
            if self._device_name == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self._device_name)

        if self._model is None:
            self._model = TabularBinaryClassifier().to(self._device)

        if self._loader is None and self._train_records:
            features = self._preprocessor.transform_batch([r["features"] for r in self._train_records])
            targets = np.array([r["target"] for r in self._train_records], dtype=np.float32)
            ds = CardioTabularDataset(features, targets)
            self._loader = DataLoader(ds, batch_size=self._batch_size, shuffle=True)

        return torch, nn, DataLoader

    def readiness(self) -> dict[str, Any]:
        base = tabular_federation_readiness(self._data_path)
        issues = list(base.get("issues", []))
        try:
            self._ensure_torch()
        except ClientLocalDataError as exc:
            issues.append(str(exc))
        if not self._train_records:
            issues.append("Empty client train partition")
        if not self._preprocessor.state.fitted:
            issues.append("Preprocessor not fitted")
        return {"ready": not issues, "issues": issues, "federation_id": self.federation_id}

    def parameter_schema(self) -> ParameterSchema:
        torch, _, _ = self._ensure_torch()
        model = self._model
        assert model is not None
        return try_schema_from_torch_state_dict(
            model.state_dict(),
            architecture_id=self.architecture_id,
            architecture_version=self.architecture_version,
        )

    def get_parameters(self) -> tuple[list[np.ndarray], list[str]]:
        self._ensure_torch()
        assert self._model is not None
        return state_dict_to_ndarrays(self._model.state_dict())

    def set_parameters(self, arrays: list[np.ndarray], names: list[str]) -> None:
        self._ensure_torch()
        assert self._model is not None
        updated = ndarrays_to_state_dict(arrays, names, template_state_dict=self._model.state_dict())
        self._model.load_state_dict(updated, strict=True)

    def num_local_examples(self) -> int:
        return len(self._train_records)

    def fit(self, config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, float]]:
        torch, nn, _ = self._ensure_torch()
        assert self._model is not None
        assert self._loader is not None

        epochs = int(config.get("local_epochs", self._local_epochs))
        lr = float(config.get("learning_rate", self._lr))
        weight_decay = float(config.get("weight_decay", self._weight_decay))

        if epochs <= 0:
            arrays, names = self.get_parameters()
            return arrays, self.num_local_examples(), {"loss": 0.0}

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        self._model.train()
        total_loss = 0.0
        n = 0
        for _ in range(epochs):
            for batch in self._loader:
                x = batch["features"].to(self._device)
                y = batch["target"].to(self._device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = self._model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)

        avg_loss = float(total_loss / max(n, 1))
        arrays, _names = self.get_parameters()
        return arrays, self.num_local_examples(), {"loss": avg_loss}
