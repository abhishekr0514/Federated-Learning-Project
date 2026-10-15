"""
Neural Tabular Encoder (Prompt 14 research component).

Architecture: Linear(10,64) → BatchNorm1d(64) → ReLU → Dropout(0.2) → Linear(64,64) → ReLU
Output: [B, 64]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.tabular.feature_schema import N_TABULAR_FEATURES

EMBED_DIM = 64
DROPOUT_P = 0.2


class NeuralTabularEncoder(nn.Module):
    def __init__(self, input_dim: int = N_TABULAR_FEATURES) -> None:
        super().__init__()
        if input_dim != N_TABULAR_FEATURES:
            raise ValueError(f"Expected input_dim={N_TABULAR_FEATURES}, got {input_dim}")
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=DROPOUT_P)
        self.fc2 = nn.Linear(64, EMBED_DIM)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input shape (B, {self.input_dim}), got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains non-finite values.")
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        if not torch.isfinite(x).all():
            raise ValueError("Encoder output contains non-finite values.")
        return x
