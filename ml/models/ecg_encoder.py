"""ECG encoder per MODEL_CONTRACT.md + binary classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
    """
    Input: (B, 12, 1000) — 12 leads, 1000 timesteps.
    Output embedding: (B, 128)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        h = self.conv(x)
        h = h.transpose(1, 2)  # (B, T', 128)
        h, _ = self.lstm(h)
        h = h[:, -1, :]  # last timestep (B, 128)
        return self.projection(h)


class ECGBinaryClassifier(nn.Module):
    """128-d embedding + single logit for BCEWithLogitsLoss."""

    def __init__(self, encoder: ECGEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or ECGEncoder()
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        logit = self.head(emb).squeeze(-1)
        return logit, emb
