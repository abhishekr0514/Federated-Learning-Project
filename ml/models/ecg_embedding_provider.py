"""Trained ECG embedding provider for multimodal GAT compatibility.

Extracts the trained ECGEncoder from the medshield.ecg_report_fusion.v1 checkpoint
and evaluates 12-lead (12, 1000) waveforms into (128,) embedding vectors per
the GAT multimodal contract (docs/MODEL_CONTRACT.md §1 & §4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ml.inference.constants import ECG_EMBED_DIM
from ml.models.ecg_encoder import ECGEncoder
from ml.preprocessing.ecg_normalization import ECGNormalizationStats
from ml.preprocessing.ecg_signal_quality import assess_ecg_signal, LEAD_ORDER

logger = logging.getLogger(__name__)

ARCHITECTURE_ID = "medshield.ecg_encoder.v1"
ARCHITECTURE_VERSION = "1.0.0"


class ECGEmbeddingProvider:
    """
    Evaluates 12-lead (12, 1000) ECG signals into frozen 128-dimensional embedding vectors.

    Enforces:
    - 12 leads, 1000 timesteps at 100 Hz in LEAD_ORDER.
    - Signal quality assessment (no NaNs, non-finite values, or flatline/saturation).
    - Per-lead z-score normalization matching training stats.
    - Exact 128-dimensional output vector matching GAT ECG_EMBED_DIM.
    """

    def __init__(
        self,
        encoder: ECGEncoder,
        normalization_stats: ECGNormalizationStats,
        *,
        device: torch.device | str = "cpu",
        provenance: str = "trained_ecg_report_fusion_checkpoint",
    ) -> None:
        self.encoder = encoder
        self.encoder.to(device)
        self.encoder.eval()
        self.normalization_stats = normalization_stats
        self.device = torch.device(device)
        self.provenance = provenance

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        normalization_path: Path | str,
        *,
        device: torch.device | str = "cpu",
    ) -> "ECGEmbeddingProvider":
        ckpt_p = Path(checkpoint_path)
        norm_p = Path(normalization_path)

        if not ckpt_p.is_file():
            raise FileNotFoundError(f"ECG checkpoint file missing: {ckpt_p}")
        if not norm_p.is_file():
            raise FileNotFoundError(f"ECG normalization stats file missing: {norm_p}")

        try:
            payload = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(ckpt_p, map_location="cpu")

        state_dict = payload.get("state_dict", {})
        ecg_sd: dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith("ecg_encoder."):
                ecg_sd[k[len("ecg_encoder."):]] = v
            elif not (k.startswith("text_encoder.") or k.startswith("head.")):
                ecg_sd[k] = v

        if not ecg_sd:
            raise ValueError(f"No ECG encoder weights found in checkpoint {ckpt_p.name}")

        encoder = ECGEncoder()
        encoder.load_state_dict(ecg_sd)
        encoder.eval()

        norm_stats = ECGNormalizationStats.load(norm_p)
        return cls(
            encoder=encoder,
            normalization_stats=norm_stats,
            device=device,
            provenance=f"extracted_from:{ckpt_p.name}",
        )

    def encode(
        self,
        signal_mv: np.ndarray,
        *,
        lead_order: Sequence[str] = LEAD_ORDER,
    ) -> list[float]:
        """
        Transform a single (12, 1000) mV array into a 128-d embedding vector.
        """
        if not isinstance(signal_mv, np.ndarray):
            signal_mv = np.asarray(signal_mv, dtype=np.float32)

        if signal_mv.shape != (12, 1000):
            raise ValueError(f"Expected ECG shape (12, 1000), got {signal_mv.shape}")

        if not np.all(np.isfinite(signal_mv)):
            raise ValueError("ECG signal contains non-finite (NaN or Inf) values")

        quality = assess_ecg_signal(signal_mv, lead_order=lead_order)
        if not quality.passed:
            reasons = "; ".join(quality.descriptions)
            raise ValueError(f"ECG signal failed quality check: {reasons}")

        normalized = self.normalization_stats.transform(signal_mv)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)  # (1, 12, 1000)

        with torch.inference_mode():
            emb = self.encoder(tensor)  # (1, 128)

        if emb.shape[-1] != ECG_EMBED_DIM:
            raise ValueError(f"ECG encoder produced dimension {emb.shape[-1]}, expected {ECG_EMBED_DIM}")

        out_vec = emb.squeeze(0).cpu().tolist()
        if len(out_vec) != ECG_EMBED_DIM:
            raise ValueError(f"Output vector dimension mismatch: {len(out_vec)} != {ECG_EMBED_DIM}")

        return out_vec
