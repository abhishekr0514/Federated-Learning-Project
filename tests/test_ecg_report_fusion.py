"""Two-modality ECG+report index and model smoke (no full PTB training)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from ml.models.ecg_report_fusion import (
    ARCHITECTURE_ID,
    ECGReportFusionClassifier,
    ECGReportFusionConfig,
    save_ecg_report_checkpoint,
)
from ml.models.clinical_text_encoder import ClinicalTextEncoder, ClinicalTextEncoderConfig
from ml.ptbxl.ecg_report_index import build_ecg_report_index

PTBXL = Path(__file__).resolve().parents[1] / "data" / "raw" / "ptb-xl"
pytestmark_ptbxl = pytest.mark.skipif(not PTBXL.exists(), reason="PTB-XL not present locally")


@pytestmark_ptbxl
def test_ecg_report_index_has_pairs_and_reports():
    records, stats = build_ecg_report_index(PTBXL)
    assert stats["included_with_report"] == len(records)
    assert stats["not_three_modality_gat"] is True
    assert all("report" in r and r["report"].strip() for r in records[:20])


def test_fusion_forward_with_mock_text_encoder():
    cfg = ClinicalTextEncoderConfig(max_sequence_length=32, embedding_dim=128)
    bert = MagicMock()
    hidden = torch.randn(2, 8, 768)
    bert.return_value = MagicMock(last_hidden_state=hidden)
    text_enc = ClinicalTextEncoder(bert, config=cfg)
    model = ECGReportFusionClassifier(text_enc, config=ECGReportFusionConfig(trained=False))
    ecg = torch.randn(2, 12, 1000)
    input_ids = torch.ones(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.long)
    logit, ecg_emb, text_emb = model(ecg, input_ids, mask)
    assert logit.shape == (2,)
    assert ecg_emb.shape == (2, 128)
    assert text_emb.shape == (2, 128)


def test_checkpoint_roundtrip_mock(tmp_path):
    cfg = ClinicalTextEncoderConfig(max_sequence_length=32, embedding_dim=128)
    bert = MagicMock()
    bert.return_value = MagicMock(last_hidden_state=torch.randn(1, 4, 768))
    text_enc = ClinicalTextEncoder(bert, config=cfg)
    model = ECGReportFusionClassifier(text_enc, config=ECGReportFusionConfig(trained=True))
    path = tmp_path / "ckpt.pt"
    save_ecg_report_checkpoint(path, model, extra_metadata={"epoch": 1})
    raw = torch.load(path, weights_only=False)
    assert raw["architecture_id"] == ARCHITECTURE_ID
    assert raw["trained"] is True
    assert raw["experiment"] == "two_modality_ecg_report_ptbxl"
