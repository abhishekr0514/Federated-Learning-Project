"""Test suite for Path B: Three-Modality GAT fusion pipeline contracts and readiness."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.fusion.errors import FusionInputError
from ml.fusion.pipeline import (
    MODALITY_CONTRACTS,
    MultimodalGATPipeline,
)
from ml.inference.constants import ECG_EMBED_DIM, TABULAR_EMBED_DIM, TEXT_EMBED_DIM
from ml.models.ecg_embedding_provider import ECGEmbeddingProvider
from ml.models.gnn_fusion import MedShieldDiagnosticNet
from ml.training.linkage_gate import evaluate_three_modality_gate


def test_modality_contracts_dimensions():
    assert MODALITY_CONTRACTS["ecg"].dimension == ECG_EMBED_DIM == 128
    assert MODALITY_CONTRACTS["clinical_text"].dimension == TEXT_EMBED_DIM == 128
    assert MODALITY_CONTRACTS["tabular"].dimension == TABULAR_EMBED_DIM == 64


def test_ecg_embedding_provider_extraction():
    provider = ECGEmbeddingProvider.from_checkpoint(
        checkpoint_path="models/ecg_report_fusion/best_checkpoint.pt",
        normalization_path="models/ecg_report_fusion/normalization.json",
    )
    assert provider.encoder is not None
    assert provider.encoder.training is False

    rng = np.random.default_rng(123)
    sig = rng.normal(0.0, 0.2, size=(12, 1000)).astype(np.float32)
    emb = provider.encode(sig)

    assert isinstance(emb, list)
    assert len(emb) == 128
    assert all(np.isfinite(emb))


def test_ecg_embedding_provider_rejects_wrong_shape():
    provider = ECGEmbeddingProvider.from_checkpoint(
        checkpoint_path="models/ecg_report_fusion/best_checkpoint.pt",
        normalization_path="models/ecg_report_fusion/normalization.json",
    )
    with pytest.raises(ValueError, match=r"Expected ECG shape \(12, 1000\)"):
        provider.encode(np.zeros((8, 1000)))


def test_ecg_embedding_provider_rejects_nan():
    provider = ECGEmbeddingProvider.from_checkpoint(
        checkpoint_path="models/ecg_report_fusion/best_checkpoint.pt",
        normalization_path="models/ecg_report_fusion/normalization.json",
    )
    bad_sig = np.zeros((12, 1000), dtype=np.float32)
    bad_sig[0, 10] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        provider.encode(bad_sig)


def test_multimodal_gat_pipeline_build_input_all_present():
    pipe = MultimodalGATPipeline()
    ecg_emb = [0.1] * 128
    text_emb = [0.2] * 128
    tab_emb = [0.3] * 64

    inp = pipe.build_fusion_input(
        ecg_emb=ecg_emb,
        text_emb=text_emb,
        tabular_emb=tab_emb,
    )
    assert inp.ecg is not None and inp.ecg.shape == (1, 128)
    assert inp.clinical_text is not None and inp.clinical_text.shape == (1, 128)
    assert inp.tabular is not None and inp.tabular.shape == (1, 64)
    assert inp.available.tolist() == [[True, True, True]]


def test_multimodal_gat_pipeline_build_input_partial_missing():
    pipe = MultimodalGATPipeline()
    ecg_emb = [0.1] * 128

    inp = pipe.build_fusion_input(ecg_emb=ecg_emb)
    assert inp.ecg is not None
    assert inp.clinical_text is None
    assert inp.tabular is None
    assert inp.available.tolist() == [[True, False, False]]


def test_multimodal_gat_pipeline_rejects_all_missing():
    pipe = MultimodalGATPipeline()
    with pytest.raises(FusionInputError, match="At least one modality"):
        pipe.build_fusion_input()


def test_multimodal_gat_pipeline_rejects_dimension_mismatch():
    pipe = MultimodalGATPipeline()
    with pytest.raises(FusionInputError, match="ECG embedding dimension"):
        pipe.build_fusion_input(ecg_emb=[0.1] * 64)  # 64 instead of 128


def test_gat_pipeline_readiness_blocked_truthfully():
    report = MultimodalGATPipeline.assess_gat_pipeline_readiness()
    assert report.ready is False
    assert report.linkage_gate_allowed is False
    assert report.linkage_gate_verdict == "BLOCKED"
    assert len(report.blockers) > 0
    assert len(report.required_to_unblock) > 0
    # Does not claim GAT is trained or operational
    safe_dict = report.to_safe_dict()
    assert safe_dict["ready"] is False
