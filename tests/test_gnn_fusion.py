"""Prompt 15 — MedShieldDiagnosticNet GAT fusion (synthetic tensors only)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.fusion.errors import FusionCheckpointError, FusionInputError
from ml.models.gnn_fusion import (
    ARCHITECTURE_VERSION,
    FLAT_FUSION_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_MODALITIES,
    NODE_DIM,
    FusionModelConfig,
    FusionModelInput,
    MedShieldDiagnosticNet,
    TABULAR_EMBED_DIM,
    TEXT_EMBED_DIM,
    ECG_EMBED_DIM,
    load_fusion_checkpoint,
    save_fusion_checkpoint,
)


def _full_input(batch: int = 4, device: str = "cpu") -> FusionModelInput:
    ones = torch.ones(batch, NUM_MODALITIES, dtype=torch.bool, device=device)
    return FusionModelInput(
        ecg=torch.randn(batch, ECG_EMBED_DIM, device=device),
        clinical_text=torch.randn(batch, TEXT_EMBED_DIM, device=device),
        tabular=torch.randn(batch, TABULAR_EMBED_DIM, device=device),
        available=ones,
        quality_ok=ones,
    )


def test_graph_node_shape_and_projections():
    model = MedShieldDiagnosticNet()
    inp = _full_input(3)
    nodes = model.build_graph_nodes(inp.ecg, inp.clinical_text, inp.tabular)
    assert nodes.shape == (3, NUM_MODALITIES, NODE_DIM)
    assert isinstance(model.proj_tabular, torch.nn.Linear)
    assert model.proj_tabular.in_features == TABULAR_EMBED_DIM
    assert model.proj_tabular.out_features == NODE_DIM


def test_gat_layers_and_head_shapes():
    model = MedShieldDiagnosticNet()
    inp = _full_input(2)
    nodes = model.build_graph_nodes(inp.ecg, inp.clinical_text, inp.tabular)
    after_gat = model.forward_gat_stack(nodes)
    assert after_gat.shape == (2, NUM_MODALITIES, NODE_DIM)
    flat = after_gat.reshape(2, FLAT_FUSION_DIM)
    assert flat.shape == (2, 384)
    logits = model.forward_head(after_gat)
    assert logits.shape == (2, NUM_CLASSES)


def test_forward_probabilities_finite():
    model = MedShieldDiagnosticNet()
    model.eval()
    out = model(_full_input(5))
    assert out.logits is not None and out.logits.shape == (5, 2)
    assert out.probabilities is not None
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert out.confidence is not None and out.uncalibrated is True
    assert out.complete.all()
    assert out.modality_used.all()


def test_invalid_shapes_and_non_finite():
    model = MedShieldDiagnosticNet()
    bad = _full_input(2)
    bad.ecg = torch.randn(2, 64)
    with pytest.raises(FusionInputError, match="ecg"):
        model(bad)
    bad2 = _full_input(2)
    bad2.ecg = torch.full((2, ECG_EMBED_DIM), float("nan"))
    with pytest.raises(FusionInputError, match="non-finite"):
        model(bad2)


def test_batch_size_mismatch():
    model = MedShieldDiagnosticNet()
    inp = _full_input(3)
    inp.tabular = torch.randn(2, TABULAR_EMBED_DIM)
    with pytest.raises(FusionInputError):
        model(inp)


def test_missing_modality_rejected_and_not_reported_used():
    model = MedShieldDiagnosticNet()
    inp = _full_input(2)
    inp.available = inp.available.clone()
    inp.available[0, 2] = False
    with pytest.raises(FusionInputError, match="Unsupported modality pattern"):
        model(inp)

    partial = FusionModelInput(
        ecg=inp.ecg,
        clinical_text=inp.clinical_text,
        tabular=None,
        available=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        quality_ok=torch.ones(2, NUM_MODALITIES, dtype=torch.bool),
    )
    status = model.describe_modality_status(partial)
    assert status.modality_used[0, 2].item() is False
    assert not status.complete[0].item()
    assert "tabular:unavailable" in status.incomplete_reasons[0]


def test_quality_vs_availability():
    model = MedShieldDiagnosticNet()
    inp = _full_input(1)
    inp.quality_ok = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    with pytest.raises(FusionInputError):
        model(inp)


def test_eval_mode_deterministic():
    torch.manual_seed(7)
    model = MedShieldDiagnosticNet()
    model.eval()
    inp = _full_input(3)
    with torch.no_grad():
        a = model(inp).logits
        b = model(inp).logits
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_checkpoint_round_trip_and_incompatible(tmp_path: Path):
    model = MedShieldDiagnosticNet(FusionModelConfig(trained=False))
    path = tmp_path / "fusion.pt"
    save_fusion_checkpoint(path, model, extra_metadata={"note": "untrained_init"})
    loaded = load_fusion_checkpoint(path)
    model.eval()
    loaded.eval()
    inp = _full_input(2)
    with torch.no_grad():
        torch.testing.assert_close(model(inp).logits, loaded(inp).logits)

    bad = tmp_path / "bad.pt"
    torch.save({"architecture_id": "other", "architecture_version": "0", "state_dict": {}}, bad)
    with pytest.raises(FusionCheckpointError):
        load_fusion_checkpoint(bad)


def test_gat_config_matches_contract():
    model = MedShieldDiagnosticNet()
    assert model.gat1.heads == 4
    assert model.gat2.heads == 1
    assert model.classifier_hidden[0].out_features == HIDDEN_DIM
    assert model.config.architecture_version == ARCHITECTURE_VERSION
