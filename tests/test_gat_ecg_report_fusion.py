"""Two-modality GAT fusion tests (synthetic tensors)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_geometric")

from ml.models.gat_ecg_report_fusion import (
    ARCHITECTURE_ID,
    GATECGReportConfig,
    GATECGReportInput,
    MedShieldGATECGReportNet,
    load_gat_ecg_report_checkpoint,
    save_gat_ecg_report_checkpoint,
)


def test_gat_forward_shape():
    model = MedShieldGATECGReportNet()
    ecg = torch.randn(4, 128)
    text = torch.randn(4, 128)
    logits = model(GATECGReportInput(ecg, text))
    assert logits.shape == (4, 2)
    assert torch.isfinite(logits).all()


def test_gat_training_step():
    model = MedShieldGATECGReportNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ecg = torch.randn(8, 128)
    text = torch.randn(8, 128)
    y = torch.randint(0, 2, (8,))
    logits = model(GATECGReportInput(ecg, text))
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    assert any(p.grad is not None for p in model.parameters())


def test_checkpoint_roundtrip(tmp_path):
    model = MedShieldGATECGReportNet(GATECGReportConfig(trained=True))
    path = tmp_path / "gat.pt"
    save_gat_ecg_report_checkpoint(path, model)
    loaded = load_gat_ecg_report_checkpoint(path)
    model.eval()
    loaded.eval()
    ecg = torch.randn(2, 128)
    text = torch.randn(2, 128)
    with torch.inference_mode():
        out1 = model(GATECGReportInput(ecg, text))
        out2 = loaded(GATECGReportInput(ecg, text))
    assert torch.allclose(out1, out2, atol=1e-5)


def test_reject_tabular_in_api_schema():
    from backend.app.schemas.gat_fusion import GATFusionEvaluateRequest
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        GATFusionEvaluateRequest(ecg_embedding=[0.0] * 128)  # type: ignore[call-arg]
