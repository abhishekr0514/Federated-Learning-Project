"""Prompt 15 / Prompt 19 — MedShieldDiagnosticNet GAT fusion (synthetic tensors only).

Architectural verification per docs/MODEL_CONTRACT.md §4 and ADR-009.
All tests use deterministic synthetic tensors for structural validation only.
Never report unit test results as clinical diagnostic performance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from ml.fusion.errors import FusionCheckpointError, FusionInputError
from ml.models.gnn_fusion import (
    _FC_EDGE_INDEX,
    ARCHITECTURE_ID,
    ARCHITECTURE_VERSION,
    DEFAULT_ECG_PROTOCOL_VERSION,
    DEFAULT_TABULAR_SCHEMA_VERSION,
    DEFAULT_TEXT_ENCODER_VERSION,
    ECG_EMBED_DIM,
    FLAT_FUSION_DIM,
    HIDDEN_DIM,
    NODE_DIM,
    NUM_CLASSES,
    NUM_MODALITIES,
    TABULAR_EMBED_DIM,
    TEXT_EMBED_DIM,
    FusionModelConfig,
    FusionModelInput,
    MedShieldDiagnosticNet,
    ModalityIndex,
    _build_batched_graph,
    checkpoint_summary,
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


# 1. Correct node projection shapes
def test_graph_node_shape_and_projections():
    model = MedShieldDiagnosticNet()
    for batch_size in (1, 3):
        inp = _full_input(batch_size)
        nodes = model.build_graph_nodes(inp.ecg, inp.clinical_text, inp.tabular)
        assert nodes.shape == (batch_size, NUM_MODALITIES, NODE_DIM)

    assert isinstance(model.proj_ecg, torch.nn.Linear)
    assert model.proj_ecg.in_features == ECG_EMBED_DIM
    assert model.proj_ecg.out_features == NODE_DIM

    assert isinstance(model.proj_text, torch.nn.Linear)
    assert model.proj_text.in_features == TEXT_EMBED_DIM
    assert model.proj_text.out_features == NODE_DIM

    assert isinstance(model.proj_tabular, torch.nn.Linear)
    assert model.proj_tabular.in_features == TABULAR_EMBED_DIM
    assert model.proj_tabular.out_features == NODE_DIM


# 2. Correct graph construction and edge connectivity
def test_graph_construction_and_edge_connectivity():
    model = MedShieldDiagnosticNet()
    assert hasattr(model, "fc_edge_index")
    assert model.fc_edge_index.shape == (2, 9)

    # Check all 9 directed edges (including self-loops) over 3 nodes
    expected_edges = {(i, j) for i in range(3) for j in range(3)}
    actual_edges = {
        (model.fc_edge_index[0, k].item(), model.fc_edge_index[1, k].item())
        for k in range(model.fc_edge_index.size(1))
    }
    assert actual_edges == expected_edges

    # Check batched graph construction for B=2
    dummy_nodes = torch.randn(2, NUM_MODALITIES, NODE_DIM)
    x, batched_edges, batch_vec = _build_batched_graph(dummy_nodes, model.fc_edge_index)
    assert x.shape == (6, NODE_DIM)
    assert batched_edges.shape == (2, 18)
    assert batch_vec.tolist() == [0, 0, 0, 1, 1, 1]

    # First graph edges in [0, 2], second in [3, 5]
    assert batched_edges[:, :9].max().item() == 2
    assert batched_edges[:, :9].min().item() == 0
    assert batched_edges[:, 9:].max().item() == 5
    assert batched_edges[:, 9:].min().item() == 3


# 3. Correct output shape [B, 2] for batch sizes 1 and greater than 1
@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_output_shapes_batch_size_one_and_greater(batch_size: int):
    model = MedShieldDiagnosticNet()
    model.eval()
    inp = _full_input(batch_size)
    out = model(inp)

    assert out.logits is not None
    assert out.logits.shape == (batch_size, NUM_CLASSES)
    assert out.probabilities is not None
    assert out.probabilities.shape == (batch_size, NUM_CLASSES)
    assert out.predicted_class is not None
    assert out.predicted_class.shape == (batch_size,)
    assert out.confidence is not None
    assert out.confidence.shape == (batch_size,)
    assert out.complete.shape == (batch_size,)
    assert out.modality_used.shape == (batch_size, NUM_MODALITIES)
    assert len(out.incomplete_reasons) == batch_size


# 4. Finite outputs for valid finite inputs
def test_forward_probabilities_finite_and_normalized():
    model = MedShieldDiagnosticNet()
    model.eval()
    out = model(_full_input(5))

    assert out.logits is not None
    assert torch.isfinite(out.logits).all()
    assert out.probabilities is not None
    assert torch.isfinite(out.probabilities).all()
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (out.probabilities >= 0.0).all() and (out.probabilities <= 1.0).all()
    assert out.confidence is not None
    assert torch.isfinite(out.confidence).all()
    assert out.uncalibrated is True
    assert out.complete.all()
    assert out.modality_used.all()


# 5. Missing-modality rejection
def test_missing_modality_rejection():
    model = MedShieldDiagnosticNet()

    for mod_idx in (ModalityIndex.ECG, ModalityIndex.CLINICAL_TEXT, ModalityIndex.TABULAR):
        inp = _full_input(2)
        inp.available = inp.available.clone()
        inp.available[0, mod_idx] = False
        with pytest.raises(FusionInputError, match="Unsupported modality pattern"):
            model(inp)

    # Tensor is None when marked available
    for mod_name in ("ecg", "clinical_text", "tabular"):
        inp = _full_input(2)
        setattr(inp, mod_name, None)
        with pytest.raises(FusionInputError, match="embedding tensor is required"):
            model(inp)


# 6. Wrong rank, dimensions, batch size, dtype, NaN, and infinity rejection
def test_invalid_rank_dims_batch_dtype_nan_inf_rejection():
    model = MedShieldDiagnosticNet()

    # Wrong rank: 3D instead of 2D
    bad_rank = _full_input(2)
    bad_rank.ecg = torch.randn(2, 12, 1000)
    with pytest.raises(FusionInputError, match="ecg must have shape"):
        model(bad_rank)

    # Wrong rank: 1D
    bad_rank_1d = _full_input(2)
    bad_rank_1d.tabular = torch.randn(TABULAR_EMBED_DIM)
    with pytest.raises(FusionInputError, match="tabular must have shape"):
        model(bad_rank_1d)

    # Wrong embedding dimension
    bad_dim = _full_input(2)
    bad_dim.ecg = torch.randn(2, 64)
    with pytest.raises(FusionInputError, match="ecg must have shape"):
        model(bad_dim)

    bad_dim_tab = _full_input(2)
    bad_dim_tab.tabular = torch.randn(2, 10)  # Raw tabular features instead of 64d embedding
    with pytest.raises(FusionInputError, match="tabular must have shape"):
        model(bad_dim_tab)

    bad_dim_text = _full_input(2)
    bad_dim_text.clinical_text = torch.randn(2, 768)  # Raw BERT CLS instead of 128d
    with pytest.raises(FusionInputError, match="clinical_text must have shape"):
        model(bad_dim_text)

    # Batch size mismatch
    mismatch_b = _full_input(3)
    mismatch_b.tabular = torch.randn(2, TABULAR_EMBED_DIM)
    with pytest.raises(FusionInputError, match="tabular must have shape"):
        model(mismatch_b)

    # Non-floating dtype
    int_dtype = _full_input(2)
    int_dtype.ecg = torch.randint(0, 10, (2, ECG_EMBED_DIM), dtype=torch.long)
    with pytest.raises(FusionInputError, match="floating-point"):
        model(int_dtype)

    # NaN in embedding
    nan_inp = _full_input(2)
    nan_inp.ecg = torch.full((2, ECG_EMBED_DIM), float("nan"))
    with pytest.raises(FusionInputError, match="non-finite"):
        model(nan_inp)

    # Infinity in embedding
    inf_inp = _full_input(2)
    inf_inp.tabular = torch.full((2, TABULAR_EMBED_DIM), float("inf"))
    with pytest.raises(FusionInputError, match="non-finite"):
        model(inf_inp)

    # Empty batch (size 0)
    with pytest.raises(FusionInputError, match="Batch size must be at least 1"):
        FusionModelInput(
            ecg=torch.randn(0, ECG_EMBED_DIM),
            clinical_text=torch.randn(0, TEXT_EMBED_DIM),
            tabular=torch.randn(0, TABULAR_EMBED_DIM),
            available=torch.ones(0, NUM_MODALITIES, dtype=torch.bool),
            quality_ok=torch.ones(0, NUM_MODALITIES, dtype=torch.bool),
        )

    # Mask with non-finite values
    with pytest.raises(FusionInputError, match="finite values"):
        FusionModelInput(
            ecg=torch.randn(2, ECG_EMBED_DIM),
            clinical_text=torch.randn(2, TEXT_EMBED_DIM),
            tabular=torch.randn(2, TABULAR_EMBED_DIM),
            available=torch.tensor([[1.0, 1.0, float("nan")], [1.0, 1.0, 1.0]]),
            quality_ok=torch.ones(2, NUM_MODALITIES),
        )

    # Mask with invalid values (not {0, 1})
    with pytest.raises(FusionInputError, match="binary"):
        FusionModelInput(
            ecg=torch.randn(2, ECG_EMBED_DIM),
            clinical_text=torch.randn(2, TEXT_EMBED_DIM),
            tabular=torch.randn(2, TABULAR_EMBED_DIM),
            available=torch.tensor([[1, 2, 1], [1, 1, 1]]),
            quality_ok=torch.ones(2, NUM_MODALITIES, dtype=torch.long),
        )


# 7. Invalid ECG quality status rejection
def test_invalid_quality_status_rejection():
    model = MedShieldDiagnosticNet()

    for mod_idx, name in (
        (ModalityIndex.ECG, "ecg"),
        (ModalityIndex.CLINICAL_TEXT, "clinical_text"),
        (ModalityIndex.TABULAR, "tabular"),
    ):
        inp = _full_input(2)
        inp.quality_ok = inp.quality_ok.clone()
        inp.quality_ok[0, mod_idx] = False
        with pytest.raises(FusionInputError, match=f"{name}:quality_rejected"):
            model(inp)


# 8. Gradient propagation through both GAT layers, modality projections, and the classification head
def test_gradient_propagation_all_parameters():
    torch.manual_seed(42)
    model = MedShieldDiagnosticNet()
    model.train()

    inp = _full_input(3)
    out = model(inp)
    assert out.logits is not None

    loss = out.logits.sum()
    loss.backward()

    # Verify every projection layer has non-zero finite gradients
    for proj_name, proj in (
        ("proj_ecg", model.proj_ecg),
        ("proj_text", model.proj_text),
        ("proj_tabular", model.proj_tabular),
    ):
        assert proj.weight.grad is not None, f"{proj_name}.weight did not receive gradients"
        assert torch.isfinite(proj.weight.grad).all()
        assert proj.weight.grad.abs().sum() > 0

    # Verify both GAT layers received gradients
    assert model.gat1.lin_src.weight.grad is not None
    assert torch.isfinite(model.gat1.lin_src.weight.grad).all()
    assert model.gat1.lin_src.weight.grad.abs().sum() > 0

    assert model.gat2.lin_src.weight.grad is not None
    assert torch.isfinite(model.gat2.lin_src.weight.grad).all()
    assert model.gat2.lin_src.weight.grad.abs().sum() > 0

    # Verify classification head received gradients
    assert model.classifier_hidden[0].weight.grad is not None
    assert torch.isfinite(model.classifier_hidden[0].weight.grad).all()
    assert model.classifier_hidden[0].weight.grad.abs().sum() > 0

    assert model.classifier_out.weight.grad is not None
    assert torch.isfinite(model.classifier_out.weight.grad).all()
    assert model.classifier_out.weight.grad.abs().sum() > 0

    # All model parameters participate in gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


# 9. Training-mode and evaluation-mode behavior
def test_train_eval_modes_dropout_behavior():
    torch.manual_seed(10)
    model = MedShieldDiagnosticNet()

    # In eval mode: dropout is inactive
    model.eval()
    inp = _full_input(4)
    with torch.no_grad():
        out1 = model(inp).logits
        out2 = model(inp).logits
    assert torch.allclose(out1, out2)

    # In train mode: dropout module active
    model.train()
    assert model.classifier_hidden[2].p == 0.3
    assert model.classifier_hidden[2].training is True


# 10. Deterministic evaluation for identical inputs and weights
def test_eval_mode_deterministic():
    torch.manual_seed(7)
    model = MedShieldDiagnosticNet()
    model.eval()
    inp = _full_input(3)
    with torch.no_grad():
        a = model(inp).logits
        b = model(inp).logits
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


# 11. Checkpoint round-trip equivalence
def test_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(99)
    model = MedShieldDiagnosticNet(FusionModelConfig(trained=False))
    path = tmp_path / "fusion.pt"
    save_fusion_checkpoint(path, model, extra_metadata={"note": "untrained_init"})

    loaded = load_fusion_checkpoint(path)
    model.eval()
    loaded.eval()
    inp = _full_input(3)
    with torch.no_grad():
        torch.testing.assert_close(model(inp).logits, loaded(inp).logits)

    assert loaded.config.architecture_id == ARCHITECTURE_ID
    assert loaded.config.architecture_version == ARCHITECTURE_VERSION
    assert loaded.config.ecg_protocol_version == DEFAULT_ECG_PROTOCOL_VERSION
    assert loaded.config.tabular_schema_version == DEFAULT_TABULAR_SCHEMA_VERSION
    assert loaded.config.text_encoder_version == DEFAULT_TEXT_ENCODER_VERSION


# 12. Explicit rejection of incompatible checkpoints
def test_checkpoint_rejection_incompatible_cases(tmp_path: Path):
    # Non-existent file
    with pytest.raises(FusionCheckpointError, match="not found"):
        load_fusion_checkpoint(tmp_path / "nonexistent.pt")

    # Incompatible architecture ID
    bad_id = tmp_path / "bad_id.pt"
    torch.save({"architecture_id": "other.id", "architecture_version": "1.0.0", "state_dict": {}}, bad_id)
    with pytest.raises(FusionCheckpointError, match="Incompatible architecture_id"):
        load_fusion_checkpoint(bad_id)

    # Incompatible architecture version
    bad_ver = tmp_path / "bad_ver.pt"
    torch.save({"architecture_id": ARCHITECTURE_ID, "architecture_version": "99.0.0", "state_dict": {}}, bad_ver)
    with pytest.raises(FusionCheckpointError, match="Incompatible architecture_version"):
        load_fusion_checkpoint(bad_ver)

    # Missing state_dict
    bad_state = tmp_path / "bad_state.pt"
    torch.save({"architecture_id": ARCHITECTURE_ID, "architecture_version": ARCHITECTURE_VERSION}, bad_state)
    with pytest.raises(FusionCheckpointError, match="state_dict"):
        load_fusion_checkpoint(bad_state)

    # Corrupt state_dict keys
    bad_keys = tmp_path / "bad_keys.pt"
    torch.save({
        "architecture_id": ARCHITECTURE_ID,
        "architecture_version": ARCHITECTURE_VERSION,
        "state_dict": {"wrong_module.weight": torch.randn(10, 10)},
    }, bad_keys)
    with pytest.raises(FusionCheckpointError, match="Failed to load state_dict"):
        load_fusion_checkpoint(bad_keys)

    # Non-dict payload
    bad_payload = tmp_path / "bad_payload.pt"
    torch.save(["not", "a", "dict"], bad_payload)
    with pytest.raises(FusionCheckpointError, match="Invalid checkpoint format"):
        load_fusion_checkpoint(bad_payload)


# 13. Correct trained=false metadata for an untrained architecture
def test_checkpoint_trained_flag_preserved(tmp_path: Path):
    model = MedShieldDiagnosticNet(FusionModelConfig(trained=False))
    path = tmp_path / "untrained_model.pt"
    save_fusion_checkpoint(path, model)

    summary = checkpoint_summary(path)
    assert summary["trained"] is False
    assert summary["architecture_id"] == ARCHITECTURE_ID
    assert summary["architecture_version"] == ARCHITECTURE_VERSION

    loaded = load_fusion_checkpoint(path)
    assert loaded.config.trained is False


# 14. Modality status reporting without running the GAT when inputs are incomplete
def test_describe_modality_status_incomplete_reasons():
    model = MedShieldDiagnosticNet()
    inp = _full_input(2)

    partial = FusionModelInput(
        ecg=inp.ecg,
        clinical_text=inp.clinical_text,
        tabular=None,
        available=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        quality_ok=torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool),
    )
    status = model.describe_modality_status(partial)

    # GAT is NOT executed
    assert status.logits is None
    assert status.probabilities is None
    assert status.predicted_class is None
    assert status.confidence is None

    # Status masks are correct
    assert status.modality_used[0, ModalityIndex.TABULAR].item() is False
    assert status.complete[0].item() is False
    assert "tabular:unavailable" in status.incomplete_reasons[0]

    assert status.modality_used[1, ModalityIndex.CLINICAL_TEXT].item() is False
    assert status.complete[1].item() is False
    assert "clinical_text:quality_rejected" in status.incomplete_reasons[1]


# 15. No silent zero-fill, random fallback, or missing-modality bypass
def test_no_silent_zero_fill_or_fallback():
    model = MedShieldDiagnosticNet()
    inp = _full_input(2)

    # When tabular embedding is missing, cannot forward even if caller desires it
    inp.tabular = None
    with pytest.raises(FusionInputError, match="tabular embedding tensor is required"):
        model(inp)

    # All-zero tensor is accepted as a valid numerical tensor ONLY IF all masks are True
    inp_zero = _full_input(2)
    inp_zero.tabular = torch.zeros(2, TABULAR_EMBED_DIM)
    out = model(inp_zero)
    assert out.logits is not None
    assert torch.isfinite(out.logits).all()


# Contract parameters check
def test_gat_config_matches_contract():
    model = MedShieldDiagnosticNet()
    assert model.gat1.heads == 4
    assert model.gat2.heads == 1
    assert model.classifier_hidden[0].out_features == HIDDEN_DIM
    assert model.classifier_hidden[0].in_features == FLAT_FUSION_DIM
    assert model.classifier_out.out_features == NUM_CLASSES
    assert model.config.architecture_id == ARCHITECTURE_ID
    assert model.config.architecture_version == ARCHITECTURE_VERSION
    assert model.config.ecg_embed_dim == ECG_EMBED_DIM
    assert model.config.text_embed_dim == TEXT_EMBED_DIM
    assert model.config.tabular_embed_dim == TABULAR_EMBED_DIM
    assert model.config.node_dim == NODE_DIM
