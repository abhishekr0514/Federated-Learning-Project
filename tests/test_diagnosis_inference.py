"""Diagnosis inference service, API gates, and persistence tests."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.models import DiagnosisPrediction
from tests.test_clinician_workspace import TestingSessionLocal, seeded_clinician_env

pytest_plugins = ("tests.test_clinician_workspace",)

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from ml.models.gnn_fusion import (  # noqa: E402
    ECG_EMBED_DIM,
    TABULAR_EMBED_DIM,
    TEXT_EMBED_DIM,
    FusionModelConfig,
    MedShieldDiagnosticNet,
    save_fusion_checkpoint,
)


def _write_trained_manifest(tmp_path, *, trained: bool = True) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "fusion.pt"
    cfg = FusionModelConfig(trained=trained)
    model = MedShieldDiagnosticNet(cfg)
    save_fusion_checkpoint(ckpt, model)
    manifest = {
        "manifest_version": "1",
        "checkpoint_relative_path": "checkpoints/fusion.pt",
        "architecture_id": "medshield.diagnostic_net.gat_fusion",
        "architecture_version": "1.0.0",
        "model_version_tag": "test-gat-proxy-v1",
        "target_definition": "PTB-XL NSR vs pathological superclasses proxy (research)",
        "prediction_type": "proxy_evaluation",
        "require_trained_checkpoint": True,
    }
    (tmp_path / "medshield_fusion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _embedding_payload() -> dict:
    return {
        "embeddings": {
            "ecg": [0.01] * ECG_EMBED_DIM,
            "clinical_text": [0.02] * TEXT_EMBED_DIM,
            "tabular": [0.03] * TABULAR_EMBED_DIM,
        }
    }


@pytest.mark.anyio
async def test_evaluate_returns_503_when_not_ready(seeded_clinician_env, monkeypatch):
    monkeypatch.setattr(settings, "INFERENCE_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "MODEL_DIR", "nonexistent_models_dir", raising=False)
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/diagnosis/evaluate?patient_id={pat_id}",
            headers=headers,
            json=_embedding_payload(),
        )
    assert res.status_code == 503
    async with TestingSessionLocal() as session:
        count = (await session.execute(select(func.count(DiagnosisPrediction.id)))).scalar_one()
        assert count == 0


@pytest.mark.anyio
async def test_evaluate_persists_prediction_with_trained_checkpoint(
    seeded_clinician_env, tmp_path, monkeypatch
):
    _write_trained_manifest(tmp_path, trained=True)
    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(settings, "INFERENCE_ENABLED", True, raising=False)

    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/diagnosis/evaluate?patient_id={pat_id}",
            headers=headers,
            json=_embedding_payload(),
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "EVALUATED"
    assert body["proxy_task"] is True
    assert 0.0 <= body["confidence_uncalibrated"] <= 1.0

    async with TestingSessionLocal() as session:
        count = (await session.execute(select(func.count(DiagnosisPrediction.id)))).scalar_one()
        assert count == 1


@pytest.mark.anyio
async def test_untrained_checkpoint_blocked(seeded_clinician_env, tmp_path, monkeypatch):
    _write_trained_manifest(tmp_path, trained=False)
    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(settings, "INFERENCE_ENABLED", True, raising=False)
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/diagnosis/evaluate?patient_id={pat_id}",
            headers=headers,
            json=_embedding_payload(),
        )
    assert res.status_code == 503


@pytest.mark.anyio
async def test_patient_role_forbidden(seeded_clinician_env, tmp_path, monkeypatch):
    _write_trained_manifest(tmp_path, trained=True)
    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(settings, "INFERENCE_ENABLED", True, raising=False)
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_pat']}"}
    pat_id = seeded_clinician_env["pat_assigned_1_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/diagnosis/evaluate?patient_id={pat_id}",
            headers=headers,
            json=_embedding_payload(),
        )
    assert res.status_code == 403


@pytest.mark.anyio
async def test_unassigned_patient_returns_404(seeded_clinician_env, tmp_path, monkeypatch):
    _write_trained_manifest(tmp_path, trained=True)
    monkeypatch.setattr(settings, "MODEL_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(settings, "INFERENCE_ENABLED", True, raising=False)
    headers = {"Authorization": f"Bearer {seeded_clinician_env['token_clin_1']}"}
    unassigned = seeded_clinician_env["pat_unassigned_1_id"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post(
            f"/api/v1/diagnosis/evaluate?patient_id={unassigned}",
            headers=headers,
            json=_embedding_payload(),
        )
    assert res.status_code == 404
