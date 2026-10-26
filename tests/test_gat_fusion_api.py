"""Tests for two-modality GAT ECG + clinical report fusion API, service, and schemas."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)


class TestGATFusionEvaluateRequest:
    def _valid_ecg(self) -> list[float]:
        return [0.1] * 128

    def _valid_text(self) -> list[float]:
        return [0.2] * 128

    def test_both_modalities_required(self):
        req = GATFusionEvaluateRequest(
            ecg_embedding=self._valid_ecg(),
            text_embedding=self._valid_text(),
        )
        assert len(req.ecg_embedding) == 128
        assert len(req.text_embedding) == 128

    def test_ecg_only_raises(self):
        with pytest.raises(ValidationError):
            GATFusionEvaluateRequest(ecg_embedding=[0.1] * 128)  # type: ignore[call-arg]

    def test_text_only_raises(self):
        with pytest.raises(ValidationError):
            GATFusionEvaluateRequest(text_embedding=[0.2] * 128)  # type: ignore[call-arg]

    def test_no_modalities_raises(self):
        with pytest.raises(ValidationError):
            GATFusionEvaluateRequest()  # type: ignore[call-arg]

    def test_ecg_wrong_dim_raises(self):
        with pytest.raises(ValidationError, match="128 dimensions"):
            GATFusionEvaluateRequest(
                ecg_embedding=[0.1] * 64,
                text_embedding=self._valid_text(),
            )

    def test_text_wrong_dim_raises(self):
        with pytest.raises(ValidationError, match="128 dimensions"):
            GATFusionEvaluateRequest(
                ecg_embedding=self._valid_ecg(),
                text_embedding=[0.1] * 32,
            )

    def test_ecg_nan_raises(self):
        emb = [0.1] * 128
        emb[5] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            GATFusionEvaluateRequest(ecg_embedding=emb, text_embedding=self._valid_text())

    def test_text_inf_raises(self):
        emb = [0.1] * 128
        emb[0] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            GATFusionEvaluateRequest(ecg_embedding=self._valid_ecg(), text_embedding=emb)

    def test_idempotency_key_max_length_ok(self):
        req = GATFusionEvaluateRequest(
            ecg_embedding=self._valid_ecg(),
            text_embedding=self._valid_text(),
            idempotency_key="k" * 128,
        )
        assert len(req.idempotency_key) == 128


class TestGATFusionReadinessResponse:
    def test_defaults(self):
        r = GATFusionReadinessResponse(ready=False)
        assert r.ready is False
        assert r.architecture_id == "medshield.diagnostic_net.gat_ecg_report"
        assert r.embedding_contracts == {"ecg": 128, "clinical_text": 128}
        assert r.pairing_gate_verdict == "BLOCKED"
        assert isinstance(r.blockers, list)


class TestGATFusionEvaluateResponse:
    def test_basic_response(self):
        r = GATFusionEvaluateResponse(
            prediction_id=uuid.uuid4(),
            confidence_uncalibrated=0.7,
            predicted_class_index=1,
            predicted_class_label="MI/STTC",
            target_definition="Research proxy only",
            architecture_id="medshield.diagnostic_net.gat_ecg_report",
            model_version_tag="gat-v1",
            disclaimer="Research only",
        )
        assert r.is_research_only is True
        assert r.modalities_used == ["ecg", "clinical_text"]


class TestGATFusionServiceReadiness:
    def test_readiness_reports_two_modality_contract(self):
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        assert isinstance(result["ready"], bool)
        assert result["embedding_contracts"]["ecg"] == 128
        assert result["embedding_contracts"]["clinical_text"] == 128
        assert "tabular" not in result["embedding_contracts"]

    def test_readiness_exposes_pairing_gate(self):
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        assert "pairing_gate_verdict" in result
        assert "pairing_gate_allowed" in result

    def test_readiness_shows_artifacts_when_deployed(self, tmp_path, monkeypatch):
        """Uses an isolated MODEL_DIR with manifest + checkpoint (deploy dir may omit checkpoint)."""
        import shutil

        from backend.app.core.config import settings
        from backend.app.services.gat_fusion_service import GATFusionService

        repo = Path(__file__).resolve().parents[1]
        deploy_manifest = repo / "models" / "gat_ecg_report" / "medshield_fusion_manifest.json"
        checkpoint_src = repo / "data" / "processed" / "gat_ecg_report" / "runs" / "20261025T182109Z" / "best_gat_checkpoint.pt"
        if not deploy_manifest.is_file() or not checkpoint_src.is_file():
            pytest.skip("GAT manifest or trained checkpoint artifact not present in workspace")

        model_dir = tmp_path / "gat_model_dir"
        model_dir.mkdir()
        shutil.copy2(deploy_manifest, model_dir / "medshield_fusion_manifest.json")
        shutil.copy2(checkpoint_src, model_dir / "best_gat_checkpoint.pt")

        monkeypatch.setattr(settings, "FUSION_MODEL_DIR", str(model_dir))
        monkeypatch.setattr(settings, "FUSION_INFERENCE_ENABLED", True)

        from ml.inference import fusion_runtime

        fusion_runtime._RUNTIME = None
        fusion_runtime.get_fusion_inference_runtime(model_dir, inference_enabled=True, reset=True)

        result = GATFusionService.get_readiness()
        assert result["manifest_present"] is True
        assert result["checkpoint_present"] is True
        assert result["checkpoint_trained"] is True

    def test_readiness_does_not_expose_filesystem_paths(self):
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        all_values = str(result)
        assert "C:\\" not in all_values
        assert "/home/" not in all_values


def _make_test_app():
    from fastapi import FastAPI
    from backend.app.api.v1.endpoints.gat_fusion import router as gat_router

    app = FastAPI()
    app.include_router(gat_router, prefix="/api/v1")
    return app


class TestGATFusionRoutes:
    def test_readiness_get_returns_200_no_auth(self):
        from fastapi.testclient import TestClient

        with patch(
            "backend.app.api.v1.endpoints.gat_fusion.GATFusionService.get_readiness",
            return_value={
                "ready": False,
                "architecture_id": "medshield.diagnostic_net.gat_ecg_report",
                "architecture_version": "1.0.0",
                "embedding_contracts": {"ecg": 128, "clinical_text": 128},
                "manifest_present": True,
                "checkpoint_present": True,
                "checkpoint_trained": True,
                "model_loaded": False,
                "pairing_gate_allowed": True,
                "pairing_gate_verdict": "ALLOWED",
                "inference_enabled": False,
                "blockers": ["FUSION_INFERENCE_ENABLED is false"],
                "required_to_unblock": ["Set FUSION_INFERENCE_ENABLED=true after verified trained checkpoint."],
                "environment": "test",
            },
        ):
            client = TestClient(_make_test_app(), raise_server_exceptions=False)
            resp = client.get("/api/v1/gat-fusion/readiness")
            assert resp.status_code == 200
            body = resp.json()
            assert body["architecture_id"] == "medshield.diagnostic_net.gat_ecg_report"

    def test_evaluate_post_requires_auth(self):
        from fastapi.testclient import TestClient

        client = TestClient(_make_test_app(), raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/gat-fusion/evaluate",
            json={"ecg_embedding": [0.1] * 128, "text_embedding": [0.2] * 128},
        )
        assert resp.status_code in (401, 403, 422)
