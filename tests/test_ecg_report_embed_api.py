"""Tests for ECG-report encoder embed endpoint (128-d contract for GAT)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from backend.app.schemas.ecg_report_inference import ECGReportEmbedRequest, ECGReportEmbedResponse


class TestECGReportEmbedRequest:
    def _ecg(self) -> list[list[float]]:
        return [[0.0] * 1000 for _ in range(12)]

    def test_valid_shape(self):
        req = ECGReportEmbedRequest(ecg_leads=self._ecg(), report_text="Normal sinus rhythm.")
        assert len(req.ecg_leads) == 12

    def test_bad_lead_count(self):
        with pytest.raises(ValidationError):
            ECGReportEmbedRequest(ecg_leads=[[0.0] * 1000], report_text="x")


class TestECGReportEmbedService:
    @pytest.mark.anyio
    async def test_embed_returns_128_vectors(self):
        from backend.app.schemas.ecg_report_inference import ECGReportEmbedRequest
        from backend.app.services.ecg_report_inference_service import ECGReportInferenceService

        body = ECGReportEmbedRequest(
            ecg_leads=[[0.0] * 1000 for _ in range(12)],
            report_text="No acute ST changes.",
        )
        fake = {
            "ecg_embedding": [0.01] * 128,
            "text_embedding": [0.02] * 128,
            "architecture_id": "medshield.ecg_report_fusion.v1",
            "model_version_tag": "ecg-report-fusion-v1-ptbxl",
            "privacy_validation_status": "VALIDATED",
        }
        with patch(
            "backend.app.services.ecg_report_inference_service.get_ecg_report_runtime"
        ) as mock_rt:
            runtime = mock_rt.return_value
            runtime.assess_readiness.return_value = SimpleNamespace(
                ready=True,
                manifest_present=True,
                checkpoint_present=True,
                to_safe_dict=lambda: {"ready": True},
            )
            runtime.extract_encoder_embeddings.return_value = fake

            resp = await ECGReportInferenceService.embed_modalities(body=body)
            assert isinstance(resp, ECGReportEmbedResponse)
            assert len(resp.ecg_embedding) == 128
            assert len(resp.text_embedding) == 128
            runtime.extract_encoder_embeddings.assert_called_once()
