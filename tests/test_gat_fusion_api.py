"""Tests for the three-modality GAT fusion API endpoint, service, and schemas.

Covers:
- Schema validation (dimensions, finiteness, at-least-one-modality)
- Service readiness response structure (always blocked until model is trained)
- API endpoint routing and authentication guards
- Integration: readiness endpoint returns truthful diagnostics
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.schemas.gat_fusion import (
    GATFusionEvaluateRequest,
    GATFusionEvaluateResponse,
    GATFusionReadinessResponse,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestGATFusionEvaluateRequest:
    def _valid_ecg(self, n: int = 128) -> List[float]:
        return [0.1] * n

    def _valid_text(self, n: int = 128) -> List[float]:
        return [0.2] * n

    def _valid_tab(self, n: int = 64) -> List[float]:
        return [0.3] * n

    def test_all_modalities_present(self):
        req = GATFusionEvaluateRequest(
            ecg_embedding=self._valid_ecg(),
            text_embedding=self._valid_text(),
            tabular_embedding=self._valid_tab(),
        )
        assert req.ecg_embedding is not None
        assert req.text_embedding is not None
        assert req.tabular_embedding is not None

    def test_ecg_only(self):
        req = GATFusionEvaluateRequest(ecg_embedding=self._valid_ecg())
        assert req.ecg_embedding is not None
        assert req.text_embedding is None
        assert req.tabular_embedding is None

    def test_text_only(self):
        req = GATFusionEvaluateRequest(text_embedding=self._valid_text())
        assert req.text_embedding is not None
        assert req.ecg_embedding is None

    def test_tabular_only(self):
        req = GATFusionEvaluateRequest(tabular_embedding=self._valid_tab())
        assert req.tabular_embedding is not None

    def test_no_modalities_raises(self):
        with pytest.raises(ValidationError, match="At least one"):
            GATFusionEvaluateRequest()

    def test_ecg_wrong_dim_raises(self):
        with pytest.raises(ValidationError, match="128 dimensions"):
            GATFusionEvaluateRequest(ecg_embedding=[0.1] * 64)

    def test_text_wrong_dim_raises(self):
        with pytest.raises(ValidationError, match="128 dimensions"):
            GATFusionEvaluateRequest(text_embedding=[0.1] * 32)

    def test_tabular_wrong_dim_raises(self):
        with pytest.raises(ValidationError, match="64 dimensions"):
            GATFusionEvaluateRequest(tabular_embedding=[0.1] * 128)

    def test_ecg_nan_raises(self):
        emb = [0.1] * 128
        emb[5] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            GATFusionEvaluateRequest(ecg_embedding=emb)

    def test_text_inf_raises(self):
        emb = [0.1] * 128
        emb[0] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            GATFusionEvaluateRequest(text_embedding=emb)

    def test_tabular_neg_inf_raises(self):
        emb = [0.3] * 64
        emb[10] = float("-inf")
        with pytest.raises(ValidationError, match="non-finite"):
            GATFusionEvaluateRequest(tabular_embedding=emb)

    def test_idempotency_key_too_long_raises(self):
        with pytest.raises(ValidationError):
            GATFusionEvaluateRequest(
                ecg_embedding=self._valid_ecg(),
                idempotency_key="x" * 129,
            )

    def test_idempotency_key_max_length_ok(self):
        req = GATFusionEvaluateRequest(
            ecg_embedding=self._valid_ecg(),
            idempotency_key="k" * 128,
        )
        assert len(req.idempotency_key) == 128


class TestGATFusionReadinessResponse:
    def test_defaults(self):
        r = GATFusionReadinessResponse(ready=False)
        assert r.ready is False
        assert r.architecture_id == "medshield.diagnostic_net.gat_fusion"
        assert r.architecture_version == "1.0.0"
        assert r.embedding_contracts == {"ecg": 128, "clinical_text": 128, "tabular": 64}
        assert r.linkage_gate_allowed is False
        assert r.linkage_gate_verdict == "BLOCKED"
        assert isinstance(r.blockers, list)
        assert isinstance(r.required_to_unblock, list)

    def test_ready_false_serialisation(self):
        r = GATFusionReadinessResponse(
            ready=False,
            blockers=["GAT not trained"],
            required_to_unblock=["Multimodal training required"],
        )
        d = r.model_dump()
        assert d["ready"] is False
        assert "GAT not trained" in d["blockers"]


class TestGATFusionEvaluateResponse:
    def test_basic_response(self):
        r = GATFusionEvaluateResponse(
            prediction_id=uuid.uuid4(),
            confidence_uncalibrated=0.7,
            predicted_class_index=1,
            predicted_class_label="CVD",
            target_definition="Research proxy only",
            architecture_id="medshield.diagnostic_net.gat_fusion",
            model_version_tag="gat-v1",
            disclaimer="Research only",
        )
        assert r.is_research_only is True
        assert r.proxy_task is True
        assert 0.0 <= r.confidence_uncalibrated <= 1.0


# ---------------------------------------------------------------------------
# Service readiness test (always blocked until model is trained)
# ---------------------------------------------------------------------------


class TestGATFusionServiceReadiness:
    def test_readiness_always_blocked_without_trained_checkpoint(self):
        """GAT readiness must always report ready=False until a trained checkpoint is present."""
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        assert isinstance(result, dict)
        assert result["ready"] is False
        assert len(result["blockers"]) > 0
        assert isinstance(result["embedding_contracts"], dict)
        assert result["embedding_contracts"]["ecg"] == 128
        assert result["embedding_contracts"]["clinical_text"] == 128
        assert result["embedding_contracts"]["tabular"] == 64

    def test_readiness_exposes_linkage_gate_verdict(self):
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        assert "linkage_gate_verdict" in result
        assert "linkage_gate_allowed" in result

    def test_readiness_does_not_expose_filesystem_paths(self):
        """Safe dict must not contain raw filesystem paths."""
        from backend.app.services.gat_fusion_service import GATFusionService

        result = GATFusionService.get_readiness()
        # Stringify all values and check no Windows absolute path leaks
        all_values = str(result)
        assert "C:\\" not in all_values
        assert "/home/" not in all_values


# ---------------------------------------------------------------------------
# API route tests (with TestClient, mocking DB and auth)
# ---------------------------------------------------------------------------


def _make_test_app():
    """Create an isolated FastAPI app with only the gat_fusion router for route testing."""
    from fastapi import FastAPI
    from backend.app.api.v1.endpoints.gat_fusion import router as gat_router

    app = FastAPI()
    app.include_router(gat_router, prefix="/api/v1")
    return app


class TestGATFusionRouteRegistration:
    def _route_paths(self, app) -> list:
        """Collect registered route paths, compatible with both APIRoute and _IncludedRouter."""
        from starlette.routing import Route
        paths = []
        for r in app.routes:
            if hasattr(r, "path"):
                paths.append(r.path)
            # Handle sub-routers that expose their own .routes
            if hasattr(r, "routes"):
                for sub in r.routes:
                    if hasattr(sub, "path"):
                        paths.append(sub.path)
        return paths

    def test_readiness_route_exists(self):
        """Probe the readiness route via HTTP; a non-404 confirms it is registered."""
        from fastapi import FastAPI
        from backend.app.api.v1.endpoints.gat_fusion import router as gat_router
        from unittest.mock import patch

        with patch(
            "backend.app.api.v1.endpoints.gat_fusion.GATFusionService.get_readiness",
            return_value={
                "ready": False,
                "architecture_id": "medshield.diagnostic_net.gat_fusion",
                "architecture_version": "1.0.0",
                "embedding_contracts": {"ecg": 128, "clinical_text": 128, "tabular": 64},
                "manifest_present": False,
                "checkpoint_present": False,
                "checkpoint_trained": None,
                "model_loaded": False,
                "linkage_gate_allowed": False,
                "linkage_gate_verdict": "BLOCKED",
                "blockers": [],
                "required_to_unblock": [],
                "environment": "test",
            },
        ):
            app = _make_test_app()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/gat-fusion/readiness")
            assert resp.status_code != 404, "readiness route not registered"

    def test_evaluate_route_exists(self):
        """Probe the evaluate route via HTTP; any non-404 confirms it is registered."""
        from fastapi import FastAPI
        from backend.app.api.v1.endpoints.gat_fusion import router as gat_router

        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        # Without auth credentials this returns 422 (unmet dependencies), not 404
        resp = client.post(
            "/api/v1/gat-fusion/evaluate",
            json={"ecg_embedding": [0.1] * 128},
        )
        assert resp.status_code != 404, "evaluate route not registered"

    def test_readiness_get_returns_200_no_auth(self):
        """The /readiness GET endpoint should return 200 without auth (diagnostics are public)."""
        from fastapi import FastAPI
        from backend.app.api.v1.endpoints.gat_fusion import router as gat_router
        from fastapi.routing import APIRoute

        # Patch the service so no runtime is needed
        with patch(
            "backend.app.api.v1.endpoints.gat_fusion.GATFusionService.get_readiness",
            return_value={
                "ready": False,
                "architecture_id": "medshield.diagnostic_net.gat_fusion",
                "architecture_version": "1.0.0",
                "embedding_contracts": {"ecg": 128, "clinical_text": 128, "tabular": 64},
                "manifest_present": False,
                "checkpoint_present": False,
                "checkpoint_trained": None,
                "model_loaded": False,
                "linkage_gate_allowed": False,
                "linkage_gate_verdict": "BLOCKED",
                "blockers": ["GAT weights not trained"],
                "required_to_unblock": ["Verified multimodal training"],
                "environment": "test",
            },
        ):
            app = FastAPI()
            app.include_router(gat_router, prefix="/api/v1")
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/gat-fusion/readiness")
            assert resp.status_code == 200
            body = resp.json()
            assert body["ready"] is False
            assert body["architecture_id"] == "medshield.diagnostic_net.gat_fusion"
            assert "ecg" in body["embedding_contracts"]

    def test_evaluate_post_requires_auth(self):
        """The /evaluate POST endpoint must reject unauthenticated requests with 422 or 403."""
        from fastapi import FastAPI
        from backend.app.api.v1.endpoints.gat_fusion import router as gat_router

        app = FastAPI()
        app.include_router(gat_router, prefix="/api/v1")
        client = TestClient(app, raise_server_exceptions=False)
        # No auth header → dependency injection will fail or return 422
        resp = client.post(
            "/api/v1/gat-fusion/evaluate",
            json={"ecg_embedding": [0.1] * 128},
        )
        # Expect 401/403/422 — any of these indicate the auth guard is present
        assert resp.status_code in (401, 403, 422)
