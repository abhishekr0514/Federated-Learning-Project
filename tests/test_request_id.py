"""Test suite for Request ID middleware generation and propagation."""

import uuid
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_request_id_auto_generation():
    """Verify X-Request-ID is automatically generated when not supplied."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    
    # Check that generated ID is a valid UUID
    req_id = response.headers["X-Request-ID"]
    parsed_uuid = uuid.UUID(req_id)
    assert str(parsed_uuid) == req_id


def test_request_id_propagation():
    """Verify custom X-Request-ID header in request is propagated to response."""
    custom_id = "test-custom-request-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
    
    data = response.json()
    assert data["metadata"]["request_id"] == custom_id
