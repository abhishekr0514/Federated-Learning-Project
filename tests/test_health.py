"""Test suite for GET /health endpoint."""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 OK with expected status, metadata, and X-Request-ID header."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "medshield-fl-backend"
    assert "version" in data
    assert "environment" in data
    assert data["components"]["health_check"] == "operational"
    assert data["components"]["gat_inference"] == "pending_implementation"
    assert data["components"]["database"] == "configured"
    
    # Metadata assertion
    assert "metadata" in data
    assert "request_id" in data["metadata"]
    assert "X-Request-ID" in response.headers
