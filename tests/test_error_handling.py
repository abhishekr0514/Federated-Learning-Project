"""Test suite for centralized exception handling and validation formatting."""

import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from backend.app.core.security import create_access_token
from backend.app.main import app

SUPER_ADMIN_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_custom_not_found_exception():
    """Verify non-existent route returns standard 404 error envelope."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/non-existent-path")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert "metadata" in data
        assert "request_id" in data["metadata"]


@pytest.mark.anyio
async def test_unhandled_exception_does_not_leak_internals():
    """Verify unimplemented feature returns 501 error envelope without leaking internal stack traces."""
    token, _ = create_access_token(user_id=str(SUPER_ADMIN_ID))
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/explanations/dice", headers=headers)
        assert response.status_code == 501

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_IMPLEMENTED"
        # Ensure no python stack trace is present in response
        assert "Traceback" not in response.text
