"""Common Pydantic response and error schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class RequestMetadata(BaseModel):
    """Metadata attached to API responses."""

    request_id: str = Field(default="unknown", description="Unique request identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Response timestamp in ISO 8601 UTC",
    )
    message: Optional[str] = Field(default=None, description="Optional informational response message")


def default_request_metadata() -> RequestMetadata:
    """Default factory generating request metadata using context request ID."""
    from backend.app.core.request_context import get_current_request_id
    req_id = get_current_request_id() or "unknown"
    return RequestMetadata(request_id=req_id)


class APIErrorDetail(BaseModel):
    """Detailed error object contained within API error responses."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Any] = Field(default=None, description="Optional error context or validation field list")


class APIErrorResponse(BaseModel):
    """Standardized error response container."""

    error: APIErrorDetail
    metadata: RequestMetadata = Field(default_factory=default_request_metadata)


class GenericSuccessEnvelope(BaseModel, Generic[T]):
    """Generic success envelope for data responses."""

    status: str = Field(default="success", description="Status string")
    data: T = Field(..., description="Response payload")
    metadata: RequestMetadata = Field(default_factory=default_request_metadata)


class HealthComponentStatus(BaseModel):
    """Status of individual backend components."""

    health_check: str = Field(default="operational")
    gat_inference: str = Field(default="pending_implementation")
    fl_orchestration: str = Field(default="pending_implementation")
    database: str = Field(default="configured")


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = Field(default="healthy")
    service: str = Field(default="medshield-fl-backend")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    message: str = Field(default="MedShield-FL backend service is running normally.")
    components: HealthComponentStatus
    metadata: RequestMetadata = Field(default_factory=default_request_metadata)
