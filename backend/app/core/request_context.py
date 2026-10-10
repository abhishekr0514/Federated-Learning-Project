"""Request context management and request ID tracking middleware."""

import uuid
from contextvars import ContextVar
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable to hold current request ID across async tasks
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id_ctx", default=None)

REQUEST_ID_HEADER = "X-Request-ID"


def get_current_request_id() -> str:
    """Retrieve the current request ID from contextvar, returning empty string if unset."""
    return request_id_ctx.get() or ""


get_request_id = get_current_request_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that extracts or generates a unique request ID (X-Request-ID) for every HTTP request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract incoming X-Request-ID or generate new UUID4
        incoming_id = request.headers.get(REQUEST_ID_HEADER)
        req_id = incoming_id if incoming_id else str(uuid.uuid4())

        # Set contextvar for access in logger and handlers
        token = request_id_ctx.set(req_id)

        # Attach request_id to request state
        request.state.request_id = req_id

        try:
            response = await call_next(request)
            # Add X-Request-ID to response headers
            response.headers[REQUEST_ID_HEADER] = req_id
            return response
        finally:
            request_id_ctx.reset(token)
