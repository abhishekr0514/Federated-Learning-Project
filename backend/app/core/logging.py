"""Structured logging setup with privacy filter and request ID context integration."""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.app.core.config import settings
from backend.app.core.request_context import get_current_request_id

# Sensitive keys to sanitize in logs to enforce privacy boundaries
SENSITIVE_KEYS = {
    "password", "secret", "token", "access_token", "refresh_token",
    "jwt", "api_key", "authorization", "ssn", "patient_name",
    "clinical_text", "clinical_notes", "ecg_data", "ecg_signal", "pii"
}


def sanitize_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub sensitive keys from log metadata dictionary."""
    clean_data = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            clean_data[key] = "[REDACTED]"
        elif isinstance(value, dict):
            clean_data[key] = sanitize_log_dict(value)
        else:
            clean_data[key] = value
    return clean_data


class JSONLogFormatter(logging.Formatter):
    """Structured JSON log formatter including timestamp, request ID, level, and metadata."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_current_request_id(),
        }

        # Include additional attributes if present
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry["extra"] = sanitize_log_dict(record.extra_data)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging() -> logging.Logger:
    """Configure system-wide logging with JSON formatting."""
    root_logger = logging.getLogger("medshield")
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = JSONLogFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Prevent duplicate propagation
    root_logger.propagate = False
    return root_logger


def get_logger(name: str = "medshield") -> logging.Logger:
    """Retrieve logger instance for specified namespace."""
    return logging.getLogger(name)


logger = setup_logging()


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware logging request method, path, status code, duration, and request ID."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Build request log entry without sensitive data
        extra = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else "unknown",
        }

        logger.info(
            f"HTTP {request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            extra={"extra_data": extra},
        )
        return response
