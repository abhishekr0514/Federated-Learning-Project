"""Centralized exception classes and FastAPI exception handlers."""

import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.request_context import get_current_request_id
from backend.app.schemas.common import APIErrorDetail, APIErrorResponse, RequestMetadata

logger = logging.getLogger("medshield.errors")


class BaseAppException(HTTPException):
    """Base application exception inheriting from FastAPI HTTPException."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: any = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.message = str(message)
        self.code = str(code)
        self.details = details


class NotFoundError(BaseAppException):
    def __init__(self, message: str = "Requested resource not found", details: any = None):
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class FeatureNotImplementedError(BaseAppException):
    def __init__(self, message: str = "This feature or domain operation is not yet implemented", details: any = None):
        super().__init__(
            message=message,
            code="NOT_IMPLEMENTED",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            details=details,
        )


class ValidationAppError(BaseAppException):
    def __init__(self, message: str = "Validation failed for input parameters", details: any = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


ValidationError = ValidationAppError


class SecurityError(BaseAppException):
    def __init__(self, message: str = "Authentication or authorization failed", details: any = None):
        super().__init__(
            message=message,
            code="SECURITY_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthenticationAppError(BaseAppException):
    def __init__(self, message: str = "Authentication token missing, invalid, or expired", details: any = None):
        super().__init__(
            message=message,
            code="UNAUTHENTICATED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class ForbiddenError(BaseAppException):
    def __init__(self, message: str = "Access forbidden for current system actor", details: any = None):
        super().__init__(
            message=message,
            code="FORBIDDEN",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class InferenceNotReadyError(BaseAppException):
    def __init__(self, message: str = "Inference pipeline is not ready", details: any = None):
        super().__init__(
            message=message,
            code="INFERENCE_NOT_READY",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class ConflictError(BaseAppException):
    def __init__(self, message: str = "Resource conflict", details: any = None):
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


def build_error_response(code: str, message: str, status_code: int, details: any = None) -> JSONResponse:
    """Helper constructing standardized APIErrorResponse payload."""
    req_id = get_current_request_id()
    payload = APIErrorResponse(
        error=APIErrorDetail(code=str(code), message=str(message), details=details),
        metadata=RequestMetadata(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on FastAPI instance."""

    @app.exception_handler(BaseAppException)
    async def custom_app_exception_handler(request: Request, exc: BaseAppException):
        logger.warning(f"Application exception [{exc.code}]: {exc.message}")
        return build_error_response(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        formatted_errors = []
        for err in exc.errors():
            formatted_errors.append({
                "field": " -> ".join(str(loc) for loc in err.get("loc", [])),
                "issue": err.get("msg", "Invalid input"),
            })
        logger.warning(f"Request validation error: {formatted_errors}")
        return build_error_response(
            code="VALIDATION_ERROR",
            message="Request validation failed for one or more fields.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=formatted_errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        code_map = {
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            501: "NOT_IMPLEMENTED",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return build_error_response(
            code=code,
            message=str(exc.detail),
            status_code=exc.status_code,
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(request: Request, exc: SQLAlchemyError):
        from backend.app.db.connectivity import database_target_summary

        target = database_target_summary()
        orig = getattr(exc, "orig", None)
        privilege_hint = ""
        if orig is not None and type(orig).__name__ == "InsufficientPrivilegeError":
            privilege_hint = (
                " (application DB role lacks table privileges; "
                "run scripts/apply_patient_questions_grants.py or scripts/migrations/002_grant_patient_clinician_questions.sql as postgres)"
            )
        logger.error(
            "Database error: %s target=%s@%s:%s/%s%s",
            type(exc).__name__,
            target.get("username"),
            target.get("host"),
            target.get("port"),
            target.get("database"),
            privilege_hint,
            exc_info=True,
        )
        return build_error_response(
            code="DATABASE_ERROR",
            message="A database operation error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {type(exc).__name__}", exc_info=True)
        return build_error_response(
            code="INTERNAL_SERVER_ERROR",
            message=str(exc) if str(exc) else "An unexpected server error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
