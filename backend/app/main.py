"""MedShield-FL FastAPI Backend Application Entry Point."""

import logging
from datetime import datetime, timezone
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.router import api_v1_router
from backend.app.core.config import settings
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import StructuredLoggingMiddleware, logger
from backend.app.core.request_context import RequestIDMiddleware, get_current_request_id
from backend.app.schemas.common import HealthComponentStatus, HealthResponse, RequestMetadata

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Privacy Masked Federated Learning for Heart Disease Diagnosis Backend API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. Register Request ID Middleware (Outer)
app.add_middleware(RequestIDMiddleware)

# 2. Register Structured Logging Middleware
app.add_middleware(StructuredLoggingMiddleware)

# 3. CORS Middleware Setup
origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Register Centralized Exception Handlers
register_exception_handlers(app)

# 5. Include Versioned API v1 Router
app.include_router(api_v1_router)


@app.on_event("startup")
async def _warm_up_ecg_report_runtime() -> None:
    """Eagerly warm-up the ECG+Report runtime on startup (non-blocking)."""
    try:
        from backend.app.services.ecg_report_inference_service import ECGReportInferenceService
        ECGReportInferenceService.initialize_runtime()
        logger.info("ECG + Report runtime warm-up complete.")
    except Exception as exc:
        logger.warning("ECG + Report runtime warm-up skipped: %s", type(exc).__name__)


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Infrastructure health check endpoint.

    Exposes system readiness status without requiring authentication or DB connection.
    Dynamically queries ECG+Report runtime readiness.
    Does NOT claim that model inference or federated campaigns are clinically validated.
    """
    req_id = get_current_request_id()

    # Query ECG+Report runtime readiness (non-raising)
    ecg_report_status = "unknown"
    try:
        from backend.app.services.ecg_report_inference_service import ECGReportInferenceService
        readiness = ECGReportInferenceService.get_readiness()
        ecg_report_status = "operational" if readiness.get("ready") else "not_ready"
    except Exception:
        ecg_report_status = "unavailable"

    # GAT is truthfully blocked until a trained checkpoint is provided
    gat_status = "blocked_awaiting_trained_checkpoint"
    if settings.FUSION_INFERENCE_ENABLED:
        try:
            from backend.app.services.gat_fusion_service import GATFusionService
            gat_readiness = GATFusionService.get_readiness()
            gat_status = "operational" if gat_readiness.get("ready") else "blocked_awaiting_trained_checkpoint"
        except Exception:
            gat_status = "unavailable"

    return HealthResponse(
        status="healthy",
        service="medshield-fl-backend",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        message="MedShield-FL backend service is running normally.",
        components=HealthComponentStatus(
            health_check="operational",
            ecg_report_inference=ecg_report_status,
            gat_inference="pending_implementation",
            fl_orchestration="pending_implementation",
            database="configured",
        ),
        metadata=RequestMetadata(
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
