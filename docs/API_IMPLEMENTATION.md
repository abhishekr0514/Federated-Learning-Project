# MedShield-FL Backend API & Infrastructure Implementation Specification

## Overview

This document details the architectural foundation, versioning strategy, middleware stack, error handling mechanisms, database session lifecycle, and security controls established for the MedShield-FL FastAPI backend.

---

## 1. API Versioning Strategy

All domain and business API operations are strictly versioned under the `/api/v1` URL prefix:

- **Unversioned Utility Endpoint**: `GET /health` (Infrastructure readiness)
- **Versioned Root Prefix**: `/api/v1`

### Registered Router Namespaces (`/api/v1/*`)
1. `/api/v1/auth`: Authentication and session management
2. `/api/v1/users`: User management & profile lookup
3. `/api/v1/hospitals`: Hospital node registration & telemetry
4. `/api/v1/patients`: Local patient record management (Hospital Boundary)
5. `/api/v1/clinicians`: Clinician portal workflow
6. `/api/v1/documents`: Clinical text reports & document metadata
7. `/api/v1/privacy`: Privacy masking & differential privacy controls
8. `/api/v1/diagnosis`: Multimodal GAT diagnostic inference (Pending ML training)
9. `/api/v1/explanations`: DiCE counterfactual explanations
10. `/api/v1/causal`: DoWhy causal DAG analysis
11. `/api/v1/federated`: Flower FL campaign orchestration
12. `/api/v1/models`: Model registry & readiness status
13. `/api/v1/metrics`: Global model evaluation metrics
14. `/api/v1/audit`: System audit logs & telemetry

---

## 2. Request Tracing & Middleware Architecture

```
Incoming Request ──► RequestIDMiddleware ──► StructuredLoggingMiddleware ──► CORSMiddleware ──► Router ──► Response
                        (Attaches X-Request-ID)   (Logs duration & scrubbed data)
```

### Request ID Handling (`X-Request-ID`)
1. **Header Extraction**: Checks incoming HTTP requests for `X-Request-ID`.
2. **Generation**: If missing, generates a unique UUIDv4 string.
3. **Context Propagation**: Sets Python `ContextVar` (`request_id_ctx`) accessible across async calls.
4. **Header Egress**: Returns `X-Request-ID` in HTTP response headers.

---

## 3. Privacy-Compliant Structured Logging

- **Format**: Structured JSON via `JSONLogFormatter`.
- **Fields**: `timestamp`, `level`, `logger`, `message`, `request_id`, `extra` (`method`, `path`, `status_code`, `duration_ms`).
- **Privacy Boundary Filter**: Sensitive fields (`password`, `secret`, `token`, `access_token`, `jwt`, `api_key`, `ssn`, `patient_name`, `clinical_text`, `ecg_data`) are automatically scrubbed and logged as `[REDACTED]`.

---

## 4. Response Conventions & Error Schema

### Generic Success Envelope
```json
{
  "status": "success",
  "data": { ... },
  "metadata": {
    "request_id": "c5f3e9b1-7a2e-4b6d-91a5-83f60d4b9102",
    "timestamp": "2026-10-09T20:15:00Z"
  }
}
```

### Standard Error Schema (`APIErrorResponse`)
```json
{
  "error": {
    "code": "NOT_IMPLEMENTED",
    "message": "This feature or domain operation is not yet implemented",
    "details": null
  },
  "metadata": {
    "request_id": "c5f3e9b1-7a2e-4b6d-91a5-83f60d4b9102",
    "timestamp": "2026-10-09T20:15:00Z"
  }
}
```

---

## 5. Exception Handling Matrix

| Exception Type | Error Code | HTTP Status | Response Details |
| :--- | :--- | :---: | :--- |
| `NotFoundError` | `NOT_FOUND` | 404 | Resource missing message |
| `FeatureNotImplementedError` | `NOT_IMPLEMENTED` | 501 | Unimplemented feature notice |
| `ValidationAppError` | `VALIDATION_ERROR` | 422 | Field-level validation issues |
| `SecurityError` | `SECURITY_ERROR` | 401 | Authentication requirement |
| `RequestValidationError` | `VALIDATION_ERROR` | 422 | Formatted Pydantic field list |
| `SQLAlchemyError` | `DATABASE_ERROR` | 500 | Generic message (SQL scrubbed) |
| `Unhandled Exception` | `INTERNAL_SERVER_ERROR` | 500 | Generic message (Trace scrubbed) |

---

## 6. Database Session Lifecycle & Layer Separation

- **Engine**: Async Engine (`AsyncEngine`) using `postgresql+asyncpg://`.
- **Session Dependency**: `get_db()` provides request-scoped `AsyncSession` with automatic commit on success and rollback on exception.
- **Repository Pattern**: `BaseRepository` handles persistence operations.
- **Service Layer**: `BaseService` handles application orchestration logic.
- **Route Handlers**: Coordinate request/response handling only; no direct SQL or inline query execution in endpoints.

---

## 7. Currently Unavailable Functionality Policy

- Unimplemented domain operations return `501 Not Implemented`.
- **Zero Fake Predictions**: Legacy formula predictions are disabled. `POST /api/v1/diagnosis/evaluate` returns HTTP 501 until empirical GAT model training is verified.
- **Zero Fake Domain Records**: No speculative database tables or hardcoded fake patient data are introduced.
