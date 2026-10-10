"""Main API v1 Router aggregating all domain router groups."""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import (
    admin,
    audit,
    auth,
    causal,
    clinician,
    clinicians,
    diagnosis,
    documents,
    explanations,
    federated,
    hospital,
    hospitals,
    metrics,
    models,
    patient,
    patients,
    privacy,
    users,
)

api_v1_router = APIRouter(prefix="/api/v1")

# Include domain router namespaces
api_v1_router.include_router(admin.router)
api_v1_router.include_router(hospital.router)
api_v1_router.include_router(auth.router)
api_v1_router.include_router(users.router)
api_v1_router.include_router(hospitals.router)
api_v1_router.include_router(patient.router)
api_v1_router.include_router(patients.router)
api_v1_router.include_router(clinician.router)
api_v1_router.include_router(clinicians.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(privacy.router)
api_v1_router.include_router(diagnosis.router)
api_v1_router.include_router(explanations.router)
api_v1_router.include_router(causal.router)
api_v1_router.include_router(federated.router)
api_v1_router.include_router(models.router)
api_v1_router.include_router(metrics.router)
api_v1_router.include_router(audit.router)
