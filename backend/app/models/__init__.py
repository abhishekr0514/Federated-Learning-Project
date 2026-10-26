"""MedShield-FL SQLAlchemy Domain Models Package."""

from backend.app.db.base import Base

from backend.app.models.identity import (
    ClinicianProfile,
    Hospital,
    HospitalMembership,
    Permission,
    Role,
    User,
    UserSession,
    role_permissions,
    user_roles,
)
from backend.app.models.patient import (
    Patient,
    PatientClinicianAssignment,
    PatientHospitalAssociation,
)
from backend.app.models.clinical import (
    ClinicalObservation,
    DocumentProcessingJob,
    DocumentVersion,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingEntityCount,
    PrivacyMaskingResult,
)
from backend.app.models.ml_models import (
    CausalAnalysis,
    CounterfactualExplanation,
    DiagnosisPrediction,
    GlobalModel,
    HospitalMetric,
    ModelMetric,
    ModelVersion,
    PredictionExplanation,
)
from backend.app.models.federated import (
    FederatedClient,
    FederatedClientUpdate,
    FederatedRound,
)
from backend.app.models.audit import (
    AuditLog,
    DataQualityEvent,
    HospitalAlert,
    SecurityEvent,
    SystemHealthSnapshot,
)
from backend.app.models.messaging import PatientClinicianQuestion

__all__ = [
    "Base",
    "User",
    "UserSession",
    "Role",
    "Permission",
    "user_roles",
    "role_permissions",
    "Hospital",
    "HospitalMembership",
    "ClinicianProfile",
    "Patient",
    "PatientClinicianAssignment",
    "PatientHospitalAssociation",
    "MedicalDocument",
    "DocumentVersion",
    "DocumentProcessingJob",
    "PrivacyMaskingResult",
    "PrivacyMaskingEntityCount",
    "ECGRecord",
    "ClinicalObservation",
    "ModelVersion",
    "GlobalModel",
    "ModelMetric",
    "HospitalMetric",
    "DiagnosisPrediction",
    "PredictionExplanation",
    "CounterfactualExplanation",
    "CausalAnalysis",
    "FederatedRound",
    "FederatedClient",
    "FederatedClientUpdate",
    "AuditLog",
    "SecurityEvent",
    "SystemHealthSnapshot",
    "DataQualityEvent",
    "HospitalAlert",
    "PatientClinicianQuestion",
]
