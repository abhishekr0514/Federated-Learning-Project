"""Hospital-scoped service and repository aggregation layer for MedShield-FL Hospital Admin Command Center.

PRIVACY & TENANT ISOLATION ARCHITECTURE:
Every query strictly enforces the authorized hospital scope server-side.
Zero cross-hospital data leakage.
Zero raw PHI/PII (names, notes, waveforms, contact info) is ever exposed.
Missing data is explicitly represented using canonical states (NO_DATA, NOT_ENABLED, NOT_MEASURED, etc.).
"""

import time
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text

from backend.app.models.identity import Hospital, HospitalMembership, ClinicianProfile, User, UserSession
from backend.app.models.patient import Patient, PatientHospitalAssociation
from backend.app.models.clinical import MedicalDocument, PrivacyMaskingResult
from backend.app.models.federated import FederatedClient, FederatedClientUpdate
from backend.app.models.ml_models import ModelVersion, HospitalMetric
from backend.app.models.audit import AuditLog, DataQualityEvent, HospitalAlert

logger = logging.getLogger("medshield.hospital_service")


def safe_isoformat(dt_val: Any) -> Optional[str]:
    """Safely format datetime object or string timestamp for JSON serialization."""
    if dt_val is None:
        return None
    if hasattr(dt_val, "isoformat"):
        return dt_val.isoformat()
    return str(dt_val)


class HospitalService:
    """Hospital Node Observability & Administration Aggregations."""

    @staticmethod
    async def get_overview(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Aggregate high-level metric counters for the authorized hospital node."""
        hosp_id = hospital.id

        # 1. Clinician count (Profiles or Memberships with CLINICIAN role)
        clinician_stmt = select(func.count(ClinicianProfile.id)).where(ClinicianProfile.hospital_id == hosp_id)
        clinician_res = await db.execute(clinician_stmt)
        clinician_count = clinician_res.scalar_one() or 0

        if clinician_count == 0:
            mem_stmt = select(func.count(HospitalMembership.id)).where(
                HospitalMembership.hospital_id == hosp_id,
                HospitalMembership.role_in_hospital == "CLINICIAN",
            )
            mem_res = await db.execute(mem_stmt)
            clinician_count = mem_res.scalar_one() or 0

        # 2. Patient reference count
        patient_stmt = select(func.count(PatientHospitalAssociation.id)).where(
            PatientHospitalAssociation.hospital_id == hosp_id
        )
        patient_res = await db.execute(patient_stmt)
        patient_count = patient_res.scalar_one() or 0

        # 3. Federated client registration count
        fed_stmt = select(func.count(FederatedClient.id)).where(FederatedClient.hospital_id == hosp_id)
        fed_res = await db.execute(fed_stmt)
        fed_client_count = fed_res.scalar_one() or 0

        # 4. Latest model version
        latest_model_stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1)
        latest_model_res = await db.execute(latest_model_stmt)
        latest_model = latest_model_res.scalar_one_or_none()

        # 5. Latest hospital metric
        latest_metric_stmt = (
            select(HospitalMetric)
            .where(HospitalMetric.hospital_id == hosp_id)
            .order_by(HospitalMetric.evaluated_at.desc())
            .limit(1)
        )
        latest_metric_res = await db.execute(latest_metric_stmt)
        latest_metric = latest_metric_res.scalar_one_or_none()

        # 6. Latest audit timestamp
        audit_ts_stmt = select(func.max(AuditLog.created_at)).where(AuditLog.hospital_id == hosp_id)
        audit_ts_res = await db.execute(audit_ts_stmt)
        latest_audit_ts = audit_ts_res.scalar_one()

        # 7. Latest alert timestamp
        alert_ts_stmt = select(func.max(HospitalAlert.created_at)).where(HospitalAlert.hospital_id == hosp_id)
        alert_ts_res = await db.execute(alert_ts_stmt)
        latest_alert_ts = alert_ts_res.scalar_one()

        return {
            "status": "AVAILABLE",
            "hospital_scope": {
                "id": str(hospital.id),
                "name": hospital.name,
                "code": hospital.code,
                "is_active": hospital.is_active,
            },
            "clinicians": {
                "status": "AVAILABLE" if clinician_count > 0 else "NO_DATA",
                "count": clinician_count,
                "message": None if clinician_count > 0 else "No clinician records available",
            },
            "patients": {
                "status": "AVAILABLE" if patient_count > 0 else "NO_DATA",
                "count": patient_count,
                "message": None if patient_count > 0 else "No patient reference records available",
            },
            "federated_clients": {
                "status": "AVAILABLE" if fed_client_count > 0 else "NO_DATA",
                "count": fed_client_count,
                "message": None if fed_client_count > 0 else "No federated client registered for this node",
            },
            "latest_model_version": {
                "status": "AVAILABLE" if latest_model else "NO_DATA",
                "version_tag": latest_model.version_tag if latest_model else None,
                "target_definition": latest_model.target_definition if latest_model else None,
                "created_at": safe_isoformat(latest_model.created_at) if latest_model else None,
            },
            "latest_metric": {
                "status": "AVAILABLE" if latest_metric else "NO_DATA",
                "metric_name": latest_metric.metric_name if latest_metric else None,
                "metric_value": latest_metric.metric_value if latest_metric else None,
                "evaluated_at": safe_isoformat(latest_metric.evaluated_at) if latest_metric else None,
            },
            "latest_audit_activity": {
                "status": "AVAILABLE" if latest_audit_ts else "NO_DATA",
                "timestamp": safe_isoformat(latest_audit_ts),
            },
            "latest_alert": {
                "status": "AVAILABLE" if latest_alert_ts else "NO_DATA",
                "timestamp": safe_isoformat(latest_alert_ts),
            },
            "local_training": {
                "status": "NOT_ENABLED",
                "message": "Local training is not enabled.",
            },
            "federated_training": {
                "status": "NOT_ENABLED",
                "message": "Federated training is not enabled.",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_system_health(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Check API, database health and surface explicit unmeasured hardware states."""
        start_time = time.perf_counter()
        try:
            await db.execute(text("SELECT 1"))
            db_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            db_status = "AVAILABLE"
        except Exception as e:
            logger.error(f"DB health ping error for hospital {hospital.code}: {e}")
            db_latency_ms = None
            db_status = "ERROR"

        # Non-revoked refresh-token sessions that have not expired (global auth table).
        now = datetime.now(timezone.utc)
        active_sessions = 0
        session_tracking = "AVAILABLE"
        session_definition = (
            "Count of non-revoked UserSession rows with expires_at in the future "
            "(refresh-token sessions, not concurrent UI tabs)."
        )
        try:
            session_stmt = select(func.count(UserSession.id)).where(
                UserSession.is_revoked.isnot(True),
                UserSession.expires_at > now,
            )
            session_res = await db.execute(session_stmt)
            active_sessions = session_res.scalar_one() or 0
        except Exception as e:
            logger.error(f"Active sessions query error: {e}")
            session_tracking = "ERROR"

        return {
            "status": "AVAILABLE",
            "hospital_scope": {
                "id": str(hospital.id),
                "code": hospital.code,
            },
            "api_health": {
                "status": "AVAILABLE",
                "service": "medshield-fl-backend",
                "version": "0.1.0",
            },
            "database": {
                "status": db_status,
                "latency_ms": db_latency_ms,
            },
            "active_sessions_count": active_sessions,
            "active_sessions": {
                "status": session_tracking,
                "count": active_sessions,
                "definition": session_definition,
                "hospital_scoped": False,
            },
            "system_hardware": {
                "cpu_usage": "NOT_MEASURED",
                "memory_usage": "NOT_MEASURED",
                "disk_usage": "NOT_MEASURED",
                "gpu_usage": "NOT_MEASURED",
                "worker_health": "NOT_MEASURED",
                "queue_depth": "NOT_MEASURED",
                "inference_latency": "NOT_MEASURED",
                "message": "Infrastructure hardware and worker metrics are not measured by current backend configuration.",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_clinicians(
        db: AsyncSession, hospital: Hospital, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch paginated hospital-scoped clinician metadata. Zero credentials/secrets exposed."""
        hosp_id = hospital.id

        # Count total clinician profiles
        count_stmt = select(func.count(ClinicianProfile.id)).where(ClinicianProfile.hospital_id == hosp_id)
        count_res = await db.execute(count_stmt)
        total_count = count_res.scalar_one() or 0

        clinicians_list = []
        if total_count > 0:
            stmt = (
                select(ClinicianProfile, User)
                .join(User, ClinicianProfile.user_id == User.id)
                .where(ClinicianProfile.hospital_id == hosp_id)
                .order_by(ClinicianProfile.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            res = await db.execute(stmt)
            for prof, usr in res.all():
                clinicians_list.append({
                    "id": str(prof.id),
                    "user_id": str(prof.user_id),
                    "department": prof.department,
                    "license_hash": prof.license_number_hash,
                    "user_status": "ACTIVE" if (usr and usr.is_active) else "INACTIVE",
                    "created_at": safe_isoformat(prof.created_at),
                    "updated_at": safe_isoformat(prof.updated_at),
                })
        else:
            # Fallback to memberships if no explicit clinician profile exists
            mem_count_stmt = select(func.count(HospitalMembership.id)).where(
                HospitalMembership.hospital_id == hosp_id,
                HospitalMembership.role_in_hospital == "CLINICIAN",
            )
            mem_count_res = await db.execute(mem_count_stmt)
            total_count = mem_count_res.scalar_one() or 0

            if total_count > 0:
                stmt = (
                    select(HospitalMembership, User)
                    .join(User, HospitalMembership.user_id == User.id)
                    .where(
                        HospitalMembership.hospital_id == hosp_id,
                        HospitalMembership.role_in_hospital == "CLINICIAN",
                    )
                    .order_by(HospitalMembership.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                res = await db.execute(stmt)
                for mem, usr in res.all():
                    clinicians_list.append({
                        "id": str(mem.id),
                        "user_id": str(mem.user_id),
                        "department": "Clinical Staff",
                        "license_hash": None,
                        "user_status": "ACTIVE" if (usr and usr.is_active) else "INACTIVE",
                        "created_at": safe_isoformat(mem.created_at),
                        "updated_at": safe_isoformat(mem.updated_at),
                    })

        return {
            "status": "AVAILABLE" if clinicians_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "clinicians": clinicians_list,
            "message": None if clinicians_list else "No clinician records available",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_patients(
        db: AsyncSession, hospital: Hospital, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch paginated opaque patient reference hashes. Zero PII/PHI or waveform arrays exposed."""
        hosp_id = hospital.id

        count_stmt = select(func.count(PatientHospitalAssociation.id)).where(
            PatientHospitalAssociation.hospital_id == hosp_id
        )
        count_res = await db.execute(count_stmt)
        total_count = count_res.scalar_one() or 0

        patients_list = []
        if total_count > 0:
            stmt = (
                select(PatientHospitalAssociation, Patient)
                .join(Patient, PatientHospitalAssociation.patient_id == Patient.id)
                .where(PatientHospitalAssociation.hospital_id == hosp_id)
                .order_by(PatientHospitalAssociation.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            res = await db.execute(stmt)
            for assoc, pat in res.all():
                patients_list.append({
                    "id": str(assoc.id),
                    "patient_id": str(pat.id),
                    "opaque_patient_id": pat.opaque_patient_id,
                    "hospital_ref_hash": assoc.hospital_patient_ref_hash,
                    "created_at": safe_isoformat(assoc.created_at),
                    "updated_at": safe_isoformat(assoc.updated_at),
                })

        return {
            "status": "AVAILABLE" if patients_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "patients": patients_list,
            "message": None if patients_list else "No patient references recorded for this hospital node",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_models(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Fetch registered model architectures and local version visibility."""
        stmt = select(ModelVersion).order_by(ModelVersion.created_at.desc())
        res = await db.execute(stmt)
        models = res.scalars().all()

        model_list = [
            {
                "id": str(m.id),
                "version_tag": m.version_tag,
                "model_family": m.model_family,
                "target_definition": m.target_definition,
                "status": m.status,
                "created_at": safe_isoformat(m.created_at),
            }
            for m in models
        ]

        return {
            "status": "AVAILABLE" if model_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "models": model_list,
            "message": None if model_list else "No model version records available",
            "local_training": {
                "status": "NOT_ENABLED",
                "loss": "NOT_AVAILABLE",
                "validation_metrics": "NOT_AVAILABLE",
                "local_model_version": "NO_DATA",
                "message": "Local training is not enabled.",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_metrics(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Fetch local evaluation metrics with explicit target definitions."""
        stmt = (
            select(HospitalMetric, ModelVersion)
            .outerjoin(ModelVersion, HospitalMetric.model_version_id == ModelVersion.id)
            .where(HospitalMetric.hospital_id == hospital.id)
            .order_by(HospitalMetric.evaluated_at.desc())
        )
        res = await db.execute(stmt)
        records = res.all()

        metrics_list = [
            {
                "id": str(hm.id),
                "metric_name": hm.metric_name,
                "metric_value": hm.metric_value,
                "model_version": mv.version_tag if mv else None,
                "target_definition": mv.target_definition if mv else "Proxy Evaluation",
                "evaluated_at": safe_isoformat(hm.evaluated_at),
            }
            for hm, mv in records
        ]

        return {
            "status": "AVAILABLE" if metrics_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "metrics": metrics_list,
            "provenance_note": (
                "Historical PTB-XL proxy experiment is NOT a production heart-disease diagnostic metric. "
                "Target definitions are explicitly declared per metric record."
            ),
            "message": None if metrics_list else "No evaluation results available.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_federated(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Fetch hospital federated client registration and historical round participation."""
        stmt = (
            select(FederatedClient)
            .where(FederatedClient.hospital_id == hospital.id)
            .order_by(FederatedClient.registered_at.desc())
        )
        res = await db.execute(stmt)
        clients = res.scalars().all()

        clients_list = []
        for c in clients:
            # Check historical round updates
            upd_stmt = (
                select(FederatedClientUpdate)
                .where(FederatedClientUpdate.client_id == c.id)
                .order_by(FederatedClientUpdate.created_at.desc())
            )
            upd_res = await db.execute(upd_stmt)
            updates = upd_res.scalars().all()

            clients_list.append({
                "id": str(c.id),
                "client_identifier": c.client_identifier,
                "status": c.status,
                "registered_at": safe_isoformat(c.registered_at),
                "record_provenance": "HISTORICAL",
                "historical_updates_count": len(updates),
            })

        return {
            "status": "NOT_ENABLED",
            "hospital_scope": str(hospital.id),
            "subsystem_state": "NOT_ENABLED",
            "message": "Federated Flower training is not enabled.",
            "historical_clients": clients_list,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_privacy(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Aggregate privacy masking metrics for medical documents belonging to this hospital."""
        stmt = (
            select(
                func.count(PrivacyMaskingResult.id),
                func.coalesce(func.sum(PrivacyMaskingResult.entities_detected_count), 0),
            )
            .join(MedicalDocument, PrivacyMaskingResult.document_id == MedicalDocument.id)
            .where(MedicalDocument.hospital_id == hospital.id)
        )
        res = await db.execute(stmt)
        doc_count, entities_count = res.one()

        has_data = (doc_count or 0) > 0

        return {
            "status": "AVAILABLE" if has_data else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "metrics": {
                "documents_processed": doc_count or 0,
                "entities_detected": entities_count or 0,
                "sanitization_status": "sanitized" if has_data else "none",
            },
            "message": None if has_data else "No privacy masking events recorded for this hospital node",
            "privacy_contract": "Zero raw notes, clinical text, PII, or replacement maps are serialized.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_data_quality(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Fetch hospital-scoped non-clinical data quality events."""
        stmt = (
            select(DataQualityEvent)
            .where(DataQualityEvent.hospital_id == hospital.id)
            .order_by(DataQualityEvent.created_at.desc())
        )
        res = await db.execute(stmt)
        events = res.scalars().all()

        events_list = [
            {
                "id": str(e.id),
                "severity": e.severity,
                "status": e.status,
                "category": e.category,
                "created_at": safe_isoformat(e.created_at),
                "resolved_at": safe_isoformat(e.resolved_at),
            }
            for e in events
        ]

        return {
            "status": "AVAILABLE" if events_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "events": events_list,
            "message": None if events_list else "No data quality events recorded for this hospital node",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_audit(
        db: AsyncSession, hospital: Hospital, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch paginated append-only audit trail strictly scoped to authorized hospital."""
        hosp_id = hospital.id

        count_stmt = select(func.count(AuditLog.id)).where(AuditLog.hospital_id == hosp_id)
        count_res = await db.execute(count_stmt)
        total_count = count_res.scalar_one() or 0

        logs_list = []
        if total_count > 0:
            stmt = (
                select(AuditLog)
                .where(AuditLog.hospital_id == hosp_id)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            res = await db.execute(stmt)
            logs = res.scalars().all()

            logs_list = [
                {
                    "id": str(log.id),
                    "event_type": log.event_type,
                    "actor_id": str(log.actor_id) if log.actor_id else None,
                    "hospital_id": str(log.hospital_id) if log.hospital_id else None,
                    "resource_type": log.resource_type,
                    "resource_id": str(log.resource_id) if log.resource_id else None,
                    "request_id": log.request_id,
                    "outcome": log.outcome,
                    "created_at": safe_isoformat(log.created_at),
                }
                for log in logs
            ]

        return {
            "status": "AVAILABLE" if logs_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "logs": logs_list,
            "message": None if logs_list else "No audit events recorded for this hospital node",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    async def get_alerts(db: AsyncSession, hospital: Hospital) -> Dict[str, Any]:
        """Fetch hospital alerts and report automatic generation status."""
        stmt = (
            select(HospitalAlert)
            .where(HospitalAlert.hospital_id == hospital.id)
            .order_by(HospitalAlert.created_at.desc())
        )
        res = await db.execute(stmt)
        alerts = res.scalars().all()

        alerts_list = [
            {
                "id": str(a.id),
                "severity": a.severity,
                "status": a.status,
                "category": a.category,
                "title": a.title,
                "created_at": safe_isoformat(a.created_at),
                "resolved_at": safe_isoformat(a.resolved_at),
            }
            for a in alerts
        ]

        return {
            "status": "AVAILABLE" if alerts_list else "NO_DATA",
            "hospital_scope": str(hospital.id),
            "automatic_generation": "NOT_ENABLED",
            "alerts": alerts_list,
            "message": None if alerts_list else "No active alerts recorded for this hospital node",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
