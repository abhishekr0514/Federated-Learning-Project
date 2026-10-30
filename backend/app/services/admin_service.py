import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from backend.app.models.identity import User, Hospital, HospitalMembership, UserSession
from backend.app.models.audit import SecurityEvent, AuditLog

logger = logging.getLogger("medshield.admin")

def safe_isoformat(dt_val: Any) -> Optional[str]:
    """Safely format datetime object or string timestamp for JSON serialization."""
    if dt_val is None:
        return None
    if hasattr(dt_val, "isoformat"):
        return dt_val.isoformat()
    return str(dt_val)

class AdminService:
    """Server-side observability and administration aggregations for Super Admin Command Center."""

    @staticmethod
    async def get_overview_summary(db: AsyncSession) -> Dict[str, Any]:
        """Aggregate high-level metric counters for Super Admin Command Center."""
        hospital_stmt = select(func.count(Hospital.id)).where(Hospital.is_active.isnot(False))
        hospitals_res = await db.execute(hospital_stmt)
        total_hospitals = hospitals_res.scalar_one() or 0

        user_stmt = select(func.count(User.id)).where(User.deleted_at.is_(None), User.is_active.isnot(False))
        users_res = await db.execute(user_stmt)
        active_users = users_res.scalar_one() or 0

        login_success_stmt = select(func.count(SecurityEvent.id)).where(SecurityEvent.event_name == "LOGIN_SUCCESS")
        login_success_res = await db.execute(login_success_stmt)
        total_logins = login_success_res.scalar_one() or 0

        login_failed_stmt = select(func.count(SecurityEvent.id)).where(SecurityEvent.event_name == "LOGIN_FAILED")
        login_failed_res = await db.execute(login_failed_stmt)
        failed_logins = login_failed_res.scalar_one() or 0

        audit_stmt = select(func.count(AuditLog.id))
        audit_res = await db.execute(audit_stmt)
        total_audit_logs = audit_res.scalar_one() or 0

        return {
            "total_hospitals": total_hospitals,
            "active_users": active_users,
            "successful_logins": total_logins,
            "failed_logins": failed_logins,
            "audit_events_count": total_audit_logs,
            "federated_training": {
                "status": "NOT_ENABLED",
                "message": "Federated training is not enabled."
            },
            "global_model": {
                "status": "NO_DATA",
                "message": "No production global model checkpoint available."
            }
        }

    @staticmethod
    async def get_system_health(db: AsyncSession) -> Dict[str, Any]:
        """Measure API and database latency."""
        start_time = time.perf_counter()
        try:
            await db.execute(text("SELECT 1"))
            db_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            db_status = "HEALTHY"
        except Exception as e:
            logger.error(f"DB health check error: {e}")
            db_latency_ms = None
            db_status = "UNHEALTHY"

        try:
            now = datetime.now(timezone.utc)
            session_stmt = select(func.count(UserSession.id)).where(
                UserSession.is_revoked.isnot(True),
                UserSession.expires_at > now,
            )
            session_res = await db.execute(session_stmt)
            active_sessions = session_res.scalar_one() or 0
        except Exception as e:
            logger.error(f"Active sessions query error: {e}")
            active_sessions = 0

        return {
            "api_health": {
                "status": "HEALTHY",
                "service": "medshield-fl-backend",
                "version": "0.1.0"
            },
            "database": {
                "status": db_status,
                "latency_ms": db_latency_ms
            },
            "active_sessions_count": active_sessions,
            "system_hardware": {
                "cpu_usage": "NOT_MEASURED",
                "memory_usage": "NOT_MEASURED",
                "disk_usage": "NOT_MEASURED",
                "message": "Hardware monitoring telemetry is not measured by current backend configuration."
            }
        }

    @staticmethod
    async def get_hospitals_overview(db: AsyncSession) -> Dict[str, Any]:
        """Fetch list of active hospitals from PostgreSQL database."""
        stmt = select(Hospital).where(Hospital.is_active.isnot(False)).order_by(Hospital.created_at.desc())
        res = await db.execute(stmt)
        hospitals = res.scalars().all()

        hospital_list = []
        for hosp in hospitals:
            mem_stmt = select(func.count(HospitalMembership.id)).where(
                HospitalMembership.hospital_id == hosp.id
            )
            mem_res = await db.execute(mem_stmt)
            mem_count = mem_res.scalar_one() or 0

            hospital_list.append({
                "id": str(hosp.id),
                "name": hosp.name,
                "code": hosp.code,
                "is_active": hosp.is_active,
                "member_count": mem_count,
                "created_at": safe_isoformat(hosp.created_at)
            })

        return {
            "status": "AVAILABLE" if hospital_list else "NO_DATA",
            "count": len(hospital_list),
            "hospitals": hospital_list
        }

    @staticmethod
    async def get_federated_status() -> Dict[str, Any]:
        """Return federated learning training status."""
        return {
            "status": "NOT_ENABLED",
            "message": "Federated Flower training is not enabled.",
            "current_round": None,
            "active_clients_count": 0,
            "rounds_history": []
        }

    @staticmethod
    async def get_models_overview() -> Dict[str, Any]:
        """Return model version artifacts overview."""
        return {
            "status": "NO_DATA",
            "message": "No model version artifacts recorded in database.",
            "models": []
        }

    @staticmethod
    async def get_metrics_overview() -> Dict[str, Any]:
        """Return offline GAT evaluation metrics from reports/model_evaluation/ when present."""
        from backend.app.services.offline_evaluation_metrics_service import (
            load_offline_gat_evaluation_summary,
        )

        summary = load_offline_gat_evaluation_summary()
        if summary.get("status") == "AVAILABLE":
            summary["proxy_experiment_note"] = (
                "PTB-XL proxy hold-out evaluation (NORM vs MI/STTC). Not clinical validation."
            )
        return summary

    @staticmethod
    async def get_privacy_metrics() -> Dict[str, Any]:
        """Return privacy masking result statistics."""
        return {
            "status": "NO_DATA",
            "message": "No PII/PHI masking events recorded in central database.",
            "metrics": {
                "documents_masked": 0,
                "entities_detected": 0,
                "masking_failures": 0
            }
        }

    @staticmethod
    async def get_security_metrics(db: AsyncSession) -> Dict[str, Any]:
        """Aggregate security events from SecurityEvent table."""
        stmt = select(SecurityEvent.event_name, func.count(SecurityEvent.id)).group_by(SecurityEvent.event_name)
        res = await db.execute(stmt)
        counts = {event_name: count for event_name, count in res.all()}

        recent_stmt = select(SecurityEvent).order_by(SecurityEvent.created_at.desc()).limit(20)
        recent_res = await db.execute(recent_stmt)
        recent_events = recent_res.scalars().all()

        event_list = [
            {
                "id": str(ev.id),
                "event_name": ev.event_name,
                "severity": ev.severity,
                "actor_id": str(ev.actor_id) if ev.actor_id else None,
                "request_id": ev.request_id,
                "created_at": safe_isoformat(ev.created_at)
            }
            for ev in recent_events
        ]

        return {
            "status": "AVAILABLE" if counts or event_list else "NO_DATA",
            "event_counts": {
                "LOGIN_SUCCESS": counts.get("LOGIN_SUCCESS", 0),
                "LOGIN_FAILED": counts.get("LOGIN_FAILED", 0),
                "LOGOUT": counts.get("LOGOUT", 0),
                "TOKEN_REFRESH": counts.get("TOKEN_REFRESH", 0),
                "REFRESH_TOKEN_REUSE": counts.get("REFRESH_TOKEN_REUSE", 0),
                "AUTHORIZATION_DENIED": counts.get("AUTHORIZATION_DENIED", 0),
            },
            "recent_events": event_list
        }

    @staticmethod
    async def get_audit_ledger(db: AsyncSession, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch paginated append-only security audit log entries."""
        count_stmt = select(func.count(AuditLog.id))
        count_res = await db.execute(count_stmt)
        total_count = count_res.scalar_one() or 0

        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        res = await db.execute(stmt)
        logs = res.scalars().all()

        log_list = [
            {
                "id": str(log.id),
                "event_type": log.event_type,
                "actor_id": str(log.actor_id) if log.actor_id else None,
                "hospital_id": str(log.hospital_id) if log.hospital_id else None,
                "resource_type": log.resource_type,
                "resource_id": str(log.resource_id) if log.resource_id else None,
                "request_id": log.request_id,
                "outcome": log.outcome,
                "created_at": safe_isoformat(log.created_at)
            }
            for log in logs
        ]

        return {
            "status": "AVAILABLE" if log_list else "NO_DATA",
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "logs": log_list
        }

    @staticmethod
    async def get_alerts_overview() -> Dict[str, Any]:
        """Return system alert status."""
        return {
            "status": "NO_DATA",
            "message": "No active system alerts recorded.",
            "alerts": []
        }
