"""Audit Log Writer — Append-Only AuditLog Entry Persistence.

Writes structured, immutable audit events to the audit_logs table.
Called by backend services when significant operations occur (patient access,
document operations, assignment changes). Never fabricates historical events.

PRIVACY: No raw clinical text, no PHI, no JWT secrets stored in audit records.
"""

import uuid
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.request_context import get_request_id
from backend.app.models.audit import AuditLog


class AuditWriter:
    """Utility for appending audit log entries to the immutable audit_logs table."""

    @staticmethod
    async def write(
        db: AsyncSession,
        *,
        event_type: str,
        resource_type: str,
        outcome: str = "SUCCESS",
        actor_id: Optional[uuid.UUID] = None,
        hospital_id: Optional[uuid.UUID] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Append a single audit log entry.

        Parameters
        ----------
        event_type:    Short event code, e.g. 'PATIENT_PORTAL_ACCESS', 'DOCUMENT_UPLOAD'
        resource_type: Entity type, e.g. 'Patient', 'MedicalDocument'
        outcome:       'SUCCESS' | 'DENIED' | 'ERROR'
        actor_id:      UUID of the acting user (None for system events)
        hospital_id:   UUID of the scoped hospital (None for global events)
        resource_id:   Optional string reference to the affected resource
        ip_address:    Optional client IP address from request
        """
        req_id = get_request_id() or str(uuid.uuid4())
        entry = AuditLog(
            event_type=event_type,
            actor_id=actor_id,
            hospital_id=hospital_id,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            request_id=req_id,
            outcome=outcome,
            ip_address=ip_address,
        )
        db.add(entry)
        # Caller is responsible for commit — this is intentionally not auto-committed
        # so the audit write participates in the same transaction as the operation.
        # If the main operation rolls back, the audit entry also rolls back (correct).
