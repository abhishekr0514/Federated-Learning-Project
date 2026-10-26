"""Patient question submission and clinician review workflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.errors import ForbiddenError, NotFoundError, ValidationAppError
from backend.app.models.identity import ClinicianProfile, User
from backend.app.models.messaging import PatientClinicianQuestion
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation
from backend.app.services.audit_writer import AuditWriter


MAX_QUESTION_LEN = 2000
MAX_RESPONSE_LEN = 4000

SUGGESTED_TEMPLATES = [
    "What do my recent cardiac test results mean for my day-to-day activities?",
    "Are there lifestyle changes you recommend based on my records on file?",
    "When should I schedule my next follow-up visit?",
    "What symptoms should prompt me to call the clinic or seek urgent care?",
]


class PatientQuestionService:
    @staticmethod
    async def submit_question(
        db: AsyncSession,
        patient: Patient,
        author: User,
        *,
        question_text: str,
        assigned_clinician_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        text = (question_text or "").strip()
        if not text:
            raise ValidationAppError("Question text is required.")
        if len(text) > MAX_QUESTION_LEN:
            raise ValidationAppError(f"Question must be at most {MAX_QUESTION_LEN} characters.")

        assoc_stmt = (
            select(PatientHospitalAssociation)
            .where(PatientHospitalAssociation.patient_id == patient.id)
            .limit(1)
        )
        assoc = (await db.execute(assoc_stmt)).scalar_one_or_none()
        if not assoc:
            raise ValidationAppError("Patient is not linked to a hospital care node.")

        if assigned_clinician_id:
            assign_stmt = select(PatientClinicianAssignment).where(
                PatientClinicianAssignment.patient_id == patient.id,
                PatientClinicianAssignment.clinician_id == assigned_clinician_id,
                PatientClinicianAssignment.hospital_id == assoc.hospital_id,
            )
            if not (await db.execute(assign_stmt)).scalar_one_or_none():
                raise ValidationAppError("Selected clinician is not assigned to your care record.")

        row = PatientClinicianQuestion(
            id=uuid.uuid4(),
            patient_id=patient.id,
            hospital_id=assoc.hospital_id,
            author_user_id=author.id,
            assigned_clinician_id=assigned_clinician_id,
            question_text=text,
            status="submitted",
        )
        db.add(row)
        await AuditWriter.write(
            db,
            event_type="PATIENT_QUESTION_SUBMITTED",
            resource_type="PatientClinicianQuestion",
            outcome="SUCCESS",
            actor_id=author.id,
            hospital_id=assoc.hospital_id,
            resource_id=str(row.id),
        )
        await db.commit()
        await db.refresh(row)
        return PatientQuestionService._serialize_question(row)

    @staticmethod
    async def list_patient_questions(db: AsyncSession, patient: Patient) -> Dict[str, Any]:
        stmt = (
            select(PatientClinicianQuestion)
            .where(PatientClinicianQuestion.patient_id == patient.id)
            .order_by(PatientClinicianQuestion.created_at.desc())
        )
        rows = (await db.execute(stmt)).scalars().all()
        return {
            "status": "AVAILABLE" if rows else "NO_DATA",
            "suggested_templates": SUGGESTED_TEMPLATES,
            "questions": [PatientQuestionService._serialize_question(q) for q in rows],
            "emergency_notice": (
                "This messaging channel is not monitored continuously and is not for emergencies. "
                "If you have urgent or life-threatening symptoms, call your local emergency number "
                "or go to the nearest emergency department."
            ),
        }

    @staticmethod
    async def list_clinician_questions(
        db: AsyncSession,
        profile: ClinicianProfile,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        stmt = (
            select(PatientClinicianQuestion, Patient)
            .join(Patient, PatientClinicianQuestion.patient_id == Patient.id)
            .join(
                PatientClinicianAssignment,
                (PatientClinicianAssignment.patient_id == Patient.id)
                & (PatientClinicianAssignment.clinician_id == profile.id)
                & (PatientClinicianAssignment.hospital_id == profile.hospital_id),
            )
            .where(PatientClinicianQuestion.hospital_id == profile.hospital_id)
            .order_by(PatientClinicianQuestion.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await db.execute(stmt)).all()
        items: List[Dict[str, Any]] = []
        for question, patient in rows:
            payload = PatientQuestionService._serialize_question(question)
            payload["opaque_patient_id"] = patient.opaque_patient_id
            items.append(payload)
        return {
            "status": "AVAILABLE" if items else "NO_DATA",
            "questions": items,
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    async def clinician_update_question(
        db: AsyncSession,
        profile: ClinicianProfile,
        question_id: uuid.UUID,
        *,
        mark_reviewed: bool = False,
        response_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        stmt = select(PatientClinicianQuestion).where(PatientClinicianQuestion.id == question_id)
        question = (await db.execute(stmt)).scalar_one_or_none()
        if not question:
            raise NotFoundError("Question resource not accessible")

        assign_stmt = select(PatientClinicianAssignment).where(
            PatientClinicianAssignment.patient_id == question.patient_id,
            PatientClinicianAssignment.clinician_id == profile.id,
            PatientClinicianAssignment.hospital_id == profile.hospital_id,
        )
        if not (await db.execute(assign_stmt)).scalar_one_or_none():
            raise NotFoundError("Question resource not accessible")

        now = datetime.now(timezone.utc)
        if mark_reviewed and question.status == "submitted":
            question.status = "reviewed"
            question.reviewed_at = now
        if response_text is not None:
            text = response_text.strip()
            if not text:
                raise ValidationAppError("Response text cannot be empty when provided.")
            if len(text) > MAX_RESPONSE_LEN:
                raise ValidationAppError(f"Response must be at most {MAX_RESPONSE_LEN} characters.")
            question.response_text = text
            question.status = "responded"
            question.responded_at = now
            if not question.reviewed_at:
                question.reviewed_at = now
        question.updated_at = now

        await AuditWriter.write(
            db,
            event_type="CLINICIAN_QUESTION_UPDATED",
            resource_type="PatientClinicianQuestion",
            outcome="SUCCESS",
            actor_id=profile.user_id,
            hospital_id=profile.hospital_id,
            resource_id=str(question.id),
        )
        await db.commit()
        await db.refresh(question)
        return PatientQuestionService._serialize_question(question)

    @staticmethod
    def _serialize_question(row: PatientClinicianQuestion) -> Dict[str, Any]:
        return {
            "question_id": str(row.id),
            "patient_id": str(row.patient_id),
            "hospital_id": str(row.hospital_id),
            "status": row.status,
            "question_text": row.question_text,
            "response_text": row.response_text,
            "assigned_clinician_id": str(row.assigned_clinician_id)
            if row.assigned_clinician_id
            else None,
            "submitted_at": row.created_at.isoformat() if row.created_at else None,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
            "responded_at": row.responded_at.isoformat() if row.responded_at else None,
        }
