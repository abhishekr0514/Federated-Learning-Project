"""Patient Portal Domain Service.

Enforces:
1. Strict Patient Ownership: Caller queries strictly their own patient reference.
2. Privacy Boundaries: Zero PII, zero raw clinical notes, zero raw ECG waveform arrays.
3. Plain Non-Technical Language: No GAT, logits, gradients, loss, or FL terminology.
4. Zero Mock Data Contract: Canonical states (AVAILABLE, NO_DATA, NOT_ENABLED, NOT_IMPLEMENTED).
   Never coercing missing data into 'Healthy', 'Normal', or 0.
5. Medical Safety Wording: Clear disclaimer whenever AI results are surfaced.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.errors import NotFoundError
from backend.app.models.patient import Patient, PatientHospitalAssociation
from backend.app.models.clinical import (
    ClinicalObservation,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingResult,
)
from backend.app.models.ml_models import DiagnosisPrediction


MANDATORY_SAFETY_DISCLAIMER = (
    "This result was generated with AI-assisted clinical decision support. "
    "It is intended to support, not replace, professional clinical judgment. "
    "Please discuss your result with your qualified healthcare professional."
)

ASSESSMENT_UNAVAILABLE_DISCLAIMER = (
    "AI-assisted assessment is not currently available. "
    "Please discuss your health questions with your qualified healthcare professional."
)


class PatientService:
    """Domain service for patient-facing health portal telemetry and data access."""

    @staticmethod
    async def get_health_overview(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve patient's high-level health overview and assessment status.
        
        Uses plain non-technical language and strictly adheres to Zero Mock Data.
        """
        # Query record counts
        doc_stmt = select(func.count(MedicalDocument.id)).where(MedicalDocument.patient_id == patient.id)
        doc_count = (await db.execute(doc_stmt)).scalar_one() or 0

        obs_stmt = select(func.count(ClinicalObservation.id)).where(ClinicalObservation.patient_id == patient.id)
        obs_count = (await db.execute(obs_stmt)).scalar_one() or 0

        ecg_stmt = select(func.count(ECGRecord.id)).where(ECGRecord.patient_id == patient.id)
        ecg_count = (await db.execute(ecg_stmt)).scalar_one() or 0

        total_records = doc_count + obs_count + ecg_count

        # Latest record timestamp
        latest_date: Optional[str] = None
        dates: List[datetime] = []

        if doc_count > 0:
            latest_doc_stmt = select(MedicalDocument.created_at).where(MedicalDocument.patient_id == patient.id).order_by(MedicalDocument.created_at.desc()).limit(1)
            dt = (await db.execute(latest_doc_stmt)).scalar_one_or_none()
            if dt:
                dates.append(dt)

        if obs_count > 0:
            latest_obs_stmt = select(ClinicalObservation.created_at).where(ClinicalObservation.patient_id == patient.id).order_by(ClinicalObservation.created_at.desc()).limit(1)
            dt = (await db.execute(latest_obs_stmt)).scalar_one_or_none()
            if dt:
                dates.append(dt)

        if ecg_count > 0:
            latest_ecg_stmt = select(ECGRecord.created_at).where(ECGRecord.patient_id == patient.id).order_by(ECGRecord.created_at.desc()).limit(1)
            dt = (await db.execute(latest_ecg_stmt)).scalar_one_or_none()
            if dt:
                dates.append(dt)

        if dates:
            latest_date = max(dates).isoformat()

        return {
            "patient": {
                "patient_id": str(patient.id),
                "opaque_id": patient.opaque_patient_id,
                "registered_date": patient.created_at.isoformat() if patient.created_at else None,
            },
            "summary": {
                "records_count": total_records,
                "reports_count": doc_count,
                "ecg_tests_count": ecg_count,
                "vitals_tests_count": obs_count,
                "latest_record_date": latest_date,
                "health_status": "AVAILABLE" if total_records > 0 else "NO_DATA",
                "health_message": None if total_records > 0 else "No health information is currently available.",
            },
            "assessment": {
                "status": "NOT_ENABLED",
                "message": "AI-assisted assessment is not currently available.",
                "safety_wording": ASSESSMENT_UNAVAILABLE_DISCLAIMER,
            },
            "capabilities": {
                "ai_assisted_assessment": "NOT_ENABLED",
                "ecg_analysis": "NOT_IMPLEMENTED",
                "explanation_summaries": "NOT_IMPLEMENTED",
                "causal_insights": "NOT_IMPLEMENTED",
                "clinical_decision_support": "NOT_IMPLEMENTED",
                "patient_messaging": "NOT_IMPLEMENTED",
            },
        }

    @staticmethod
    async def get_records(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve list of authorized health records for the patient.
        
        Strictly excludes raw clinical text, raw ECG waveform arrays, and engineering metrics.
        """
        records: List[Dict[str, Any]] = []

        # 1. Medical Documents metadata
        doc_stmt = (
            select(MedicalDocument)
            .where(MedicalDocument.patient_id == patient.id)
            .order_by(MedicalDocument.created_at.desc())
        )
        doc_rows = (await db.execute(doc_stmt)).scalars().all()
        for d in doc_rows:
            records.append({
                "record_id": str(d.id),
                "record_type": "Medical Document",
                "category": d.document_type.replace("_", " ").title(),
                "status": "Available",
                "date": d.created_at.isoformat() if d.created_at else None,
                "details": "Sanitized clinical document record on file with your care team.",
            })

        # 2. Vitals & Clinical Observations metadata
        obs_stmt = (
            select(ClinicalObservation)
            .where(ClinicalObservation.patient_id == patient.id)
            .order_by(ClinicalObservation.created_at.desc())
        )
        obs_rows = (await db.execute(obs_stmt)).scalars().all()
        for o in obs_rows:
            records.append({
                "record_id": str(o.id),
                "record_type": "Health Observation",
                "category": o.observation_type.replace("_", " ").title(),
                "status": "Recorded",
                "date": o.created_at.isoformat() if o.created_at else None,
                "details": f"Recorded observation ({o.feature_schema_version}).",
            })

        # 3. ECG Records metadata (Acquisition parameters only, Zero raw waveform floats)
        ecg_stmt = (
            select(ECGRecord)
            .where(ECGRecord.patient_id == patient.id)
            .order_by(ECGRecord.created_at.desc())
        )
        ecg_rows = (await db.execute(ecg_stmt)).scalars().all()
        for e in ecg_rows:
            records.append({
                "record_id": str(e.id),
                "record_type": "Cardiac Recording",
                "category": "12-Lead Electrocardiogram (ECG)",
                "status": "Acquired",
                "date": e.acquired_at.isoformat() if e.acquired_at else (e.created_at.isoformat() if e.created_at else None),
                "details": f"Standard {e.lead_count}-lead test recording ({e.duration_seconds}s duration).",
            })

        if not records:
            return {
                "status": "NO_DATA",
                "total_count": 0,
                "records": [],
                "message": "No records are currently available.",
            }

        return {
            "status": "AVAILABLE",
            "total_count": len(records),
            "records": records,
            "message": None,
        }

    @staticmethod
    async def get_reports(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve patient diagnostic reports.
        
        Only returns actual stored reports. Zero fabricated diagnoses or interpretations.
        """
        doc_stmt = (
            select(MedicalDocument)
            .options(selectinload(MedicalDocument.masking_results))
            .where(MedicalDocument.patient_id == patient.id)
            .order_by(MedicalDocument.created_at.desc())
        )
        doc_rows = (await db.execute(doc_stmt)).scalars().all()

        reports: List[Dict[str, Any]] = []
        for d in doc_rows:
            protection_status = "Privacy Protected"
            if d.masking_results and d.masking_results[0].sanitization_status == "sanitized":
                protection_status = "Privacy Sanitized"

            reports.append({
                "report_id": str(d.id),
                "report_title": f"{d.document_type.replace('_', ' ').title()} Report",
                "report_type": d.document_type,
                "date": d.created_at.isoformat() if d.created_at else None,
                "status": protection_status,
                "summary": "Privacy-sanitized health document stored securely within your hospital node.",
            })

        if not reports:
            return {
                "status": "NO_DATA",
                "total_count": 0,
                "reports": [],
                "message": "No reports are currently available.",
            }

        return {
            "status": "AVAILABLE",
            "total_count": len(reports),
            "reports": reports,
            "message": None,
        }

    @staticmethod
    async def get_diagnosis(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve patient's diagnostic assessment and genuine predictions.
        
        Never fabricates predictions, risk scores, or treatment recommendations.
        Strictly distinguishes individual model confidence from model accuracy.
        """
        stmt = (
            select(DiagnosisPrediction)
            .where(DiagnosisPrediction.patient_id == patient.id)
            .order_by(DiagnosisPrediction.created_at.desc())
        )
        res = await db.execute(stmt)
        prediction = res.scalar_one_or_none()

        if not prediction:
            return {
                "status": "NO_DATA",
                "assessment_status": "NOT_ENABLED",
                "prediction": None,
                "message": "AI-assisted assessment is not currently available.",
                "safety_wording": ASSESSMENT_UNAVAILABLE_DISCLAIMER,
                "explanation_status": "NOT_IMPLEMENTED",
                "causal_status": "NOT_IMPLEMENTED",
            }

        confidence_val = None
        if prediction.confidence is not None:
            confidence_val = f"{prediction.confidence * 100:.1f}%"

        return {
            "status": "AVAILABLE",
            "assessment_status": "AVAILABLE",
            "prediction": {
                "prediction_id": str(prediction.id),
                "assessment_target": prediction.target_definition,
                "assessment_date": prediction.created_at.isoformat() if prediction.created_at else None,
                "model_confidence": confidence_val,
                "status": prediction.status,
            },
            "message": None,
            "safety_wording": MANDATORY_SAFETY_DISCLAIMER,
            "explanation_status": "NOT_IMPLEMENTED",
            "causal_status": "NOT_IMPLEMENTED",
        }
