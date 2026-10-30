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
from backend.app.services.audit_writer import AuditWriter
from backend.app.services.capability_registry import get_patient_portal_capabilities
from backend.app.models.patient import Patient, PatientHospitalAssociation
from backend.app.models.clinical import (
    ClinicalObservation,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingResult,
)
from backend.app.models.ml_models import DiagnosisPrediction, ModelVersion
from backend.app.services.prediction_proxy_class import resolve_patient_portal_proxy_class


MANDATORY_SAFETY_DISCLAIMER = (
    "This result was generated with AI-assisted clinical decision support. "
    "It is intended to support, not replace, professional clinical judgment. "
    "Please discuss your result with your qualified healthcare professional."
)

ASSESSMENT_UNAVAILABLE_DISCLAIMER = (
    "AI-assisted assessment is not currently available. "
    "Please discuss your health questions with your qualified healthcare professional."
)

RESEARCH_PROXY_VALIDATION_NOTICE = (
    "Not clinically validated: outputs come from a PTB-XL research proxy task "
    "(NORM vs MI/STTC classification), not from an approved diagnostic device."
)

PROXY_CLASS_EXPLAINER = {
    "NORM": (
        "Within this research dataset, class NORM (index 0) groups records labeled as "
        "normal or otherwise not in the MI/STTC super-class used for the proxy task."
    ),
    "MI/STTC": (
        "Within this research dataset, class MI/STTC (index 1) groups myocardial infarction "
        "and ST/T-change labels used as a combined proxy super-class—not a individual clinical diagnosis."
    ),
}

MODEL_CONFIDENCE_EXPLAINER = (
    "Model confidence is the neural network's uncalibrated score for its chosen proxy class. "
    "It is not the probability that you have a disease, not sensitivity/specificity, "
    "and not proof of clinical accuracy."
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
        Writes an audit log entry recording this portal access event.
        """
        # Resolve hospital scope for audit entry
        hosp_assoc_stmt = select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id
        ).limit(1)
        hosp_assoc = (await db.execute(hosp_assoc_stmt)).scalar_one_or_none()
        audit_hospital_id = hosp_assoc.hospital_id if hosp_assoc else None

        # Write audit event for patient portal overview access
        await AuditWriter.write(
            db,
            event_type="PATIENT_PORTAL_OVERVIEW_ACCESS",
            resource_type="Patient",
            outcome="SUCCESS",
            actor_id=patient.user_id,
            hospital_id=audit_hospital_id,
            resource_id=str(patient.id),
        )

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

        pred_stmt = select(func.count(DiagnosisPrediction.id)).where(
            DiagnosisPrediction.patient_id == patient.id
        )
        pred_count = (await db.execute(pred_stmt)).scalar_one() or 0

        caps = get_patient_portal_capabilities()
        ai_cap = caps.get("ai_assisted_assessment", "NOT_ENABLED")
        if pred_count > 0:
            assessment_status = "RESULTS_ON_FILE"
            assessment_message = (
                "Stored AI-assisted assessment results are available under My Results. "
                "These are research/decision-support outputs, not a clinical diagnosis."
            )
        elif ai_cap == "NOT_ENABLED":
            assessment_status = "NOT_ENABLED"
            assessment_message = (
                "AI-assisted assessment is not enabled in this environment. "
                "Your care team uses standard clinical evaluation."
            )
        elif ai_cap == "NOT_IMPLEMENTED":
            assessment_status = "NOT_IMPLEMENTED"
            assessment_message = "Automated patient assessment is not implemented on the portal."
        elif ai_cap in ("CONFIGURATION_REQUIRED", "MISSING_DEPENDENCY", "UNAVAILABLE"):
            assessment_status = ai_cap
            assessment_message = (
                "The assessment service is not ready (configuration or dependencies missing). "
                "No automated result will be shown until the platform is configured."
            )
        elif ai_cap == "AVAILABLE":
            assessment_status = "NO_DATA"
            assessment_message = (
                "Assessment models are configured, but no completed assessment is stored for your account yet. "
                "When your authorized clinician runs an evaluation, results may appear under My Results."
            )
        else:
            assessment_status = ai_cap
            assessment_message = "AI-assisted assessment is not currently available."

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
                "status": assessment_status,
                "message": assessment_message,
                "safety_wording": ASSESSMENT_UNAVAILABLE_DISCLAIMER,
            },
            "results": {
                "status": "AVAILABLE" if pred_count > 0 else "NO_DATA",
                "stored_count": int(pred_count),
                "message": None
                if pred_count > 0
                else "No stored assessment results for your account.",
            },
            "capabilities": caps,
        }

    @staticmethod
    async def get_records(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve list of authorized health records for the patient.
        
        Strictly excludes raw clinical text, raw ECG waveform arrays, and engineering metrics.
        Writes an audit log entry for records access.
        """
        hosp_assoc_stmt = select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id
        ).limit(1)
        hosp_assoc = (await db.execute(hosp_assoc_stmt)).scalar_one_or_none()
        audit_hospital_id = hosp_assoc.hospital_id if hosp_assoc else None

        await AuditWriter.write(
            db,
            event_type="PATIENT_RECORDS_ACCESS",
            resource_type="Patient",
            outcome="SUCCESS",
            actor_id=patient.user_id,
            hospital_id=audit_hospital_id,
            resource_id=str(patient.id),
        )

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
        Writes an audit log entry for reports access.
        """
        hosp_assoc_stmt = select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id
        ).limit(1)
        hosp_assoc = (await db.execute(hosp_assoc_stmt)).scalar_one_or_none()
        audit_hospital_id = hosp_assoc.hospital_id if hosp_assoc else None

        await AuditWriter.write(
            db,
            event_type="PATIENT_REPORTS_ACCESS",
            resource_type="MedicalDocument",
            outcome="SUCCESS",
            actor_id=patient.user_id,
            hospital_id=audit_hospital_id,
            resource_id=str(patient.id),
        )

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
    async def _load_prediction_history(
        db: AsyncSession,
        patient_id: uuid.UUID,
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(DiagnosisPrediction, ModelVersion)
            .join(ModelVersion, DiagnosisPrediction.model_version_id == ModelVersion.id)
            .where(DiagnosisPrediction.patient_id == patient_id)
            .order_by(DiagnosisPrediction.created_at.asc())
        )
        rows = (await db.execute(stmt)).all()
        history: List[Dict[str, Any]] = []
        for pred, model in rows:
            proxy = resolve_patient_portal_proxy_class(
                confidence=pred.confidence,
                predicted_class_index=pred.predicted_class_index,
                model_family=model.model_family,
                class_meaning_map=PROXY_CLASS_EXPLAINER,
            )
            conf_pct = f"{pred.confidence * 100:.1f}%" if pred.confidence is not None else None
            history.append(
                {
                    "prediction_id": str(pred.id),
                    "assessment_date": pred.created_at.isoformat() if pred.created_at else None,
                    "assessment_target": pred.target_definition,
                    "model_confidence": conf_pct,
                    "model_confidence_raw": pred.confidence,
                    "predicted_proxy_class": proxy["predicted_proxy_class"],
                    "predicted_class_index": proxy["predicted_class_index"],
                    "proxy_class_status": proxy["proxy_class_status"],
                    "model_version_tag": model.version_tag,
                    "model_family": model.model_family,
                    "architecture_config": model.architecture_config,
                    "provenance": pred.provenance,
                    "prediction_type": pred.prediction_type,
                    "status": pred.status,
                    "is_research_proxy": True,
                    "validation_status": "NOT_CLINICALLY_VALIDATED",
                }
            )
        return history

    @staticmethod
    def _build_chart_payload(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        points = [
            {
                "prediction_id": h["prediction_id"],
                "date": h["assessment_date"],
                "model_confidence_raw": h["model_confidence_raw"],
                "predicted_class_index": h["predicted_class_index"],
                "predicted_proxy_class": h["predicted_proxy_class"],
            }
            for h in history
            if h.get("assessment_date") is not None
        ]
        count = len(points)
        trend_supported = count >= 2
        return {
            "timeline_points": points,
            "assessment_count": count,
            "trend_supported": trend_supported,
            "trend_message": (
                "Multiple stored assessments are available for timeline and confidence charts."
                if trend_supported
                else "Only one stored assessment exists; trend charts are not meaningful yet."
            ),
            "confidence_axis_label": "Model confidence",
            "class_axis_label": "Research class (NORM vs MI/STTC)",
        }

    @staticmethod
    async def get_diagnosis(
        db: AsyncSession,
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve patient's diagnostic assessment and genuine predictions.
        
        Never fabricates predictions, risk scores, or treatment recommendations.
        Strictly distinguishes individual model confidence from model accuracy.
        Writes an audit log entry for diagnosis access.
        """
        hosp_assoc_stmt = select(PatientHospitalAssociation).where(
            PatientHospitalAssociation.patient_id == patient.id
        ).limit(1)
        hosp_assoc = (await db.execute(hosp_assoc_stmt)).scalar_one_or_none()
        audit_hospital_id = hosp_assoc.hospital_id if hosp_assoc else None

        await AuditWriter.write(
            db,
            event_type="PATIENT_DIAGNOSIS_ACCESS",
            resource_type="DiagnosisPrediction",
            outcome="SUCCESS",
            actor_id=patient.user_id,
            hospital_id=audit_hospital_id,
            resource_id=str(patient.id),
        )

        caps = get_patient_portal_capabilities()
        ai_cap = caps.get("ai_assisted_assessment", "NOT_ENABLED")
        history = await PatientService._load_prediction_history(db, patient.id)
        latest = history[-1] if history else None

        if not latest:
            empty_status = "NOT_ENABLED" if ai_cap == "NOT_ENABLED" else "NO_DATA"
            return {
                "status": "NO_DATA",
                "assessment_status": empty_status,
                "inference_capability": ai_cap,
                "prediction": None,
                "history": [],
                "charts": PatientService._build_chart_payload([]),
                "message": (
                    "No stored assessment results exist for your account yet."
                    if empty_status == "NO_DATA"
                    else "AI-assisted assessment is not enabled for new evaluations in this environment."
                ),
                "safety_wording": ASSESSMENT_UNAVAILABLE_DISCLAIMER,
                "validation_notice": RESEARCH_PROXY_VALIDATION_NOTICE,
                "model_confidence_explainer": MODEL_CONFIDENCE_EXPLAINER,
                "proxy_class_explainer": PROXY_CLASS_EXPLAINER,
                "explanation_status": "NOT_IMPLEMENTED",
                "causal_status": "NOT_IMPLEMENTED",
            }

        return {
            "status": "AVAILABLE",
            "assessment_status": "RESULTS_ON_FILE",
            "inference_capability": ai_cap,
            "prediction": latest,
            "history": history,
            "charts": PatientService._build_chart_payload(history),
            "message": None,
            "safety_wording": MANDATORY_SAFETY_DISCLAIMER,
            "validation_notice": RESEARCH_PROXY_VALIDATION_NOTICE,
            "model_confidence_explainer": MODEL_CONFIDENCE_EXPLAINER,
            "proxy_class_explainer": PROXY_CLASS_EXPLAINER,
            "explanation_status": "NOT_IMPLEMENTED",
            "causal_status": "NOT_IMPLEMENTED",
        }
