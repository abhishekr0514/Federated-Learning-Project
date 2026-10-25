"""Clinician Workspace Domain Service.

Enforces:
1. Strict Patient Assignment Scoping: Clinicians query ONLY explicitly assigned patients.
2. Privacy Boundaries: Zero PII/PHI, zero raw clinical notes, zero raw ECG waveform arrays.
3. Zero Mock Data Contract: Canonical honest states (AVAILABLE, NO_DATA, NOT_IMPLEMENTED, NOT_ENABLED).
"""

import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.errors import ConflictError, ForbiddenError, NotFoundError
from backend.app.services.capability_registry import get_clinician_workspace_capabilities
from backend.app.models.identity import ClinicianProfile, Hospital, User
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation
from backend.app.models.clinical import (
    ClinicalObservation,
    DocumentProcessingJob,
    DocumentVersion,
    ECGRecord,
    MedicalDocument,
    PrivacyMaskingResult,
)
from backend.app.models.ml_models import (
    CausalAnalysis,
    CounterfactualExplanation,
    DiagnosisPrediction,
    ModelVersion,
    PredictionExplanation,
)


class ClinicianService:
    """Domain service for clinician workspace telemetry and data access."""

    @staticmethod
    async def _get_assigned_patient_ids(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
    ) -> List[uuid.UUID]:
        """Retrieve list of patient UUIDs explicitly assigned to the clinician."""
        if not profile:
            # Super admin or unprofiled context
            res = await db.execute(select(Patient.id))
            return res.scalars().all()

        stmt = select(PatientClinicianAssignment.patient_id).where(
            PatientClinicianAssignment.clinician_id == profile.id,
            PatientClinicianAssignment.hospital_id == profile.hospital_id,
        )
        res = await db.execute(stmt)
        return res.scalars().all()

    @staticmethod
    async def get_overview(
        db: AsyncSession,
        user: User,
        profile: Optional[ClinicianProfile],
    ) -> Dict[str, Any]:
        """Retrieve high-level workspace metrics and subsystem capabilities."""
        hospital_name = None
        hospital_code = None
        if profile and profile.hospital_id:
            hosp_res = await db.execute(select(Hospital).where(Hospital.id == profile.hospital_id))
            hosp = hosp_res.scalar_one_or_none()
            if hosp:
                hospital_name = hosp.name
                hospital_code = hosp.code

        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        patient_count = len(assigned_patient_ids)

        doc_count = 0
        ecg_count = 0
        if patient_count > 0:
            doc_stmt = select(func.count(MedicalDocument.id)).where(MedicalDocument.patient_id.in_(assigned_patient_ids))
            doc_res = await db.execute(doc_stmt)
            doc_count = doc_res.scalar_one() or 0

            ecg_stmt = select(func.count(ECGRecord.id)).where(ECGRecord.patient_id.in_(assigned_patient_ids))
            ecg_res = await db.execute(ecg_stmt)
            ecg_count = ecg_res.scalar_one() or 0

        return {
            "clinician": {
                "profile_id": str(profile.id) if profile else None,
                "user_id": str(user.id),
                "department": profile.department if profile else "General Clinical",
                "hospital_id": str(profile.hospital_id) if profile else None,
                "hospital_name": hospital_name,
                "hospital_code": hospital_code,
                "status": "active" if user.is_active else "inactive",
            },
            "metrics": {
                "assigned_patients": {
                    "count": patient_count,
                    "status": "AVAILABLE" if patient_count > 0 else "NO_DATA",
                    "message": None if patient_count > 0 else "No patients assigned to your clinician profile",
                },
                "clinical_documents": {
                    "count": doc_count,
                    "status": "AVAILABLE" if doc_count > 0 else "NO_DATA",
                    "message": None if doc_count > 0 else "No medical documents stored for assigned patients",
                },
                "ecg_records": {
                    "count": ecg_count,
                    "status": "AVAILABLE" if ecg_count > 0 else "NO_DATA",
                    "message": None if ecg_count > 0 else "No ECG records acquired for assigned patients",
                },
            },
            "capabilities": get_clinician_workspace_capabilities(),
        }

    @staticmethod
    async def get_assigned_patients(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Retrieve paginated list of authorized patient references."""
        if not profile:
            count_stmt = select(func.count(Patient.id))
            query_stmt = select(Patient).limit(limit).offset(offset)
            total = (await db.execute(count_stmt)).scalar_one() or 0
            patients = (await db.execute(query_stmt)).scalars().all()
            
            patient_list = []
            for p in patients:
                patient_list.append({
                    "patient_id": str(p.id),
                    "opaque_patient_id": p.opaque_patient_id,
                    "hospital_id": None,
                    "assigned_at": p.created_at.isoformat() if p.created_at else None,
                    "documents_count": 0,
                    "ecg_records_count": 0,
                })
            return {
                "status": "AVAILABLE" if total > 0 else "NO_DATA",
                "total_count": total,
                "limit": limit,
                "offset": offset,
                "patients": patient_list,
            }

        # Query explicit clinician assignments
        count_stmt = select(func.count(PatientClinicianAssignment.id)).where(
            PatientClinicianAssignment.clinician_id == profile.id,
            PatientClinicianAssignment.hospital_id == profile.hospital_id,
        )
        total = (await db.execute(count_stmt)).scalar_one() or 0

        query_stmt = (
            select(PatientClinicianAssignment, Patient)
            .join(Patient, PatientClinicianAssignment.patient_id == Patient.id)
            .where(
                PatientClinicianAssignment.clinician_id == profile.id,
                PatientClinicianAssignment.hospital_id == profile.hospital_id,
            )
            .order_by(PatientClinicianAssignment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await db.execute(query_stmt)).all()

        hosp_res = await db.execute(select(Hospital).where(Hospital.id == profile.hospital_id))
        hospital = hosp_res.scalar_one_or_none()
        hospital_code = hospital.code if hospital else None

        patient_list = []
        for assign, pat in rows:
            doc_cnt_stmt = select(func.count(MedicalDocument.id)).where(MedicalDocument.patient_id == pat.id)
            doc_cnt = (await db.execute(doc_cnt_stmt)).scalar_one() or 0

            ecg_cnt_stmt = select(func.count(ECGRecord.id)).where(ECGRecord.patient_id == pat.id)
            ecg_cnt = (await db.execute(ecg_cnt_stmt)).scalar_one() or 0

            patient_list.append({
                "patient_id": str(pat.id),
                "opaque_patient_id": pat.opaque_patient_id,
                "hospital_id": str(assign.hospital_id),
                "hospital_code": hospital_code,
                "assigned_at": assign.created_at.isoformat() if assign.created_at else None,
                "documents_count": doc_cnt,
                "ecg_records_count": ecg_cnt,
            })

        return {
            "status": "AVAILABLE" if total > 0 else "NO_DATA",
            "total_count": total,
            "limit": limit,
            "offset": offset,
            "patients": patient_list,
        }

    @staticmethod
    async def get_eligible_patients_for_assignment(
        db: AsyncSession,
        profile: ClinicianProfile,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Patients registered at the clinician's hospital without an assignment to this clinician."""
        assigned_subq = (
            select(PatientClinicianAssignment.patient_id)
            .where(PatientClinicianAssignment.clinician_id == profile.id)
        )
        base = (
            select(Patient)
            .join(
                PatientHospitalAssociation,
                PatientHospitalAssociation.patient_id == Patient.id,
            )
            .where(
                PatientHospitalAssociation.hospital_id == profile.hospital_id,
                Patient.id.not_in(assigned_subq),
            )
        )
        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await db.execute(count_stmt)).scalar_one() or 0
        rows = (
            await db.execute(base.order_by(Patient.created_at.desc()).limit(limit).offset(offset))
        ).scalars().all()
        return {
            "status": "AVAILABLE" if total > 0 else "NO_DATA",
            "total_count": total,
            "limit": limit,
            "offset": offset,
            "patients": [
                {
                    "patient_id": str(p.id),
                    "opaque_patient_id": p.opaque_patient_id,
                    "hospital_id": str(profile.hospital_id),
                }
                for p in rows
            ],
        }

    @staticmethod
    async def assign_patient(
        db: AsyncSession,
        profile: ClinicianProfile,
        patient_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Create explicit clinician-patient assignment within hospital boundary."""
        if profile is None:
            raise ForbiddenError("Clinician profile required for patient assignment")

        pat_res = await db.execute(select(Patient).where(Patient.id == patient_id))
        patient = pat_res.scalar_one_or_none()
        if patient is None:
            raise NotFoundError("Patient reference resource not accessible")

        assoc_res = await db.execute(
            select(PatientHospitalAssociation).where(
                PatientHospitalAssociation.patient_id == patient_id,
                PatientHospitalAssociation.hospital_id == profile.hospital_id,
            )
        )
        if assoc_res.scalar_one_or_none() is None:
            raise NotFoundError("Patient reference resource not accessible")

        existing = await db.execute(
            select(PatientClinicianAssignment).where(
                PatientClinicianAssignment.patient_id == patient_id,
                PatientClinicianAssignment.clinician_id == profile.id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ConflictError("Patient is already assigned to this clinician")

        assignment = PatientClinicianAssignment(
            id=uuid.uuid4(),
            patient_id=patient_id,
            clinician_id=profile.id,
            hospital_id=profile.hospital_id,
        )
        db.add(assignment)
        await db.flush()
        return {
            "status": "ASSIGNED",
            "assignment_id": str(assignment.id),
            "patient_id": str(patient_id),
            "clinician_profile_id": str(profile.id),
            "hospital_id": str(profile.hospital_id),
            "opaque_patient_id": patient.opaque_patient_id,
        }

    @staticmethod
    async def get_patient_detail(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve full clinical metadata dossier for an authorized patient."""
        # Hospital information
        hospital_id = str(profile.hospital_id) if profile else None
        hospital_name = None
        hospital_code = None
        if profile and profile.hospital_id:
            hosp_res = await db.execute(select(Hospital).where(Hospital.id == profile.hospital_id))
            hosp = hosp_res.scalar_one_or_none()
            if hosp:
                hospital_name = hosp.name
                hospital_code = hosp.code

        # Clinical Observations
        obs_stmt = select(ClinicalObservation).where(ClinicalObservation.patient_id == patient.id)
        obs_rows = (await db.execute(obs_stmt)).scalars().all()
        observations = [
            {
                "id": str(o.id),
                "observation_type": o.observation_type,
                "feature_schema_version": o.feature_schema_version,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in obs_rows
        ]

        # Medical Documents (Metadata only, Zero raw notes)
        doc_stmt = (
            select(MedicalDocument)
            .options(
                selectinload(MedicalDocument.versions),
                selectinload(MedicalDocument.jobs),
                selectinload(MedicalDocument.masking_results),
            )
            .where(MedicalDocument.patient_id == patient.id)
            .order_by(MedicalDocument.created_at.desc())
        )
        doc_rows = (await db.execute(doc_stmt)).scalars().all()
        documents = []
        for d in doc_rows:
            masking_status = "NOT_PROCESSED"
            entities_count = 0
            if d.masking_results:
                masking_status = d.masking_results[0].sanitization_status
                entities_count = d.masking_results[0].entities_detected_count

            documents.append({
                "id": str(d.id),
                "document_type": d.document_type,
                "status": d.status,
                "versions_count": len(d.versions),
                "masking_status": masking_status,
                "entities_detected_count": entities_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            })

        # ECG Records (Acquisition parameters only, Zero raw signal arrays)
        ecg_stmt = (
            select(ECGRecord)
            .where(ECGRecord.patient_id == patient.id)
            .order_by(ECGRecord.created_at.desc())
        )
        ecg_rows = (await db.execute(ecg_stmt)).scalars().all()
        ecg_records = [
            {
                "id": str(e.id),
                "sampling_rate_hz": e.sampling_rate_hz,
                "duration_seconds": e.duration_seconds,
                "lead_count": e.lead_count,
                "dataset_source": e.dataset_source,
                "acquired_at": e.acquired_at.isoformat() if e.acquired_at else None,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in ecg_rows
        ]

        # Diagnosis Predictions (Genuine DB records only, Zero fake predictions)
        diag_stmt = (
            select(DiagnosisPrediction)
            .options(selectinload(DiagnosisPrediction.model_version))
            .where(DiagnosisPrediction.patient_id == patient.id)
            .order_by(DiagnosisPrediction.created_at.desc())
        )
        diag_rows = (await db.execute(diag_stmt)).scalars().all()
        prediction_data = None
        if diag_rows:
            pred = diag_rows[0]
            prediction_data = {
                "id": str(pred.id),
                "model_version_id": str(pred.model_version_id),
                "model_version_tag": pred.model_version.version_tag if pred.model_version else "Unknown",
                "target_definition": pred.target_definition,
                "prediction_type": pred.prediction_type,
                "provenance": pred.provenance,
                "status": pred.status,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
            }

        return {
            "patient": {
                "patient_id": str(patient.id),
                "opaque_patient_id": patient.opaque_patient_id,
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "hospital_code": hospital_code,
                "created_at": patient.created_at.isoformat() if patient.created_at else None,
            },
            "observations": {
                "status": "AVAILABLE" if len(observations) > 0 else "NO_DATA",
                "count": len(observations),
                "items": observations,
            },
            "documents": {
                "status": "AVAILABLE" if len(documents) > 0 else "NO_DATA",
                "count": len(documents),
                "items": documents,
            },
            "ecg_records": {
                "status": "AVAILABLE" if len(ecg_records) > 0 else "NO_DATA",
                "count": len(ecg_records),
                "items": ecg_records,
            },
            "diagnosis": {
                "status": "AVAILABLE" if prediction_data else "NO_DATA",
                "prediction": prediction_data,
                "message": (
                    None
                    if prediction_data
                    else "Prediction unavailable — clinical inference is not implemented."
                ),
            },
            "ai_capabilities": {
                k: v
                for k, v in get_clinician_workspace_capabilities().items()
                if k
                in (
                    "clinical_ai_inference",
                    "dice_counterfactuals",
                    "dowhy_causal",
                    "clinical_decision_support",
                )
            },
        }

    @staticmethod
    async def get_documents(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        patient_id: Optional[uuid.UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Retrieve paginated documents for authorized assigned patients."""
        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        if not assigned_patient_ids:
            return {"status": "NO_DATA", "total_count": 0, "limit": limit, "offset": offset, "documents": []}

        if patient_id and patient_id not in assigned_patient_ids:
            raise NotFoundError("Patient reference resource not accessible")

        target_ids = [patient_id] if patient_id else assigned_patient_ids

        count_stmt = select(func.count(MedicalDocument.id)).where(MedicalDocument.patient_id.in_(target_ids))
        total = (await db.execute(count_stmt)).scalar_one() or 0

        query_stmt = (
            select(MedicalDocument, Patient)
            .join(Patient, MedicalDocument.patient_id == Patient.id)
            .options(
                selectinload(MedicalDocument.versions),
                selectinload(MedicalDocument.masking_results),
            )
            .where(MedicalDocument.patient_id.in_(target_ids))
            .order_by(MedicalDocument.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await db.execute(query_stmt)).all()

        doc_list = []
        for d, p in rows:
            masking_status = "NOT_PROCESSED"
            entities_count = 0
            if d.masking_results:
                masking_status = d.masking_results[0].sanitization_status
                entities_count = d.masking_results[0].entities_detected_count

            doc_list.append({
                "id": str(d.id),
                "patient_id": str(d.patient_id),
                "opaque_patient_id": p.opaque_patient_id,
                "document_type": d.document_type,
                "status": d.status,
                "versions_count": len(d.versions),
                "masking_status": masking_status,
                "entities_detected_count": entities_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None,
            })

        return {
            "status": "AVAILABLE" if total > 0 else "NO_DATA",
            "total_count": total,
            "limit": limit,
            "offset": offset,
            "documents": doc_list,
        }

    @staticmethod
    async def get_document_detail(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        document_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Retrieve detailed document metadata and processing job history."""
        stmt = (
            select(MedicalDocument, Patient)
            .join(Patient, MedicalDocument.patient_id == Patient.id)
            .options(
                selectinload(MedicalDocument.versions),
                selectinload(MedicalDocument.jobs),
                selectinload(MedicalDocument.masking_results).selectinload(PrivacyMaskingResult.entity_counts),
            )
            .where(MedicalDocument.id == document_id)
        )
        res = await db.execute(stmt)
        row = res.first()
        if not row:
            raise NotFoundError("Medical document resource not accessible")

        doc, pat = row

        # Check authorization scope
        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        if doc.patient_id not in assigned_patient_ids:
            raise NotFoundError("Medical document resource not accessible")

        versions = [
            {
                "id": str(v.id),
                "version_number": v.version_number,
                "checksum_sha256": v.checksum_sha256,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in doc.versions
        ]

        jobs = [
            {
                "id": str(j.id),
                "status": j.status,
                "error_message": j.error_message,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in doc.jobs
        ]

        masking_results = [
            {
                "id": str(m.id),
                "masking_method": m.masking_method,
                "entities_detected_count": m.entities_detected_count,
                "sanitization_status": m.sanitization_status,
                "entity_counts": {
                    ec.entity_category: ec.count for ec in getattr(m, "entity_counts", [])
                },
                "processed_at": m.processed_at.isoformat() if m.processed_at else None,
            }
            for m in doc.masking_results
        ]

        return {
            "id": str(doc.id),
            "patient_id": str(doc.patient_id),
            "opaque_patient_id": pat.opaque_patient_id,
            "document_type": doc.document_type,
            "status": doc.status,
            "versions": versions,
            "jobs": jobs,
            "masking_results": masking_results,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        }

    @staticmethod
    async def get_diagnosis(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
    ) -> Dict[str, Any]:
        """Retrieve genuine diagnosis predictions for assigned patients."""
        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        if not assigned_patient_ids:
            return {
                "status": "NO_DATA",
                "count": 0,
                "predictions": [],
                "inference_capability": "NOT_IMPLEMENTED",
                "message": "Clinical AI inference is not yet available.",
            }

        stmt = (
            select(DiagnosisPrediction, Patient)
            .join(Patient, DiagnosisPrediction.patient_id == Patient.id)
            .options(selectinload(DiagnosisPrediction.model_version))
            .where(DiagnosisPrediction.patient_id.in_(assigned_patient_ids))
            .order_by(DiagnosisPrediction.created_at.desc())
        )
        rows = (await db.execute(stmt)).all()

        pred_list = []
        for dp, p in rows:
            pred_list.append({
                "id": str(dp.id),
                "patient_id": str(dp.patient_id),
                "opaque_patient_id": p.opaque_patient_id,
                "model_version_tag": dp.model_version.version_tag if dp.model_version else "Unknown",
                "target_definition": dp.target_definition,
                "prediction_type": dp.prediction_type,
                "provenance": dp.provenance,
                "status": dp.status,
                "created_at": dp.created_at.isoformat() if dp.created_at else None,
            })

        return {
            "status": "AVAILABLE" if len(pred_list) > 0 else "NO_DATA",
            "count": len(pred_list),
            "predictions": pred_list,
            "inference_capability": "NOT_IMPLEMENTED",
            "message": None if len(pred_list) > 0 else "No genuine diagnosis predictions recorded.",
        }

    @staticmethod
    async def get_patient_diagnosis(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        patient: Patient,
    ) -> Dict[str, Any]:
        """Retrieve diagnosis prediction for a specific authorized patient."""
        stmt = (
            select(DiagnosisPrediction)
            .options(selectinload(DiagnosisPrediction.model_version))
            .where(DiagnosisPrediction.patient_id == patient.id)
            .order_by(DiagnosisPrediction.created_at.desc())
        )
        res = await db.execute(stmt)
        pred = res.scalar_one_or_none()

        if not pred:
            return {
                "status": "NO_DATA",
                "prediction": None,
                "inference_capability": "NOT_IMPLEMENTED",
                "message": "Prediction unavailable — clinical inference is not implemented.",
            }

        return {
            "status": "AVAILABLE",
            "prediction": {
                "id": str(pred.id),
                "patient_id": str(pred.patient_id),
                "opaque_patient_id": patient.opaque_patient_id,
                "model_version_tag": pred.model_version.version_tag if pred.model_version else "Unknown",
                "target_definition": pred.target_definition,
                "prediction_type": pred.prediction_type,
                "provenance": pred.provenance,
                "status": pred.status,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
            },
            "inference_capability": "NOT_IMPLEMENTED",
        }

    @staticmethod
    async def get_prediction_explanation(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        prediction_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Retrieve DiCE explanations for a genuine diagnosis prediction."""
        # Check prediction existence and patient authorization
        stmt = select(DiagnosisPrediction).where(DiagnosisPrediction.id == prediction_id)
        res = await db.execute(stmt)
        pred = res.scalar_one_or_none()
        if not pred:
            raise NotFoundError("Diagnosis prediction resource not accessible")

        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        if pred.patient_id not in assigned_patient_ids:
            raise NotFoundError("Diagnosis prediction resource not accessible")

        # Query genuine explanation record if present
        exp_stmt = (
            select(PredictionExplanation)
            .options(selectinload(PredictionExplanation.counterfactuals))
            .where(PredictionExplanation.prediction_id == prediction_id)
        )
        exp_res = await db.execute(exp_stmt)
        explanation = exp_res.scalar_one_or_none()

        if not explanation:
            return {
                "status": "NOT_IMPLEMENTED",
                "method": "DiCE-Counterfactual",
                "explanation": None,
                "message": "Explanation generation is not yet available.",
            }

        counterfactuals = [
            {
                "id": str(c.id),
                "feature_perturbation_summary": c.feature_perturbation_summary,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in explanation.counterfactuals
        ]

        return {
            "status": "AVAILABLE",
            "method": explanation.method,
            "explanation": {
                "id": str(explanation.id),
                "prediction_id": str(explanation.prediction_id),
                "status": explanation.status,
                "counterfactuals": counterfactuals,
                "created_at": explanation.created_at.isoformat() if explanation.created_at else None,
            },
        }

    @staticmethod
    async def get_prediction_causal(
        db: AsyncSession,
        profile: Optional[ClinicianProfile],
        prediction_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """Retrieve DoWhy causal analyses for a genuine diagnosis prediction."""
        stmt = select(DiagnosisPrediction).where(DiagnosisPrediction.id == prediction_id)
        res = await db.execute(stmt)
        pred = res.scalar_one_or_none()
        if not pred:
            raise NotFoundError("Diagnosis prediction resource not accessible")

        assigned_patient_ids = await ClinicianService._get_assigned_patient_ids(db, profile)
        if pred.patient_id not in assigned_patient_ids:
            raise NotFoundError("Diagnosis prediction resource not accessible")

        causal_stmt = select(CausalAnalysis).where(CausalAnalysis.prediction_id == prediction_id)
        causal_res = await db.execute(causal_stmt)
        analysis = causal_res.scalar_one_or_none()

        if not analysis:
            return {
                "status": "NOT_IMPLEMENTED",
                "method": "DoWhy-DAG-Intervention",
                "analysis": None,
                "message": "Causal analysis is not yet available.",
            }

        return {
            "status": "AVAILABLE",
            "method": analysis.method,
            "analysis": {
                "id": str(analysis.id),
                "prediction_id": str(analysis.prediction_id),
                "status": analysis.status,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            },
        }
