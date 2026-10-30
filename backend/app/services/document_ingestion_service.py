"""Medical Document Ingestion and Privacy Masking Orchestrator Service.

HARD PRIVACY CONTRACT:
1. Zero Raw Text in PostgreSQL: Original clinical notes, OCR outputs, and raw unmasked texts
   are strictly kept in local protected storage (storage/protected_documents/) and never written
   to central database columns.
2. Aggregate Entity Counts Only: Only aggregate counts per category are stored in
   privacy_masking_entity_counts. Raw detected entity values and masking spans are NEVER stored.
3. Downstream AI Privacy Gate: Verifies that only fully sanitized representations reach
   DOCUMENT_READY_FOR_AI. No AI diagnosis is invoked.
"""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.errors import NotFoundError, ValidationError, ForbiddenError
from backend.app.models.audit import AuditLog
from backend.app.models.clinical import (
    DocumentProcessingJob,
    DocumentVersion,
    MedicalDocument,
    PrivacyMaskingEntityCount,
    PrivacyMaskingResult,
)
from backend.app.models.identity import ClinicianProfile, Hospital, User
from backend.app.models.patient import Patient, PatientClinicianAssignment, PatientHospitalAssociation
from backend.app.services.document_classifier import DocumentClassifier
from backend.app.services.document_storage import default_document_storage
from backend.app.services.document_validation import DocumentValidator, SecurityScanner
from backend.app.services.privacy_gate import MaskingValidator, document_safe_for_ai
from backend.app.services.privacy_masking import PrivacyMasker
from backend.app.services.text_extraction import TextExtractor


class DocumentIngestionService:
    """End-to-end orchestrator for medical document ingestion and privacy masking."""

    @staticmethod
    async def upload_document(
        db: AsyncSession,
        user: User,
        profile: Optional[ClinicianProfile],
        filename: str,
        content_type: str,
        content: bytes,
        patient_id: Optional[uuid.UUID] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate, store original file in protected storage, and persist metadata in DB."""
        req_id = request_id or str(uuid.uuid4())

        # Determine hospital ID
        if profile and profile.hospital_id:
            hospital_id = profile.hospital_id
        else:
            # Fallback to query hospital membership
            hospital_id = getattr(user, "hospital_id", None)
            if not hospital_id:
                # Query first hospital as fallback
                hosp = (await db.execute(select(Hospital))).scalars().first()
                if not hosp:
                    raise ValidationError("No active hospital node found in system.")
                hospital_id = hosp.id

        # patient_id is required: medical_documents.patient_id is NOT NULL in PostgreSQL.
        if not patient_id:
            raise ValidationError(
                "patient_id is required. Select an assigned patient before uploading a clinical document."
            )

        # 1. Authorize patient association
        patient = None
        if patient_id:
            stmt = select(Patient).where(Patient.id == patient_id)
            patient = (await db.execute(stmt)).scalars().first()
            if not patient:
                raise NotFoundError("Specified patient does not exist in your authorized hospital node.")

            assoc_stmt = select(PatientHospitalAssociation).where(
                PatientHospitalAssociation.patient_id == patient_id,
                PatientHospitalAssociation.hospital_id == hospital_id,
            )
            assoc = (await db.execute(assoc_stmt)).scalars().first()
            if not assoc:
                raise NotFoundError("Specified patient does not exist in your authorized hospital node.")

            # If clinician profile exists, verify explicit assignment
            if profile:
                assign_stmt = select(PatientClinicianAssignment).where(
                    PatientClinicianAssignment.clinician_id == profile.id,
                    PatientClinicianAssignment.patient_id == patient_id,
                )
                assignment = (await db.execute(assign_stmt)).scalars().first()
                if not assignment:
                    raise ForbiddenError("You are not assigned to manage this patient.")

        # 2. File Validation
        try:
            val_meta = DocumentValidator.validate_file(filename, content_type, content)
        except ValidationError as ve:
            # Audit validation failure
            audit_fail = AuditLog(
                event_type="DOCUMENT_VALIDATION_FAILED",
                actor_id=user.id,
                hospital_id=hospital_id,
                resource_type="medical_document",
                resource_id=None,
                request_id=req_id,
                outcome="DENIED",
            )
            db.add(audit_fail)
            await db.commit()
            raise

        # 3. Compute SHA256 checksum
        checksum = hashlib.sha256(content).hexdigest()

        # 4. Generate Document ID and Store in Protected Storage
        doc_id = uuid.uuid4()
        local_uri = default_document_storage.store_original(
            hospital_id=hospital_id,
            doc_id=doc_id,
            ext=val_meta["extension"],
            content=content,
        )

        initial_status = "UPLOADED"

        # 5. Persist MedicalDocument & DocumentVersion
        doc = MedicalDocument(
            id=doc_id,
            patient_id=patient.id,
            hospital_id=hospital_id,
            local_document_uri=local_uri,
            document_type="UNKNOWN",
            status=initial_status,
        )
        db.add(doc)

        version = DocumentVersion(
            document_id=doc_id,
            version_number=1,
            checksum_sha256=checksum,
        )
        db.add(version)

        # 6. Audit Successful Upload
        audit_success = AuditLog(
            event_type="DOCUMENT_UPLOADED",
            actor_id=user.id,
            hospital_id=hospital_id,
            resource_type="medical_document",
            resource_id=str(doc_id),
            request_id=req_id,
            outcome="SUCCESS",
        )
        db.add(audit_success)

        await db.commit()
        await db.refresh(doc)

        return {
            "document_id": str(doc.id),
            "status": doc.status,
            "patient_associated": bool(patient),
            "patient_id": str(patient.id) if patient else None,
            "opaque_patient_id": patient.opaque_patient_id if patient else None,
            "version": 1,
            "checksum_sha256": checksum,
            "file_metadata": {
                "extension": val_meta["extension"],
                "mime_type": val_meta["mime_type"],
                "size_bytes": val_meta["size_bytes"],
                "page_count": val_meta["page_count"],
            },
            "message": "Document uploaded successfully to protected storage.",
        }

    @staticmethod
    async def process_document(
        db: AsyncSession,
        user: User,
        profile: Optional[ClinicianProfile],
        document_id: uuid.UUID,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end security check, OCR/extraction, classification, masking, and AI gate."""
        req_id = request_id or str(uuid.uuid4())

        # Retrieve document with versions and patient
        stmt = (
            select(MedicalDocument)
            .options(
                selectinload(MedicalDocument.patient),
                selectinload(MedicalDocument.versions),
                selectinload(MedicalDocument.jobs),
            )
            .where(MedicalDocument.id == document_id)
        )
        doc = (await db.execute(stmt)).scalars().first()
        if not doc:
            raise NotFoundError("Medical document resource not found.")

        # Check hospital and clinician authorization
        if profile and profile.hospital_id and doc.hospital_id != profile.hospital_id:
            raise NotFoundError("Medical document resource not accessible.")

        if profile and doc.patient_id:
            assign_stmt = select(PatientClinicianAssignment).where(
                PatientClinicianAssignment.clinician_id == profile.id,
                PatientClinicianAssignment.patient_id == doc.patient_id,
            )
            assignment = (await db.execute(assign_stmt)).scalars().first()
            if not assignment:
                raise ForbiddenError("You are not authorized to process documents for this patient.")

        # Retrieve raw content from protected storage
        try:
            raw_content = default_document_storage.retrieve_for_authorized_processing(doc.local_document_uri)
        except Exception as e:
            raise NotFoundError(f"Protected document storage file missing: {str(e)}")

        # Create processing job record
        job = DocumentProcessingJob(
            document_id=doc.id,
            status="processing",
            started_at=datetime.now(timezone.utc),
        )
        db.add(job)
        await db.commit()

        # Step 1: Security Validation (Antivirus)
        sec_res = SecurityScanner.scan_content(raw_content)

        # Step 2: Text Extraction / OCR
        _, ext = os.path.splitext(doc.local_document_uri)
        ext_res = TextExtractor.extract_text(raw_content, ext)

        if ext_res["status"] == "FAILED":
            job.status = "failed"
            job.error_message = ext_res["message"]
            job.completed_at = datetime.now(timezone.utc)
            doc.status = "TEXT_EXTRACTION_FAILED"
            await db.commit()
            return {
                "document_id": str(doc.id),
                "status": "TEXT_EXTRACTION_FAILED",
                "error": ext_res["message"],
            }

        extracted_text = ext_res["text"]

        # Step 3: Document Classification
        class_res = DocumentClassifier.classify(extracted_text, filename_hint=ext)
        doc.document_type = class_res["category"]

        # Step 4: Privacy Masking & Entity Detection
        masked_text, entity_counts, detected_raw_vals = PrivacyMasker.mask_document(extracted_text)

        # Step 5: Masking Validation
        is_masking_valid, val_msg = MaskingValidator.validate_masking(
            original_text=extracted_text,
            masked_text=masked_text,
            detected_values=detected_raw_vals,
        )

        total_pii_detected = sum(entity_counts.values())

        if not is_masking_valid:
            job.status = "failed"
            job.error_message = val_msg
            job.completed_at = datetime.now(timezone.utc)
            doc.status = "MASKING_VALIDATION_FAILED"

            audit_fail = AuditLog(
                event_type="DOCUMENT_MASKING_VALIDATION_FAILED",
                actor_id=user.id,
                hospital_id=doc.hospital_id,
                resource_type="medical_document",
                resource_id=str(doc.id),
                request_id=req_id,
                outcome="ERROR",
            )
            db.add(audit_fail)
            await db.commit()

            return {
                "document_id": str(doc.id),
                "status": "MASKING_VALIDATION_FAILED",
                "error": val_msg,
            }

        # Step 6: Store Masked Text in Protected Storage
        masked_uri = default_document_storage.store_masked(
            hospital_id=doc.hospital_id,
            doc_id=doc.id,
            masked_text=masked_text,
        )

        # Step 7: Persist Masking Result and Aggregate Entity Counts (Zero PII in DB)
        masking_result = PrivacyMaskingResult(
            document_id=doc.id,
            masking_method="BioClinicalBERT-PII-Masker",
            entities_detected_count=total_pii_detected,
            sanitization_status="sanitized",
            processed_at=datetime.now(timezone.utc),
        )
        db.add(masking_result)
        await db.flush()  # populate masking_result.id

        # Insert aggregate entity counts
        for category, count in entity_counts.items():
            if count > 0:
                ec = PrivacyMaskingEntityCount(
                    masking_result_id=masking_result.id,
                    entity_category=category,
                    count=count,
                )
                db.add(ec)

        # Step 8: Evaluate Downstream Privacy Gate
        doc_metadata = {
            "file_validation_status": "VALIDATED",
            "security_status": sec_res["status"],
            "extraction_status": ext_res["status"],
            "document_type": doc.document_type,
            "masking_completed": True,
            "masking_validation_status": "VALIDATED",
            "patient_associated": bool(doc.patient_id),
            "patient_id": str(doc.patient_id) if doc.patient_id else None,
            "total_pii_count": total_pii_detected,
            "_original_text_in_memory": extracted_text,
        }

        # Downstream safe payload (Strictly zero raw text)
        downstream_payload = {
            "document_id": str(doc.id),
            "document_type": doc.document_type,
            "masked_text": masked_text,
            "masked_uri": masked_uri,
            "metadata_version": "v1.0",
        }

        is_safe_for_ai, final_state = document_safe_for_ai(doc_metadata, downstream_payload)

        # Update Document and Job Status
        doc.status = final_state if is_safe_for_ai else "MASKED_READY"
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)

        # Step 9: Audit Lifecycle Completion
        audit_comp = AuditLog(
            event_type="DOCUMENT_PROCESSING_COMPLETED",
            actor_id=user.id,
            hospital_id=doc.hospital_id,
            resource_type="medical_document",
            resource_id=str(doc.id),
            request_id=req_id,
            outcome="SUCCESS",
        )
        db.add(audit_comp)

        await db.commit()

        return {
            "document_id": str(doc.id),
            "status": doc.status,
            "classification": {
                "category": doc.document_type,
                "confidence": class_res["confidence"],
                "matched_keywords": class_res["matched_keywords"],
            },
            "security_scan": {
                "status": sec_res["status"],
                "engine": sec_res.get("engine"),
                "message": sec_res.get("message"),
            },
            "text_extraction": {
                "status": ext_res["status"],
                "mode": ext_res["extraction_mode"],
                "page_count": ext_res["page_count"],
            },
            "privacy_masking": {
                "sanitization_status": "sanitized",
                "entities_detected_count": total_pii_detected,
                "entity_counts": {k: v for k, v in entity_counts.items() if v > 0},
            },
            "downstream_ai_gate": {
                "safe_for_ai": is_safe_for_ai,
                "final_state": final_state,
            },
            "message": "Document processed and privacy-masked successfully.",
        }
