# MedShield-FL — Medical Document Ingestion & Privacy Masking Pipeline

## 1. Executive Summary & Core Mandate

The **Medical Document Ingestion & Privacy Masking Pipeline** is a core privacy-safe capability of MedShield-FL. Its objective is to ingest heterogeneous clinical documents (e.g., PDF lab panels, discharge summaries, cardiology consult notes, and medical images) and produce sanitized, privacy-masked artifacts suitable for future downstream AI processing, **without leaking raw Protected Health Information (PHI) or unmasked clinical text into the central relational database**.

> [!IMPORTANT]
> **Explicit Trust & Privacy Declarations**:
> 1. *"Raw unmasked clinical document content is not stored in the central PostgreSQL database."*
> 2. *"Document ingestion does not establish clinical model-training eligibility."*
> 3. *"The pipeline terminates at `DOCUMENT_READY_FOR_AI`. Downstream diagnostic inference and federated training are decoupled and not executed during document ingestion."*

---

## 2. Trust Boundary Architecture

```
                                    TRUST BOUNDARY
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  Client / Clinician Browser                                                           │
│  (Authenticated JWT, Hospital & Patient Scoped RBAC)                                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Multipart POST
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Gateway & Ingestion Service (backend/app/services/)                           │
│  - File Format, MIME, and Magic Byte Verification                                     │
│  - Security Scanning (Antivirus Telemetry)                                            │
│  - In-Memory Text Extraction (pdfplumber)                                              │
│  - Rule-Based Document Classification                                                 │
│  - PII Entity Detection & Semantic Masking                                             │
│  - Masking Integrity Validation                                                       │
└───────────────┬──────────────────────────────────────────────┬─────────────────────────┘
                │                                              │
                │ Opaque Key / UUID File Storage               │ Safe Metadata & Aggregate Counts
                ▼                                              ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│  Protected Local Storage                     │ │  Central Relational DB (PostgreSQL)    │
│  storage/protected_documents/                │ │  - medical_documents                   │
│  - {hospital_id}/{doc_id}/original.{ext}     │ │  - document_versions                   │
│  - {hospital_id}/{doc_id}/masked.txt         │ │  - document_processing_jobs            │
│                                              │ │  - privacy_masking_results             │
│  * Strictly isolated outside web/public dirs │ │  - privacy_masking_entity_counts       │
│  * Excluded from version control (.gitignore)│ │  * ZERO raw text columns               │
│  * Production: S3/MinIO with SSE-KMS         │ │  * ZERO raw PII or masking spans       │
└──────────────────────────────────────────────┘ └────────────────────────────────────────┘
```

---

## 3. Pipeline Lifecycle Stages

Every uploaded document transitions through an immutable state machine:

```
[UPLOAD]
   │
   ▼
1. FILE VALIDATION
   ├── Check allowed extension (.pdf, .jpg, .jpeg, .png)
   ├── Check MIME type header
   ├── Check magic bytes / file signatures (%PDF-, \xFF\xD8\xFF, \x89PNG)
   ├── Enforce size limit (<= 10 MB)
   └── Verify structural integrity (openable, non-zero pages/dimensions)
   │
   ▼
2. SECURITY VALIDATION
   ├── Inspect for host antivirus engine (e.g. ClamAV daemon)
   └── If engine missing: record status as NOT_IMPLEMENTED (never false 'clean')
   │
   ▼
3. TEXT EXTRACTION / OCR
   ├── DIGITAL_PDF_TEXT: extract embedded text stream via pdfplumber
   └── SCANNED_DOCUMENT_OCR / IMAGE_OCR: check Tesseract binary; if absent, NOT_IMPLEMENTED
   │
   ▼
4. DOCUMENT CLASSIFICATION
   ├── Match clinical keywords against standard categories (e.g., LIPID_PROFILE, ECG_REPORT)
   └── Report confidence as NOT_MEASURED (honest scientific telemetry)
   │
   ▼
5. PII / DIRECT IDENTIFIER DETECTION
   ├── Detect PATIENT_NAME, MRN, PHONE, EMAIL, ADDRESS, GOVERNMENT_ID, REG_NO, DOB
   └── Compute aggregate counts per category (safe for DB persistence)
   │
   ▼
6. PRIVACY MASKING
   ├── Replace detected spans with semantic tokens: [PATIENT_NAME], [MRN], [PHONE], etc.
   └── Strictly preserve clinical findings, lab values, and units (e.g., LDL 160 mg/dL)
   │
   ▼
7. MASKING VALIDATION
   ├── Assert detected raw PII values are absent from masked text
   ├── Assert unmasked pattern regexes (SSN, email, phone) do not match
   ├── Assert clinical terminology and numbers remain present
   └── Rejects to MASKING_VALIDATION_FAILED if any check fails
   │
   ▼
8. DOWNSTREAM AI PRIVACY GATE
   ├── document_safe_for_ai() checks all validations + patient assignment authorization
   ├── Blocks transmission if raw or unmasked text is present in payload
   └── Reaches final state: DOCUMENT_READY_FOR_AI
```

---

## 4. Preserving Clinical Integrity While Protecting Privacy

Privacy masking must never compromise the medical utility of clinical text. The `PrivacyMasker` replaces direct identifying labels while leaving laboratory results and clinical metrics intact:

| Original Snippet | Masked Output | Status |
| :--- | :--- | :--- |
| `Patient Name: John Doe` | `Patient Name: [PATIENT_NAME]` | Identifier Masked |
| `MRN: 12345-AB` | `MRN: [MRN]` | Identifier Masked |
| `Phone: 555-123-4567` | `Phone: [PHONE]` | Identifier Masked |
| `Email: jane@test.org` | `Email: [EMAIL]` | Identifier Masked |
| `LDL 160 mg/dL` | `LDL 160 mg/dL` | **Preserved Intact** |
| `Troponin I 0.04 ng/mL` | `Troponin I 0.04 ng/mL` | **Preserved Intact** |
| `Ejection Fraction 55%` | `Ejection Fraction 55%` | **Preserved Intact** |
| `Sinus rhythm with ST elevation` | `Sinus rhythm with ST elevation` | **Preserved Intact** |

---

## 5. Central Relational Database Usage

The central PostgreSQL / SQLite database stores **operational and telemetry metadata only**:

- `medical_documents`:
  - `id`: Unique GUID
  - `hospital_id`: Owning hospital GUID
  - `patient_id`: Associated patient GUID
  - `document_type`: Classified category (e.g., `LIPID_PROFILE`, `ECG_REPORT`, `UNKNOWN`)
  - `status`: Lifecycle state (`UPLOADED`, `DOCUMENT_READY_FOR_AI`, `MASKING_VALIDATION_FAILED`)
  - `local_document_uri`: Opaque storage reference (`protected://hospitals/...`)
- `document_versions`:
  - `version_number`: Integer
  - `checksum_sha256`: Cryptographic digest of uploaded file
- `document_processing_jobs`:
  - `status`: `processing`, `completed`, `failed`
  - `error_message`: Sanitized error description
- `privacy_masking_results`:
  - `masking_method`: Sanitization algorithm identifier
  - `entities_detected_count`: Total aggregate count of direct identifiers found
  - `sanitization_status`: `sanitized`
- `privacy_masking_entity_counts`:
  - `entity_category`: Category label (`PATIENT_NAME`, `MRN`, `PHONE`, etc.)
  - `count`: Aggregate count of detections for this category

**Prohibited in Database**:
- No raw file bytes
- No raw OCR text
- No detected PII strings or values
- No character offsets or masking spans that could reveal sensitive data

---

## 6. Audit Logging Safety

All security and processing events are logged to `audit_logs` append-only table:
- `DOCUMENT_UPLOADED`
- `DOCUMENT_VALIDATION_FAILED`
- `DOCUMENT_PROCESSING_COMPLETED`
- `DOCUMENT_MASKING_VALIDATION_FAILED`

Audit entries record `actor_id`, `hospital_id`, `resource_type`, `resource_id`, `request_id`, and `outcome`. **Zero clinical text, zero OCR buffers, and zero patient PII are ever written to audit logs.**
