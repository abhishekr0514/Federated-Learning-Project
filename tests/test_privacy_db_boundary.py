"""Security and Privacy Boundary Test Suite for Central Database Schema."""

import re
from pathlib import Path
from backend.app.models import (
    Patient,
    MedicalDocument,
    ECGRecord,
    User,
    AuditLog,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "backend" / "app" / "db" / "migrations"


def test_patient_model_has_no_pii_columns():
    """Verify Patient model contains NO central PII attributes (name, ssn, dob, phone, etc.)."""
    patient_columns = [col.name for col in Patient.__table__.columns]
    prohibited_pii = ["name", "first_name", "last_name", "ssn", "dob", "address", "phone", "email", "mrn", "government_id"]
    
    for pii in prohibited_pii:
        assert pii not in patient_columns, f"Prohibited PII column '{pii}' found in central Patient model!"


def test_ecg_record_has_no_raw_signal_waveform_columns():
    """Verify ECGRecord model contains NO central raw signal waveform array columns."""
    ecg_columns = [col.name for col in ECGRecord.__table__.columns]
    prohibited_raw_data = ["raw_signal", "signal_array", "ecg_data", "waveforms", "raw_lead_tensors"]
    
    for raw_col in prohibited_raw_data:
        assert raw_col not in ecg_columns, f"Prohibited raw ECG signal column '{raw_col}' found in central ECGRecord model!"


def test_medical_document_has_no_raw_clinical_text_columns():
    """Verify MedicalDocument model contains NO raw clinical note text columns."""
    doc_columns = [col.name for col in MedicalDocument.__table__.columns]
    prohibited_text_cols = ["raw_text", "clinical_notes", "unmasked_content", "document_text", "full_text"]
    
    for text_col in prohibited_text_cols:
        assert text_col not in doc_columns, f"Prohibited raw clinical text column '{text_col}' found in central MedicalDocument model!"


def test_user_password_field_is_hashed_only():
    """Verify User model names password attribute as hashed_password to prevent plaintext usage."""
    user_columns = [col.name for col in User.__table__.columns]
    assert "hashed_password" in user_columns
    assert "password" not in user_columns, "Plain 'password' column found in User model!"


def test_audit_log_has_no_sensitive_data_fields():
    """Verify AuditLog model contains NO raw text or token fields."""
    audit_columns = [col.name for col in AuditLog.__table__.columns]
    prohibited_audit_cols = ["password", "token", "jwt", "clinical_text", "ecg_values", "phi"]
    
    for col in prohibited_audit_cols:
        assert col not in audit_columns, f"Prohibited field '{col}' found in AuditLog model!"


def test_no_hardcoded_secrets_in_migration_files():
    """Verify migration files contain no hardcoded production passwords or JWT keys."""
    suspicious_patterns = [
        r'(?i)secret_key\s*=\s*["\'][^"\']{10,}["\']',
        r'(?i)password\s*=\s*["\'](?!default|fake|test)[^"\']{8,}["\']',
    ]

    for py_file in MIGRATIONS_DIR.glob("**/*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pattern in suspicious_patterns:
            match = re.search(pattern, content)
            assert match is None, f"Potential hardcoded secret found in migration {py_file.name}: {match.group(0)}"
