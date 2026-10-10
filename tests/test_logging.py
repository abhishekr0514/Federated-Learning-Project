"""Test suite for structured JSON logging and privacy filter scrubbing."""

import json
import logging
from backend.app.core.logging import JSONLogFormatter, sanitize_log_dict


def test_sanitize_log_dict_redaction():
    """Verify sensitive key fields (passwords, tokens, PII, ECG) are redacted from logs."""
    sensitive_data = {
        "username": "clinician_1",
        "password": "SuperSecretPassword123",
        "access_token": "eyJhbGciOi...",
        "patient_name": "John Doe",
        "clinical_notes": "Patient complains of severe chest pain.",
        "nested": {
            "secret": "DeepSecretValue",
            "normal_field": "public_val"
        }
    }

    scrubbed = sanitize_log_dict(sensitive_data)
    
    assert scrubbed["username"] == "clinician_1"
    assert scrubbed["password"] == "[REDACTED]"
    assert scrubbed["access_token"] == "[REDACTED]"
    assert scrubbed["patient_name"] == "[REDACTED]"
    assert scrubbed["clinical_notes"] == "[REDACTED]"
    assert scrubbed["nested"]["secret"] == "[REDACTED]"
    assert scrubbed["nested"]["normal_field"] == "public_val"


def test_json_log_formatter():
    """Verify log record is formatted as valid JSON with request metadata."""
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="medshield.test",
        level=logging.INFO,
        pathname="test_logging.py",
        lineno=10,
        msg="Test log entry",
        args=(),
        exc_info=None,
    )
    
    output = formatter.format(record)
    parsed = json.loads(output)
    
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "medshield.test"
    assert parsed["message"] == "Test log entry"
    assert "timestamp" in parsed
    assert "request_id" in parsed
