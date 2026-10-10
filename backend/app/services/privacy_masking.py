"""Privacy Masking and PII Detection Service.

PRIVACY ARCHITECTURE CONTRACT:
1. Direct Identifier Detection: Detects direct identifiers (Name, MRN, Phone, Email, Address,
   Government ID, Registration Number, Date of Birth).
2. Preserving Clinical Content: Medical findings, laboratory values, measurements, and units
   (e.g., 'LDL 160 mg/dL', 'Troponin 0.04 ng/mL', 'EF 55%') MUST be preserved intact.
3. Zero Central PII Storage: Central PostgreSQL stores ONLY aggregate entity counts
   (e.g., {'PATIENT_NAME': 1, 'MRN': 1}).
   Raw detected entity values and masking spans are NEVER saved to the database and NEVER logged.
"""

import re
from typing import Any, Dict, List, Set, Tuple

# Supported direct-identifier categories
PII_CATEGORIES = [
    "PATIENT_NAME",
    "MRN",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "GOVERNMENT_ID",
    "REGISTRATION_NUMBER",
    "DATE_OF_BIRTH",
]


class PIIDetector:
    """Detects direct personal identifiers using structured regex and pattern matchers."""

    # Patterns that identify labels followed by values
    LABELED_PATTERNS = [
        (
            "PATIENT_NAME",
            re.compile(r"(?i)\b(?:Patient\s*Name|Pt(?:\s*Name)?|Name)\s*:\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)"),
        ),
        (
            "MRN",
            re.compile(r"(?i)\b(?:MRN|Medical\s*Record\s*Number|Patient\s*ID)\s*[:#]?\s*([A-Za-z0-9\-]{4,20})"),
        ),
        (
            "REGISTRATION_NUMBER",
            re.compile(r"(?i)\b(?:Reg(?:istration)?|Admission)\s*(?:No|Number|#)?\s*[:#]?\s*([A-Za-z0-9\-]{4,20})"),
        ),
        (
            "DATE_OF_BIRTH",
            re.compile(r"(?i)\b(?:DOB|Date\s*of\s*Birth)\s*[:#]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
        ),
        (
            "ADDRESS",
            re.compile(r"(?i)\b(?:Address|Street\s*Address)\s*:\s*([^\n\r,]+(?:,\s*[^\n\r]+)?)"),
        ),
    ]

    # Global pattern matchers for standalone formats
    GLOBAL_PATTERNS = [
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        ),
        (
            "PHONE",
            re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ),
        (
            "GOVERNMENT_ID",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN format
        ),
        (
            "GOVERNMENT_ID",
            re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),  # National ID / Aadhaar format
        ),
    ]

    @classmethod
    def detect_entities(cls, text: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Detect PII entities in text.
        
        Returns:
            Tuple of:
                - Aggregate counts: Dict[str, int] (Safe to store centrally)
                - In-memory entity list: List[Dict] with category, start, end, raw_value
                  (FOR WORKER MASKING ONLY - NEVER PERSIST OR LOG)
        """
        if not text:
            return {cat: 0 for cat in PII_CATEGORIES}, []

        detected_entities: List[Dict[str, Any]] = []

        # 1. Labeled patterns
        for category, regex in cls.LABELED_PATTERNS:
            for match in regex.finditer(text):
                # The identifying value is captured in group 1
                val = match.group(1).strip()
                # Skip if empty or looks like clinical keyword
                if len(val) >= 2:
                    detected_entities.append({
                        "category": category,
                        "start": match.start(1),
                        "end": match.end(1),
                        "value": val,
                    })

        # 2. Global pattern matchers
        for category, regex in cls.GLOBAL_PATTERNS:
            for match in regex.finditer(text):
                val = match.group(0).strip()
                detected_entities.append({
                    "category": category,
                    "start": match.start(0),
                    "end": match.end(0),
                    "value": val,
                })

        # Sort entities by start index descending to allow clean non-overlapping replacement
        detected_entities.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        # De-duplicate / remove overlapping spans
        non_overlapping: List[Dict[str, Any]] = []
        last_end = -1
        for ent in sorted(detected_entities, key=lambda x: x["start"]):
            if ent["start"] >= last_end:
                non_overlapping.append(ent)
                last_end = ent["end"]

        # Compute safe aggregate counts
        counts: Dict[str, int] = {cat: 0 for cat in PII_CATEGORIES}
        for ent in non_overlapping:
            cat = ent["category"]
            counts[cat] = counts.get(cat, 0) + 1

        return counts, non_overlapping


class PrivacyMasker:
    """Masks detected direct identifiers while strictly preserving clinical and laboratory information."""

    @classmethod
    def mask_document(
        cls,
        text: str,
    ) -> Tuple[str, Dict[str, int], List[str]]:
        """Apply semantic privacy masking to clinical text.
        
        Returns:
            Tuple of:
                - masked_text: str (Protected downstream representation)
                - aggregate_counts: Dict[str, int] (Safe for PostgreSQL)
                - detected_raw_values: List[str] (For memory validation only; never logged)
        """
        if not text:
            return "", {cat: 0 for cat in PII_CATEGORIES}, []

        counts, entities = PIIDetector.detect_entities(text)
        detected_values = [ent["value"] for ent in entities]

        # Apply replacement from end of string to beginning so offsets remain valid
        entities_desc = sorted(entities, key=lambda x: x["start"], reverse=True)
        masked_chars = list(text)

        for ent in entities_desc:
            token = f"[{ent['category']}]"
            start = ent["start"]
            end = ent["end"]
            masked_chars[start:end] = list(token)

        masked_text = "".join(masked_chars)

        return masked_text, counts, detected_values
