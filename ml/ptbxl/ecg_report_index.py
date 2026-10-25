"""PTB-XL within-cohort ECG + report index (two-modality experiment)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ml.preprocessing.clinical_text_privacy import mask_clinical_text_for_encoding
from ml.ptbxl.labels import PROTOCOL_VERSION, build_labeled_index
from ml.text.errors import MaskingFailedError


def load_report_by_ecg_id(ptbxl_root: Path, database_csv: str = "ptbxl_database.csv") -> dict[int, str]:
    reports: dict[int, str] = {}
    with open(ptbxl_root / database_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ecg_id = int(row["ecg_id"])
            text = (row.get("report") or "").strip()
            if text:
                reports[ecg_id] = text
    return reports


def build_ecg_report_index(ptbxl_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Included binary-label records with non-empty PTB-XL `report` field (same ecg_id).
    """
    records, label_stats = build_labeled_index(ptbxl_root)
    reports = load_report_by_ecg_id(ptbxl_root)
    paired: list[dict[str, Any]] = []
    missing_report = 0
    masking_failed = 0
    for rec in records:
        text = reports.get(rec["ecg_id"])
        if not text:
            missing_report += 1
            continue
        try:
            mask_clinical_text_for_encoding(text)
        except MaskingFailedError:
            masking_failed += 1
            continue
        paired.append({**rec, "report": text})

    stats = {
        **label_stats,
        "experiment": "two_modality_ecg_report_ptbxl",
        "protocol_version": PROTOCOL_VERSION,
        "included_with_report": len(paired),
        "included_missing_report": missing_report,
        "excluded_masking_failed": masking_failed,
        "linkage_key": "ecg_id",
        "target_definition": (
            "PTB-XL proxy: NORM (class 0) vs MI or STTC superclasses (class 1). "
            "NOT validated clinical heart-disease diagnosis."
        ),
        "modalities": ["ecg_waveform_12x1000", "ptbxl_machine_report_text"],
        "not_three_modality_gat": True,
    }
    return paired, stats
