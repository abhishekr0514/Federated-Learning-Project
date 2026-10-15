"""NBME dataset schema adapter (concept extraction — not cardiovascular diagnosis)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REQUIRED_NOTE_COLUMNS = ("pn_num", "case_num", "pn_history")
REQUIRED_TRAIN_COLUMNS = ("id", "case_num", "pn_num", "feature_num", "annotation", "location")
REQUIRED_FEATURE_COLUMNS = ("feature_num", "case_num", "feature_text")


def validate_nbme_schema(datasets_dir: Path) -> dict[str, Any]:
    """Verify expected NBME files and columns without loading note bodies into reports."""
    paths = {
        "patient_notes.csv": datasets_dir / "patient_notes.csv",
        "train.csv": datasets_dir / "train.csv",
        "features.csv": datasets_dir / "features.csv",
    }
    result: dict[str, Any] = {"present": {}, "columns_ok": {}, "task": "NBME clinical phrase span identification"}
    for name, path in paths.items():
        if not path.exists():
            result["present"][name] = False
            continue
        result["present"][name] = True
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = tuple(reader.fieldnames or ())
        if name == "patient_notes.csv":
            result["columns_ok"][name] = cols == REQUIRED_NOTE_COLUMNS
        elif name == "train.csv":
            result["columns_ok"][name] = cols == REQUIRED_TRAIN_COLUMNS
        else:
            result["columns_ok"][name] = cols == REQUIRED_FEATURE_COLUMNS
    result["linkage_to_ptbxl_or_cardio"] = False
    result["note"] = "Do not join NBME to PTB-XL or cardio_train.csv without verified crosswalk."
    return result
