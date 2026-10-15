"""
Read-only MIMIC-IV / MIMIC-IV-ECG / MIMIC-IV-Note install and linkage auditor.

Uses only documented local paths. Does not download data, join PTB-XL/NBME/cardio,
materialize merged patient tables, or print identifiers / notes / waveforms.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Documented install roots (MULTIMODAL_DATASET_ACQUISITION_PLAN.md / Prompt 16).
DOCUMENTED_ECG_ROOTS = (
    Path("data/raw/mimic-iv-ecg"),
    Path("data/raw/mimic-iv-ecg/1.0"),
)
DOCUMENTED_CLINICAL_ROOTS = (
    Path("data/raw/mimic-iv"),
    Path("data/raw/mimiciv"),
    Path("data/raw/mimic-iv/2.2"),
)
DOCUMENTED_NOTE_ROOTS = (
    Path("data/raw/mimic-iv-note"),
    Path("data/raw/mimic-iv-note/2.2"),
)

FORBIDDEN_CROSSWALK_DATASETS = ("ptb-xl", "cardio_train.csv", "patient_notes.csv", "nbme")

ECG_METADATA_CANDIDATES = (
    "record_list.csv",
    "machine_measurements.csv",
    "waveform_note_links.csv",
)
CLINICAL_METADATA_CANDIDATES = (
    "hosp/patients.csv",
    "hosp/patients.csv.gz",
    "hosp/admissions.csv",
    "hosp/admissions.csv.gz",
    "hosp/omr.csv",
    "hosp/omr.csv.gz",
    "hosp/labevents.csv",
    "hosp/labevents.csv.gz",
    "hosp/diagnoses_icd.csv",
    "hosp/diagnoses_icd.csv.gz",
)
NOTE_METADATA_CANDIDATES = (
    "note/discharge.csv",
    "note/discharge.csv.gz",
    "discharge.csv",
    "discharge.csv.gz",
    "note/radiology.csv",
    "note/radiology.csv.gz",
    "radiology.csv",
    "radiology.csv.gz",
)

OFFICIAL_DOCS = {
    "mimic_iv_ecg": {
        "name": "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset",
        "version": "1.0",
        "url": "https://physionet.org/content/mimic-iv-ecg/1.0/",
        "doi": "10.13026/4nqg-sb35",
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "dua": "PhysioNet Credentialed Health Data Use Agreement 1.5.0",
        "training": "CITI Data or Specimens Only Research",
    },
    "mimic_iv": {
        "name": "MIMIC-IV",
        "version": "2.2",
        "url": "https://physionet.org/content/mimiciv/2.2/",
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "dua": "PhysioNet Credentialed Health Data Use Agreement 1.5.0",
        "training": "CITI Data or Specimens Only Research",
    },
    "mimic_iv_note": {
        "name": "MIMIC-IV-Note: Deidentified free-text clinical notes",
        "version": "2.2",
        "url": "https://physionet.org/content/mimic-iv-note/2.2/",
        "doi": "10.13026/1n74-ne17",
        "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "dua": "PhysioNet Credentialed Health Data Use Agreement 1.5.0",
        "training": "CITI Data or Specimens Only Research",
    },
}

# Public documentation of key semantics (PhysioNet pages fetched 2026-10-15).
# Local schema is UNVERIFIED until files exist.
DOCUMENTATION_KEY_SEMANTICS = {
    "subject_id": (
        "Deidentified patient identifier shared across MIMIC-IV modules. "
        "MIMIC-IV-ECG record_list.csv maps each WFDB record to subject_id and study_id "
        "(https://physionet.org/content/mimic-iv-ecg/1.0/)."
    ),
    "study_id": (
        "Deidentified ECG study identifier unique per diagnostic ECG record; "
        "directory/file names follow s{study_id}."
    ),
    "hadm_id": (
        "Hospital admission identifier in MIMIC-IV hosp.admissions. "
        "Not a native column of MIMIC-IV-ECG waveform tables; encounter overlap is temporal."
    ),
    "note_id": (
        "Unique note identifier in MIMIC-IV-Note. waveform_note_links.csv provides note_id "
        "for cardiologist ECG reports (domain abbreviation EK). MIMIC-IV-Note v2.2 currently "
        "documents discharge and radiology tables; ECG EK notes are described as forthcoming."
    ),
    "ecg_time": (
        "Shifted ECG timestamp in machine_measurements.csv. Relative within-subject times "
        "are preserved; machine clocks may be unsynchronized with EHR timestamps."
    ),
}


class MimicAuditError(ValueError):
    """Invalid audit configuration or forbidden cross-cohort join attempt."""


@dataclass
class DatasetPresence:
    dataset_id: str
    documented_roots: list[str]
    found_root: str | None
    present: bool
    metadata_files_found: list[str] = field(default_factory=list)
    missing_expected_files: list[str] = field(default_factory=list)


@dataclass
class ReferentialIntegrityResult:
    name: str
    left: str
    right: str
    left_unique: int
    right_unique: int
    intersection: int
    left_unresolved: int
    notes: str


def _first_existing(roots: Sequence[Path], repo: Path) -> Path | None:
    for rel in roots:
        candidate = (repo / rel) if not rel.is_absolute() else rel
        if candidate.is_dir():
            return candidate
    return None


def _find_rel(root: Path, candidates: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for rel in candidates:
        path = root / rel
        if path.is_file():
            found.append(path)
    return found


def locate_documented_installs(repo: Path) -> dict[str, DatasetPresence]:
    """Locate MIMIC modules using documented paths only. Does not search the filesystem broadly."""
    mapping = {
        "mimic_iv_ecg": (DOCUMENTED_ECG_ROOTS, ECG_METADATA_CANDIDATES),
        "mimic_iv": (DOCUMENTED_CLINICAL_ROOTS, CLINICAL_METADATA_CANDIDATES),
        "mimic_iv_note": (DOCUMENTED_NOTE_ROOTS, NOTE_METADATA_CANDIDATES),
    }
    out: dict[str, DatasetPresence] = {}
    for ds_id, (roots, expected) in mapping.items():
        found = _first_existing(roots, repo)
        files = _find_rel(found, expected) if found is not None else []
        missing = []
        if found is not None:
            missing = [rel for rel in expected if not (found / rel).is_file()]
        out[ds_id] = DatasetPresence(
            dataset_id=ds_id,
            documented_roots=[str(r) for r in roots],
            found_root=str(found) if found else None,
            present=found is not None,
            metadata_files_found=[str(p.relative_to(found)) for p in files] if found else [],
            missing_expected_files=missing,
        )
    return out


def read_column_set(path: Path, column: str, *, max_rows: int | None = None) -> set[str]:
    """Read a single identifier column as strings. Callers must not log the values."""
    values: set[str] = set()
    opener = path.open
    with opener(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise MimicAuditError(f"Column {column!r} missing from metadata schema (filename only).")
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            raw = (row.get(column) or "").strip()
            if raw:
                values.add(raw)
    return values


def referential_integrity(
    left_values: set[str],
    right_values: set[str],
    *,
    name: str,
    left: str,
    right: str,
    notes: str = "",
) -> ReferentialIntegrityResult:
    inter = left_values & right_values
    return ReferentialIntegrityResult(
        name=name,
        left=left,
        right=right,
        left_unique=len(left_values),
        right_unique=len(right_values),
        intersection=len(inter),
        left_unresolved=len(left_values - right_values),
        notes=notes,
    )


def parse_naive_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def count_ecg_within_admission_windows(
    ecg_rows: Iterable[Mapping[str, str]],
    admission_rows: Iterable[Mapping[str, str]],
) -> dict[str, int]:
    """
    Encounter-level temporal overlap: same subject_id and ecg_time in [admittime, dischtime].

    Aggregate counts only. Does not emit identifiers.
    """
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in admission_rows:
        sid = (row.get("subject_id") or "").strip()
        start = parse_naive_datetime(row.get("admittime") or "")
        end = parse_naive_datetime(row.get("dischtime") or "")
        if not sid or start is None or end is None:
            continue
        windows.setdefault(sid, []).append((start, end))

    total = 0
    overlapping = 0
    missing_subject = 0
    unparsable_time = 0
    for row in ecg_rows:
        total += 1
        sid = (row.get("subject_id") or "").strip()
        ts = parse_naive_datetime(row.get("ecg_time") or "")
        if ts is None:
            unparsable_time += 1
            continue
        if sid not in windows:
            missing_subject += 1
            continue
        if any(start <= ts <= end for start, end in windows[sid]):
            overlapping += 1
    return {
        "ecg_rows": total,
        "overlapping_admission_window": overlapping,
        "subject_without_admission_window": missing_subject,
        "unparsable_ecg_time": unparsable_time,
    }


def reject_cross_cohort_join(source_names: Sequence[str]) -> None:
    lowered = " ".join(source_names).lower()
    for forbidden in FORBIDDEN_CROSSWALK_DATASETS:
        if forbidden in lowered:
            raise MimicAuditError(
                "Cross-cohort joins to PTB-XL, NBME, or cardio_train.csv are forbidden "
                "(ADR-004 / ADR-010). Numeric ID overlap is not linkage evidence."
            )


def audit_synthetic_metadata(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    """Referential and temporal checks on in-memory synthetic tables (unit tests only)."""
    reject_cross_cohort_join(tuple(tables.keys()))
    records = list(tables.get("record_list", []))
    patients = list(tables.get("patients", []))
    admissions = list(tables.get("admissions", []))
    notes = list(tables.get("discharge", []))
    links = list(tables.get("waveform_note_links", []))
    measurements = list(tables.get("machine_measurements", records))

    rec_subjects = {r["subject_id"] for r in records if r.get("subject_id")}
    pat_subjects = {r["subject_id"] for r in patients if r.get("subject_id")}
    note_ids = {r["note_id"] for r in notes if r.get("note_id")}
    link_note_ids = {r["note_id"] for r in links if r.get("note_id")}

    patient_ri = referential_integrity(
        rec_subjects,
        pat_subjects,
        name="ecg_to_patient",
        left="record_list.subject_id",
        right="patients.subject_id",
        notes="Patient-level key per MIMIC-IV-ECG record_list documentation.",
    )
    note_ri = referential_integrity(
        link_note_ids,
        note_ids,
        name="ecg_to_note",
        left="waveform_note_links.note_id",
        right="discharge.note_id",
        notes="Cardiologist EK notes may be absent from MIMIC-IV-Note v2.2 discharge/radiology tables.",
    )
    temporal = count_ecg_within_admission_windows(measurements, admissions)
    return {
        "ecg_to_patient": asdict(patient_ri),
        "ecg_to_note": asdict(note_ri),
        "temporal_admission_overlap": temporal,
        "merged_dataset_materialized": False,
    }


def run_local_install_audit(repo: Path) -> dict[str, Any]:
    """Stop if documented installs are missing. Never search unauthorized locations."""
    presence = locate_documented_installs(repo)
    missing = [k for k, v in presence.items() if not v.present]
    all_present = not missing
    decision = _blocked_matrix(
        reason="Required MIMIC modules absent at documented paths."
        if missing
        else "Install present but shared-target protocol not yet approved."
    )
    if missing:
        for key in decision:
            if key.endswith("_readiness") or "linkage" in key or "target" in key or "temporal" in key:
                pass
    return {
        "audit_id": "medshield.mimic_linkage_audit",
        "audit_version": "1.0.0",
        "training_conducted": False,
        "merged_dataset_materialized": False,
        "production_inference_changed": False,
        "cross_cohort_joins_attempted": False,
        "official_documentation": OFFICIAL_DOCS,
        "documentation_key_semantics": DOCUMENTATION_KEY_SEMANTICS,
        "documented_paths_only": True,
        "presence": {k: asdict(v) for k, v in presence.items()},
        "missing_datasets": missing,
        "local_referential_integrity": None,
        "local_schema_verified": False,
        "decision_matrix": decision if missing else _blocked_matrix(
            reason="Install detected; remaining clinical-target and temporal protocol gates unmet."
        ),
        "all_required_modules_present": all_present,
        "verdict_three_modality_gat_training": "BLOCKED",
    }


def _blocked_matrix(reason: str) -> dict[str, dict[str, str]]:
    return {
        "ecg_to_patient_linkage": {"verdict": "BLOCKED", "evidence": reason},
        "ecg_to_note_linkage": {"verdict": "BLOCKED", "evidence": reason},
        "ecg_to_tabular_linkage": {"verdict": "BLOCKED", "evidence": reason},
        "shared_target_validity": {"verdict": "BLOCKED", "evidence": reason},
        "temporal_alignment": {"verdict": "BLOCKED", "evidence": reason},
        "leakage_safe_patient_splits": {"verdict": "BLOCKED", "evidence": reason},
        "two_modality_research_readiness": {"verdict": "BLOCKED", "evidence": reason},
        "three_modality_gat_training_readiness": {"verdict": "BLOCKED", "evidence": reason},
    }
