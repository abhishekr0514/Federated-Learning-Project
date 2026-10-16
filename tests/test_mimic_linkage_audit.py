"""Prompt 16B — MIMIC linkage auditor (synthetic metadata only; no real PHI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.datasets.mimic_linkage_audit import (
    MimicAuditError,
    audit_synthetic_metadata,
    locate_documented_installs,
    referential_integrity,
    reject_cross_cohort_join,
    run_local_install_audit,
)

REPO = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO / "docs" / "data-audit" / "mimic_linkage_audit.json"


def _synth_tables(*, dangling_note: bool = False, out_of_window: bool = False):
    records = [
        {"subject_id": "900001", "study_id": "910001", "path": "files/p9000/p900001/s910001/910001"},
        {"subject_id": "900002", "study_id": "910002", "path": "files/p9000/p900002/s910002/910002"},
    ]
    patients = [
        {"subject_id": "900001"},
        {"subject_id": "900002"},
        {"subject_id": "900099"},
    ]
    admissions = [
        {
            "subject_id": "900001",
            "hadm_id": "920001",
            "admittime": "2113-08-25 07:15:00",
            "dischtime": "2113-08-30 14:15:00",
        }
    ]
    measurements = [
        {"subject_id": "900001", "study_id": "910001", "ecg_time": "2113-08-25 13:58:00"},
        {
            "subject_id": "900002",
            "study_id": "910002",
            "ecg_time": "2110-01-01 00:00:00" if out_of_window else "2113-08-26 10:00:00",
        },
    ]
    notes = [
        {"note_id": "900001-DS-1", "subject_id": "900001", "hadm_id": "920001", "charttime": "2113-08-30 12:00:00"}
    ]
    links = [
        {"subject_id": "900001", "study_id": "910001", "note_id": "900001-DS-1", "note_seq": "1"},
        {
            "subject_id": "900002",
            "study_id": "910002",
            "note_id": "900002-EK-1" if dangling_note else "900001-DS-1",
            "note_seq": "1",
        },
    ]
    return {
        "record_list": records,
        "patients": patients,
        "admissions": admissions,
        "machine_measurements": measurements,
        "discharge": notes,
        "waveform_note_links": links,
    }


def test_documented_paths_do_not_guess_unauthorized_roots():
    presence = locate_documented_installs(REPO)
    for ds in ("mimic_iv_ecg", "mimic_iv", "mimic_iv_note"):
        assert presence[ds].present is False
        assert presence[ds].found_root is None
        assert all("ptb-xl" not in r for r in presence[ds].documented_roots)


def test_local_install_audit_blocks_when_mimic_absent():
    result = run_local_install_audit(REPO)
    assert set(result["missing_datasets"]) == {"mimic_iv_ecg", "mimic_iv", "mimic_iv_note"}
    assert result["all_required_modules_present"] is False
    assert result["merged_dataset_materialized"] is False
    assert result["training_conducted"] is False
    assert result["verdict_three_modality_gat_training"] == "BLOCKED"
    matrix = result["decision_matrix"]
    for key in (
        "ecg_to_patient_linkage",
        "ecg_to_note_linkage",
        "ecg_to_tabular_linkage",
        "shared_target_validity",
        "temporal_alignment",
        "leakage_safe_patient_splits",
        "two_modality_research_readiness",
        "three_modality_gat_training_readiness",
    ):
        assert matrix[key]["verdict"] == "BLOCKED"


def test_synthetic_patient_referential_integrity():
    out = audit_synthetic_metadata(_synth_tables())
    ri = out["ecg_to_patient"]
    assert ri["left_unique"] == 2
    assert ri["intersection"] == 2
    assert ri["left_unresolved"] == 0
    assert out["merged_dataset_materialized"] is False


def test_synthetic_unresolved_note_links():
    out = audit_synthetic_metadata(_synth_tables(dangling_note=True))
    assert out["ecg_to_note"]["left_unresolved"] == 1
    assert out["ecg_to_note"]["intersection"] == 1


def test_synthetic_temporal_admission_window():
    in_window = audit_synthetic_metadata(_synth_tables(out_of_window=False))
    assert in_window["temporal_admission_overlap"]["overlapping_admission_window"] == 1
    out_window = audit_synthetic_metadata(_synth_tables(out_of_window=True))
    assert out_window["temporal_admission_overlap"]["overlapping_admission_window"] == 1
    assert out_window["temporal_admission_overlap"]["subject_without_admission_window"] == 1


def test_reject_cross_cohort_join_names():
    with pytest.raises(MimicAuditError, match="Cross-cohort"):
        reject_cross_cohort_join(["mimic_iv_ecg", "cardio_train.csv"])
    with pytest.raises(MimicAuditError, match="Cross-cohort"):
        reject_cross_cohort_join(["ptb-xl", "mimic_iv"])


def test_referential_integrity_aggregates_only():
    result = referential_integrity(
        {"a", "b", "c"},
        {"b", "c", "d"},
        name="demo",
        left="L",
        right="R",
    )
    assert result.intersection == 2
    assert result.left_unresolved == 1
    dumped = json.dumps(result.__dict__)
    assert "900001" not in dumped


def test_audit_json_artifact_matches_blocked_training():
    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert data["verdict_three_modality_gat_training"] == "BLOCKED"
    assert data["merged_dataset_materialized"] is False
    assert data["training_conducted"] is False
    assert set(data["missing_datasets"]) == {"mimic_iv_ecg", "mimic_iv", "mimic_iv_note"}


def test_metadata_audit_runs_without_ml_stack():
    import sys

    # Verify metadata-only modules do not import torch
    assert "torch" not in sys.modules, "torch was unexpectedly imported prior to metadata audit"
    import ml.datasets.mimic_linkage_audit
    import ml.datasets.nbme_text

    assert "torch" not in sys.modules, "importing metadata modules caused eager torch import"
    result = ml.datasets.mimic_linkage_audit.run_local_install_audit(REPO)
    assert result["verdict_three_modality_gat_training"] == "BLOCKED"
    assert "torch" not in sys.modules, "running install audit caused eager torch import"

