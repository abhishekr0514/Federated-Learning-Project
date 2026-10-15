"""Prompt 16 — multimodal linkage audit artifacts (no dataset joins)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO / "docs" / "data-audit" / "multimodal_linkage_audit.json"
INVENTORY = REPO / "docs" / "data-audit" / "dataset_inventory.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_linkage_audit_json_verdicts_block_cross_modal_training():
    data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    matrix = data["decision_matrix"]
    assert matrix["three_modality_gat_training_readiness"]["verdict"] == "BLOCKED"
    assert matrix["ecg_text_linkage_cross_local_cohorts"]["verdict"] == "BLOCKED"
    assert matrix["ecg_tabular_linkage_cross_local_cohorts"]["verdict"] == "BLOCKED"
    assert matrix["text_tabular_linkage_cross_local_cohorts"]["verdict"] == "BLOCKED"
    assert matrix["shared_target_validity_multimodal_local"]["verdict"] == "BLOCKED"
    assert data["datasets_joined"] is False
    assert data["training_conducted"] is False


def test_dataset_inventory_checksums_unchanged():
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for rel, meta in inv["files"].items():
        path = REPO / rel.replace("/", "\\") if "\\" not in rel else REPO / rel
        path = REPO / Path(rel)
        assert path.is_file(), rel
        assert _sha256(path) == meta["sha256"]
