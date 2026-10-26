"""Static checks for synthetic demo seed script (no live DB required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SEED_PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_synthetic.py"
_spec = importlib.util.spec_from_file_location("seed_demo_synthetic", _SEED_PATH)
assert _spec and _spec.loader
seed_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_mod)


def test_demo_hospitals_are_synthetic_and_stable():
    codes = {h["code"] for h in seed_mod.DEMO_HOSPITALS}
    assert codes == {"DEMO-METRO", "DEMO-RIVERSIDE"}
    for h in seed_mod.DEMO_HOSPITALS:
        assert "[SYNTHETIC]" in h["name"]


def test_demo_patient_count_and_distribution():
    assert len(seed_mod.DEMO_PATIENTS) == 15
    metro = sum(1 for p in seed_mod.DEMO_PATIENTS if p["hospital_code"] == "DEMO-METRO")
    riverside = sum(1 for p in seed_mod.DEMO_PATIENTS if p["hospital_code"] == "DEMO-RIVERSIDE")
    assert metro == 8
    assert riverside == 7


def test_demo_patient_opaque_ids_unique():
    ids = [p["opaque_id"] for p in seed_mod.DEMO_PATIENTS]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("DEMO-SYNTH-") for i in ids)
