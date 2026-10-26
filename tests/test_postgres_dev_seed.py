"""Safety checks for PostgreSQL dev seed script (no live DB required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from backend.app.core.config import settings
from backend.app.db.connectivity import assert_postgres_configuration

_SEED_PATH = Path(__file__).resolve().parents[1] / "scripts" / "seed_postgres_dev.py"
_spec = importlib.util.spec_from_file_location("seed_postgres_dev", _SEED_PATH)
assert _spec and _spec.loader
seed_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_mod)


def test_seed_refuses_sqlite_database_url(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite+aiosqlite:///medshield_dev.db", raising=False)
    monkeypatch.setattr(settings, "POSTGRES_DB", "medshield_fl_db", raising=False)
    with pytest.raises(RuntimeError, match="SQLite"):
        assert_postgres_configuration()


def test_seed_refuses_wrong_database_name(monkeypatch):
    monkeypatch.setattr(
        settings,
        "DATABASE_URL",
        "postgresql+asyncpg://u:p@localhost:5432/other_db",
        raising=False,
    )
    monkeypatch.setattr(settings, "POSTGRES_DB", "other_db", raising=False)
    with pytest.raises(RuntimeError, match="medshield_fl_db"):
        assert_postgres_configuration()


def test_accounts_match_requirements():
    emails = {a["email"] for a in seed_mod.ACCOUNTS}
    roles = {a["role"] for a in seed_mod.ACCOUNTS}
    assert "dr_metro@metro.org" in emails
    assert "patient_alice@example.com" in emails
    assert roles == {"CLINICIAN", "PATIENT", "HOSPITAL_ADMIN", "SUPER_ADMIN"}
    super_admin = next(a for a in seed_mod.ACCOUNTS if a["email"] == "admin@medshield.org")
    assert super_admin.get("is_super_admin") is True
    assert super_admin["role"] == "SUPER_ADMIN"


def test_dev_hospital_constants_for_clinician_workspace():
    assert seed_mod.DEV_HOSPITAL_CODE == "HOSP-METRO"
    assert seed_mod._ensure_clinician_workspace is not None
    assert seed_mod._ensure_patient_portal_link is not None
    assert seed_mod.PATIENT_DEV_EMAIL == "patient_alice@example.com"
    assert len(seed_mod.DEV_PATIENT_OPAQUE_ID) >= 32
