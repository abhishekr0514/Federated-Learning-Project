"""Regression tests for repository-root .env loading and DATABASE_URL assembly."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from backend.app.core.config import Settings, _ENV_FILE, _REPO_ROOT


def test_env_file_points_at_repository_root():
    assert _ENV_FILE == _REPO_ROOT / ".env"
    assert _ENV_FILE.name == ".env"


def test_database_url_built_from_postgres_components():
    settings = Settings(
        POSTGRES_USER="medshield_admin",
        POSTGRES_PASSWORD="p@ss:w/rd#special",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_DB="medshield_fl_db",
    )
    url = make_url(settings.DATABASE_URL)
    assert url.drivername == "postgresql+asyncpg"
    assert url.username == "medshield_admin"
    assert url.password == "p@ss:w/rd#special"
    assert url.host == "localhost"
    assert url.port == 5432
    assert url.database == "medshield_fl_db"


def test_stale_database_url_env_does_not_override_postgres_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://wrong:wrong@localhost:5432/medshield_fl_db")
    monkeypatch.setenv("POSTGRES_PASSWORD", "correct-secret-value")
    monkeypatch.setenv("POSTGRES_USER", "medshield_admin")
    monkeypatch.setenv("POSTGRES_DB", "medshield_fl_db")
    settings = Settings()
    url = make_url(settings.DATABASE_URL)
    assert url.password == "correct-secret-value"
    assert url.username == "medshield_admin"


def test_settings_load_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        POSTGRES_HOST="h",
        POSTGRES_PORT=5432,
        POSTGRES_DB="medshield_fl_db",
    )
    assert settings.POSTGRES_DB == "medshield_fl_db"
    assert "postgresql+asyncpg://" in settings.DATABASE_URL
