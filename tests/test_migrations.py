"""Test suite verifying Alembic migration script integrity."""

import importlib.util
from pathlib import Path
from sqlalchemy import create_engine

MIGRATION_PATH = Path(__file__).parent.parent / "backend" / "app" / "db" / "migrations" / "versions" / "001_initial_domain_schema.py"


def load_migration_module():
    """Dynamically load 001_initial_domain_schema migration module."""
    spec = importlib.util.spec_from_file_location("initial_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_module_metadata():
    """Verify initial migration script defines revision ID and upgrade/downgrade callables."""
    migration = load_migration_module()
    assert migration.revision == "001_initial_domain_schema"
    assert migration.down_revision is None
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
