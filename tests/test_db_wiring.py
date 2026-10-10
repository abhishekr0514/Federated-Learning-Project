"""Test suite verifying database infrastructure dependency wiring."""

import pytest
from backend.app.db.base import Base
from backend.app.db.session import async_engine, sync_engine
from backend.app.repositories.base import BaseRepository


def test_db_engine_configuration():
    """Verify async and sync database engines are correctly configured."""
    assert async_engine is not None
    assert sync_engine is not None
    assert Base is not None


@pytest.mark.asyncio
async def test_base_repository_instantiation():
    """Verify BaseRepository can be instantiated with generic model and session."""

    class DummyModel(Base):
        __tablename__ = "test_dummy_model"
        from sqlalchemy import Column, Integer
        id = Column(Integer, primary_key=True)

    # Repository pattern instantiation verification
    repo_class = BaseRepository(DummyModel, session=None)
    assert repo_class.model == DummyModel
