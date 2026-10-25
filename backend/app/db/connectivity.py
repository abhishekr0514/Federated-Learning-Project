"""Safe PostgreSQL connectivity checks (no credential logging)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from backend.app.core.config import settings


@dataclass(frozen=True)
class DatabaseConnectivityResult:
    ok: bool
    driver: str
    host: str
    port: int
    database: str
    username: str
    error_type: str | None = None
    error_message: str | None = None


def database_target_summary() -> dict[str, str | int]:
    """Non-secret connection target for diagnostics."""
    return {
        "driver": "postgresql+asyncpg",
        "host": settings.POSTGRES_HOST,
        "port": settings.POSTGRES_PORT,
        "database": settings.POSTGRES_DB,
        "username": settings.POSTGRES_USER,
    }


def assert_postgres_configuration(*, expected_database: str = "medshield_fl_db") -> None:
    url = settings.DATABASE_URL.lower()
    if "sqlite" in url:
        raise RuntimeError("Refusing operation: configured database is SQLite, not PostgreSQL.")
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("Refusing operation: DATABASE_URL must use postgresql+asyncpg.")
    if settings.POSTGRES_DB != expected_database:
        raise RuntimeError(
            f"Refusing operation: expected PostgreSQL database {expected_database!r}, "
            f"got {settings.POSTGRES_DB!r}."
        )


async def verify_async_connection(engine: AsyncEngine | None = None) -> DatabaseConnectivityResult:
    """Run SELECT 1; never log passwords or full DATABASE_URL."""
    summary = database_target_summary()
    owned = engine is None
    if owned:
        engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return DatabaseConnectivityResult(
            ok=True,
            driver=str(summary["driver"]),
            host=str(summary["host"]),
            port=int(summary["port"]),
            database=str(summary["database"]),
            username=str(summary["username"]),
        )
    except Exception as exc:
        return DatabaseConnectivityResult(
            ok=False,
            driver=str(summary["driver"]),
            host=str(summary["host"]),
            port=int(summary["port"]),
            database=str(summary["database"]),
            username=str(summary["username"]),
            error_type=type(exc).__name__,
            error_message=str(exc).split("\n")[0][:200],
        )
    finally:
        if owned and engine is not None:
            await engine.dispose()
