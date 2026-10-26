"""Print non-secret PostgreSQL dev DB counts (users, roles, public tables)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.app.db.session import AsyncSessionLocal  # noqa: E402


async def main() -> None:
    async with AsyncSessionLocal() as session:
        user_count = (await session.execute(text("SELECT count(*) FROM users"))).scalar_one()
        roles = [
            row[0]
            for row in (await session.execute(text("SELECT name FROM roles ORDER BY name"))).fetchall()
        ]
        tables = [
            row[0]
            for row in (
                await session.execute(
                    text(
                        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
                    )
                )
            ).fetchall()
        ]
        required = {"users", "roles", "hospitals", "alerts", "data_quality_events"}
        missing = sorted(required - set(tables))
        print(f"user_count={user_count}")
        print(f"roles={roles}")
        print(f"public_table_count={len(tables)}")
        print(f"required_tables_missing={missing or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
