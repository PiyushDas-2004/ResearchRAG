"""Programmatic Alembic entry points with a single-run guard for concurrent starts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Arbitrary constant key for pg_advisory_lock so only one process migrates at a time.
MIGRATION_LOCK_KEY = 72_731_001


def alembic_config(database_url: str) -> Config:
    """Build an Alembic config pointing at the packaged migrations."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def head_revision() -> str:
    """Return the head revision id of the packaged migrations."""
    script = ScriptDirectory.from_config(alembic_config("postgresql+asyncpg://unused/unused"))
    head = script.get_current_head()
    if head is None:  # pragma: no cover - migrations always exist
        raise RuntimeError("no migrations found")
    return head


async def current_revision(database_url: str) -> str | None:
    """Return the revision the database is at (``None`` if never migrated)."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(lambda c: MigrationContext.configure(c).get_current_revision())
    finally:
        await engine.dispose()


async def migrate(database_url: str) -> str:
    """Upgrade to head under a Postgres advisory lock; safe when several instances start at once."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": MIGRATION_LOCK_KEY})
            try:
                await asyncio.to_thread(command.upgrade, alembic_config(database_url), "head")
            finally:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": MIGRATION_LOCK_KEY})
    finally:
        await engine.dispose()
    return head_revision()


async def reset(database_url: str) -> None:
    """Drop and recreate the public schema, then migrate (development only)."""
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()
    await migrate(database_url)
