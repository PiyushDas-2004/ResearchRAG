"""Database fixtures shared by integration, API and end-to-end tests.

Uses ``RR_TEST_DATABASE_URL`` when set (CI service container, local compose); otherwise tries
testcontainers with the ``pgvector/pgvector:pg16`` image; otherwise skips database tests.
The schema is reset and migrated once per test session.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from researchrag.config.settings import Settings
from researchrag.storage.db import create_engine, session_factory
from researchrag.storage.migrate import reset


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """URL of a disposable test database."""
    url = os.environ.get("RR_TEST_DATABASE_URL")
    if url:
        yield url
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover
        pytest.skip("no RR_TEST_DATABASE_URL and testcontainers not installed")
    try:
        container = PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg")
        container.start()
    except Exception as exc:
        pytest.skip(f"no RR_TEST_DATABASE_URL and docker unavailable: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
async def migrated(database_url: str) -> str:
    """Reset the schema and apply migrations once per session."""
    await reset(database_url)
    return database_url


@pytest.fixture(scope="session")
async def engine(migrated: str) -> AsyncIterator[AsyncEngine]:
    """Session-wide async engine."""
    eng = create_engine(Settings(database_url=migrated, db_pool_size=5))
    yield eng
    await eng.dispose()


@pytest.fixture(scope="session")
def factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory."""
    return session_factory(engine)
