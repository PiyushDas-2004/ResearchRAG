"""Async engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from researchrag.config.settings import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Create the async SQLAlchemy engine (asyncpg driver)."""
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory that does not expire objects on commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def transaction(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Open a session with a transaction that commits on success and rolls back on error."""
    async with factory() as session, session.begin():
        yield session


async def pgvector_version(session: AsyncSession) -> tuple[int, ...]:
    """Return the installed pgvector extension version, e.g. ``(0, 8, 0)``."""
    raw = (
        await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
    ).scalar()
    if not raw:
        return (0,)
    return tuple(int(p) for p in str(raw).split(".") if p.isdigit())
