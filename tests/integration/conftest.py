"""Database fixtures: a real PostgreSQL + pgvector, migrated from scratch.

Uses ``RR_TEST_DATABASE_URL`` when set (CI service container, local compose); otherwise tries
testcontainers with the ``pgvector/pgvector:pg16`` image; otherwise skips database tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from researchrag.config.pipeline import PipelineConfig
from researchrag.config.settings import Settings
from researchrag.embeddings.service import EmbeddingService, build_embedder
from researchrag.ingestion.service import IngestionService, ParseOptions
from researchrag.retrieval.types import EmbeddingVersionRef
from researchrag.storage.db import create_engine, session_factory
from researchrag.storage.migrate import reset
from researchrag.storage.search import PgSearchBackend
from researchrag.storage.store import PgDocumentStore
from tests.conftest import corpus_papers, paper_pdf

pytestmark = pytest.mark.db


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


@dataclass
class IndexedCorpus:
    """The synthetic corpus ingested into Postgres with the default pipeline."""

    config: PipelineConfig
    store: PgDocumentStore
    backend: PgSearchBackend
    embeddings: EmbeddingService
    ingestion: IngestionService
    version: EmbeddingVersionRef
    doc_ids: dict[str, str]


@pytest.fixture(scope="session")
async def indexed(factory: async_sessionmaker[AsyncSession]) -> IndexedCorpus:
    """Ingest all synthetic papers once (offline fake embedder)."""
    config = PipelineConfig()
    store = PgDocumentStore(factory)
    embeddings = EmbeddingService(build_embedder(config.embedding), config.embedding)
    ingestion = IngestionService(store, embeddings, config, ParseOptions(isolation="inline"))
    doc_ids: dict[str, str] = {}
    for paper in corpus_papers():
        result = await ingestion.ingest_bytes(
            paper_pdf(paper.key), f"{paper.key}.pdf", collection="synthetic"
        )
        doc_ids[paper.key] = result.document_id
    return IndexedCorpus(
        config, store, PgSearchBackend(factory), embeddings, ingestion, await ingestion.version(), doc_ids
    )
