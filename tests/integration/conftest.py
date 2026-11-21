"""Integration fixtures: the synthetic corpus indexed in PostgreSQL (see tests/db_fixtures.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchrag.config.pipeline import PipelineConfig
from researchrag.embeddings.service import EmbeddingService, build_embedder
from researchrag.ingestion.service import IngestionService, ParseOptions
from researchrag.retrieval.types import EmbeddingVersionRef
from researchrag.storage.search import PgSearchBackend
from researchrag.storage.store import PgDocumentStore
from tests.conftest import corpus_papers, paper_pdf

pytestmark = pytest.mark.db


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
