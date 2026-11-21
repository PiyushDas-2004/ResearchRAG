"""Integration tests against real PostgreSQL + pgvector: ingestion, hybrid retrieval, filters, indexes."""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchrag.config.pipeline import PipelineConfig
from researchrag.errors import NotFoundError, ProviderError, UploadRejectedError
from researchrag.ingestion.service import IngestionService, ParseOptions
from researchrag.observability.tracing import StageRecorder
from researchrag.rerank.rerankers import build_reranker
from researchrag.retrieval.retriever import Retriever
from researchrag.retrieval.types import SearchFilters
from researchrag.storage.models import Chunk, ChunkEmbedding, ChunkSet, Document, DocumentPage
from researchrag.storage.search import PgSearchBackend, tsquery_terms
from researchrag.textutils import content_terms
from tests.conftest import EVAL_DIR, paper_pdf
from tests.integration.conftest import IndexedCorpus

pytestmark = pytest.mark.db


async def _count(factory: async_sessionmaker[AsyncSession], model: type) -> int:
    async with factory() as session:
        return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_corpus_ingested_with_pages_chunks_embeddings(indexed: IndexedCorpus, factory) -> None:  # type: ignore[no-untyped-def]
    assert len(indexed.doc_ids) == 16
    assert await _count(factory, Document) == 16
    assert await _count(factory, DocumentPage) >= 32
    chunks = await indexed.store.count_chunks(indexed.config.chunking.config_hash())
    assert chunks > 300
    async with factory() as session:
        retrievable = await session.scalar(select(func.count()).select_from(Chunk).where(Chunk.retrievable))
    assert await _count(factory, ChunkEmbedding) == retrievable


async def test_reingest_same_pdf_is_noop(indexed: IndexedCorpus, factory) -> None:  # type: ignore[no-untyped-def]
    before = await _count(factory, Chunk)
    result = await indexed.ingestion.ingest_bytes(paper_pdf("ir-01"), "again.pdf")
    assert not result.created
    assert result.document_id == indexed.doc_ids["ir-01"]
    assert await _count(factory, Chunk) == before


async def test_new_chunking_config_adds_chunk_set_without_touching_existing(
    indexed: IndexedCorpus, factory
) -> None:  # type: ignore[no-untyped-def]
    config = indexed.config.with_overrides({"chunking.strategy": "fixed", "chunking.chunk_tokens": 256})
    service = IngestionService(indexed.store, indexed.embeddings, config, ParseOptions(isolation="inline"))
    before_default = await indexed.store.count_chunks(indexed.config.chunking.config_hash())
    added = await service.ensure_chunk_set(indexed.doc_ids["clim-01"])
    assert added > 0
    assert await service.ensure_chunk_set(indexed.doc_ids["clim-01"]) == 0
    assert await indexed.store.count_chunks(indexed.config.chunking.config_hash()) == before_default
    async with factory() as session:
        sets = await session.scalar(
            select(func.count())
            .select_from(ChunkSet)
            .where(ChunkSet.document_id == uuid.UUID(indexed.doc_ids["clim-01"]))
        )
    assert sets == 2


async def test_failed_embedding_rolls_back_ingest(indexed: IndexedCorpus, factory) -> None:  # type: ignore[no-untyped-def]
    class Broken:
        provider, model, dimension = "fake", "fake-hash-v1", 384

        async def embed(self, texts, kind):  # type: ignore[no-untyped-def]
            raise ProviderError("embedding provider down mid-ingest")

    from researchrag.embeddings.service import EmbeddingService
    from researchrag.evaluation.synthetic import load_papers, render_paper
    from tests.conftest import PAPERS_DIR

    paper = load_papers(PAPERS_DIR)[0]
    data = render_paper(
        paper, extra_paragraphs=[("Appendix", "A unique appendix used only by the rollback test.")]
    )
    service = IngestionService(
        indexed.store,
        EmbeddingService(Broken(), indexed.config.embedding),
        indexed.config,
        ParseOptions(isolation="inline"),
    )
    docs_before = await _count(factory, Document)
    with pytest.raises(ProviderError):
        await service.ingest_bytes(data, "broken.pdf")
    assert await _count(factory, Document) == docs_before


async def test_rejects_non_pdf_before_touching_db(indexed: IndexedCorpus) -> None:
    with pytest.raises(UploadRejectedError):
        await indexed.ingestion.ingest_bytes(b"<html>" + b"x" * 500, "paper.pdf")


def _retriever(indexed: IndexedCorpus, **overrides: object) -> Retriever:
    config = indexed.config.with_overrides(dict(overrides)) if overrides else indexed.config
    reranker = build_reranker(config.rerank.provider, config.rerank.model, None, 100)
    return Retriever(indexed.backend, indexed.embeddings, indexed.version, config, reranker=reranker)


@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
async def test_retrieval_finds_gold_evidence(indexed: IndexedCorpus, mode: str) -> None:
    retriever = _retriever(indexed, **{"retrieval.mode": mode})
    key_of = {v: k for k, v in indexed.doc_ids.items()}
    hits = total = 0
    for line in (EVAL_DIR / "questions_ir.jsonl").read_text().splitlines():
        q = json.loads(line)
        if not q["answerable"]:
            continue
        result = await retriever.retrieve(q["question"], SearchFilters(), StageRecorder())
        ev = q["evidence"][0]
        terms = set(content_terms(ev["passage"]))
        total += 1
        hits += any(
            key_of[c.document_id] == ev["doc_key"]
            and len(terms & set(content_terms(c.text))) / len(terms) >= 0.6
            for c in result.candidates[:8]
        )
    assert hits / total >= 0.6, f"{mode}: {hits}/{total}"


async def test_metadata_filters_are_applied_in_sql(indexed: IndexedCorpus) -> None:
    backend = indexed.backend
    ch = indexed.config.chunking.config_hash()
    vec = await indexed.embeddings.embed_query("carbon burial in marsh sediment cores")
    only = SearchFilters(document_ids=(indexed.doc_ids["ir-02"],))
    found = await backend.vector_search(
        vec, version=indexed.version, chunk_config_hash=ch, k=5, filters=only, ef_search=40
    )
    assert found
    assert {c.document_id for c in found} == {indexed.doc_ids["ir-02"]}
    years = SearchFilters(year_min=2024, year_max=2025)
    lex = await backend.lexical_search("retrieval recall chunk", chunk_config_hash=ch, k=20, filters=years)
    assert lex
    assert all(c.year is not None and 2024 <= c.year <= 2025 for c in lex)
    author = (await indexed.store.load_document(indexed.doc_ids["clim-02"])).metadata.authors[0].split()[-1]
    by_author = await backend.lexical_search(
        "roof cooling", chunk_config_hash=ch, k=10, filters=SearchFilters(authors=(author.lower(),))
    )
    assert by_author
    assert {c.document_id for c in by_author} == {indexed.doc_ids["clim-02"]}
    coll = await backend.lexical_search(
        "roof", chunk_config_hash=ch, k=3, filters=SearchFilters(collection="none")
    )
    assert coll == []


async def test_references_excluded_unless_requested(indexed: IndexedCorpus) -> None:
    ch = indexed.config.chunking.config_hash()
    query = "Journal 2019 pp"
    plain = await indexed.backend.lexical_search(query, chunk_config_hash=ch, k=50, filters=SearchFilters())
    assert all(c.section != "references" for c in plain)
    with_refs = await indexed.backend.lexical_search(
        query, chunk_config_hash=ch, k=50, filters=SearchFilters(include_references=True)
    )
    assert any(c.section == "references" for c in with_refs)


async def test_parents_fetched_for_children(indexed: IndexedCorpus) -> None:
    result = await _retriever(indexed).retrieve("BraidRank fusion weights", SearchFilters(), StageRecorder())
    parent_ids = [c.parent_id for c in result.candidates if c.parent_id]
    parents = await indexed.backend.get_parents(parent_ids)
    assert set(parents) == set(parent_ids)
    for cand in result.candidates:
        assert cand.parent_id is not None
        assert cand.text in parents[cand.parent_id].text
    assert await indexed.backend.get_parents([]) == {}


async def test_hnsw_and_gin_indexes_exist_and_are_used(indexed: IndexedCorpus, factory) -> None:  # type: ignore[no-untyped-def]
    async with factory() as session:
        names = set(
            (
                await session.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename IN ('chunks','chunk_embeddings')")
                )
            ).scalars()
        )
    assert f"ix_emb_hnsw_v{indexed.version.id}" in names
    assert "ix_chunks_tsv" in names
    sql, params = PgSearchBackend.vector_sql(indexed.version, SearchFilters())
    vec = await indexed.embeddings.embed_query("glacier meltwater river temperature")
    from researchrag.storage.search import vector_literal

    params |= {"q": vector_literal(vec), "ch": indexed.config.chunking.config_hash(), "k": 10}
    async with factory() as session, session.begin():
        await session.execute(text("ANALYZE chunk_embeddings"))
        await session.execute(text("ANALYZE chunks"))
        plan = "\n".join((await session.execute(text("EXPLAIN " + sql), params)).scalars())
    assert f"ix_emb_hnsw_v{indexed.version.id}" in plan, plan
    # At this corpus size the planner may legitimately prefer the config_hash B-tree for the
    # full lexical query, so prove that the GIN index serves the @@ predicate itself.
    lex_sql = "EXPLAIN SELECT id FROM chunks c, to_tsquery('english', :q) q WHERE c.tsv @@ q"
    async with factory() as session, session.begin():
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        lex_plan = "\n".join(
            (await session.execute(text(lex_sql), {"q": tsquery_terms("BraidRank")})).scalars()
        )
    assert "ix_chunks_tsv" in lex_plan, lex_plan


async def test_reembedding_new_version_keeps_old_vectors(indexed: IndexedCorpus, factory) -> None:  # type: ignore[no-untyped-def]
    from researchrag.embeddings.service import EmbeddingService, build_embedder

    config = indexed.config.with_overrides(
        {"embedding.dimension": 128, "embedding.model": "fake-hash-v1-small"}
    )
    embeddings = EmbeddingService(build_embedder(config.embedding), config.embedding)
    service = IngestionService(indexed.store, embeddings, config, ParseOptions(isolation="inline"))
    old_count = await _count(factory, ChunkEmbedding)
    embedded = await service.embed_missing(config.chunking.config_hash())
    assert embedded > 300
    new_version = await service.version()
    assert new_version.id != indexed.version.id
    assert await _count(factory, ChunkEmbedding) == old_count + embedded
    assert await service.embed_missing(config.chunking.config_hash()) == 0
    retriever = Retriever(indexed.backend, embeddings, new_version, config)
    result = await retriever.retrieve(
        "pollinator decline farmland transects", SearchFilters(), StageRecorder()
    )
    assert result.candidates
    async with factory() as session:
        names = set(
            (
                await session.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunk_embeddings'")
                )
            ).scalars()
        )
    assert f"ix_emb_hnsw_v{new_version.id}" in names


async def test_store_errors(indexed: IndexedCorpus) -> None:
    with pytest.raises(NotFoundError):
        await indexed.store.load_document(str(uuid.uuid4()))
    with pytest.raises(NotFoundError):
        await indexed.store.write_chunk_set(
            str(uuid.uuid4()), PipelineConfig().chunking, [], indexed.version, {}
        )
    assert await indexed.store.find_document("0" * 64) is None
    ids = await indexed.store.list_document_ids()
    assert len(ids) >= 16
    with pytest.raises(Exception, match="HNSW"):
        await indexed.store.register_embedding_version("fake", "huge", 3072)
