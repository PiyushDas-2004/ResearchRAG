"""Document store interface and its PostgreSQL implementation.

Writes are transactional: a document, its pages, its chunk set, chunks and embeddings are
written in one transaction, so a failure mid-ingest never leaves a half-written document.
Chunk ids are deterministic (UUIDv5 of content hash, chunking config hash and index), which
makes re-ingestion and re-chunking idempotent.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchrag.config.pipeline import ChunkingConfig
from researchrag.errors import ConfigurationError, ConflictError, NotFoundError
from researchrag.ingestion.types import (
    ChunkDraft,
    DocumentMetadata,
    ParsedDocument,
    document_from_structure,
    structure_to_json,
)
from researchrag.retrieval.types import EmbeddingVersionRef
from researchrag.storage.models import (
    Chunk,
    ChunkEmbedding,
    ChunkSet,
    Document,
    DocumentPage,
    EmbeddingVersion,
)
from researchrag.storage.search import vector_literal

CHUNK_NAMESPACE = uuid.UUID("8f4b7a52-2a9e-4c38-9a59-6f6c0b6d2a11")
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
MAX_HNSW_DIM = 2000  # pgvector's HNSW limit for the `vector` type


def chunk_uuid(content_hash: str, config_hash: str, index: int) -> uuid.UUID:
    """Deterministic chunk id."""
    return uuid.uuid5(CHUNK_NAMESPACE, f"{content_hash}:{config_hash}:{index}")


def text_hash(value: str) -> str:
    """SHA-256 of text."""
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class IngestBundle:
    """Everything needed to persist one parsed document in one transaction."""

    parsed: ParsedDocument
    filename: str
    size_bytes: int
    chunking: ChunkingConfig
    drafts: list[ChunkDraft]
    version: EmbeddingVersionRef
    vectors: dict[int, list[float]] = field(default_factory=dict)
    collection: str | None = None


class DocumentStore(Protocol):
    """Persistence operations used by ingestion and evaluation."""

    async def find_document(self, content_hash: str) -> str | None:
        """Return the id of a document with these bytes, if any."""
        ...

    async def register_embedding_version(
        self, provider: str, model: str, dimension: int
    ) -> EmbeddingVersionRef:
        """Get or create an embedding version (and its ANN index)."""
        ...

    async def chunk_set_exists(self, document_id: str, config_hash: str) -> bool:
        """True if the document was already chunked with this config."""
        ...

    async def write_document(self, bundle: IngestBundle) -> str:
        """Persist a new document with its chunks and embeddings; return its id."""
        ...

    async def write_chunk_set(
        self,
        document_id: str,
        chunking: ChunkingConfig,
        drafts: list[ChunkDraft],
        version: EmbeddingVersionRef,
        vectors: dict[int, list[float]],
    ) -> int:
        """Add a chunk set (re-chunking) to an existing document; return chunk count."""
        ...

    async def load_document(self, document_id: str) -> ParsedDocument:
        """Rebuild the parsed document (for re-chunking)."""
        ...

    async def list_document_ids(self) -> list[str]:
        """All document ids."""
        ...

    async def chunks_missing_embeddings(
        self, version: EmbeddingVersionRef, config_hash: str | None, limit: int
    ) -> list[tuple[str, str, str | None, str]]:
        """Retrievable chunks lacking an embedding: ``[(chunk_id, text, doc_title, section)]``."""
        ...

    async def write_embeddings(
        self, version: EmbeddingVersionRef, items: list[tuple[str, list[float]]]
    ) -> None:
        """Upsert embeddings for chunks."""
        ...


class PgDocumentStore:
    """:class:`DocumentStore` backed by PostgreSQL."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def find_document(self, content_hash: str) -> str | None:
        """Return the id of a document with these bytes, if any."""
        async with self._factory() as session:
            found = await session.scalar(select(Document.id).where(Document.content_hash == content_hash))
        return str(found) if found else None

    async def register_embedding_version(
        self, provider: str, model: str, dimension: int
    ) -> EmbeddingVersionRef:
        """Get or create an embedding version and its partial HNSW index."""
        if dimension > MAX_HNSW_DIM:
            raise ConfigurationError(
                f"HNSW supports at most {MAX_HNSW_DIM} dimensions; configure a smaller embedding dimension."
            )
        model_id = f"{provider}/{model}"
        async with self._factory() as session, session.begin():
            await session.execute(
                pg_insert(EmbeddingVersion)
                .values(
                    model_id=model_id,
                    provider=provider,
                    model=model,
                    dimension=dimension,
                    hnsw_m=HNSW_M,
                    hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                )
                .on_conflict_do_nothing(constraint="uq_embedding_versions_model_dim")
            )
            version_id = await session.scalar(
                select(EmbeddingVersion.id).where(
                    EmbeddingVersion.model_id == model_id, EmbeddingVersion.dimension == dimension
                )
            )
            assert version_id is not None  # noqa: S101 - guaranteed by the upsert above
            vid, dim = int(version_id), int(dimension)
            await session.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS ix_emb_hnsw_v{vid} ON chunk_embeddings "
                    f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops) "
                    f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}) WHERE version_id = {vid}"
                )
            )
        return EmbeddingVersionRef(vid, dim, model_id)

    async def chunk_set_exists(self, document_id: str, config_hash: str) -> bool:
        """True if the document already has a chunk set for this config."""
        async with self._factory() as session:
            found = await session.scalar(
                select(ChunkSet.id).where(
                    ChunkSet.document_id == uuid.UUID(document_id), ChunkSet.config_hash == config_hash
                )
            )
        return found is not None

    async def write_document(self, bundle: IngestBundle) -> str:
        """Persist document + pages + chunk set + chunks + embeddings in one transaction."""
        parsed = bundle.parsed
        doc_id = uuid.uuid4()
        try:
            await self._write_document(bundle, doc_id)
        except IntegrityError as exc:
            existing = await self.find_document(parsed.content_hash)
            if existing is not None:
                raise ConflictError("Document already ingested", detail={"document_id": existing}) from exc
            raise
        return str(doc_id)

    async def _write_document(self, bundle: IngestBundle, doc_id: uuid.UUID) -> None:
        parsed = bundle.parsed
        meta = parsed.metadata
        async with self._factory() as session, session.begin():
            session.add(
                Document(
                    id=doc_id,
                    content_hash=parsed.content_hash,
                    text_hash=text_hash(parsed.text),
                    filename=bundle.filename,
                    size_bytes=bundle.size_bytes,
                    page_count=len(parsed.pages),
                    title=meta.title,
                    authors=list(meta.authors),
                    year=meta.year,
                    abstract=meta.abstract,
                    doi=meta.doi,
                    arxiv_id=meta.arxiv_id,
                    venue=meta.venue,
                    collection=bundle.collection,
                    text=parsed.text,
                    structure=structure_to_json(parsed),
                    warnings={"items": list(parsed.warnings)},
                )
            )
            session.add_all(
                DocumentPage(
                    document_id=doc_id,
                    page_number=p.number,
                    start_offset=p.start,
                    end_offset=p.end,
                    text=parsed.text[p.start : p.end],
                    ocr=p.ocr,
                )
                for p in parsed.pages
            )
            await session.flush()
            await self._insert_chunks(
                session,
                doc_id,
                parsed.content_hash,
                bundle.chunking,
                bundle.drafts,
                bundle.version,
                bundle.vectors,
            )

    async def write_chunk_set(
        self,
        document_id: str,
        chunking: ChunkingConfig,
        drafts: list[ChunkDraft],
        version: EmbeddingVersionRef,
        vectors: dict[int, list[float]],
    ) -> int:
        """Add a chunk set to an existing document in one transaction."""
        doc_uuid = uuid.UUID(document_id)
        async with self._factory() as session, session.begin():
            content_hash = await session.scalar(select(Document.content_hash).where(Document.id == doc_uuid))
            if content_hash is None:
                raise NotFoundError(f"Document {document_id} not found")
            await self._insert_chunks(session, doc_uuid, content_hash, chunking, drafts, version, vectors)
        return len(drafts)

    async def _insert_chunks(
        self,
        session: AsyncSession,
        doc_id: uuid.UUID,
        content_hash: str,
        chunking: ChunkingConfig,
        drafts: list[ChunkDraft],
        version: EmbeddingVersionRef,
        vectors: dict[int, list[float]],
    ) -> None:
        config_hash = chunking.config_hash()
        chunk_set = ChunkSet(
            document_id=doc_id, config_hash=config_hash, config=chunking.model_dump(), chunk_count=len(drafts)
        )
        session.add(chunk_set)
        await session.flush()
        ids = {d.index: chunk_uuid(content_hash, config_hash, d.index) for d in drafts}
        # Parents first so the self-referencing foreign key is satisfied.
        for draft in sorted(drafts, key=lambda d: (d.kind != "parent", d.index)):
            session.add(
                Chunk(
                    id=ids[draft.index],
                    chunk_set_id=chunk_set.id,
                    document_id=doc_id,
                    config_hash=config_hash,
                    chunk_index=draft.index,
                    kind=draft.kind,
                    parent_id=ids[draft.parent_index] if draft.parent_index is not None else None,
                    text=draft.text,
                    text_hash=text_hash(draft.text),
                    start_offset=draft.start,
                    end_offset=draft.end,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    section=draft.section,
                    token_count=draft.token_count,
                    retrievable=draft.retrievable,
                    is_reference=draft.section == "references",
                )
            )
        await session.flush()
        if vectors:
            session.add_all(
                ChunkEmbedding(chunk_id=ids[index], version_id=version.id, embedding=vec)
                for index, vec in vectors.items()
            )
            await session.flush()

    async def load_document(self, document_id: str) -> ParsedDocument:
        """Rebuild the parsed document from stored text and structure."""
        async with self._factory() as session:
            doc = await session.get(Document, uuid.UUID(document_id))
            if doc is None:
                raise NotFoundError(f"Document {document_id} not found")
            meta = DocumentMetadata(
                title=doc.title,
                authors=list(doc.authors),
                year=doc.year,
                abstract=doc.abstract,
                doi=doc.doi,
                arxiv_id=doc.arxiv_id,
                venue=doc.venue,
            )
            return document_from_structure(doc.text, doc.structure, meta, doc.content_hash)

    async def list_document_ids(self) -> list[str]:
        """All document ids, oldest first."""
        async with self._factory() as session:
            rows = (
                await session.execute(select(Document.id).order_by(Document.created_at, Document.id))
            ).scalars()
            return [str(r) for r in rows]

    async def chunks_missing_embeddings(
        self, version: EmbeddingVersionRef, config_hash: str | None, limit: int
    ) -> list[tuple[str, str, str | None, str]]:
        """Retrievable chunks without an embedding for ``version``."""
        stmt = (
            select(Chunk.id, Chunk.text, Document.title, Chunk.section)
            .join(Document, Document.id == Chunk.document_id)
            .outerjoin(
                ChunkEmbedding,
                (ChunkEmbedding.chunk_id == Chunk.id) & (ChunkEmbedding.version_id == version.id),
            )
            .where(Chunk.retrievable.is_(True), ChunkEmbedding.chunk_id.is_(None))
            .order_by(Chunk.id)
            .limit(limit)
        )
        if config_hash is not None:
            stmt = stmt.where(Chunk.config_hash == config_hash)
        async with self._factory() as session:
            return [
                (str(cid), body, title, sec) for cid, body, title, sec in (await session.execute(stmt)).all()
            ]

    async def write_embeddings(
        self, version: EmbeddingVersionRef, items: list[tuple[str, list[float]]]
    ) -> None:
        """Upsert embeddings in one transaction."""
        if not items:
            return
        async with self._factory() as session, session.begin():
            for chunk_id, vec in items:
                await session.execute(
                    text(
                        "INSERT INTO chunk_embeddings (chunk_id, version_id, embedding) "
                        "VALUES (CAST(:cid AS uuid), :vid, CAST(:emb AS vector)) "
                        "ON CONFLICT (chunk_id, version_id) DO UPDATE SET embedding = EXCLUDED.embedding"
                    ),
                    {"cid": chunk_id, "vid": version.id, "emb": vector_literal(vec)},
                )

    async def count_chunks(self, config_hash: str | None = None) -> int:
        """Number of chunks (optionally for one chunking config)."""
        stmt = select(func.count()).select_from(Chunk)
        if config_hash is not None:
            stmt = stmt.where(Chunk.config_hash == config_hash)
        async with self._factory() as session:
            return int(await session.scalar(stmt) or 0)
