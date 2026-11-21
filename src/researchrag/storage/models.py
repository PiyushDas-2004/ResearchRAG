"""SQLAlchemy 2.0 ORM models for the ResearchRAG schema.

Key design points (see docs/adr):

* Documents are keyed by the SHA-256 of their bytes, so re-ingesting is a no-op.
* Chunks belong to a *chunk set* identified by ``(document, chunking config hash)``, so
  several chunkings of the same corpus can coexist (needed for evaluation sweeps).
* Embeddings are stored per *embedding version* ``(provider/model, dimension)``. The vector
  column is dimension-less; each version gets its own partial HNSW expression index
  (``embedding::vector(D)`` ... ``WHERE version_id = N``), created when the version is
  registered. Switching models therefore never corrupts or rewrites existing vectors.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all tables."""

    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB, list[str]: ARRAY(Text)}


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Document(Base):
    """An ingested paper."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    abstract: Mapped[str | None] = mapped_column(Text)
    doi: Mapped[str | None] = mapped_column(String(255))
    arxiv_id: Mapped[str | None] = mapped_column(String(64))
    venue: Mapped[str | None] = mapped_column(Text)
    collection: Mapped[str | None] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    structure: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    warnings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    pages: Mapped[list[DocumentPage]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentPage(Base):
    """Per-page text and offsets (for showing citation context)."""

    __tablename__ = "document_pages"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ocr: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    document: Mapped[Document] = relationship(back_populates="pages")


class ChunkSet(Base):
    """All chunks of one document produced by one chunking configuration."""

    __tablename__ = "chunk_sets"
    __table_args__ = (UniqueConstraint("document_id", "config_hash", name="uq_chunk_sets_doc_config"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now()


class Chunk(Base):
    """A chunk of document text with page, offsets, section and a generated tsvector."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_config_doc", "config_hash", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    chunk_set_id: Mapped[int] = mapped_column(ForeignKey("chunk_sets.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str] = mapped_column(String(32), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    retrievable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_reference: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=True
    )


class EmbeddingVersion(Base):
    """An embedding model + dimension; each version has its own HNSW partial index."""

    __tablename__ = "embedding_versions"
    __table_args__ = (UniqueConstraint("model_id", "dimension", name="uq_embedding_versions_model_dim"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    hnsw_m: Mapped[int] = mapped_column(Integer, nullable=False, default=16)
    hnsw_ef_construction: Mapped[int] = mapped_column(Integer, nullable=False, default=64)
    created_at: Mapped[datetime] = _now()


class ChunkEmbedding(Base):
    """An embedding of one chunk under one embedding version."""

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_versions.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = _now()


class IngestionJob(Base):
    """A background job (ingest a PDF, or re-embed chunks under a new embedding version)."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("api_key_id", "idempotency_key", name="uq_jobs_idempotency"),
        Index("ix_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="ingest")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    stage: Mapped[str | None] = mapped_column(String(32))
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    filename: Mapped[str | None] = mapped_column(String(255))
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    payload: Mapped[bytes | None] = mapped_column(LargeBinary)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    api_key_id: Mapped[str | None] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QueryTrace(Base):
    """Everything that happened while answering one query."""

    __tablename__ = "query_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = _now()
    query: Mapped[str] = mapped_column(Text, nullable=False)
    standalone_query: Mapped[str | None] = mapped_column(Text)
    conversation_id: Mapped[str | None] = mapped_column(String(128), index=True)
    config_name: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    stages: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    api_key_id: Mapped[str | None] = mapped_column(String(64))


class Answer(Base):
    """The final answer returned for a trace."""

    __tablename__ = "answers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("query_traces.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False)
    abstain_reason: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _now()


class AnswerClaim(Base):
    """An atomic claim of an answer and its verification verdict."""

    __tablename__ = "answer_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), nullable=False)
    claim_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    removed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)


class Citation(Base):
    """A resolved citation marker of an answer."""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), nullable=False)
    marker: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chunks.id", ondelete="SET NULL"))
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    page: Mapped[int | None] = mapped_column(Integer)
    quote: Mapped[str | None] = mapped_column(Text)
    quote_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quote_score: Mapped[float | None] = mapped_column(Float)


class Feedback(Base):
    """User feedback (thumbs up/down) on an answer trace."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("query_traces.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    api_key_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()


class ConversationTurn(Base):
    """Short-term conversational memory."""

    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    api_key_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()


class EvalRun(Base):
    """One evaluation run of one pipeline configuration over one dataset."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = _now()


class EvalResult(Base):
    """Per-question metrics of an evaluation run."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)
    question_id: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
