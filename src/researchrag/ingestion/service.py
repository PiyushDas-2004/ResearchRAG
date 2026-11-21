"""Ingestion orchestration: validate -> dedupe -> parse -> chunk -> embed -> transactional write.

Idempotency: documents are keyed by the SHA-256 of their bytes. Re-ingesting the same file
is a no-op, except that a missing chunk set for the *current* chunking config is created from
the stored text (so switching chunking strategies never requires re-uploading PDFs).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from researchrag.config.pipeline import PipelineConfig
from researchrag.embeddings.service import EmbeddingService
from researchrag.errors import ConflictError
from researchrag.ingestion.chunking import build_chunker
from researchrag.ingestion.isolation import parse_document
from researchrag.ingestion.pdf_parser import ParseLimits
from researchrag.ingestion.types import ChunkDraft, ParsedDocument
from researchrag.retrieval.types import EmbeddingVersionRef
from researchrag.security.upload import UploadPolicy, validate_upload
from researchrag.storage.store import DocumentStore, IngestBundle

Progress = Callable[[str, float], Awaitable[None]]


@dataclass(frozen=True)
class ParseOptions:
    """How untrusted PDFs are parsed."""

    limits: ParseLimits = field(default_factory=ParseLimits)
    isolation: str = "subprocess"
    timeout_s: float = 60.0
    memory_mb: int = 1024
    upload: UploadPolicy = field(default_factory=UploadPolicy)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of ingesting one file."""

    document_id: str
    created: bool
    chunk_count: int
    embedded: int
    title: str | None = None
    warnings: tuple[str, ...] = ()


async def _noop(stage: str, fraction: float) -> None:
    return None


def embedding_input(draft_text: str, title: str | None, section: str, enabled: bool) -> str:
    """Text actually sent to the embedder (optionally with a contextual header)."""
    if not enabled:
        return draft_text
    header = f"{title}. {section.replace('_', ' ')}:" if title else f"{section.replace('_', ' ')}:"
    return f"{header} {draft_text}"


class IngestionService:
    """Ingests PDFs into a :class:`DocumentStore` using one pipeline configuration."""

    def __init__(
        self,
        store: DocumentStore,
        embeddings: EmbeddingService,
        config: PipelineConfig,
        options: ParseOptions | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.config = config
        self.options = options or ParseOptions()
        self.chunker = build_chunker(config.chunking)

    async def version(self) -> EmbeddingVersionRef:
        """The embedding version for the configured embedder (created on first use)."""
        return await self.store.register_embedding_version(
            self.embeddings.embedder.provider, self.embeddings.embedder.model, self.embeddings.dimension
        )

    async def _embed(self, parsed: ParsedDocument, drafts: list[ChunkDraft]) -> dict[int, list[float]]:
        retrievable = [d for d in drafts if d.retrievable]
        inputs = [
            embedding_input(d.text, parsed.metadata.title, d.section, self.config.embedding.contextual_header)
            for d in retrievable
        ]
        vectors = await self.embeddings.embed(inputs, "document")
        return {d.index: v for d, v in zip(retrievable, vectors, strict=True)}

    async def ingest_bytes(
        self,
        data: bytes,
        filename: str | None,
        *,
        collection: str | None = None,
        progress: Progress | None = None,
    ) -> IngestResult:
        """Ingest one PDF (validated, parsed in isolation, written in one transaction)."""
        report = progress or _noop
        name = validate_upload(data, filename, self.options.upload)
        content_hash = hashlib.sha256(data).hexdigest()
        existing = await self.store.find_document(content_hash)
        if existing is not None:
            created = await self.ensure_chunk_set(existing)
            return IngestResult(existing, False, created, created)
        await report("parse", 0.1)
        parsed = await asyncio.to_thread(
            parse_document,
            data,
            self.options.limits,
            isolation=self.options.isolation,
            timeout_s=self.options.timeout_s,
            memory_mb=self.options.memory_mb,
        )
        await report("chunk", 0.4)
        drafts = self.chunker.chunk(parsed)
        await report("embed", 0.5)
        version = await self.version()
        vectors = await self._embed(parsed, drafts)
        await report("store", 0.9)
        bundle = IngestBundle(
            parsed, name, len(data), self.config.chunking, drafts, version, vectors, collection
        )
        try:
            doc_id = await self.store.write_document(bundle)
        except ConflictError as exc:
            # A concurrent ingest of the same bytes won the race: treat as a no-op.
            return IngestResult(str(exc.detail["document_id"]), False, 0, 0)
        await report("done", 1.0)
        return IngestResult(
            doc_id, True, len(drafts), len(vectors), parsed.metadata.title, tuple(parsed.warnings)
        )

    async def ensure_chunk_set(self, document_id: str) -> int:
        """Create the chunk set for the current chunking config if it is missing; return chunks added."""
        config_hash = self.config.chunking.config_hash()
        if await self.store.chunk_set_exists(document_id, config_hash):
            await self.embed_missing(config_hash)
            return 0
        parsed = await self.store.load_document(document_id)
        drafts = self.chunker.chunk(parsed)
        vectors = await self._embed(parsed, drafts)
        version = await self.version()
        return await self.store.write_chunk_set(document_id, self.config.chunking, drafts, version, vectors)

    async def embed_missing(self, config_hash: str | None = None, batch: int = 256) -> int:
        """Embed chunks that lack a vector for the current embedding version (re-embedding job)."""
        version = await self.version()
        total = 0
        while True:
            todo = await self.store.chunks_missing_embeddings(version, config_hash, batch)
            if not todo:
                return total
            header = self.config.embedding.contextual_header
            inputs = [embedding_input(body, title, section, header) for _, body, title, section in todo]
            vectors = await self.embeddings.embed(inputs, "document")
            await self.store.write_embeddings(
                version, [(t[0], v) for t, v in zip(todo, vectors, strict=True)]
            )
            total += len(todo)
