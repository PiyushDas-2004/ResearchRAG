"""Repositories for traces, feedback, conversations, documents and background jobs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from researchrag.errors import ConflictError, NotFoundError
from researchrag.generation.schema import QueryResult
from researchrag.storage.models import (
    Answer,
    AnswerClaim,
    Chunk,
    Citation,
    ConversationTurn,
    Document,
    Feedback,
    IngestionJob,
    QueryTrace,
)


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise NotFoundError(f"Invalid id: {value}") from exc


# --------------------------------------------------------------------------- traces
class TraceSink(Protocol):
    """Persists query traces."""

    async def save(
        self, result: QueryResult, details: dict[str, Any], *, api_key_id: str | None = None
    ) -> None:
        """Store a finished query."""
        ...

    async def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return a stored trace (result + stage details) or ``None``."""
        ...


class MemoryTraceSink:
    """Keeps traces in a dict (tests, evaluation)."""

    def __init__(self) -> None:
        self.traces: dict[str, dict[str, Any]] = {}

    async def save(
        self, result: QueryResult, details: dict[str, Any], *, api_key_id: str | None = None
    ) -> None:
        """Store a finished query."""
        self.traces[result.trace_id] = {"result": result.model_dump(mode="json"), "stages": details}

    async def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return a stored trace."""
        return self.traces.get(trace_id)


class PgTraceRepository:
    """Writes query_traces, answers, answer_claims and citations in one transaction."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def save(
        self, result: QueryResult, details: dict[str, Any], *, api_key_id: str | None = None
    ) -> None:
        """Store a finished query."""
        trace_id = uuid.UUID(result.trace_id)
        async with self._factory() as session, session.begin():
            session.add(
                QueryTrace(
                    id=trace_id,
                    query=result.question,
                    standalone_query=result.standalone_question,
                    conversation_id=details.get("conversation_id"),
                    config_name=result.config_name,
                    config_hash=str(details.get("config_hash", "")),
                    filters=result.searched.filters,
                    stages=details,
                    latency_ms=result.latency_ms,
                    cost_usd=result.usage.cost_usd,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    abstained=result.abstained,
                    faithfulness=result.faithfulness,
                    api_key_id=api_key_id,
                )
            )
            await session.flush()  # the trace row must exist before its answer (FK)
            answer = Answer(
                trace_id=trace_id,
                answer_text=result.answer,
                abstained=result.abstained,
                abstain_reason=result.abstain_reason,
                confidence=result.confidence,
                faithfulness=result.faithfulness,
            )
            session.add(answer)
            await session.flush()
            for index, claim in enumerate(result.claims):
                cited = [c.chunk_id for c in result.citations if c.marker in claim.citations]
                session.add(
                    AnswerClaim(
                        answer_id=answer.id,
                        claim_index=index,
                        text=claim.text,
                        verdict=claim.verdict,
                        rationale=claim.rationale,
                        removed=claim.removed,
                        chunk_ids=cited,
                    )
                )
            for citation in result.citations:
                session.add(
                    Citation(
                        answer_id=answer.id,
                        marker=citation.marker,
                        chunk_id=uuid.UUID(citation.chunk_id),
                        document_id=uuid.UUID(citation.document_id),
                        page=citation.page,
                        quote=citation.quote,
                        quote_verified=citation.quote_verified,
                        quote_score=citation.quote_score,
                    )
                )

    async def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return the stored trace and answer."""
        async with self._factory() as session:
            trace = await session.get(QueryTrace, _uuid(trace_id))
            if trace is None:
                return None
            answer = await session.scalar(select(Answer).where(Answer.trace_id == trace.id))
            return {
                "trace_id": str(trace.id),
                "created_at": trace.created_at.isoformat(),
                "query": trace.query,
                "standalone_query": trace.standalone_query,
                "config_name": trace.config_name,
                "config_hash": trace.config_hash,
                "filters": trace.filters,
                "latency_ms": trace.latency_ms,
                "cost_usd": trace.cost_usd,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "abstained": trace.abstained,
                "faithfulness": trace.faithfulness,
                "answer": answer.answer_text if answer else None,
                "stages": trace.stages,
            }


# --------------------------------------------------------------------------- feedback
class PgFeedbackRepository:
    """Stores thumbs up/down feedback and exports low-rated traces."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def add(self, trace_id: str, rating: int, comment: str | None, api_key_id: str | None) -> int:
        """Record feedback for an existing trace; returns the feedback id."""
        tid = _uuid(trace_id)
        async with self._factory() as session, session.begin():
            if await session.get(QueryTrace, tid) is None:
                raise NotFoundError(f"Trace {trace_id} not found")
            row = Feedback(trace_id=tid, rating=rating, comment=comment, api_key_id=api_key_id)
            session.add(row)
            await session.flush()
            return int(row.id)

    async def low_rated(self, limit: int = 100) -> list[dict[str, Any]]:
        """Traces with negative feedback, newest first (for building candidate eval sets)."""
        stmt = (
            select(Feedback, QueryTrace, Answer)
            .join(QueryTrace, QueryTrace.id == Feedback.trace_id)
            .outerjoin(Answer, Answer.trace_id == QueryTrace.id)
            .where(Feedback.rating < 0)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
        )
        async with self._factory() as session:
            rows = (await session.execute(stmt)).all()
        return [
            {
                "trace_id": str(trace.id),
                "question": trace.standalone_query or trace.query,
                "answer": answer.answer_text if answer else None,
                "abstained": trace.abstained,
                "comment": fb.comment,
                "retrieved": [c.get("chunk_id") for c in trace.stages.get("passages", [])],
            }
            for fb, trace, answer in rows
        ]


# --------------------------------------------------------------------------- conversations
class ConversationStore(Protocol):
    """Short-term conversational memory."""

    async def history(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        """Most recent ``limit`` turns as ``(role, content)``, oldest first."""
        ...

    async def append(
        self, conversation_id: str, role: str, content: str, api_key_id: str | None = None
    ) -> None:
        """Append a turn."""
        ...


class MemoryConversationStore:
    """In-process conversation memory."""

    def __init__(self) -> None:
        self.turns: dict[str, list[tuple[str, str]]] = {}

    async def history(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        """Recent turns."""
        return self.turns.get(conversation_id, [])[-limit:] if limit else []

    async def append(
        self, conversation_id: str, role: str, content: str, api_key_id: str | None = None
    ) -> None:
        """Append a turn."""
        self.turns.setdefault(conversation_id, []).append((role, content))


class PgConversationStore:
    """Conversation memory in the ``conversation_turns`` table (shared across instances)."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def history(self, conversation_id: str, limit: int) -> list[tuple[str, str]]:
        """Recent turns, oldest first."""
        if limit <= 0:
            return []
        stmt = (
            select(ConversationTurn.role, ConversationTurn.content)
            .where(ConversationTurn.conversation_id == conversation_id)
            .order_by(ConversationTurn.id.desc())
            .limit(limit)
        )
        async with self._factory() as session:
            rows = (await session.execute(stmt)).all()
        return [(r, c) for r, c in reversed(rows)]

    async def append(
        self, conversation_id: str, role: str, content: str, api_key_id: str | None = None
    ) -> None:
        """Append a turn."""
        async with self._factory() as session, session.begin():
            session.add(
                ConversationTurn(
                    conversation_id=conversation_id, role=role, content=content, api_key_id=api_key_id
                )
            )


# --------------------------------------------------------------------------- documents
_EDITABLE = {"title", "authors", "year", "abstract", "doi", "arxiv_id", "venue", "collection"}


def document_summary(doc: Document, chunk_count: int | None = None) -> dict[str, Any]:
    """Public representation of a document row."""
    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "title": doc.title,
        "authors": list(doc.authors),
        "year": doc.year,
        "abstract": doc.abstract,
        "doi": doc.doi,
        "arxiv_id": doc.arxiv_id,
        "venue": doc.venue,
        "collection": doc.collection,
        "page_count": doc.page_count,
        "size_bytes": doc.size_bytes,
        "content_hash": doc.content_hash,
        "created_at": doc.created_at.isoformat(),
        "chunk_count": chunk_count,
    }


class PgDocumentRepository:
    """Document listing, lookup, metadata edits and deletion."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def list(
        self, *, limit: int, offset: int, collection: str | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        """A page of documents (newest first) and the total count."""
        base = select(Document)
        count = select(func.count()).select_from(Document)
        if collection:
            base = base.where(Document.collection == collection)
            count = count.where(Document.collection == collection)
        async with self._factory() as session:
            total = int(await session.scalar(count) or 0)
            docs = (
                await session.execute(
                    base.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset)
                )
            ).scalars()
            return [document_summary(d) for d in docs], total

    async def get(self, document_id: str) -> dict[str, Any]:
        """One document with its chunk count."""
        async with self._factory() as session:
            doc = await session.get(Document, _uuid(document_id))
            if doc is None:
                raise NotFoundError(f"Document {document_id} not found")
            chunks = int(
                await session.scalar(
                    select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
                )
                or 0
            )
            return document_summary(doc, chunks)

    async def update_metadata(self, document_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Edit user-correctable metadata fields."""
        unknown = set(fields) - _EDITABLE
        if unknown:
            raise ConflictError(f"Fields not editable: {sorted(unknown)}")
        async with self._factory() as session, session.begin():
            doc = await session.get(Document, _uuid(document_id))
            if doc is None:
                raise NotFoundError(f"Document {document_id} not found")
            for key, value in fields.items():
                setattr(doc, key, value)
        return await self.get(document_id)

    async def delete(self, document_id: str) -> None:
        """Delete a document and (by cascade) its pages, chunks and embeddings."""
        async with self._factory() as session, session.begin():
            result = await session.execute(delete(Document).where(Document.id == _uuid(document_id)))
            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise NotFoundError(f"Document {document_id} not found")


# --------------------------------------------------------------------------- jobs
def job_view(job: IngestionJob) -> dict[str, Any]:
    """Public representation of a job."""
    return {
        "id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "filename": job.filename,
        "document_id": str(job.document_id) if job.document_id else None,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


class PgJobRepository:
    """A small durable job queue on PostgreSQL (``FOR UPDATE SKIP LOCKED``)."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def enqueue(
        self,
        *,
        kind: str,
        payload: bytes | None,
        filename: str | None,
        content_hash: str | None,
        params: dict[str, Any],
        idempotency_key: str | None,
        api_key_id: str | None,
        max_attempts: int,
    ) -> tuple[dict[str, Any], bool]:
        """Create a job; with an idempotency key, return the existing job instead. ``(job, created)``."""
        if idempotency_key:
            existing = await self._by_idempotency(api_key_id, idempotency_key)
            if existing is not None:
                if existing.content_hash != content_hash:
                    raise ConflictError("Idempotency-Key was already used with a different file")
                return job_view(existing), False
        job = IngestionJob(
            kind=kind,
            payload=payload,
            filename=filename,
            content_hash=content_hash,
            params=params,
            idempotency_key=idempotency_key,
            api_key_id=api_key_id,
            max_attempts=max_attempts,
        )
        try:
            async with self._factory() as session, session.begin():
                session.add(job)
        except IntegrityError as exc:  # a concurrent request with the same key won
            existing = await self._by_idempotency(api_key_id, idempotency_key or "")
            if existing is None:  # pragma: no cover - only reachable on unrelated integrity errors
                raise
            raise ConflictError("Duplicate idempotency key", detail={"job_id": str(existing.id)}) from exc
        return job_view(job), True

    async def _by_idempotency(self, api_key_id: str | None, key: str) -> IngestionJob | None:
        stmt = select(IngestionJob).where(IngestionJob.idempotency_key == key)
        stmt = stmt.where(
            IngestionJob.api_key_id.is_(None) if api_key_id is None else IngestionJob.api_key_id == api_key_id
        )
        async with self._factory() as session:
            return await session.scalar(stmt)

    async def get(self, job_id: str) -> dict[str, Any]:
        """Public view of a job."""
        async with self._factory() as session:
            job = await session.get(IngestionJob, _uuid(job_id))
            if job is None:
                raise NotFoundError(f"Job {job_id} not found")
            return job_view(job)

    async def claim(self) -> IngestionJob | None:
        """Atomically claim the oldest queued job (safe with many workers)."""
        async with self._factory() as session, session.begin():
            job = await session.scalar(
                select(IngestionJob)
                .where(IngestionJob.status == "queued")
                .order_by(IngestionJob.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            now = datetime.now(UTC)
            job.status = "running"
            job.attempts += 1
            job.started_at = now
            job.heartbeat_at = now
            job.stage = "starting"
        return job

    async def progress(self, job_id: uuid.UUID, stage: str, fraction: float) -> None:
        """Update progress and heartbeat."""
        async with self._factory() as session, session.begin():
            await session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(stage=stage, progress=fraction, heartbeat_at=datetime.now(UTC))
            )

    async def succeed(self, job_id: uuid.UUID, document_id: str | None) -> None:
        """Mark a job done and drop its payload."""
        async with self._factory() as session, session.begin():
            await session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id)
                .values(
                    status="succeeded",
                    stage="done",
                    progress=1.0,
                    payload=None,
                    document_id=uuid.UUID(document_id) if document_id else None,
                    finished_at=datetime.now(UTC),
                    error=None,
                )
            )

    async def fail(self, job_id: uuid.UUID, error: dict[str, Any], *, retryable: bool) -> str:
        """Record a failure; requeue if retryable and attempts remain. Returns the new status."""
        async with self._factory() as session, session.begin():
            job = await session.get(IngestionJob, job_id, with_for_update=True)
            if job is None:  # pragma: no cover - deleted while running
                return "failed"
            requeue = retryable and job.attempts < job.max_attempts
            job.status = "queued" if requeue else "failed"
            job.error = error
            if not requeue:
                job.payload = None
                job.finished_at = datetime.now(UTC)
            return job.status

    async def requeue_stale(self, stale_after_s: float) -> int:
        """Requeue running jobs whose worker stopped heart-beating (crash recovery)."""
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
        async with self._factory() as session, session.begin():
            result = await session.execute(
                update(IngestionJob)
                .where(IngestionJob.status == "running", IngestionJob.heartbeat_at < cutoff)
                .values(status="queued", stage="requeued")
            )
            return int(result.rowcount or 0)  # type: ignore[attr-defined]
