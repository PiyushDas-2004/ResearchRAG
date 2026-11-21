"""Background ingestion worker over the durable PostgreSQL job queue.

Jobs are claimed with ``FOR UPDATE SKIP LOCKED`` (safe with several instances). Retryable
failures (provider timeouts, rate limits) are re-queued up to ``max_attempts``; rejected
uploads fail immediately with a clear error. Jobs whose worker died mid-run are re-queued
after ``stale_after_s`` without a heartbeat, which makes ingestion resumable.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import structlog

from researchrag.errors import ResearchRAGError, RetryableProviderError
from researchrag.ingestion.service import IngestionService
from researchrag.observability.metrics import INGESTION_JOBS
from researchrag.storage.repositories import PgJobRepository

log = structlog.get_logger("researchrag.jobs")


class JobWorker:
    """Polls the job table and runs ingestion jobs one at a time."""

    def __init__(
        self,
        jobs: PgJobRepository,
        ingestion: IngestionService,
        *,
        poll_interval_s: float = 1.0,
        stale_after_s: float = 300.0,
    ) -> None:
        self.jobs = jobs
        self.ingestion = ingestion
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the polling loop in the background."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="researchrag-job-worker")

    async def stop(self, timeout_s: float = 20.0) -> None:
        """Ask the loop to stop and wait for the current job (graceful shutdown)."""
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout_s)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

    async def _loop(self) -> None:
        await self.jobs.requeue_stale(self.stale_after_s)
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except Exception:
                log.exception("job_worker.loop_error")
                worked = False
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)

    async def run_once(self) -> bool:
        """Claim and run one job; returns False when the queue was empty."""
        job = await self.jobs.claim()
        if job is None:
            return False
        job_id: uuid.UUID = job.id
        log.info("job.started", job_id=str(job_id), kind=job.kind, attempt=job.attempts)

        async def progress(stage: str, fraction: float) -> None:
            await self.jobs.progress(job_id, stage, fraction)

        try:
            if job.kind == "reembed":
                count = await self.ingestion.embed_missing(job.params.get("config_hash"))
                await self.jobs.succeed(job_id, None)
                log.info("job.succeeded", job_id=str(job_id), embedded=count)
            else:
                if job.payload is None:
                    raise ResearchRAGError("Job payload missing", detail={"reason": "payload_missing"})
                result = await self.ingestion.ingest_bytes(
                    job.payload, job.filename, collection=job.params.get("collection"), progress=progress
                )
                await self.jobs.succeed(job_id, result.document_id)
                log.info(
                    "job.succeeded",
                    job_id=str(job_id),
                    document_id=result.document_id,
                    created=result.created,
                )
            INGESTION_JOBS.labels(outcome="succeeded").inc()
        except RetryableProviderError as exc:
            status = await self.jobs.fail(job_id, {"code": exc.code, "message": exc.message}, retryable=True)
            INGESTION_JOBS.labels(outcome="retry" if status == "queued" else "failed").inc()
            log.warning("job.failed", job_id=str(job_id), code=exc.code, status=status)
        except ResearchRAGError as exc:
            await self.jobs.fail(
                job_id, {"code": exc.code, "message": exc.message, **exc.detail}, retryable=False
            )
            INGESTION_JOBS.labels(outcome="failed").inc()
            log.warning("job.failed", job_id=str(job_id), code=exc.code)
        except Exception as exc:
            await self.jobs.fail(
                job_id, {"code": "internal_error", "message": type(exc).__name__}, retryable=False
            )
            INGESTION_JOBS.labels(outcome="failed").inc()
            log.exception("job.crashed", job_id=str(job_id))
        return True
