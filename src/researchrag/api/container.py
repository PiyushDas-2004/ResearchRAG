"""Application container: everything the API needs, built once at startup."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from researchrag.api.security import ApiKeyAuth, RateLimiter
from researchrag.config.pipeline import PipelineConfig, load_pipeline_config
from researchrag.config.pricing import load_pricing
from researchrag.config.settings import Settings
from researchrag.factory import Stack, build_stack
from researchrag.ingestion.jobs import JobWorker
from researchrag.ingestion.pdf_parser import ParseLimits
from researchrag.ingestion.service import ParseOptions
from researchrag.llm.base import LLMClient
from researchrag.security.upload import UploadPolicy
from researchrag.storage.db import create_engine, session_factory
from researchrag.storage.migrate import migrate
from researchrag.storage.repositories import (
    PgConversationStore,
    PgDocumentRepository,
    PgFeedbackRepository,
    PgJobRepository,
    PgTraceRepository,
)
from researchrag.storage.search import PgSearchBackend
from researchrag.storage.store import PgDocumentStore


@dataclass
class AppContainer:
    """Long-lived services shared by all requests."""

    settings: Settings
    config: PipelineConfig
    engine: AsyncEngine
    factory: async_sessionmaker[AsyncSession]
    stack: Stack
    documents: PgDocumentRepository
    jobs: PgJobRepository
    traces: PgTraceRepository
    feedback: PgFeedbackRepository
    auth: ApiKeyAuth
    limiter: RateLimiter
    upload_policy: UploadPolicy
    worker: JobWorker | None = None

    async def close(self) -> None:
        """Stop the worker and close the pool (graceful shutdown)."""
        if self.worker is not None:
            await self.worker.stop()
        await self.engine.dispose()


def parse_options(settings: Settings) -> ParseOptions:
    """Parsing limits from settings."""
    return ParseOptions(
        ParseLimits(settings.max_pages, settings.max_extracted_chars, settings.ocr_enabled),
        settings.parse_isolation,
        settings.parse_timeout_s,
        settings.parse_memory_limit_mb,
        UploadPolicy(settings.max_upload_bytes),
    )


async def build_container(
    settings: Settings, *, config: PipelineConfig | None = None, llm: LLMClient | None = None
) -> AppContainer:
    """Create the engine, repositories and pipeline; optionally migrate and start the worker."""
    if settings.run_migrations_on_startup:
        await migrate(settings.database_url)
    engine = create_engine(settings)
    factory = session_factory(engine)
    cfg = config or load_pipeline_config(settings.pipeline_config)
    traces = PgTraceRepository(factory)
    stack = await build_stack(
        cfg,
        store=PgDocumentStore(factory),
        backend=PgSearchBackend(factory),
        traces=traces,
        conversations=PgConversationStore(factory),
        pricing=load_pricing(settings.pricing_file),
        parse=parse_options(settings),
        llm=llm,
        store_prompts=settings.store_prompts_in_traces,
        log_prompts=settings.log_prompts,
    )
    jobs = PgJobRepository(factory)
    container = AppContainer(
        settings=settings,
        config=cfg,
        engine=engine,
        factory=factory,
        stack=stack,
        documents=PgDocumentRepository(factory),
        jobs=jobs,
        traces=traces,
        feedback=PgFeedbackRepository(factory),
        auth=ApiKeyAuth(settings),
        limiter=RateLimiter(settings.rate_limit_per_minute),
        upload_policy=UploadPolicy(settings.max_upload_bytes),
    )
    if settings.run_worker:
        container.worker = JobWorker(
            jobs,
            stack.ingestion,
            poll_interval_s=settings.worker_poll_interval_s,
            stale_after_s=settings.job_stale_after_s,
        )
        container.worker.start()
    return container
