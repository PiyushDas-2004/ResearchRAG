"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from researchrag import __version__
from researchrag.api.container import AppContainer, build_container
from researchrag.api.middleware import RequestContextMiddleware
from researchrag.api.problems import install_problem_handlers
from researchrag.api.routers import documents, feedback, query, system
from researchrag.config.settings import Settings, get_settings
from researchrag.observability.logging import configure_logging
from researchrag.observability.tracing import configure_tracing

UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
DESCRIPTION = """Citation-grounded question answering over academic papers.

Every answer is decomposed into claims that are verified against the cited passages;
unsupported claims are removed or flagged and the service abstains when the indexed papers
do not contain enough evidence. Errors are returned as RFC 7807 `application/problem+json`.
Authenticate with the `X-API-Key` header."""


def create_app(settings: Settings | None = None, *, container: AppContainer | None = None) -> FastAPI:
    """Build the app; the container is created in the lifespan unless one is injected (tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json, redact=settings.redact_pii)
    configure_tracing(settings.otel_service_name, settings.otel_exporter_endpoint, settings.otel_console)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or await build_container(settings)
        try:
            yield
        finally:
            if owned:
                await app.state.container.close()

    app = FastAPI(
        title="ResearchRAG API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "query", "description": "Ask questions and inspect traces"},
            {"name": "documents", "description": "Upload and manage papers"},
            {"name": "feedback", "description": "Rate answers"},
            {"name": "system", "description": "Health and metrics"},
        ],
    )
    install_problem_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=settings.max_request_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type", "Idempotency-Key", "X-Request-ID", "Authorization"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )
    for router in (system.router, query.router, documents.router, feedback.router):
        app.include_router(router)
    if UI_DIR.exists():
        app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return app
