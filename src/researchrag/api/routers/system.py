"""Liveness, readiness and Prometheus metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from researchrag import __version__
from researchrag.api.deps import ContainerDep
from researchrag.storage.db import pgvector_version
from researchrag.storage.migrate import head_revision

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness")
async def health() -> dict[str, str]:
    """The process is up."""
    return {"status": "ok", "version": __version__}


@router.get("/ready", summary="Readiness (database, migrations, pgvector)")
async def ready(request: Request, container: ContainerDep) -> JSONResponse:
    """Ready when the database answers, migrations are at head and pgvector is installed."""
    checks: dict[str, Any] = {}
    ok = True
    try:
        async with container.factory() as session:
            await session.execute(text("SELECT 1"))
            revision = (await session.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            checks["database"] = "ok"
            checks["migrations"] = {"current": revision, "head": head_revision()}
            checks["pgvector"] = ".".join(map(str, await pgvector_version(session)))
            ok = revision == head_revision() and checks["pgvector"] != "0"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
        ok = False
    checks["worker"] = "running" if container.worker is not None else "disabled"
    return JSONResponse(
        {"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503
    )


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
