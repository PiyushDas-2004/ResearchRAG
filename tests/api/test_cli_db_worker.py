"""Database-backed CLI commands and job-worker failure handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update
from typer.testing import CliRunner

from researchrag.api.container import AppContainer, build_container
from researchrag.cli import app
from researchrag.config.settings import Settings, get_settings
from researchrag.errors import ProviderTimeoutError
from researchrag.ingestion.jobs import JobWorker
from researchrag.storage.models import IngestionJob
from tests.conftest import paper_pdf

pytestmark = pytest.mark.db
runner = CliRunner()


@pytest.fixture
def db_env(migrated: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("RR_DATABASE_URL", migrated)
    monkeypatch.setenv("RR_PARSE_ISOLATION", "inline")
    get_settings.cache_clear()
    yield migrated
    get_settings.cache_clear()


def test_cli_db_ingest_query_feedback(db_env: str, tmp_path: Path) -> None:
    assert runner.invoke(app, ["db", "migrate"]).exit_code == 0
    pdf = tmp_path / "bio-04.pdf"
    pdf.write_bytes(paper_pdf("bio-04"))
    (tmp_path / "junk.pdf").write_bytes(b"not a pdf at all, just text" * 10)
    result = runner.invoke(app, ["ingest", str(tmp_path), "--collection", "cli"])
    assert result.exit_code == 0, result.output
    assert "REJECTED junk.pdf" in result.output
    assert "bio-04.pdf" in result.output
    again = runner.invoke(app, ["ingest", str(pdf)])
    assert "unchanged" in again.output
    result = runner.invoke(
        app, ["query", "How did sleep duration relate to reaction time in the shift workers?"]
    )
    assert result.exit_code == 0, result.output
    assert "trace" in result.output
    result = runner.invoke(app, ["query", "--json", "What is the capital of Mars?"])
    assert json.loads(result.output)["abstained"] is True
    stats = runner.invoke(app, ["db", "stats"])
    assert stats.exit_code == 0, stats.output
    data = json.loads(stats.output)
    assert data["counts"]["documents"] >= 1
    assert any(i["index"].startswith("ix_emb_hnsw_v") for i in data["indexes"])
    out = tmp_path / "cands.jsonl"
    result = runner.invoke(app, ["feedback", "export", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


@pytest.fixture
async def worker_container(migrated: str) -> AppContainer:
    c = await build_container(
        Settings(
            database_url=migrated,
            run_worker=False,
            parse_isolation="inline",
            pipeline_config="configs/default.yaml",
        )
    )
    yield c  # type: ignore[misc]
    await c.close()


async def _enqueue(c: AppContainer, data: bytes, name: str, attempts: int = 2) -> str:
    import hashlib

    job, _ = await c.jobs.enqueue(
        kind="ingest", payload=data, filename=name, content_hash=hashlib.sha256(data).hexdigest(),
        params={}, idempotency_key=None, api_key_id=None, max_attempts=attempts,
    )  # fmt: skip
    return str(job["id"])


async def test_worker_retries_retryable_errors_then_fails(
    worker_container: AppContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = await _enqueue(worker_container, paper_pdf("mat-01"), "mat-01.pdf", attempts=2)
    calls = {"n": 0}

    async def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        raise ProviderTimeoutError("embedding provider timed out")

    monkeypatch.setattr(worker_container.stack.ingestion, "ingest_bytes", flaky)
    worker = JobWorker(worker_container.jobs, worker_container.stack.ingestion)
    assert await worker.run_once()
    assert (await worker_container.jobs.get(job_id))["status"] == "queued"
    assert await worker.run_once()
    final = await worker_container.jobs.get(job_id)
    assert final["status"] == "failed"
    assert final["error"]["code"] == "provider_timeout"
    assert calls["n"] == 2

    async def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bug")

    job2 = await _enqueue(worker_container, paper_pdf("mat-02"), "mat-02.pdf")
    monkeypatch.setattr(worker_container.stack.ingestion, "ingest_bytes", crash)
    assert await worker.run_once()
    assert (await worker_container.jobs.get(job2))["error"]["code"] == "internal_error"


async def test_stale_jobs_are_requeued_and_worker_loop(worker_container: AppContainer) -> None:
    import uuid
    from datetime import UTC, datetime, timedelta

    job_id = await _enqueue(worker_container, paper_pdf("mat-03"), "mat-03.pdf")
    async with worker_container.factory() as session, session.begin():
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == uuid.UUID(job_id))
            .values(status="running", heartbeat_at=datetime.now(UTC) - timedelta(hours=1))
        )
    assert await worker_container.jobs.requeue_stale(60) >= 1
    worker = JobWorker(
        worker_container.jobs, worker_container.stack.ingestion, poll_interval_s=0.05, stale_after_s=600
    )
    worker.start()
    import asyncio

    for _ in range(100):
        if (await worker_container.jobs.get(job_id))["status"] == "succeeded":
            break
        await asyncio.sleep(0.1)
    await worker.stop()
    assert (await worker_container.jobs.get(job_id))["status"] == "succeeded"


async def test_reembed_job_kind(worker_container: AppContainer) -> None:
    job, _ = await worker_container.jobs.enqueue(
        kind="reembed", payload=None, filename=None, content_hash=None, params={}, idempotency_key=None,
        api_key_id=None, max_attempts=1,
    )  # fmt: skip
    worker = JobWorker(worker_container.jobs, worker_container.stack.ingestion)
    assert await worker.run_once()
    assert (await worker_container.jobs.get(str(job["id"])))["status"] == "succeeded"
