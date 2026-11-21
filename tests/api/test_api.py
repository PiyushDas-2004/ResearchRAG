"""API tests with httpx against the real app, container and PostgreSQL (offline components)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from researchrag.api.app import create_app
from researchrag.api.container import AppContainer, build_container
from researchrag.api.security import RateLimiter
from researchrag.config.settings import Settings
from researchrag.ingestion.jobs import JobWorker
from tests.conftest import corpus_papers, paper_pdf
from tests.unit.test_pipeline import CANARY_PAYLOAD, poisoned_pdf

pytestmark = pytest.mark.db
KEY = "test-key-123"
AUTH = {"X-API-Key": KEY}


@pytest.fixture(scope="module")
async def container(migrated: str) -> AsyncIterator[AppContainer]:
    settings = Settings(
        database_url=migrated,
        api_keys=[SecretStr(KEY)],
        run_worker=False,
        parse_isolation="inline",
        rate_limit_per_minute=10_000,
        max_upload_bytes=2_000_000,
        max_request_bytes=3_000_000,
        log_json=True,
        pipeline_config="configs/default.yaml",
        cors_origins=["http://localhost:8000"],
    )
    c = await build_container(settings)
    yield c
    await c.close()


@pytest.fixture(scope="module")
async def client(container: AppContainer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container.settings, container=container)
    app.state.container = container
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _drain(container: AppContainer) -> None:
    worker = JobWorker(container.jobs, container.stack.ingestion)
    while await worker.run_once():
        pass


def _problem(resp: httpx.Response, status: int) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    assert resp.headers["content-type"].startswith("application/problem+json")
    body: dict[str, Any] = resp.json()
    assert body["status"] == status
    assert body["type"].startswith("https://")
    assert body["request_id"]
    return body


async def _upload(
    client: httpx.AsyncClient, data: bytes, name: str = "paper.pdf", **headers: str
) -> httpx.Response:
    return await client.post(
        "/v1/documents", files={"file": (name, data, "application/pdf")}, headers={**AUTH, **headers}
    )


async def test_health_ready_metrics(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).json()["status"] == "ok"
    ready = await client.get("/ready")
    assert ready.status_code == 200, ready.text
    assert ready.json()["checks"]["migrations"]["current"] == ready.json()["checks"]["migrations"]["head"]
    metrics = await client.get("/metrics")
    assert "researchrag_http_requests_total" in metrics.text
    assert (await client.get("/")).status_code == 200
    assert "ResearchRAG" in (await client.get("/")).text
    assert (await client.get("/static/app.js")).status_code == 200
    openapi = (await client.get("/openapi.json")).json()
    assert "/v1/query/stream" in openapi["paths"]


async def test_auth_and_problem_json(client: httpx.AsyncClient) -> None:
    _problem(await client.get("/v1/documents"), 401)
    _problem(await client.get("/v1/documents", headers={"X-API-Key": "wrong"}), 401)
    ok = await client.get("/v1/documents", headers={"Authorization": f"Bearer {KEY}"})
    assert ok.status_code == 200
    _problem(await client.get("/v1/nope", headers=AUTH), 404)
    body = _problem(await client.post("/v1/query", json={"question": "x"}, headers=AUTH), 422)
    assert body["errors"]
    _problem(
        await client.post("/v1/query", json={"question": "valid question", "bogus": 1}, headers=AUTH), 422
    )
    _problem(await client.get("/v1/documents/not-a-uuid", headers=AUTH), 404)
    _problem(await client.get("/v1/jobs/00000000-0000-0000-0000-000000000000", headers=AUTH), 404)


async def test_request_id_cors_and_security_headers(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health", headers={"X-Request-ID": "req-abcdef123"})
    assert resp.headers["X-Request-ID"] == "req-abcdef123"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    generated = await client.get("/health", headers={"X-Request-ID": "bad id!"})
    assert generated.headers["X-Request-ID"] != "bad id!"
    pre = await client.options(
        "/v1/query", headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST"}
    )
    assert pre.headers.get("access-control-allow-origin") == "http://localhost:8000"


async def test_rate_limit(client: httpx.AsyncClient, container: AppContainer) -> None:
    original = container.limiter
    container.limiter = RateLimiter(2)
    try:
        assert (await client.get("/v1/documents", headers=AUTH)).status_code == 200
        assert (await client.get("/v1/documents", headers=AUTH)).status_code == 200
        resp = await client.get("/v1/documents", headers=AUTH)
        _problem(resp, 429)
        assert int(resp.headers["Retry-After"]) >= 1
    finally:
        container.limiter = original


async def test_upload_validation(client: httpx.AsyncClient) -> None:
    _problem(await _upload(client, b"PK\x03\x04" + b"0" * 500, "paper.pdf"), 415)
    _problem(await _upload(client, b"%PDF-1.4 tiny"), 422)
    _problem(await _upload(client, b"%PDF-1.4\n" + b"0" * 2_100_000), 413)
    big = await client.post(
        "/v1/query",
        content=b"x" * 10,
        headers={**AUTH, "Content-Length": "999999999", "Content-Type": "application/json"},
    )
    _problem(big, 413)
    _problem(await _upload(client, paper_pdf("ir-01"), **{"Idempotency-Key": "bad key with spaces"}), 422)


async def test_upload_ingest_and_document_lifecycle(
    client: httpx.AsyncClient, container: AppContainer
) -> None:
    resp = await _upload(client, paper_pdf("ir-02"), "ir-02.pdf", **{"Idempotency-Key": "upload-ir-02"})
    assert resp.status_code == 202, resp.text
    job = resp.json()["job"]
    assert job["status"] == "queued"
    again = await _upload(client, paper_pdf("ir-02"), "ir-02.pdf", **{"Idempotency-Key": "upload-ir-02"})
    assert again.status_code == 200
    assert again.json()["job"]["id"] == job["id"]
    _problem(await _upload(client, paper_pdf("ir-03"), **{"Idempotency-Key": "upload-ir-02"}), 409)
    await _drain(container)
    done = (await client.get(f"/v1/jobs/{job['id']}", headers=AUTH)).json()
    assert done["status"] == "succeeded"
    assert done["progress"] == 1.0
    doc_id = done["document_id"]
    doc = (await client.get(f"/v1/documents/{doc_id}", headers=AUTH)).json()
    paper = next(p for p in corpus_papers() if p.key == "ir-02")
    assert doc["title"] == paper.title
    assert doc["chunk_count"] > 0
    patched = await client.patch(
        f"/v1/documents/{doc_id}", json={"year": 2020, "venue": "Edited"}, headers=AUTH
    )
    assert patched.json()["year"] == 2020
    _problem(await client.patch(f"/v1/documents/{doc_id}", json={}, headers=AUTH), 422)
    page = (await client.get("/v1/documents?limit=1", headers=AUTH)).json()
    assert page["limit"] == 1
    assert page["total"] >= 1
    assert (await client.delete(f"/v1/documents/{doc_id}", headers=AUTH)).status_code == 204
    _problem(await client.get(f"/v1/documents/{doc_id}", headers=AUTH), 404)
    _problem(await client.delete(f"/v1/documents/{doc_id}", headers=AUTH), 404)


async def test_corrupted_pdf_job_fails_cleanly(client: httpx.AsyncClient, container: AppContainer) -> None:
    corrupted = b"%PDF-1.7\n" + b"\x00\xff garbage stream " * 200
    job = (await _upload(client, corrupted, "broken.pdf")).json()["job"]
    await _drain(container)
    failed = (await client.get(f"/v1/jobs/{job['id']}", headers=AUTH)).json()
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "upload_rejected"
    assert failed["document_id"] is None


@pytest.fixture(scope="module")
async def indexed_api(client: httpx.AsyncClient, container: AppContainer) -> dict[str, str]:
    ids = {}
    for key in ("clim-01", "clim-02", "ir-01"):
        resp = await _upload(client, paper_pdf(key), f"{key}.pdf")
        ids[key] = resp.json()["job"]["id"]
    resp = await _upload(client, poisoned_pdf(), "poisoned.pdf")
    ids["poisoned"] = resp.json()["job"]["id"]
    await _drain(container)
    return ids


async def test_query_returns_verified_citations(
    client: httpx.AsyncClient, container: AppContainer, indexed_api: dict[str, str]
) -> None:
    resp = await client.post(
        "/v1/query",
        json={"question": "What afternoon cooling did the CoolCrown roofs deliver in Harbourside?"},
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not body["abstained"]
    assert body["citations"]
    assert all(c["quote_verified"] and c["page"] >= 1 for c in body["citations"])
    assert body["claims"][0]["verdict"] == "supported"
    trace = (await client.get(f"/v1/traces/{body['trace_id']}", headers=AUTH)).json()
    assert trace["stages"]["passages"]
    assert trace["stages"]["generate"]["prompt"]
    assert trace["stages"]["stage_latency_ms"]["generate"] >= 0
    _problem(await client.get("/v1/traces/00000000-0000-0000-0000-000000000000", headers=AUTH), 404)
    fb = await client.post(
        "/v1/feedback", json={"trace_id": body["trace_id"], "rating": "down", "comment": "test"}, headers=AUTH
    )
    assert fb.status_code == 201
    assert fb.json()["rating"] == "down"
    _problem(
        await client.post(
            "/v1/feedback",
            json={"trace_id": "00000000-0000-0000-0000-000000000000", "rating": "up"},
            headers=AUTH,
        ),
        404,
    )
    low = await container.feedback.low_rated(10)
    assert any(row["trace_id"] == body["trace_id"] for row in low)


async def test_query_abstains_on_off_topic(client: httpx.AsyncClient, indexed_api: dict[str, str]) -> None:
    body = (
        await client.post(
            "/v1/query", json={"question": "Who painted the ceiling of the Sistine Chapel?"}, headers=AUTH
        )
    ).json()
    assert body["abstained"]
    assert body["answer"] == ""
    assert body["abstain_reason"].startswith("Not enough evidence")


async def test_query_with_filters_and_conversation(
    client: httpx.AsyncClient, indexed_api: dict[str, str]
) -> None:
    first = await client.post(
        "/v1/query",
        json={
            "question": "What afternoon cooling did the CoolCrown roofs deliver?",
            "conversation_id": "api-conv",
            "filters": {"year_min": 2023},
        },
        headers=AUTH,
    )
    assert first.status_code == 200
    second = (
        await client.post(
            "/v1/query",
            json={"question": "What about its limitations?", "conversation_id": "api-conv"},
            headers=AUTH,
        )
    ).json()
    assert "coolcrown" in second["standalone_question"].lower()


async def test_poisoned_upload_is_neutralised_end_to_end(
    client: httpx.AsyncClient, indexed_api: dict[str, str]
) -> None:
    body = (
        await client.post(
            "/v1/query",
            json={"question": "Who holds the raw sensor archive of the CoolCrown evaluation?"},
            headers=AUTH,
        )
    ).json()
    assert CANARY_PAYLOAD not in json.dumps(body["answer"])
    assert any("instruction-like" in w for w in body["warnings"])


async def test_sse_stream_order(client: httpx.AsyncClient, indexed_api: dict[str, str]) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    async with client.stream(
        "POST",
        "/v1/query/stream",
        json={"question": "What afternoon cooling did the CoolCrown roofs deliver in Harbourside?"},
        headers=AUTH,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        async for chunk in resp.aiter_text():
            buffer += chunk
        for block in buffer.strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
            events.append((lines["event"], json.loads(lines["data"])))
    names = [e for e, _ in events]
    assert names[0] == "status"
    assert names[-1] == "final"
    assert "token" in names
    assert names.index("token") < names.index("final")
    streamed = "".join(d["text"] for e, d in events if e == "token")
    final = events[-1][1]
    assert final["trace_id"]
    assert streamed == final["answer"]


async def test_sse_reports_errors_as_events(
    client: httpx.AsyncClient, container: AppContainer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from researchrag.errors import ProviderError

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise ProviderError("upstream down")

    monkeypatch.setattr(container.stack.pipeline, "answer", boom)
    async with client.stream(
        "POST", "/v1/query/stream", json={"question": "anything at all"}, headers=AUTH
    ) as resp:
        text = "".join([c async for c in resp.aiter_text()])
    assert "event: error" in text
    assert "upstream down" in text
    _problem(await client.post("/v1/query", json={"question": "anything at all"}, headers=AUTH), 502)

    async def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(container.stack.pipeline, "answer", crash)
    body = _problem(await client.post("/v1/query", json={"question": "anything at all"}, headers=AUTH), 500)
    assert "bug" not in body["detail"]
    async with client.stream(
        "POST", "/v1/query/stream", json={"question": "anything at all"}, headers=AUTH
    ) as resp:
        text = "".join([c async for c in resp.aiter_text()])
    assert "internal_error" in text
