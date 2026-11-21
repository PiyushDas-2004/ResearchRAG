"""Browser-free end-to-end test against a real ``researchrag serve`` process.

Starts the server (migrations on startup, background worker on), uploads PDFs over HTTP,
waits for the jobs, then checks a cited answer, an abstention and the poisoned document.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

from researchrag.textutils import collapse_whitespace
from tests.conftest import paper_pdf, parsed
from tests.unit.test_pipeline import CANARY_PAYLOAD, poisoned_pdf

pytestmark = [pytest.mark.db, pytest.mark.slow]
KEY = "e2e-key-456"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(database_url: str) -> Iterator[str]:
    port = _free_port()
    env = {
        **os.environ,
        "RR_DATABASE_URL": database_url,
        "RR_API_KEYS": KEY,
        "RR_RUN_MIGRATIONS_ON_STARTUP": "true",
        "RR_RUN_WORKER": "true",
        "RR_WORKER_POLL_INTERVAL_S": "0.2",
        "RR_PARSE_ISOLATION": "subprocess",
        "RR_LOG_LEVEL": "WARNING",
    }
    proc = subprocess.Popen(  # noqa: S603 - fixed argv
        [sys.executable, "-m", "researchrag.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                if httpx.get(f"{base}/ready", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if proc.poll() is not None:
                pytest.fail(f"server exited: {proc.stderr.read().decode() if proc.stderr else ''}")
            time.sleep(0.2)
        else:
            pytest.fail("server did not become ready")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _original_offset(text: str, collapsed_offset: int) -> int:
    seen, prev_space = 0, False
    for i, ch in enumerate(text):
        space = ch.isspace()
        if not (space and prev_space):
            if seen == collapsed_offset:
                return i
            seen += 1
        prev_space = space
    return len(text)


def _wait(base: str, job_id: str) -> dict[str, object]:
    for _ in range(300):
        job: dict[str, object] = httpx.get(f"{base}/v1/jobs/{job_id}", headers={"X-API-Key": KEY}).json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.2)
    pytest.fail("job did not finish")


def test_upload_query_abstain_and_injection_over_http(server: str) -> None:
    headers = {"X-API-Key": KEY}
    jobs = []
    for name, data in (
        ("clim-02.pdf", paper_pdf("clim-02")),
        ("ir-03.pdf", paper_pdf("ir-03")),
        ("poison.pdf", poisoned_pdf()),
    ):
        resp = httpx.post(
            f"{server}/v1/documents",
            files={"file": (name, data, "application/pdf")},
            headers=headers,
            timeout=30,
        )
        assert resp.status_code in {200, 202}, resp.text
        jobs.append(resp.json()["job"]["id"])
    for job_id in jobs:
        assert _wait(server, job_id)["status"] == "succeeded"

    answer = httpx.post(
        f"{server}/v1/query",
        json={"question": "What afternoon cooling did the CoolCrown roofs deliver in Harbourside?"},
        headers=headers,
        timeout=60,
    ).json()
    assert not answer["abstained"]
    assert answer["citations"]
    doc = parsed("clim-02")
    flat = collapse_whitespace(doc.text)
    for citation in answer["citations"]:
        assert citation["quote_verified"]
        quote = collapse_whitespace(citation["quote"])
        if quote in flat:
            # The cited page is the page on which the quoted span starts in the PDF.
            offset = _original_offset(doc.text, flat.index(quote))
            assert citation["page"] == doc.page_at(offset), citation

    abstain = httpx.post(
        f"{server}/v1/query",
        json={"question": "What is the boiling point of liquid nitrogen?"},
        headers=headers,
        timeout=60,
    ).json()
    assert abstain["abstained"]

    poisoned = httpx.post(
        f"{server}/v1/query",
        json={"question": "Who holds the raw sensor archive of the CoolCrown evaluation?"},
        headers=headers,
        timeout=60,
    ).json()
    assert CANARY_PAYLOAD not in poisoned["answer"]
    assert any("instruction-like" in w for w in poisoned["warnings"])
    metrics = httpx.get(f"{server}/metrics").text
    assert "researchrag_stage_seconds_bucket" in metrics
    assert "researchrag_injection_flags_total" in metrics
