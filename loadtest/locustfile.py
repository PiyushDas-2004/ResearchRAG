"""Small load test for the offline stack.

    locust -f loadtest/locustfile.py --headless -u 10 -r 5 -t 60s --host http://localhost:8000

Uses questions from the evaluation set; set RR_API_KEY if the server requires a key.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    json.loads(line)["question"]
    for path in sorted(Path(__file__).resolve().parents[1].joinpath("data", "eval").glob("questions_*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
HEADERS = {"X-API-Key": os.environ.get("RR_API_KEY", "dev-local-key")}


class ResearchUser(HttpUser):
    """Mostly asks questions; occasionally lists documents."""

    wait_time = between(0.5, 1.5)

    @task(8)
    def ask(self) -> None:
        self.client.post(
            "/v1/query", json={"question": random.choice(QUESTIONS)}, headers=HEADERS, name="/v1/query"
        )

    @task(1)
    def documents(self) -> None:
        self.client.get("/v1/documents?limit=20", headers=HEADERS, name="/v1/documents")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
