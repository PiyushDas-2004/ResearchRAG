"""Prometheus metrics (exposed at ``/metrics``)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

STAGE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)

REQUESTS = Counter("researchrag_http_requests_total", "HTTP requests", ["method", "route", "status"])
REQUEST_LATENCY = Histogram(
    "researchrag_http_request_seconds", "HTTP request latency", ["method", "route"], buckets=STAGE_BUCKETS
)
STAGE_LATENCY = Histogram(
    "researchrag_stage_seconds", "Pipeline stage latency", ["stage"], buckets=STAGE_BUCKETS
)
TOKENS = Counter("researchrag_tokens_total", "Model tokens", ["model", "kind"])
QUERIES = Counter("researchrag_queries_total", "Answered queries", ["outcome"])
CLAIM_VERDICTS = Counter("researchrag_claim_verdicts_total", "Claim verification verdicts", ["verdict"])
VERIFICATION_FAILURES = Counter(
    "researchrag_verification_failures_total", "Answers with at least one unsupported or contradicted claim"
)
INJECTIONS = Counter(
    "researchrag_injection_flags_total", "Passages or answers flagged by injection defences", ["kind"]
)
INGESTION_JOBS = Counter("researchrag_ingestion_jobs_total", "Ingestion job outcomes", ["outcome"])
COST = Counter("researchrag_cost_usd_total", "Estimated model cost in USD", ["stage"])
