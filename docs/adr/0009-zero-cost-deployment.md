# ADR 0009: Zero-cost deployment profile

- Status: accepted
- Date: 2026-11-21

## Context
The owner requires the project to cost nothing to run, test and demo. The first cloud design used
Cloud SQL (an always-on instance, roughly $10+/month) and Cloud Run with CPU always allocated for
the background ingestion worker (instance-time billing).

## Decision
- The default Terraform profile uses an **external free-tier PostgreSQL with pgvector**
  (`database_mode = "external"`, e.g. Neon Free) and creates no Cloud SQL instance.
- Cloud Run uses **request-based billing** (`cpu_idle = true`, min 0 / max 1 instances). Because
  CPU is throttled between requests, ingestion runs **inline** in the upload request
  (`RR_INGEST_MODE=inline`); the background worker mode remains for always-on deployments.
- The default image contains no provider SDKs (`EXTRAS` build argument) and the Artifact Registry
  repository keeps only the newest image, staying inside the free storage allowance.
- An optional 1 USD billing budget alert is created when a billing account id is supplied.
- Paid options (Cloud SQL, always-on CPU, OpenAI/Gemini) remain available but are opt-in and
  documented with their costs.

## Consequences
- A demo URL can run at $0 within free tiers; free tiers change, so docs date the figures.
- Inline ingestion makes uploads slower to return (seconds per paper) and bounded by the request
  timeout; very large PDFs need the background mode.
- Free databases suspend when idle, so the first request after a pause is slower.
